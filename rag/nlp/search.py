#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import re
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from rag.nlp import rag_tokenizer, query
import numpy as np
from common.doc_store.doc_store_base import MatchDenseExpr, FusionExpr, OrderByExpr, DocStoreConnection
from common.string_utils import remove_redundant_spaces
from common.float_utils import get_float
from common.constants import PAGERANK_FLD, TAG_FLD
from common import settings

from common.misc_utils import thread_pool_exec

def index_name(uid): return f"ragflow_{uid}"


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    async def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        # 通过共享线程池异步调用 embedding 模型。
        # 这样做是因为大多数 embedding SDK 调用都是阻塞的，可能涉及 I/O
        # 或较重的 CPU/GPU 计算；把它们丢到线程池里可以避免卡住事件循环。
        qv, _ = await thread_pool_exec(emb_mdl.encode_queries, txt)

        # 先把 embedding 结果转成 numpy shape，用来校验返回结构。
        # 因为下游文档存储层期望拿到的是“一个查询向量”，而不是一整个 batch。
        shape = np.array(qv).shape

        # 显式拒绝多维输出。
        # 如果模型这里返回的是 batch 形状的数组，后续代码就无法判断
        # 到底应该拿一个向量去搜，还是拿多个向量去搜，生成的 MatchDenseExpr
        # 也会变得含义不清甚至非法。
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")

        # 把每个元素都规范成普通 float。
        # 这样可以避免 numpy 标量、decimal，或者模型 SDK 自己的数值包装类型
        # 混进文档存储适配层；下游更希望拿到一个干净的 Python float 列表。
        embedding_data = [get_float(v) for v in qv]

        # 根据 embedding 维度推导向量字段名。
        # RAGFlow 把不同维度的向量存进不同列里，比如 q_768_vec、q_1024_vec，
        # 所以查询侧必须精确命中和模型输出维度一致的那一列。
        vector_column_name = f"q_{len(embedding_data)}_vec"

        # 构造文档存储适配层要消费的稠密向量检索表达式
        # （例如 ES、Infinity 等底层引擎都会读这个对象）。
        # - 'float' 表示向量元素类型
        # - 'cosine' 表示使用余弦相似度
        # - topk 控制底层向量召回多少候选
        # - similarity 作为底层阈值，用来过滤过弱的匹配
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` is nullable however infinity doesn't support nullable columns.
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd",
                    "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    async def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight: bool | list | None = None,
               rank_feature: dict | None = None
               ):
        # 默认关闭高亮，除非调用方显式要求。
        if highlight is None:
            highlight = False

        # 从请求里提取底层过滤条件，例如 kb_id、doc_id、available_int 等。
        filters = self.get_filters(req)
        # 初始化排序表达式对象。
        orderBy = OrderByExpr()

        # 解析分页参数。
        # 这里是底层搜索分页，不一定等同于最终业务层返回页。
        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        # 确定本次需要从底层取回哪些字段。
        # 默认字段既包含展示用文本，也包含排序、引用、图谱、标签、向量等后续处理所需字段。
        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "chunk_order_int", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks", "doc_type_kwd",
                       "available_int", "content_with_weight", "mom_id", PAGERANK_FLD, TAG_FLD, "row_id()"])
        # `kwds` 用来收集问题关键词，后面生成高亮时会用到。
        kwds = set([])

        # 读取问题文本。
        qst = req.get("question", "")
        # `q_vec` 保存查询向量；只有使用 embedding 检索时才会填充。
        q_vec = []
        # 没有问题文本时，退化成“带过滤条件的列表查询”。
        if not qst:
            # 如果请求要求排序，就按 chunk 在文档中的自然顺序和创建时间排序。
            if req.get("sort"):
                orderBy.asc("chunk_order_int")
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            # 不做全文检索，只按过滤和排序直接查。
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.get_total(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            # 有问题文本时，先确定是否需要高亮字段。
            highlightFields = ["content_ltks", "title_tks"]
            if not highlight:
                highlightFields = []
            elif isinstance(highlight, list):
                highlightFields = highlight
            # 把自然语言问题解析成全文检索表达式和关键词列表。
            matchText, keywords = self.qryr.question(qst, min_match=0.3)
            # 如果没有 embedding 模型，就只能走纯文本检索。
            if emb_mdl is None:
                matchExprs = [matchText]
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                # 有 embedding 模型时，额外构造查询向量表达式。
                matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                # 非 Infinity 引擎需要把向量字段显式取回，供后续重排和相似度计算使用。
                if not settings.DOC_ENGINE_INFINITY:
                    src.append(f"q_{len(q_vec)}_vec")

                # 构造文本检索 + 向量检索的融合表达式。
                # 当前权重固定为 0.05 / 0.95，说明底层搜索阶段更偏向向量召回。
                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
                matchExprs = [matchText, matchDense, fusionExpr]

                # 执行混合检索。
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # 如果一次混合检索完全没有结果，就做一次更宽松的兜底重试。
                if total == 0:
                    # 如果已经限定了 doc_id，说明调用方只关心这些文档。
                    # 这时直接按文档过滤查询，不再坚持文本/向量匹配。
                    if filters.get("doc_id"):
                        res = await thread_pool_exec(self.dataStore.search, src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
                        total = self.dataStore.get_total(res)
                    else:
                        # 否则降低全文匹配要求，并稍微放宽向量相似度阈值，再试一次。
                        matchText, _ = self.qryr.question(qst, min_match=0.1)
                        matchDense.extra_options["similarity"] = 0.17
                        res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                                    orderBy, offset, limit, idx_names, kb_ids,
                                                    rank_feature=rank_feature)
                        total = self.dataStore.get_total(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            # 把主关键词和细粒度分词都收集起来，用于后续高亮和提示。
            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        # 抽取命中的文档 ID 列表。
        ids = self.dataStore.get_doc_ids(res)
        # 关键词集合转成列表，后续用于高亮处理。
        keywords = list(kwds)
        # 生成高亮结果。
        highlight = self.dataStore.get_highlight(res, keywords, "content_with_weight")
        # 提取按文档名聚合的底层聚合结果。
        aggs = self.dataStore.get_aggregation(res, "docnm_kwd")
        # 统一封装搜索结果对象返回。
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.get_fields(res, src + ["_score"]),
            keywords=keywords
        )

    @staticmethod
    def trans2floats(txt):
        # 把以制表符分隔的向量字符串转成 float 列表。
        # 这样做是为了兼容底层向量字段有时以字符串形式存储、而后续相似度计算需要数值数组的情况。
        return [get_float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    # Sentence boundary regex includes Arabic punctuation (، ؛ ؟ ۔)
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            # Sentence boundary regex includes Arabic punctuation (، ؛ ؟ ۔)
            pieces = re.split(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0] * len(ans_v[0])
                logging.warning(
                    "The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" [ID:{c}]"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        # 计算额外排序特征分数。
        # 目前主要是两部分：
        # 1. 文档自身的 PageRank
        # 2. 查询侧 rank_feature 与 chunk 侧 TAG_FLD 的相似度
        rank_fea = []
        pageranks = []
        # 先把每个 chunk 的 PageRank 取出来，作为基础先验分。
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        # 如果查询侧没有额外 rank_feature，就只返回 pagerank。
        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        # 先算查询侧 rank_feature 向量的模长。
        # 注意这里把 PageRank 从归一化计算里排除，因为它不是 tag 相似度的一部分，而是直接加成项。
        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            # 当前 chunk 没有标签特征时，额外特征相似度记为 0。
            if not search_res.field[i].get(TAG_FLD):
                rank_fea.append(0)
                continue
            # 计算查询侧 rank_feature 与当前 chunk 标签特征的点积与模长。
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            # 当前 chunk 标签向量为空时，相似度记为 0。
            if denor == 0:
                rank_fea.append(0)
            else:
                # 这里本质上在算一个余弦相似度。
                rank_fea.append(nor / np.sqrt(denor) / q_denor)
        # 把标签特征分数放大后与 pagerank 相加，形成最终额外排序分。
        return np.array(rank_fea) * 10. + pageranks

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        # 从查询里提取关键词，供词法相似度计算使用。
        _, keywords = self.qryr.question(query)
        # 根据查询向量维度推断 chunk 里的向量字段名。
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        # 收集每个 chunk 的向量表示。
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            # 向量如果是字符串形式，先转成 float 数组。
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        # 没有任何向量可用时，直接返回空分数。
        if not ins_embd:
            return [], [], []

        # 统一把 `important_kwd` 规范成列表。
        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        # 构造每个 chunk 的词法特征 token 列表。
        # 这里对不同来源的 token 做了人工加权：
        # title * 2, important_kwd * 5, question_tks * 6
        # 目的是让更强语义信号在 token 相似度里占更大权重。
        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        # 计算额外排序特征分数，例如标签特征和 pagerank。
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        # 同时计算：
        # 1. 综合相似度 sim
        # 2. 词法相似度 tksim
        # 3. 向量相似度 vtsim
        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        # 最终综合分 = 混合相似度 + 额外排序特征。
        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        # 先从查询里提取关键词，供词法相似度部分使用。
        _, keywords = self.qryr.question(query)

        # 统一把 `important_kwd` 规范成列表，避免后面拼接 token 时出现字符串逐字符展开的问题。
        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        # 构造每个 chunk 的 token 序列。
        # 与 `rerank()` 不同，这里权重更轻，不做 title / important_kwd 的重复放大，
        # 因为主要语义判断会交给独立 rerank 模型完成。
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        # 先计算一个轻量的词法相似度分数。
        tksim = self.qryr.token_similarity(keywords, ins_tw)
        # 再调用独立 rerank 模型，基于 query 与 chunk 文本计算语义相似度。
        # 这里把 token 列表重新拼成字符串，是因为大多数 rerank 模型吃的是纯文本输入。
        vtsim, _ = rerank_mdl.similarity(query, [remove_redundant_spaces(" ".join(tks)) for tks in ins_tw])
        # 计算额外排序特征分数。
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        # 最终综合分 = 词法相似度 * tkweight + rerank 模型分 * vtweight + 额外排序特征。
        # 这一步的作用是把“可解释的词法分”“更强的语义分”“先验特征分”融合成一个总分。
        return tkweight * np.array(tksim) + vtweight * vtsim + rank_fea, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        # 这是一个对底层 `qryr.hybrid_similarity` 的薄封装。
        # 作用是把原始文本先走 tokenizer，再统一交给 queryer 计算混合相似度。
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    async def retrieval(
            self,
            question,
            embd_mdl,
            tenant_ids,
            kb_ids,
            page,
            page_size,
            similarity_threshold=0.2,
            vector_similarity_weight=0.3,
            top=1024,
            doc_ids=None,
            aggs=True,
            rerank_mdl=None,
            highlight=False,
            rank_feature: dict | None = {PAGERANK_FLD: 10},
    ):
        # 初始化统一返回结构。
        # `chunks` 保存当前页真正返回的片段，`doc_aggs` 保存按文档聚合的命中统计。
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        # 没有问题时直接返回空结果，避免无意义检索。
        if not question:
            return ranks

        # 计算一次实际召回并参与重排的候选上限。
        # 这里保证它是 `page_size` 的倍数，并至少为 30，这样分页后的重排结果更稳定。
        RERANK_LIMIT = math.ceil(64 / page_size) * page_size if page_size > 1 else 1
        RERANK_LIMIT = max(30, RERANK_LIMIT)
        # 组装底层搜索请求。
        # 注意这里的 `page/size` 不是最终返回页，而是“候选召回页”，后面还会在重排后再次分页。
        req = {
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "page": math.ceil(page_size * page / RERANK_LIMIT),
            "size": RERANK_LIMIT,
            "question": question,
            "vector": True,
            "topk": top,
            "similarity": similarity_threshold,
            "available_int": 1,
        }

        # 兼容 `tenant_ids` 既可能是字符串，也可能是列表的调用方式。
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        # 先做一次底层召回。
        # 这里真正打到的是索引/向量检索层，返回原始候选集合 `sres`。
        sres = await self.search(req, [index_name(tid) for tid in tenant_ids], kb_ids, embd_mdl, highlight,
                           rank_feature=rank_feature)

        # 如果显式提供了 rerank 模型，并且底层召回有结果，则优先使用外部 rerank 模型重排。
        if rerank_mdl and sres.total > 0:
            sim, tsim, vsim = self.rerank_by_model(
                rerank_mdl,
                sres,
                question,
                1 - vector_similarity_weight,
                vector_similarity_weight,
                rank_feature=rank_feature,
            )
        else:
            # 没有外部 rerank 模型时，按底层引擎特性选择内置排序逻辑。
            if settings.DOC_ENGINE_INFINITY:
                # Infinity 已经在底层做了较好的分数归一和融合，因此这里直接复用 `_score`。
                sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sim = [s if s is not None else 0.0 for s in sim]
                tsim = sim
                vsim = sim
            else:
                # Elastic/OpenSearch 的文本分和向量分还需要在应用层再做一次融合排序。
                sim, tsim, vsim = self.rerank(
                    sres,
                    question,
                    1 - vector_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )

        # 转成 numpy 数组，方便后面统一排序和阈值过滤。
        sim_np = np.array(sim, dtype=np.float64)
        # 没有任何候选分数时，直接返回空结果。
        if sim_np.size == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 按融合分从高到低排序。
        sorted_idx = np.argsort(sim_np * -1)

        # 当 `vector_similarity_weight=0` 时，说明当前基本是纯词法检索。
        # 这时向量相似度阈值没有实际意义，因此把后置阈值降为 0。
        post_threshold = 0.0 if vector_similarity_weight <= 0 else similarity_threshold

        # 如果调用方显式指定了 doc_ids，则跳过相似度阈值过滤。
        # 这样做是因为此时用户要的是“限定在这些文档里检索”，而不是再按分数把这些文档过滤掉。
        if doc_ids:
            post_threshold = 0.0

        # 过滤掉低于阈值的候选，并记录命中总数。
        valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
        filtered_count = len(valid_idx)
        ranks["total"] = int(filtered_count)

        # 全部候选都被阈值裁掉时，直接返回空结果。
        if filtered_count == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 在重排后的候选集合上再做分页。
        # 这样做不是“先分页再排序”，而是“先多召回、先重排、再分页”，排序质量更稳定。
        max_pages = max(RERANK_LIMIT // max(page_size, 1), 1)
        page_index = (page - 1) % max_pages
        begin = page_index * page_size
        end = begin + page_size
        page_idx = valid_idx[begin:end]

        # 动态推断查询向量维度，并确定当前 chunk 中向量字段名。
        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim

        # 组装当前页真正返回的 chunk 列表。
        for i in page_idx:
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")

            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "tag_kwd": chunk.get("tag_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": float(sim_np[i]),
                "vector_similarity": float(vsim[i]),
                "term_similarity": float(tsim[i]),
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
                "mom_id": chunk.get("mom_id", ""),
                "row_id": chunk.get("row_id()"),
            }
            if highlight and sres.highlight:
                # 如果开启高亮，就优先使用高亮内容；否则回退到原始文本。
                if id in sres.highlight:
                    d["highlight"] = remove_redundant_spaces(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)

        # 可选地构造按文档聚合的命中统计。
        # 注意聚合统计基于全部有效候选 `valid_idx`，而不是仅当前页，所以它反映的是整体命中情况。
        if aggs:
            for i in valid_idx:
                id = sres.ids[i]
                chunk = sres.field[id]
                dnm = chunk.get("docnm_kwd", "")
                did = chunk.get("doc_id", "")
                if dnm not in ranks["doc_aggs"]:
                    ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
                ranks["doc_aggs"][dnm]["count"] += 1

            ranks["doc_aggs"] = [
                {
                    "doc_name": k,
                    "doc_id": v["doc_id"],
                    "count": v["count"],
                }
                for k, v in sorted(
                    ranks["doc_aggs"].items(),
                    key=lambda x: x[1]["count"] * -1,
                )
            ]
        else:
            # 如果调用方不需要聚合结果，就返回空列表。
            ranks["doc_aggs"] = []

        # 返回本次检索结果。
        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"],
                   sort_by_position: bool = False):
        condition = {"doc_id": doc_id}

        fields_set = set(fields or [])
        if sort_by_position:
            for need in ("page_num_int", "position_int", "top_int"):
                if need not in fields_set:
                    fields_set.add(need)
        fields = list(fields_set)

        orderBy = OrderByExpr()
        if sort_by_position:
            orderBy.asc("page_num_int")
            orderBy.asc("position_int")
            orderBy.asc("top_int")

        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            limit = min(bs, max_count - p)
            if limit <= 0:
                break
            es_res = self.dataStore.search(fields, [], condition, [], orderBy, p, limit, index_name(tenant_id),
                                           kb_ids)
            dict_chunks = self.dataStore.get_fields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            chunk_count = len(dict_chunks)
            if chunk_count == 0 or chunk_count < limit:
                break
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        if not self.dataStore.index_exist(index_name(tenant_id), kb_ids[0]):
            return []
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.get_aggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.get_aggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []),
                                        keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}
    # retrieval_by_toc 的作用是：
    #
    # 在“第一轮已经召回到一些 chunk”之后，利用文档的目录结构 TOC，再补召回或重排一批更相关的 chunk，让结果更贴近“用户真正关心的章节”。
    #
    # 定义在 search.py (line 731)，调用入口之一在 dialog_service.py (line 789)。
    #
    # 它解决的核心问题
    # 普通向量检索有时只能命中零散段落，但用户的问题其实是针对某个“章节主题”。
    # 这时如果文档提前抽出了 TOC，系统就可以：
    #
    # 先看第一轮召回最像哪篇文档
    # 再读这篇文档的目录
    # 让大模型判断目录里哪些章节和问题最相关
    # 把这些章节对应的 chunk 补进结果，或给已有 chunk 加分
    # 最后重新排序，返回更完整、更成体系的内容
    # 你可以把它理解成“目录增强检索”。
    #
    # 它大致怎么做
    # 结合 search.py (line 731) 这段实现，流程是：
    #
    # 输入当前已经召回的 chunks
    # 这些通常是第一轮向量/关键词检索得到的候选结果。
    #
    # 按文档累计分数
    # 函数会把同一篇文档下多个命中 chunk 的 similarity 累加，选出当前“最相关的那篇文档”。
    # 关键位置在 search.py (line 753)。
    #
    # 读取这篇文档的 TOC chunk
    # 系统会去索引里找这篇文档的目录数据，目录是特殊 chunk，用 toc_kwd == "toc" 标记。
    #
    # 让 LLM 根据“问题 + 目录”挑相关章节
    # 这里调用了 relevant_chunks_with_toc(...)，返回一批目录推荐的 chunk id 和相似度分数。
    # 关键位置在 search.py (line 779)。
    #
    # 对已有结果加分，或把新 chunk 补进来
    # 如果目录挑中的 chunk 本来就在结果里，就提高它的分数。
    # 如果不在，就去底层索引把它取回来，补进结果集。
    #
    # 重新排序，只保留 topn
    # 最终按更新后的 similarity 排序返回。
    # 关键位置在 search.py (line 831)。
    #
    # 举个直观例子
    #
    # 假设一份技术文档的目录是：
    #
    # 1. 产品概述
    # 2. 安装部署
    # 3. 鉴权机制
    # 4. 向量检索流程
    # 5. 常见故障排查
    # 用户问：
    #
    # 向量检索为什么召回不稳定？
    # 第一轮普通检索，可能召回到这些 chunk：
    #
    # chunk A: “召回率受 embedding 模型影响”
    # chunk B: “索引刷新会影响查询结果”
    # chunk C: “向量维度配置说明”
    # 这些 chunk 可能来自同一篇文档，但比较碎。
    #
    # 这时 retrieval_by_toc 会做的事是：
    #
    # 发现这些 chunk 大多来自同一篇文档
    # 读取这篇文档的目录
    # 让模型判断最相关章节是“4. 向量检索流程”和“5. 常见故障排查”
    # 找到这两个目录章节对应的 chunk id
    # 把这些 chunk 补回结果集
    # 于是最后返回给生成模型的上下文，可能变成：
    #
    # 原来命中的 A/B/C
    # 新补进的 D: “向量检索流程总览”
    # 新补进的 E: “召回不稳定的常见原因”
    # 新补进的 F: “embedding、chunk 切分、topk 配置的影响”
    # 这样回答就更容易从“单段命中”升级成“章节级理解”。
    #
    # 再举一个业务文档例子
    #
    # 文档目录：
    #
    # 1. 合同总则
    # 2. 付款条款
    # 3. 违约责任
    # 4. 保密条款
    # 用户问：
    #
    # 这个合同逾期付款怎么处理？
    # 普通检索可能只打到一句“逾期按日计息”。
    # 但 retrieval_by_toc 会意识到这个问题更可能属于“付款条款”和“违约责任”两章，于是把这两个章节的 chunk 一并召回，最终答案就不只是一句话，而是能把：
    #
    # 付款时间
    # 逾期定义
    # 违约责任
    # 利息或罚则
    # 一起交代出来。
    #
    # 它和普通检索的区别
    #
    # 普通检索：
    # 直接按 chunk 文本相似度找片段。
    #
    # retrieval_by_toc：
    # 先用普通检索找候选，再借助目录判断“应该展开看哪一章”。
    #
    # 所以它更像“二阶段增强”，不是替代第一轮检索。
    #
    # 适合什么场景
    #
    # 它特别适合：
    #
    # 长文档
    # 章节结构清晰的 PDF / 手册 / 合同 / 规范文档
    # 用户问题更偏“主题”而不是精确句子匹配
    # 希望召回结果更完整、更成块
    # 也有一个限制：
    # 它只会围绕“当前得分最高的那篇文档”做 TOC 增强，不会同时对很多篇文档都展开，所以它本质上偏向“在最相关主文档内部做纵深补召回”。
    async def retrieval_by_toc(self, query: str, chunks: list[dict], tenant_ids: list[str], chat_mdl, topn: int = 6):
        # 延迟导入目录增强相关逻辑，避免模块加载阶段产生循环依赖。
        from rag.prompts.generator import \
            relevant_chunks_with_toc  # moved from the top of the file to avoid circular import
        # 没有任何候选 chunk 时，目录增强没有对象可扩展，直接返回空。
        if not chunks:
            return []
        # 把租户 ID 转成底层索引名，后面查目录 chunk 和补充正文 chunk 都要用到。
        idx_nms = [index_name(tid) for tid in tenant_ids]
        # ranks 用来累计“每个文档当前命中 chunk 的总相似度”，
        # doc_id2kb_id 用来记录文档属于哪个知识库，方便后面精确回查。
        ranks, doc_id2kb_id = {}, {}
        for ck in chunks:
            # 初始化当前文档的累计得分。
            if ck["doc_id"] not in ranks:
                ranks[ck["doc_id"]] = 0
            # 把同一文档下多个命中 chunk 的相似度加总。
            # 这样可以选出“当前最值得继续沿目录扩展”的主文档。
            ranks[ck["doc_id"]] += ck["similarity"]
            # 记录 doc_id -> kb_id 的映射，后面查目录时要带上对应知识库过滤。
            doc_id2kb_id[ck["doc_id"]] = ck["kb_id"]
        # 选出累计得分最高的那个文档，目录增强只在这个文档内部进行。
        # 这样做是为了避免把目录扩展范围放得太宽，稀释当前最相关文档的上下文。
        doc_id = sorted(ranks.items(), key=lambda x: x[1] * -1.)[0][0]
        # 当前主文档所在的知识库 ID。
        kb_ids = [doc_id2kb_id[doc_id]]
        # 去索引里查这个文档对应的目录 chunk。
        # 目录 chunk 通过 toc_kwd == "toc" 标识，核心内容放在 content_with_weight 里。
        # TOC chunk 就是“目录块”。
        #
        # 在 RAGFlow 里，它不是普通正文 chunk，而是一种专门存“文档目录结构”的特殊 chunk，用来做目录增强检索。
        #
        # 你可以把它理解成：
        #
        # 普通 chunk：存正文内容
        # TOC chunk：存这篇文档有哪些章节、每个章节对应哪些 chunk
        # 从代码看，retrieval_by_toc 会专门查 toc_kwd == "toc" 的记录来拿目录信息，位置在 search.py (line 756)。
        #
        # 它里面通常放的不是自然段正文，而是类似这种结构化目录数据：
        #
        # [
        #   {"title": "1. 产品概述", "ids": ["chunk_1", "chunk_2"]},
        #   {"title": "2. 安装部署", "ids": ["chunk_3", "chunk_4", "chunk_5"]},
        #   {"title": "3. 常见问题", "ids": ["chunk_6", "chunk_7"]}
        # ]
        # 意思是：
        #
        # 目录项 “产品概述” 对应哪些正文 chunk
        # 目录项 “安装部署” 对应哪些正文 chunk
        # 后续如果用户问题更像“安装部署”，系统就能直接把 chunk_3~5 补召回回来
        # 它是怎么来的
        # 在抽取流程里，系统会从文档内容里生成 TOC，然后把 TOC 单独存成一个 chunk。
        # 相关逻辑在 extractor.py (line 40) 和 extractor.py (line 62)。
        #
        # 那里做的事情大致是：
        #
        # 从一批正文 chunk 提取目录
        # 给每个目录项挂上对应的 chunk id 列表
        # 把整份目录序列化后放进一个特殊 chunk 的 content_with_weight
        # 再打上 toc_kwd = "toc" 标记
        # 举个例子
        #
        # 假设一篇员工手册被切成这些正文 chunk：
        #
        # chunk_101: 公司介绍
        # chunk_102: 企业文化
        # chunk_103: 请假制度
        # chunk_104: 病假流程
        # chunk_105: 报销规范
        # 那生成的 TOC chunk 可能是：
        #
        # [
        #   {"title": "一、公司介绍", "ids": ["chunk_101", "chunk_102"]},
        #   {"title": "二、请假制度", "ids": ["chunk_103", "chunk_104"]},
        #   {"title": "三、报销规范", "ids": ["chunk_105"]}
        # ]
        # 如果用户问：
        #
        # 病假怎么申请？
        # 系统先普通检索命中一点内容后，再看 TOC chunk，会发现“二、请假制度”最相关，于是把 chunk_103 和 chunk_104 一起补进来。
        es_res = self.dataStore.search(["content_with_weight"], [], {"doc_id": doc_id, "toc_kwd": "toc"}, [],
                                       OrderByExpr(), 0, 128, idx_nms,
                                       kb_ids)
        # toc 用来汇总目录节点列表。
        toc = []
        # 把搜索结果转成字段字典，便于逐条解析目录内容。
        dict_chunks = self.dataStore.get_fields(es_res, ["content_with_weight"])
        for _, doc in dict_chunks.items():
            try:
                # 每个目录 chunk 的 content_with_weight 存的是 JSON 数组，
                # 这里把它展开后合并成一份完整目录。
                toc.extend(json.loads(doc["content_with_weight"]))
            except Exception as e:
                # 目录 JSON 解析失败时只记日志，不让整个检索流程失败。
                logging.exception(e)
        # 当前文档没有可用目录时，目录增强无法继续，直接返回原始候选。
        if not toc:
            return chunks

        # 让聊天模型根据“用户问题 + 目录结构”挑出最相关的目录项 / chunk id。
        # 先取 topn * 2 个，是为了给后面的合并和去重预留余量。
        ids = await relevant_chunks_with_toc(query, toc, chat_mdl, topn * 2)
        # 模型如果没有挑出任何目录命中项，就保留原始结果。
        if not ids:
            return chunks

        # 默认向量维度兜底值；如果后面读到真实向量字段，会用真实维度覆盖。
        vector_size = 1024
        # 先建立当前结果里已有 chunk 的索引表。
        # 这样目录增强命中的 chunk 如果已经在候选里，只需要加分，不必重复插入。
        id2idx = {ck["chunk_id"]: i for i, ck in enumerate(chunks)}
        for cid, sim in ids:
            # 如果目录增强命中的 chunk 原本就存在，就把目录相关性分直接加到原 similarity 上。
            if cid in id2idx:
                chunks[id2idx[cid]]["similarity"] += sim
                continue
            # 否则去底层索引里把这个 chunk 取回来，补充进结果集。
            chunk = self.dataStore.get(cid, idx_nms[0], kb_ids)
            # 目录里指到但索引里取不到的 chunk，直接跳过。
            if not chunk:
                continue
            # 组装成和普通 retrieval 返回结构一致的 chunk 字典，
            # 这样后续排序和上层消费逻辑无需区分“原始召回”还是“目录补召回”。
            d = {
                "chunk_id": cid,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                # 目录增强只在当前主文档内部扩展，所以这里沿用选出的主 doc_id。
                "doc_id": doc_id,
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                # 目录增强给出的 sim 在这里同时作为综合分、词法分、向量分的占位值。
                # 这样做主要是为了兼容统一结果结构，并不意味着这三种分真的分别算过。
                "similarity": sim,
                "vector_similarity": sim,
                "term_similarity": sim,
                # 先放零向量占位；如果真实 chunk 里存在 *_vec 字段，后面再替换成真实向量。
                "vector": [0.0] * vector_size,
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", "")
            }
            for k in chunk.keys():
                # 如果取到真实向量字段，就同步写回返回结构，并更新当前向量维度。
                if k[-4:] == "_vec":
                    d["vector"] = chunk[k]
                    vector_size = len(chunk[k])
                    break
            # 把目录增强补出的新 chunk 加入候选集合。
            chunks.append(d)

        # 最后按更新后的 similarity 重新排序，并只保留 topn 条结果。
        return sorted(chunks, key=lambda x: x["similarity"] * -1)[:topn]
    # 检索阶段命中的多个“子块”合并提升成它们对应的“母块”，这样后续给模型的上下文会更完整，不只是一些碎片段。
    #
    # 它在 search.py (line 1001)。
    #
    # 它解决的问题
    # 很多文档在切分时，会有这种结构：
    #
    # 一个较大的母块
    # 母块下面再切成多个更细的 children chunk
    # 检索时往往先命中的是细粒度子块
    # 但回答问题时，单个子块上下文可能太碎
    # 所以这个函数做的是：
    #
    # 找出命中的子块
    # 按它们的 mom_id 分组
    # 回查对应的母块正文
    # 用这些子块的分数，生成一个新的母块候选
    # 把这个母块放回结果集，再统一排序
    # 你可以把它理解成：
    #
    # “子块负责召回精度，母块负责回答上下文完整性。”
    #
    # 它具体怎么做
    #
    # 看 search.py (line 1007) 往后这段逻辑：
    #
    # 遍历当前命中的 chunks
    # 如果某个 chunk 有 mom_id，说明它只是某个母块下面的子块
    # 把这些子块从原结果里移出去，并按 mom_id 分组
    # 每组子块去存储层 get(id, ...) 查回对应母块
    # 组装一个新的母块结果：
    # content_with_weight 用母块完整正文
    # similarity 用这一组子块相似度的均值
    # important_kwd 合并所有子块关键词
    # 把母块追加回候选结果
    # 最后按相似度排序返回
    # 举个简单例子
    #
    # 假设原始文档里有一个母块 M1，内容是整段“请假制度”，然后它被切成 3 个子块：
    #
    # C1: 年假规则，mom_id = M1
    # C2: 病假流程，mom_id = M1
    # C3: 事假审批，mom_id = M1
    # 用户问：
    #
    # 病假怎么申请？
    # 第一轮检索可能命中的是：
    #
    # C2 相似度 0.91
    # C3 相似度 0.63
    # 这时候如果直接把 C2 和 C3 丢给模型，虽然也能回答，但上下文还是碎的。
    #
    # retrieval_by_children 会做成这样：
    #
    # 发现 C2 和 C3 都有同一个 mom_id = M1
    # 把它们归成一组
    # 回查母块 M1
    # 生成一个新的候选块：
    # {
    #     "chunk_id": "M1",
    #     "content_with_weight": "整段请假制度全文",
    #     "similarity": mean([0.91, 0.63])  # 约 0.77
    # }
    # 最后给模型的，不再只是零碎句子，而是整段“请假制度”的完整内容。
    #
    # 再举一个更像 PDF 的例子
    #
    # 一份技术手册里，一个章节母块是：
    #
    # M20: “4. 向量检索参数说明”
    # 它下面有几个子块：
    #
    # C201: top_k 配置
    # C202: similarity_threshold
    # C203: rerank 策略
    # 用户问：
    #
    # similarity_threshold 应该怎么设置？
    # 第一轮检索很可能只命中 C202。
    # 但真正回答这个问题时，往往还需要看到：
    #
    # 这个参数在整章里的上下文
    # 它和 top_k、rerank 的关系
    # 这一节前后说明
    # 所以 retrieval_by_children 会把 C202 提升成 M20，让模型读到整章母块，而不是只读一句参数说明。
    #
    # 一句话总结
    def retrieval_by_children(self, chunks: list[dict], tenant_ids: list[str]):
        # 如果没有任何候选 chunk，就没有子块可合并，直接返回空列表。
        if not chunks:
            return []
        # 把 tenant_id 转换成底层索引名，后面回查父块时会用到。
        idx_nms = [index_name(tid) for tid in tenant_ids]
        # mom_chunks 用来按父块 ID 分组保存所有命中的子块。
        mom_chunks = defaultdict(list)
        # 用 while + pop 的方式原地扫描并移除带 mom_id 的子块。
        i = 0
        # 遍历当前候选结果，找出哪些是“有父块”的子 chunk。
        while i < len(chunks):
            # 取出当前位置的候选 chunk。
            ck = chunks[i]
            # mom_id 表示当前 chunk 对应的父块 ID。
            mom_id = ck.get("mom_id")
            # 如果没有合法的父块 ID，说明它本身就是普通块或父块，保留在原列表中继续向后扫描。
            if not isinstance(mom_id, str) or not mom_id.strip():
                i += 1
                continue
            # 如果有父块 ID，就把该子块从原结果集中移出，并按 mom_id 归入对应父块分组。
            mom_chunks[ck["mom_id"]].append(chunks.pop(i))

        # 如果最终一个需要提升为父块的分组都没有，说明输入里没有子块，直接返回原结果。
        if not mom_chunks:
            return chunks

        # 理论上此时 chunks 里保留的是原本没有 mom_id 的候选块。
        # 如果全被移走了，就显式重置为空列表，便于后面 append 父块结果。
        if not chunks:
            chunks = []

        # 默认向量维度占位值；如果后面从子块里读到真实向量，会用真实维度覆盖。
        vector_size = 1024
        # 逐个父块分组处理，把多个命中的子块合并成一个父块结果。
        for id, cks in mom_chunks.items():
            # 从存储层按父块 ID 回查父块正文内容。
            # kb_id 取该组子块所属知识库列表，兼容底层检索接口。
            chunk = self.dataStore.get(id, idx_nms[0], [ck["kb_id"] for ck in cks])
            # 组装统一的返回结构，把多个子块的信号提升为一个父块候选。
            d = {
                # 父块自己的 chunk_id。
                "chunk_id": id,
                # 把所有命中子块的分词文本拼起来，保留检索语义痕迹。
                "content_ltks": " ".join([ck["content_ltks"] for ck in cks]),
                # 父块的完整正文内容，作为后续生成阶段真正可用的上下文。
                "content_with_weight": chunk["content_with_weight"],
                # 父块所属文档 ID。
                "doc_id": chunk["doc_id"],
                # 文档名，缺省时给空字符串。
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                # 父块所属知识库 ID。
                "kb_id": chunk["kb_id"],
                # 把所有子块的重要关键词合并起来，尽量保留原检索特征。
                "important_kwd": [kwd for ck in cks for kwd in ck.get("important_kwd", [])],
                # 父块关联的图片 ID。
                "image_id": chunk.get("img_id", ""),
                # 父块的综合分数用子块相似度均值表示，避免子块数多的父块天然占优。
                "similarity": np.mean([ck["similarity"] for ck in cks]),
                # 这里沿用同一份均值作为向量分和词项分，占位兼容上层统一结构。
                "vector_similarity": np.mean([ck["similarity"] for ck in cks]),
                "term_similarity": np.mean([ck["similarity"] for ck in cks]),
                # 先放一个默认零向量，后面如发现真实向量字段再替换。
                "vector": [0.0] * vector_size,
                # 父块在原文中的位置坐标。
                "positions": chunk.get("position_int", []),
                # 文档类型字段，便于后续统一处理。
                "doc_type_kwd": chunk.get("doc_type_kwd", "")
            }
            # 尝试从第一个子块里继承真实向量字段。
            # 这里只取第一条，是因为同组子块的向量格式通常一致。
            for k in cks[0].keys():
                # 约定所有向量字段都以 _vec 结尾。
                if k[-4:] == "_vec":
                    # 用真实向量替换默认零向量。
                    d["vector"] = cks[0][k]
                    # 同步记录真实向量维度，供后续父块继续复用。
                    vector_size = len(cks[0][k])
                    break
            # 把合并后的父块结果追加回候选列表。
            chunks.append(d)

        # 最后按相似度降序返回，保证父块提升后仍遵循统一排序规则。
        return sorted(chunks, key=lambda x: x["similarity"] * -1)
