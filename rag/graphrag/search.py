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
import asyncio
import json
import logging
from collections import defaultdict
from copy import deepcopy
import json_repair
import pandas as pd

from common.misc_utils import get_uuid
from rag.graphrag.query_analyze_prompt import PROMPTS
from rag.graphrag.utils import get_entity_type2samples, get_llm_cache, set_llm_cache, get_relation
from common.token_utils import num_tokens_from_string

from rag.nlp.search import Dealer, index_name
from common.float_utils import get_float
from common import settings
from common.doc_store.doc_store_base import OrderByExpr


class KGSearch(Dealer):
    async def _chat(self, llm_bdl, system, history, gen_conf):
        response = get_llm_cache(llm_bdl.llm_name, system, history, gen_conf)
        if response:
            return response
        response = await llm_bdl.async_chat(system, history, gen_conf)
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        set_llm_cache(llm_bdl.llm_name, system, response, history, gen_conf)
        return response

    async def query_rewrite(self, llm, question, idxnms, kb_ids):
        # 先从当前知识图谱中取出“实体类型 -> 示例实体”的映射，
        # 给 LLM 一份类型候选池，帮助它更准确地识别问题涉及的实体类型。
        ty2ents = await get_entity_type2samples(idxnms, kb_ids)
        # 用问题文本和类型样本池渲染提示词，
        # 目标是让模型输出 answer_type_keywords 和 entities_from_query 这两个字段。
        hint_prompt = PROMPTS["minirag_query2kwd"].format(query=question,
                                                          TYPE_POOL=json.dumps(ty2ents, ensure_ascii=False, indent=2))
        # 调用内部带缓存的聊天接口，请模型只返回结构化结果。
        result = await self._chat(llm, hint_prompt, [{"role": "user", "content": "Output:"}], {})
        try:
            # 优先按标准 JSON 解析模型返回。
            keywords_data = json_repair.loads(result)
            # 取出模型判断的“答案可能涉及哪些实体类型”。
            type_keywords = keywords_data.get("answer_type_keywords", [])
            # 取出模型从问题里识别出的实体词，并最多保留前 5 个。
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]
            # 返回“实体类型关键词 + 实体关键词”供后续 KG 检索使用。
            return type_keywords, entities_from_query
        except json_repair.JSONDecodeError:
            try:
                # 如果模型输出混入了提示词、角色名或其他噪声文本，
                # 先做一次字符串级清洗，再尽量截取最外层 JSON 对象。
                result = result.replace(hint_prompt[:-1], '').replace('user', '').replace('model', '').strip()
                result = '{' + result.split('{')[1].split('}')[0] + '}'
                # 对清洗后的结果再尝试解析。
                keywords_data = json_repair.loads(result)
                # 再次提取实体类型关键词。
                type_keywords = keywords_data.get("answer_type_keywords", [])
                # 再次提取实体关键词，并限制最多 5 个。
                entities_from_query = keywords_data.get("entities_from_query", [])[:5]
                # 清洗成功后返回解析结果。
                return type_keywords, entities_from_query
            # Handle parsing error
            except Exception as e:
                # 两轮解析都失败时记录异常，交给上层决定如何兜底。
                logging.exception(f"JSON parsing error: {result} -> {e}")
                raise e

    def _ent_info_from_(self, es_res, sim_thr=0.3):
        res = {}
        flds = ["content_with_weight", "_score", "entity_kwd", "rank_flt", "n_hop_with_weight"]
        es_res = self.dataStore.get_fields(es_res, flds)
        for _, ent in es_res.items():
            for f in flds:
                if f in ent and ent[f] is None:
                    del ent[f]
            if get_float(ent.get("_score", 0)) < sim_thr:
                continue
            if isinstance(ent["entity_kwd"], list):
                ent["entity_kwd"] = ent["entity_kwd"][0]
            res[ent["entity_kwd"]] = {
                "sim": get_float(ent.get("_score", 0)),
                "pagerank": get_float(ent.get("rank_flt", 0)),
                "n_hop_ents": json.loads(ent.get("n_hop_with_weight", "[]")),
                "description": ent.get("content_with_weight", "{}")
            }
        return res

    def _relation_info_from_(self, es_res, sim_thr=0.3):
        res = {}
        es_res = self.dataStore.get_fields(es_res, ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd",
                                                   "weight_int"])
        for _, ent in es_res.items():
            if get_float(ent.get("_score", 0)) < sim_thr:
                continue
            f, t = sorted([ent["from_entity_kwd"], ent["to_entity_kwd"]])
            if isinstance(f, list):
                f = f[0]
            if isinstance(t, list):
                t = t[0]
            res[(f, t)] = {
                "sim": get_float(ent.get("_score", 0)),
                "pagerank": get_float(ent.get("weight_int", 0)),
                "description": ent["content_with_weight"]
            }
        return res

    # 作用：
    # 根据一组实体关键词，到知识图谱里的“实体索引”中做向量召回，找出最相关的实体节点。
    #
    # 例子：
    # 如果 keywords = ["诸葛亮", "蜀汉"]，
    # 这个函数会先把 "诸葛亮, 蜀汉" 编码成向量，再到实体索引里搜索，
    # 最终可能返回：
    # {
    #   "诸葛亮": {"sim": 0.92, "pagerank": 0.41, ...},
    #   "蜀汉": {"sim": 0.88, "pagerank": 0.35, ...}
    # }
    # 这些结果会被后续 KG 检索流程继续用于关系扩展和排序。
    def get_relevant_ents_by_keywords(self, keywords, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        # 没有关键词时无法做实体召回，直接返回空结果。
        if not keywords:
            return {}
        # 复制一份过滤条件，避免修改外部传入的原始 filters。
        filters = deepcopy(filters)
        # 只在 knowledge graph 的实体记录中搜索，不查关系或其他图数据。
        filters["knowledge_graph_kwd"] = "entity"
        # 把多个关键词拼成一段文本，再编码成向量检索表达式。
        # sim_thr 会作为向量匹配的最低相似度阈值。
        matchDense = self.get_vector(", ".join(keywords), emb_mdl, 1024, sim_thr)
        # 到底层存储中搜索实体，取回实体描述、实体名和 pagerank 分数字段。
        es_res = self.dataStore.search(["content_with_weight", "entity_kwd", "rank_flt"], [], filters, [matchDense],
                                       OrderByExpr(), 0, N,
                                       idxnms, kb_ids)
        # 把底层搜索结果统一整理成 {实体名: 实体信息} 的结构，并按 sim_thr 过滤低分结果。
        return self._ent_info_from_(es_res, sim_thr)

    # 作用：
    # 根据一段问题文本，到知识图谱里的“关系索引”中做向量召回，找出语义上最相关的关系边。
    #
    # 例子：
    # 如果 txt = "诸葛亮辅佐了哪个政权"，
    # 这个函数会把整句文本编码成向量，再到 relation 记录中搜索，
    # 最终可能返回：
    # {
    #   ("诸葛亮", "蜀汉"): {"sim": 0.89, "pagerank": 12, ...},
    #   ("刘备", "蜀汉"): {"sim": 0.63, "pagerank": 9, ...}
    # }
    # 这些关系会被后续 KG 检索流程继续做重排和上下文拼装。
    def get_relevant_relations_by_txt(self, txt, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        # 没有输入文本时无法做关系语义检索，直接返回空结果。
        if not txt:
            return {}
        # 复制一份过滤条件，避免污染调用方传进来的 filters。
        filters = deepcopy(filters)
        # 只在知识图谱里的 relation 记录上检索，不查 entity 或其他类型数据。
        filters["knowledge_graph_kwd"] = "relation"
        # 把整段问题文本编码成向量检索表达式。
        # sim_thr 会作为召回时的最低相似度阈值。
        matchDense = self.get_vector(txt, emb_mdl, 1024, sim_thr)
        # 执行底层搜索，取回关系描述、相似度、起点实体、终点实体和边权重。
        es_res = self.dataStore.search(
            ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd", "weight_int"],
            [], filters, [matchDense], OrderByExpr(), 0, N, idxnms, kb_ids)
        # 把底层结果统一整理成 {(from_entity, to_entity): 关系信息} 的结构并返回。
        return self._relation_info_from_(es_res, sim_thr)

    def get_relevant_ents_by_types(self, types, filters, idxnms, kb_ids, N=56):
        if not types:
            return {}
        filters = deepcopy(filters)
        filters["knowledge_graph_kwd"] = "entity"
        filters["entity_type_kwd"] = types
        ordr = OrderByExpr()
        ordr.desc("rank_flt")
        es_res = self.dataStore.search(["entity_kwd", "rank_flt"], [], filters, [], ordr, 0, N,
                                       idxnms, kb_ids)
        return self._ent_info_from_(es_res, 0)

    async def retrieval(self, question: str,
               tenant_ids: str | list[str],
               kb_ids: list[str],
               emb_mdl,
               llm,
               max_token: int = 8196,
               ent_topn: int = 6,
               rel_topn: int = 6,
               comm_topn: int = 1,
               ent_sim_threshold: float = 0.3,
               rel_sim_threshold: float = 0.3,
                  **kwargs
               ):
        # 保留原始问题文本，后面实体、关系检索都围绕这个问题展开。
        qst = question
        # 构造知识图谱检索的基础过滤条件，至少限定在当前 kb_ids 范围内。
        filters = self.get_filters({"kb_ids": kb_ids})
        # 兼容 tenant_ids 既可能传单个逗号分隔字符串，也可能直接传列表。
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")
        # 把租户 ID 转换成底层索引名，后续检索实体、关系、社区报告都会用到。
        idxnms = [index_name(tid) for tid in tenant_ids]
        # ty_kwds 用来保存问题改写后识别出的实体类型关键词。
        ty_kwds = []
        try:
            # 先让 LLM 对问题做改写/解析，提取“可能的实体类型”和“显式实体关键词”。
            ty_kwds, ents = await self.query_rewrite(llm, qst, [index_name(tid) for tid in tenant_ids], kb_ids)
            # 记录改写结果，方便排查 KG 检索链路为什么命中了这些实体。
            logging.info(f"Q: {qst}, Types: {ty_kwds}, Entities: {ents}")
        except Exception as e:
            # 改写失败时记录异常，但不中断检索流程。
            logging.exception(e)
            # 兜底把原始问题整体当成一个实体关键词继续检索。
            ents = [qst]
            pass

        # 根据实体关键词做实体召回，拿到“问题里直接提到”的候选实体。
        ents_from_query = self.get_relevant_ents_by_keywords(ents, filters, idxnms, kb_ids, emb_mdl, ent_sim_threshold)
        # 根据识别出的实体类型再补召回一批实体，作为类型先验信号。
        ents_from_types = self.get_relevant_ents_by_types(ty_kwds, filters, idxnms, kb_ids, 10000)
        # 直接根据问题文本检索关系边，找到与问题语义相近的图关系。
        rels_from_txt = self.get_relevant_relations_by_txt(qst, filters, idxnms, kb_ids, emb_mdl, rel_sim_threshold)
        # 保存从实体的 n-hop 邻接路径推出来的候选关系及其分数。
        nhop_pathes = defaultdict(dict)
        # 遍历通过关键词召回到的实体，尝试从它们的 n-hop 邻接信息里推导更多关系。
        for _, ent in ents_from_query.items():
            # n_hop_ents 存的是该实体向外扩展若干跳得到的路径信息。
            nhops = ent.get("n_hop_ents", [])
            # 数据异常时跳过这条实体，避免整个检索失败。
            if not isinstance(nhops, list):
                logging.warning(f"Abnormal n_hop_ents: {nhops}")
                continue
            # 遍历每一条 n-hop 路径候选。
            for nbr in nhops:
                # path 是路径上的实体序列，weights 是每一跳对应的重要性权重。
                path = nbr["path"]
                wts = nbr["weights"]
                # 把路径拆成一条条相邻边，用于累积关系分数。
                for i in range(len(path) - 1):
                    f, t = path[i], path[i + 1]
                    # 如果同一条边已经出现过，就继续累加来自不同实体/路径的相似度贡献。
                    if (f, t) in nhop_pathes:
                        nhop_pathes[(f, t)]["sim"] += ent["sim"] / (2 + i)
                    else:
                        # 新边则初始化一个基于实体相似度衰减后的分数。
                        nhop_pathes[(f, t)]["sim"] = ent["sim"] / (2 + i)
                    # 记录这条边在图中的 pagerank/权重信息。
                    nhop_pathes[(f, t)]["pagerank"] = wts[i]

        # 输出召回摘要，方便理解 KG 检索命中了哪些实体和关系。
        logging.info("Retrieved entities: {}".format(list(ents_from_query.keys())))
        logging.info("Retrieved relations: {}".format(list(rels_from_txt.keys())))
        logging.info("Retrieved entities from types({}): {}".format(ty_kwds, list(ents_from_types.keys())))
        logging.info("Retrieved N-hops: {}".format(list(nhop_pathes.keys())))

        # P(E|Q) => P(E) * P(Q|E) => pagerank * sim
        # 如果某个实体既被关键词召回，又命中了类型约束，就额外提升它的相似度。
        for ent in ents_from_types.keys():
            if ent not in ents_from_query:
                continue
            ents_from_query[ent]["sim"] *= 2

        # 对文本检索到的关系边做二次加权：
        # 同时参考 n-hop 推导结果和实体类型命中情况，增强更可信的边。
        for (f, t) in rels_from_txt.keys():
            # 关系边按无向对处理，便于和 nhop_pathes 中的边匹配。
            pair = tuple(sorted([f, t]))
            s = 0
            # 如果这条边也出现在 n-hop 推导结果里，把该信号加进来并消费掉它。
            if pair in nhop_pathes:
                s += nhop_pathes[pair]["sim"]
                del nhop_pathes[pair]
            # 如果边的起点实体命中了类型约束，再加一份权重。
            if f in ents_from_types:
                s += 1
            # 如果边的终点实体也命中了类型约束，再加一份权重。
            if t in ents_from_types:
                s += 1
            # 用综合信号放大文本检索得到的关系分数。
            rels_from_txt[(f, t)]["sim"] *= s + 1

        # This is for the relations from n-hop but not by query search
        # 对那些“没被文本直接召回，但被 n-hop 推导出来”的关系边，也补进结果集中。
        for (f, t) in nhop_pathes.keys():
            s = 0
            # 如果边两端实体命中了类型约束，就继续提高这条边的分数。
            if f in ents_from_types:
                s += 1
            if t in ents_from_types:
                s += 1
            # 组装成与 rels_from_txt 同结构的候选关系记录。
            rels_from_txt[(f, t)] = {
                "sim": nhop_pathes[(f, t)]["sim"] * (s + 1),
                "pagerank": nhop_pathes[(f, t)]["pagerank"]
            }

        # 以 sim * pagerank 为最终排序分数，截取 top N 实体。
        ents_from_query = sorted(ents_from_query.items(), key=lambda x: x[1]["sim"] * x[1]["pagerank"], reverse=True)[
                          :ent_topn]
        # 同样按 sim * pagerank 排序并截取 top N 关系。
        rels_from_txt = sorted(rels_from_txt.items(), key=lambda x: x[1]["sim"] * x[1]["pagerank"], reverse=True)[
                        :rel_topn]

        # ents 用来保存最终输出到上下文中的实体表。
        ents = []
        # relas 用来保存最终输出到上下文中的关系表。
        relas = []
        # 依次组装实体结果，并控制总 token 不超过预算。
        for n, ent in ents_from_query:
            ents.append({
                # 实体名称。
                "Entity": n,
                # 实体最终得分，格式化成两位小数。
                "Score": "%.2f" % (ent["sim"] * ent["pagerank"]),
                # 实体描述通常以 JSON 字符串存储，这里尽量取出 description 字段。
                "Description": json.loads(ent["description"]).get("description", "") if ent["description"] else ""
            })
            # 每加入一条实体，就扣减一次剩余 token 预算。
            max_token -= num_tokens_from_string(str(ents[-1]))
            # 如果预算耗尽，就丢掉刚刚添加的这一条并停止继续追加。
            if max_token <= 0:
                ents = ents[:-1]
                break

        # 依次组装关系结果，同样受 max_token 限制。
        for (f, t), rel in rels_from_txt:
            # 某些关系记录本身没有 description，需要回存储层补查。
            if not rel.get("description"):
                # 逐个租户索引尝试获取这条关系的原始描述。
                for tid in tenant_ids:
                    rela = await get_relation(tid, kb_ids, f, t)
                    if rela:
                        break
                else:
                    # 全部租户都没查到关系描述，就跳过这条关系。
                    continue
                # 把补查到的 description 写回当前关系记录。
                rel["description"] = rela["description"]
            # 先取出关系描述原文。
            desc = rel["description"]
            try:
                # 如果描述是 JSON 字符串，就尽量解出 description 字段。
                desc = json.loads(desc).get("description", "")
            except Exception:
                # 不是合法 JSON 时就保留原始字符串。
                pass
            relas.append({
                # 关系起点实体。
                "From Entity": f,
                # 关系终点实体。
                "To Entity": t,
                # 关系最终得分。
                "Score": "%.2f" % (rel["sim"] * rel["pagerank"]),
                # 关系文本描述。
                "Description": desc
            })
            # 每加入一条关系，也同步扣减 token 预算。
            max_token -= num_tokens_from_string(str(relas[-1]))
            # 超预算时移除刚加的这一条并停止继续追加。
            if max_token <= 0:
                relas = relas[:-1]
                break

        # 如果有实体结果，就转成 CSV 文本块，便于直接拼进最终上下文。
        if ents:
            ents = "\n---- Entities ----\n{}".format(pd.DataFrame(ents).to_csv())
        else:
            # 没有实体时返回空字符串，避免污染最终内容。
            ents = ""
        # 关系结果同样转成 CSV 文本块。
        if relas:
            relas = "\n---- Relations ----\n{}".format(pd.DataFrame(relas).to_csv())
        else:
            relas = ""

        # 最终返回一个伪 chunk 结构，方便与普通 RAG 检索结果走统一下游处理链路。
        return {
                # 生成一个临时 chunk_id，标识这份 KG 检索结果。
                "chunk_id": get_uuid(),
                # KG 结果不参与词法检索，所以这里留空。
                "content_ltks": "",
                # 最终内容由实体表、关系表和社区报告三部分拼接而成。
                "content_with_weight": ents + relas + self._community_retrieval_([n for n, _ in ents_from_query], filters, kb_ids, idxnms,
                                                        comm_topn, max_token),
                # KG 结果不是某篇具体文档里的 chunk，因此 doc_id 置空。
                "doc_id": "",
                # 给这个伪文档一个固定名称，便于上层展示来源。
                "docnm_kwd": "Related content in Knowledge Graph",
                # 记录当前关联的知识库范围。
                "kb_id": kb_ids,
                # 暂无额外重要关键词。
                "important_kwd": [],
                # KG 结果不对应图片。
                "image_id": "",
                # 作为增强结果，直接给一个固定高分。
                "similarity": 1.,
                "vector_similarity": 1.,
                "term_similarity": 0,
                # KG 结果没有向量和位置信息。
                "vector": [],
                "positions": [],
            }

    def _community_retrieval_(self, entities, condition, kb_ids, idxnms, topn, max_token):
        ## Community retrieval
        fields = ["docnm_kwd", "content_with_weight"]
        odr = OrderByExpr()
        odr.desc("weight_flt")
        fltr = deepcopy(condition)
        fltr["knowledge_graph_kwd"] = "community_report"
        fltr["entities_kwd"] = entities
        comm_res = self.dataStore.search(fields, [], fltr, [],
                                         odr, 0, topn, idxnms, kb_ids)
        comm_res_fields = self.dataStore.get_fields(comm_res, fields)
        txts = []
        for ii, (_, row) in enumerate(comm_res_fields.items()):
            obj = json.loads(row["content_with_weight"])
            txts.append("# {}. {}\n## Content\n{}\n## Evidences\n{}\n".format(
                ii + 1, row["docnm_kwd"], obj["report"], obj["evidences"]))
            max_token -= num_tokens_from_string(str(txts[-1]))

        if not txts:
            return ""
        return "\n---- Community Report ----\n" + "\n".join(txts)


if __name__ == "__main__":
    import argparse
    from common.constants import LLMType
    from api.db.services.knowledgebase_service import KnowledgebaseService
    from api.db.services.llm_service import LLMBundle
    from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type, get_model_config_by_id, get_model_config_by_type_and_name
    from rag.nlp import search

    settings.init_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tenant_id', default=False, help="Tenant ID", action='store', required=True)
    parser.add_argument('-d', '--kb_id', default=False, help="Knowledge base ID", action='store', required=True)
    parser.add_argument('-q', '--question', default=False, help="Question", action='store', required=True)
    args = parser.parse_args()

    kb_id = args.kb_id
    llm_config = get_tenant_default_model_by_type(args.tenant_id, LLMType.CHAT)
    llm_bdl = LLMBundle(args.tenant_id, llm_config)
    _, kb = KnowledgebaseService.get_by_id(kb_id)
    if kb.tenant_embd_id:
        embd_model_config = get_model_config_by_id(kb.tenant_embd_id)
    else:
        embd_model_config = get_model_config_by_type_and_name(args.tenant_id, LLMType.EMBEDDING, kb.embd_id)
    embed_bdl = LLMBundle(args.tenant_id, embd_model_config)

    kg = KGSearch(settings.docStoreConn)
    print(asyncio.run(kg.retrieval({"question": args.question, "kb_ids": [kb_id]},
                    search.index_name(kb.tenant_id), [kb_id], embed_bdl, llm_bdl)))
