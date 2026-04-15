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

import logging
import json
import re
from collections import defaultdict

from common.query_base import QueryBase
from common.doc_store.doc_store_base import MatchTextExpr
from rag.nlp import rag_tokenizer, term_weight, synonym


class FulltextQueryer(QueryBase):
    def __init__(self):
        self.tw = term_weight.Dealer()
        self.syn = synonym.Dealer()
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]

    def question(self, txt, tbl="qa", min_match: float = 0.6):
        # 保留用户原始问题，供日志、调试和界面回显使用。
        # 后面会对查询文本做较强的归一化处理；如果不保存原文，
        # 就很难解释“用户输入”和“最终检索表达式”为什么不同。
        original_query = txt

        # 在中英文交界处补空格，避免混合文本在分词时被黏成一个整体 token。
        # 这样后续的分词、加权和扩展会更稳定。
        txt = self.add_space_between_eng_zh(txt)

        # 去掉 Infinity 会当成查询语法的字符，并把文本归一化到更接近索引侧的形态：
        # - 英文转小写，减少大小写差异
        # - 全角转半角，减少输入法差异
        # - 繁体转简体，减少字形差异
        txt = re.sub(
            r"[ :|\r\n\t,锛屻€傦紵?/`!锛?^%%()\[\]{}<>*~'\"\\]+",
            " ",
            rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
        ).strip()

        # 保留一份归一化后的兜底版本。
        # 如果后面的结构化扩展结果为空，至少还能退回到这个版本做基础检索。
        otxt = txt

        # 提前去掉 when why what
        # 它们通常语义价值很低，却很容易污染 token 权重和关键词扩展。
        txt = self.rmWWW(txt)

        # 中英文查询适合不同的扩展策略：
        # - 英文更适合按 token、同义词和相邻短语扩展；
        # - 中文更依赖分词、语义片段和细粒度 token 扩展。
        if not self.is_chinese(txt):
            # 英文分支再做一次 URL 清理，确保进入 tokenizer 前尽量干净。
            txt = self.rmWWW(txt)
            # 英文分支先做基础分词。
            # 这些 token 一方面用于构造检索查询，另一方面会回传给上层做高亮关键词。
            # 它在做什么
            # tokenize 本质上是在做“把连续文本切成词元边界”。
            #
            # 比如原始输入：
            #
            # txt = "RAGFlow知识库检索怎么做"
            # 在进入 tokenize 之前，前面通常还会先做这些归一化：
            #
            # 小写化
            # 全角转半角
            # 繁体转简体
            # 中英边界补空格
            # 去掉部分噪音字符
            # 所以更接近它真正吃到的文本可能是：
            #
            # "ragflow 知识库检索怎么做"
            # 然后 tokenize(txt) 可能输出类似：
            #
            # "ragflow 知识库 检索 怎么 做"
            # 再 .split() 之后就是：
            #
            # ["ragflow", "知识库", "检索", "怎么", "做"]
            # 为什么要 token 化
            # 因为后面的逻辑都依赖“词”而不是“整句字符流”：
            #
            # weights() 要按 token 算权重
            # syn.lookup(tk) 要按 token 查同义词
            # query string 要按 token 拼 boost
            # 高亮关键词也要按 token 返回
            # 如果不先切开，整句像：
            #
            # "ragflow知识库检索怎么做"
            # 就很难判断：
            #
            # 哪部分是品牌词
            # 哪部分是主题词
            # 哪部分是停用词/问句壳子
            # 具体举几个例子
            # 例子 1：中文技术问题
            # 输入：
            # txt = "知识库权限管理"
            # 可能 token 化后：
            # "知识库 权限 管理"
            # 再 .split()：
            # ["知识库", "权限", "管理"]
            # 这时候后面可以判断：
            # 知识库 是主题词
            # 权限 是关键业务词
            # 管理 是动作词
            # 例子 2：中英混写
            # 输入：
            # txt = "RAGFlow向量检索"
            # 前面先补空格后更像：
            # "ragflow 向量检索"
            # tokenize 可能输出：
            # "ragflow 向量 检索"
            # 最后：
            # ["ragflow", "向量", "检索"]
            # 这样就比把整串当成一个 token 好很多。
            # 如果不拆，可能会变成：
            # ["ragflow向量检索"]
            # 那后面的：
            # 权重计算
            # 同义词扩展
            # 倒排召回
            # 都会差很多。
            #
            # 例子 3：英文短语
            # 输入：
            #
            # txt = "large language model"
            # token 化后通常还是：
            #
            # "large language model"
            # .split() 后：
            #
            # ["large", "language", "model"]
            # 然后后面还能继续补 bigram：
            #
            # "large language"
            # "language model"
            # 这样检索时既保留散词，也保留短语顺序信息。
            #
            # 例子 4：复合中文词
            # 输入：
            #
            # txt = "向量数据库"
            # 可能 token 化后是：
            #
            # "向量 数据库"
            # 得到：
            #
            # ["向量", "数据库"]
            # 这比整块不拆更适合检索，因为文档里很可能也是这两个词分别建索引的。
            #
            # 但如果某些场景切得过碎，后面 token_merge() 还会再做一次修正。
            tks = rag_tokenizer.tokenize(txt).split()
            keywords = [t for t in tks if t]

            # 词权重会转成最终查询串里的 boost。
            # 这样重要词会比普通词更强地影响召回和排序。
            tks_w = self.tw.weights(tks, preprocess=False)

            # 去掉对 query_string 语法敏感的字符，避免 token 本身把查询语义带偏。
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]
            syns = []

            # 第一段在做什么
            # for tk, w in tks_w[:256]:
            #     syn = [rag_tokenizer.tokenize(s).replace("'", "") for s in self.syn.lookup(tk)]
            #     keywords.extend(syn)
            #     syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
            #     syns.append(" ".join(syn))
            # 逐步解释
            # for tk, w in tks_w[:256]:
            # 遍历前 256 个高权重 token。
            #
            # tks_w 里面长这样：
            #
            # [("ragflow", 0.35), ("knowledge", 0.25), ("base", 0.20)]
            # 为什么只取前 256 个：
            #
            # 防止 query 太长
            # 防止同义词扩展爆炸
            # 防止搜索变慢、精度下降
            # self.syn.lookup(tk)
            # 给当前 token 查同义词。
            #
            # 比如：
            #
            # tk = "car"
            # self.syn.lookup("car")
            # # 可能返回 ["automobile", "motorcar"]
            # rag_tokenizer.tokenize(s).replace("'", "")
            # 把同义词也做一次 token 化，并去掉单引号。
            #
            # 比如：
            #
            # "large language model" -> "large language model"
            # "cat-o'-nine-tails" -> "cat-o-nine-tails"
            # 为什么去单引号：
            #
            # 因为底层 Infinity 查询语法里，单引号可能被当成特殊词法符号，导致解析报错。
            #
            # keywords.extend(syn)
            # 把这些同义词也放进 keywords 列表。
            #
            # 这个列表后面可能被拿去：
            #
            # 高亮
            # 关键词展示
            # 上层逻辑复用
            # syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
            # 把每个同义词改造成带权重的查询表达式。
            #
            # 比如：
            #
            # tk = "car", w = 0.8
            # syn = ["automobile", "motorcar"]
            # 处理后变成：
            #
            # ['"automobile"^0.2000', '"motorcar"^0.2000']
            # 注意这里用的是：
            #
            # w / 4.
            # 也就是同义词权重只有原词的 1/4。
            #
            # 为什么这么做：
            #
            # 同义词只是补召回
            # 用户原词才是最可信的意图
            # 不想让扩展词“反客为主”
            # syns.append(" ".join(syn))
            # 把当前 token 的所有同义词表达式拼成一个字符串，存起来。
            #
            # 例如：
            #
            # '"automobile"^0.2000 "motorcar"^0.2000'
            # 第二段在做什么
            # q = ["({}^{:.4f}".format(tk, w) + " {})".format(syn) for (tk, w), syn in zip(tks_w, syns) if
            #      tk and not re.match(r"[.^+\(\)-]", tk)]
            # 这段是在给每个 token 构造一个“局部查询子句”。
            #
            # 它拼出来长什么样
            # 假设：
            #
            # tks_w = [("car", 0.8), ("price", 0.5)]
            # syns = [
            #     '"automobile"^0.2000 "motorcar"^0.2000',
            #     '"cost"^0.1250 "pricing"^0.1250'
            # ]
            # 那么 q 可能变成：
            #
            # [
            #   '(car^0.8000 "automobile"^0.2000 "motorcar"^0.2000)',
            #   '(price^0.5000 "cost"^0.1250 "pricing"^0.1250)'
            # ]
            # 意思可以理解成：
            #
            # car 是主词，权重 0.8
            # automobile / motorcar 是补充词，权重 0.2
            # price 是主词，权重 0.5
            # cost / pricing 是补充词，权重更低
            # 只扩展排在前面的高权重 token，控制查询规模。
            # 同义词确实能补召回，但扩太多会让查询过宽，精度明显下降。
            for tk, w in tks_w[:256]:
                # 去掉同义词里的单引号，避免触发 Infinity 词法解析错误。
                # 同义词权重故意设得更低，它们的职责是补召回，而不是压过用户原词。
                syn = [rag_tokenizer.tokenize(s).replace("'", "") for s in self.syn.lookup(tk)]
                keywords.extend(syn)
                syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
                syns.append(" ".join(syn))

            # 为每个 token 构造一个局部子句：
            # 原词作为主信号，同义词作为较弱的 OR 分支。
            q = ["({}^{:.4f}".format(tk, w) + " {})".format(syn) for (tk, w), syn in zip(tks_w, syns) if
                 tk and not re.match(r"[.^+\(\)-]", tk)]

            # 再补相邻 bigram，保留短语顺序信息。
            # 这样像 "large model" 这种短语，不会退化成两个互不相关的散词。
            # 逐步解释
            # for i in range(1, len(tks_w)):
            # 从第 2 个 token 开始往后遍历。
            #
            # 因为它每次要看一对相邻词：
            #
            # 前一个词 tks_w[i - 1]
            # 当前词 tks_w[i]
            # left, right = ...
            # 取出这两个相邻 token 的文本，并去掉两边空格。
            #
            # 比如：
            #
            # tks_w = [
            #     ("large", 0.4),
            #     ("language", 0.8),
            #     ("model", 0.7)
            # ]
            # 那么第一次循环时：
            #
            # left = "large"
            # right = "language"
            # 第二次循环时：
            #
            # left = "language"
            # right = "model"
            # if not left or not right: continue
            # 如果某个 token 空了，就跳过。
            #
            # 这是防御式写法，避免拼出非法短语。
            #
            # q.append('"%s %s"^%.4f' % (...))
            # 把这两个相邻词拼成一个带 boost 的短语查询。
            #
            # 例如：
            #
            # "large language"^1.6000
            # 或者：
            #
            # "language model"^1.6000
            # 注意这里加了双引号：
            #
            # "large language"
            # 表示它不是两个散词，而是一个短语匹配。
            #
            # max(...) * 2
            # 短语的权重取：
            #
            # max(前一个词权重, 当前词权重) * 2
            # 比如：
            #
            # ("large", 0.4)
            # ("language", 0.8)
            # 那么短语权重就是：
            #
            # max(0.4, 0.8) * 2 = 1.6
            # 也就是说：
            #
            # 短语匹配的权重，故意比单词匹配更高。
            #
            # 为什么要这样做
            # 因为单词匹配只能说明：
            #
            # 文档里出现过这两个词
            # 但不能说明：
            #
            # 这两个词是连着出现的
            # 它们在这个上下文里真的是一个概念
            # 而短语匹配能保留“顺序和邻接关系”。
            #
            # 举个很直观的例子
            # 用户 query：
            #
            # "large language model"
            # 前面单词子句可能已经有：
            #
            # (large^0.4 ...)
            # (language^0.8 ...)
            # (model^0.7 ...)
            # 如果只靠这些散词，下面这种文档也可能匹配：
            #
            # This model is large. The language support is good.
            # 因为它同时包含：
            #
            # large
            # language
            # model
            # 但其实这不是用户要的“large language model”这个概念。
            #
            # 所以这里再补：
            #
            # "large language"^1.6
            # "language model"^1.6
            # 这样文档里真正包含相邻短语的内容，会得到更高分。
            #
            # 更贴近你这个项目的中文例子
            # 假设经过 token 化后：
            #
            # tks_w = [
            #     ("知识库", 0.7),
            #     ("检索", 0.9),
            #     ("优化", 0.5)
            # ]
            # 这段代码会额外加：
            #
            # "知识库 检索"^1.8000
            # "检索 优化"^1.8000
            # 为什么是 1.8：
            #
            # max(0.7, 0.9) * 2 = 1.8
            # max(0.9, 0.5) * 2 = 1.8
            # 这样如果某篇文档里正好有：
            #
            # 知识库检索
            # 检索优化
            # 这种紧邻概念，就会被明显抬高。
            #
            # 它和前面“单词 + 同义词子句”的关系
            # 前面那段代码是在做：
            #
            # 单个 token 的主匹配
            # 同义词的补召回
            # 这一段是在补：
            #
            # 相邻 token 的短语匹配
            # 所以整体是三层信号一起工作：
            #
            # 单词匹配
            # 同义词匹配
            # 相邻短语匹配
            # 这样比只做单词匹配更稳。
            #
            # 为什么只看相邻两个词，不直接整句
            # 因为相邻 bigram 是一个很划算的折中：
            #
            # 比单词更能保住局部语义
            # 比整句短语更不容易过严
            # 数量也可控，不会让 query 爆炸
            # 比如：
            #
            # ["知识库", "向量", "检索", "优化"]
            # 只补 3 个 bigram：
            #
            # "知识库 向量"
            # "向量 检索"
            # "检索 优化"
            # 成本低，但效果通常不错。
            #
            # 如果直接拼整句：
            #
            # "知识库 向量 检索 优化"
            for i in range(1, len(tks_w)):
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                if not left or not right:
                    continue
                q.append(
                    '"%s %s"^%.4f'
                    % (
                        tks_w[i - 1][0],
                        tks_w[i][0],
                        max(tks_w[i - 1][1], tks_w[i][1]) * 2,
                    )
                )

            # 如果清洗后一个可用子句都没剩下，就退回到归一化文本，
            # 至少保留一次粗召回机会。
            if not q:
                q.append(txt)
            query = " ".join(q)
            return MatchTextExpr(
                self.query_fields, query, 100, {"original_query": original_query}
            ), keywords

        def need_fine_grained_tokenize(tk):
            # 太短的 token 继续细切往往会损失语义；
            # 纯数字或纯 ASCII 类 token 本身就比较稳定，也不太受益于中文式细粒度切分。
            if len(tk) < 3:
                return False
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False
            return True

        # 中文分支先去掉 URL 噪音，再做分词和加权。
        txt = self.rmWWW(txt)
        qs, keywords = [], []

        # 先把整句拆成若干语义片段，再分别扩展每个片段。
        # 这样比把整句一次性摊平更能保住局部语义。
        for tt in self.tw.split(txt)[:256]:  # .split():
            if not tt:
                continue
            keywords.append(tt)

            # 给当前片段内部的词加权，让更有信息量的部分主导查询表达式。
            twts = self.tw.weights([tt])

            # 中文里同一个意思经常存在短语级别的替代表达，
            # 所以片段级同义词扩展通常比逐字扩展更有价值。
            syns = self.syn.lookup(tt)
            if syns and len(keywords) < 32:
                keywords.extend(syns)
            logging.debug(json.dumps(twts, ensure_ascii=False))
            tms = []

            # 先处理高权重词，让查询围绕片段里最关键的语义信号展开。
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                sm = (
                    rag_tokenizer.fine_grained_tokenize(tk).split()
                    if need_fine_grained_tokenize(tk)
                    else []
                )

                # 清理细粒度 token，确保它们能安全进入 query_string。
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|锛屻€傦紱鈥樷€欍€愩€戙€侊紒锟μό€︹€︼紙锛麼€斺€斻€娿€嬶紵锛氣€溾€?]+",
                        "",
                        m,
                    )
                    for m in sm
                ]
                sm = [self.sub_special_char(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                # 关键词列表需要控制规模，因为后面还会被高亮和界面逻辑复用；
                # 列表太大只会放大噪音。
                if len(keywords) < 32:
                    keywords.append(re.sub(r"[ \\\"']+", "", tk))
                    keywords.extend(sm)

                # token 级同义词用来补足“说法不同但语义接近”带来的召回缺口。
                tk_syns = self.syn.lookup(tk)
                tk_syns = [self.sub_special_char(s) for s in tk_syns]
                if len(keywords) < 32:
                    keywords.extend([s for s in tk_syns if s])

                # 同义词也要细分一次，尽量和索引侧 token 形态对齐。
                tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
                tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]

                # 关键词池够大就停止扩展，否则查询会变得太宽，伤害精度。
                if len(keywords) >= 32:
                    break

                # 当前主 token 进入查询语法前也要先做安全清理。
                tk = self.sub_special_char(tk)
                if tk.find(" ") > 0:
                    tk = '"%s"' % tk

                # 同义词分支保持比原词更弱，避免“补召回的词”反客为主。
                if tk_syns:
                    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)

                # 如果这个 token 适合进一步切分，就同时补：
                # - 精确的细粒度短语
                # - 允许轻微位置偏移的近邻短语
                # 这样能降低查询侧和索引侧切分不一致导致的漏召回。
                if sm:
                    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

            # 把当前片段内部的加权子句重新合成为一个片段级子句。
            tms = " ".join([f"({t})^{w}" for t, w in tms])

            # 如果片段里有多个加权词，再补一次整片段的近邻匹配，
            # 避免局部词序信息丢失。
            if len(twts) > 1:
                tms += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(tt)

            # 片段级同义词作为较弱的旁路分支。
            # 主表达式仍然优先，同义表达主要负责兜底补召回。
            syns = " OR ".join(
                [
                    '"%s"'
                    % rag_tokenizer.tokenize(self.sub_special_char(s))
                    for s in syns
                ]
            )
            if syns and tms:
                tms = f"({tms})^5 OR ({syns})^0.7"

            # 每个语义片段形成一个局部查询块，最后再把这些块拼成整句查询。
            qs.append(tms)

        if qs:
            # 片段之间用 OR 连接，优先保证召回率。
            # 更细的精排交给后续 rerank 和 hybrid scoring。
            query = " OR ".join([f"({t})" for t in qs if t])
            if not query:
                # 如果扩展后空了，就退回到归一化文本做兜底。
                query = otxt
            return MatchTextExpr(
                self.query_fields, query, 100, {"minimum_should_match": min_match, "original_query": original_query}
            ), keywords

        # 显式返回 None，让调用方知道这次没能构造出可靠的全文查询表达式。
        return None, keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        sims = cosine_similarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        if np.sum(sims[0]) == 0:
            return np.array(tksim), tksim, sims[0]
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        def to_dict(tks):
            # 兼容传入字符串或 token 列表两种形式。
            # 如果是字符串，就先按空格切开，统一成 token 序列再处理。
            if isinstance(tks, str):
                tks = tks.split()
            # d 用来累积“token/bigram -> 权重”的字典表示。
            # 后面比较两个文本相似度时，不直接看原始 token 列表，
            # 而是看它们在这个加权字典空间里有多少重合。
            d = defaultdict(int)
            # 先对 token 序列做一次权重计算。
            # 这里得到的权重已经综合考虑了词频、文档频次、词性、实体类型等因素。
            wts = self.tw.weights(tks, preprocess=False)
            for i, (t, c) in enumerate(wts):
                # 单个 token 本身也作为一个匹配信号放进去，但权重只占 0.4。
                # 这样做是为了给后面的相邻 bigram 留出更大的发言权。
                d[t] += c * 0.4
                # 如果后面还有相邻 token，就额外构造一个 bigram 特征。
                # 这样可以让“相邻词一起出现”比“两个散词分别出现”更有区分度。
                if i+1 < len(wts):
                    _t, _c = wts[i+1]
                    # bigram 权重取两个相邻词中较大的那个，再乘 0.6。
                    # 这里故意让 bigram 比单词更重，是为了保留局部短语语义和词序信息。
                    d[t+_t] += max(c, _c) * 0.6
            return d

        # 先把查询侧 token 转成加权字典。
        atks = to_dict(atks)
        # 再把每个候选 chunk 的 token 序列都转成同样形式的加权字典。
        btkss = [to_dict(tks) for tks in btkss]
        # 逐个比较“查询字典”和“候选字典”的重合程度，输出每个候选的词法相似度。
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v  # * dtwt[k]
        q = 1e-9
        for k, v in qtwt.items():
            q += v  # * v
        return s / q  # math.sqrt(3. * (s / q / math.log10( len(dtwt.keys()) + 512 )))

    def paragraph(self, content_tks: str, keywords: list = [], keywords_topn=30):
        if isinstance(content_tks, str):
            content_tks = [c.strip() for c in content_tks.strip() if c.strip()]
        tks_w = self.tw.weights(content_tks, preprocess=False)

        origin_keywords = keywords.copy()
        keywords = [f'"{k.strip()}"' for k in keywords]
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            tk_syns = self.syn.lookup(tk)
            tk_syns = [self.sub_special_char(s) for s in tk_syns]
            tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
            tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]
            tk = self.sub_special_char(tk)
            if tk.find(" ") > 0:
                tk = '"%s"' % tk
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
            if tk:
                keywords.append(f"{tk}^{w}")

        return MatchTextExpr(self.query_fields, " ".join(keywords), 100,
                             {"minimum_should_match": min(3, round(len(keywords) / 10)),
                              "original_query": " ".join(origin_keywords)})
