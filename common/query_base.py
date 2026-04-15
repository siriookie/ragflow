#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
import re
from abc import ABC, abstractmethod


class QueryBase(ABC):

    @staticmethod
    def is_chinese(line):
        # 先按空格和制表符切开，得到一组粗粒度 token。
        # 这里不是做严格分词，只是想快速看这段输入里“像英文单词的片段”占比有多少。
        arr = re.split(r"[ \t]+", line)
        # 如果切出来的片段很少，直接按中文场景处理。
        # 这样做偏保守：短查询通常信息不足，没必要因为夹了少量英文就走英文分支。
        if len(arr) <= 3:
            return True
        # e 用来统计“不是纯英文单词”的片段数量。
        # 这些片段通常包括中文、数字混写、符号混写，或者其他非纯英文内容。
        e = 0
        for t in arr:
            # 只要某个片段不是纯英文字母，就记为“更像中文/混合文本”的证据。
            # 这里故意把判断写得简单，是为了低成本快速分流，而不是做语言学上的精确识别。
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1
        # 当“非纯英文片段”占比达到 70% 及以上时，就把整句当成中文查询。
        # 这样能把大多数中文、中文夹少量英文术语、中文数字混写的输入送到中文检索策略里。
        return e * 1.0 / len(arr) >= 0.7

    @staticmethod
    def sub_special_char(line):
        # Strip single quotes first to avoid Infinity's lexer treating them as string delimiters,
        # then escape remaining Infinity/Lucene special characters.
        return re.sub(r"([:\{\}/\[\]\-\*\?\"\(\)\|\+~\^])", r"\\\1", line.replace("'", "")).strip()

    @staticmethod
    def rmWWW(txt):
        patts = [
            (
                r"是*(怎么办|什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
                "",
            ),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
                " ")
        ]
        otxt = txt
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        if not txt:
            txt = otxt
        return txt

    @staticmethod
    def add_space_between_eng_zh(txt):
        # 在“英文或英文+数字”后面紧跟中文时补一个空格。
        # 例如 "RAGFlow2检索" -> "RAGFlow2 检索"。
        # 这样做是为了避免中英文混在一起时被后续 tokenizer 黏成一个整体 token。
        txt = re.sub(r'([A-Za-z]+[0-9]+)([\u4e00-\u9fa5]+)', r'\1 \2', txt)
        # 在“纯英文”后面紧跟中文时补一个空格。
        # 例如 "RAG检索" -> "RAG 检索"。
        # 这是最常见的中英混写场景，拆开后更利于英文词和中文词分别参与检索。
        txt = re.sub(r'([A-Za-z])([\u4e00-\u9fa5]+)', r'\1 \2', txt)
        # 在“中文”后面紧跟“英文或英文+数字”时补一个空格。
        # 例如 "检索RAGFlow2" -> "检索 RAGFlow2"。
        # 需要同时处理反向顺序，否则只处理英文在前的情况会漏掉一半混写文本。
        txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z]+[0-9]+)', r'\1 \2', txt)
        # 在“中文”后面紧跟“纯英文”时补一个空格。
        # 例如 "检索RAG" -> "检索 RAG"。
        # 四条规则合起来，基本覆盖了中英文相邻但中间没有分隔符的常见输入形式。
        txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z])', r'\1 \2', txt)
        # 返回补过空格的结果，让后续分词和权重计算能在更清晰的边界上工作。
        return txt

    @abstractmethod
    def question(self, text, tbl, min_match):
        """
        Returns a query object based on the input text, table, and minimum match criteria.
        """
        raise NotImplementedError("Not implemented")
