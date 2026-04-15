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
import math
import json
import re
import os
import numpy as np
from rag.nlp import rag_tokenizer
from common.file_utils import get_project_base_directory


class Dealer:
    def __init__(self):
        self.stop_words = set(["请问",
                               "您",
                               "你",
                               "我",
                               "他",
                               "是",
                               "的",
                               "就",
                               "有",
                               "于",
                               "及",
                               "即",
                               "在",
                               "为",
                               "最",
                               "有",
                               "从",
                               "以",
                               "了",
                               "将",
                               "与",
                               "吗",
                               "吧",
                               "中",
                               "#",
                               "什么",
                               "怎么",
                               "哪个",
                               "哪些",
                               "啥",
                               "相关"])

        def load_dict(fnm):
            res = {}
            with open(fnm, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    arr = line.replace("\n", "").split("\t")
                    if len(arr) < 2:
                        res[arr[0]] = 0
                    else:
                        res[arr[0]] = int(arr[1])

            c = 0
            for _, v in res.items():
                c += v
            if c == 0:
                return set(res.keys())
            return res

        fnm = os.path.join(get_project_base_directory(), "rag/res")
        self.ne, self.df = {}, {}
        try:
            with open(os.path.join(fnm, "ner.json"), "r") as f:
                self.ne = json.load(f)
        except Exception:
            logging.warning("Load ner.json FAIL!")
        try:
            self.df = load_dict(os.path.join(fnm, "term.freq"))
        except Exception:
            logging.warning("Load term.freq FAIL!")

    def pretoken(self, txt, num=False, stpwd=True):
        patt = [
            r"[~—\t @#%!<>,\.\?\":;'\{\}\[\]_=\(\)\|，。？》•●○↓《；‘’：“”【¥ 】…￥！、·（）×`&\\/「」\\]"
        ]
        rewt = [
        ]
        for p, r in rewt:
            txt = re.sub(p, r, txt)

        res = []
        for t in rag_tokenizer.tokenize(txt).split():
            tk = t
            if (stpwd and tk in self.stop_words) or (
                    re.match(r"[0-9]$", tk) and not num):
                continue
            for p in patt:
                if re.match(p, t):
                    tk = "#"
                    break
            # tk = re.sub(r"([\+\\-])", r"\\\1", tk)
            if tk != "#" and tk:
                res.append(tk)
        return res

    def token_merge(self, tks):
        def one_term(t):
            # 把“单个字符”或“长度很短的数字/英文片段”视为可以继续合并的碎片。
            # 这些 token 单独参与检索时信息量通常太低，更适合作为更长术语的一部分。
            return len(t) == 1 or re.match(r"[0-9a-z]{1,2}$", t)

        # res 用来收集合并后的结果，i 是当前扫描位置。
        res, i = [], 0
        while i < len(tks):
            # j 负责向后探测，看从 i 开始能连续吃掉多少个“可合并碎片”。
            j = i
            # 特判开头位置：
            # 如果第一个 token 很短，而第二个 token 是长度大于 1 的非英文串，
            # 就直接把前两个 token 合起来。
            # 这样是为了照顾像 “多 工位” 这种场景，避免句首短词被孤立出来。
            if i == 0 and one_term(tks[i]) and len(
                    tks) > 1 and (len(tks[i + 1]) > 1 and not re.match(r"[0-9a-zA-Z]", tks[i + 1])):  # 多 工位
                res.append(" ".join(tks[0:2]))
                i = 2
                continue

            # 从当前位置开始，尽量向后吞并连续的短碎片：
            # - token 不能为空
            # - 不能是停用词
            # - 必须满足 one_term，说明它本身太短，单独保留价值不高
            while j < len(
                    tks) and tks[j] and tks[j] not in self.stop_words and one_term(tks[j]):
                j += 1
            # 如果从 i 到 j 之间连续出现了多个短碎片，说明这里大概率本来是一个被切散的术语。
            if j - i > 1:
                # 如果连续短碎片数量不多，就一次性全并起来。
                # 这样可以尽量保住原本术语的完整性。
                if j - i < 5:
                    res.append(" ".join(tks[i:j]))
                    i = j
                else:
                    # 如果连续碎片太多，就不要整段全拼，否则会形成过长、过噪的 token。
                    # 这里只取前两个先合并，后面的下一轮再继续处理，避免查询词失控变长。
                    res.append(" ".join(tks[i:i + 2]))
                    i = i + 2
            else:
                # 如果当前位置不是一串可合并碎片，就按原样保留当前 token。
                if len(tks[i]) > 0:
                    res.append(tks[i])
                i += 1
        # 最后再过滤一次空字符串，确保输出 token 列表干净可用。
        return [t for t in res if t]

    def ner(self, t):
        if not self.ne:
            return ""
        res = self.ne.get(t, "")
        if res:
            return res

    def split(self, txt):
        tks = []
        for t in re.sub(r"[ \t]+", " ", txt).split():
            if tks and re.match(r".*[a-zA-Z]$", tks[-1]) and \
                    re.match(r".*[a-zA-Z]$", t) and tks and \
                    self.ne.get(t, "") != "func" and self.ne.get(tks[-1], "") != "func":
                tks[-1] = tks[-1] + " " + t
            else:
                tks.append(t)
        return tks

    def weights(self, tks, preprocess=True):
        # 纯数字、金额、小数这类 token 往往有较强区分度，
        # 这里先准备一个模式，后面会把它们识别出来并给予更高权重。
        num_pattern = re.compile(r"[0-9,.]{2,}$")
        # 很短的英文片段（如 a、an、of 的残片）通常信息量低，
        # 后面会故意把它们的权重压得很低，避免噪音过大。
        short_letter_pattern = re.compile(r"[a-z]{1,2}$")
        # 只由数字、点、空格、短横线组成的片段，
        # 常见于编号、日期、型号，频率统计时单独走一套保守逻辑。
        num_space_pattern = re.compile(r"[0-9. -]{2,}$")
        # 只包含英文字母、点、空格、短横线的片段，
        # 一般是英文术语、缩写或型号名，字典里缺失时也需要特殊对待。
        letter_pattern = re.compile(r"[a-z. -]+$")

        def ner(t):
            # 数字类 token 通常在检索里很关键，比如版本号、编号、金额、日期，
            # 所以直接给一个偏高的实体权重。
            if num_pattern.match(t):
                return 2
            # 超短英文 token 往往是停用词、缩写残片或分词噪音，
            # 给极低权重，防止它们在最终查询里“抢戏”。
            if short_letter_pattern.match(t):
                return 0.01
            # 如果没有命名实体识别结果，默认按普通词处理。
            if not self.ne or t not in self.ne:
                return 1
            # 对不同实体类型做经验性加权：
            # 公司、地点、学校、股票等专名通常辨识度更强；
            # 人名和功能词相关实体相对保守一些。
            m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3, "stock": 3,
                 "firstnm": 1}
            return m[self.ne[t]]

        def postag(t):
            # 用词性再做一层启发式调权。
            t = rag_tokenizer.tag(t)
            # 代词、连词、副词通常检索价值低，降权。
            if t in set(["r", "c", "d"]):
                return 0.3
            # 地名、机构名通常更有区分度，升权。
            if t in set(["ns", "nt"]):
                return 3
            # 普通名词通常是主题词，也给较高权重。
            if t in set(["n"]):
                return 2
            # 数字相关标签也偏重要。
            if re.match(r"[0-9-]+", t):
                return 2
            # 其他词性使用默认权重。
            return 1

        def freq(t):
            # 对编号/数字串直接返回一个固定值，
            # 避免它们在通用词频表里因为缺失而被错误压低。
            if num_space_pattern.match(t):
                return 3
            # 从 tokenizer 词典里取词频。
            s = rag_tokenizer.freq(t)
            # 英文术语如果词典里没有，通常不代表它不重要，
            # 更可能只是词典覆盖不到，所以人为给一个较大的频率值。
            if not s and letter_pattern.match(t):
                return 300
            # 非英文且没查到频率时先置 0，后面继续尝试细粒度拆分兜底。
            if not s:
                s = 0

            # 长 token 如果查不到词频，说明它可能是复合词或短语粘连，
            # 就把它细切后看子词频率，再折算回一个较保守的频率值。
            if not s and len(t) >= 4:
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    # 取子词中的最小值再缩小，等于说：
                    # “这个长词至少不像最稀有的子词那么常见，但也不能过分自信”。
                    s = np.min([freq(tt) for tt in s]) / 6.
                else:
                    s = 0

            # 设一个下限，避免极小频率把 idf 放得过高，导致权重失真。
            return max(s, 10)

        def df(t):
            # 数字/编号类 token 的文档频次同样走固定兜底逻辑。
            if num_space_pattern.match(t):
                return 5
            # 优先使用离线统计好的文档频次表。
            if t in self.df:
                return self.df[t] + 3
            # 英文术语即使不在 df 表里，也不直接当成超稀有词，
            # 否则会被过度放大，所以给一个较大的经验值。
            elif letter_pattern.match(t):
                return 300
            # 对较长复合词做细粒度拆分，用子词 df 推一个保守估计。
            elif len(t) >= 4:
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    return max(3, np.min([df(tt) for tt in s]) / 6.)

            # 默认给一个小但非零的 df，避免完全缺统计时数值爆炸。
            return 3

        def idf(s, N):
            # 一个平滑过的 IDF：
            # - s 越小，说明词越稀有，idf 越高
            # - 加 10 和 0.5 是为了避免极端值不稳定
            # - N 是经验语料规模，不要求严格等于真实语料大小
            return math.log10(10 + ((N - s + 0.5) / (s + 0.5)))

        tw = []
        # preprocess=False 表示外部已经完成分词，
        # 这里直接对传入 token 列表打权重，不再做二次预处理。
        if not preprocess:
            # idf1 更像“词频稀有度”，反映这个词本身常不常见。
            idf1 = np.array([idf(freq(t), 10000000) for t in tks])
            # idf2 更像“文档分布稀有度”，反映这个词跨文档是否稀有。
            idf2 = np.array([idf(df(t), 1000000000) for t in tks])
            # 最终基础分数 = 词频稀有度 * 0.3 + 文档稀有度 * 0.7，
            # 再乘上实体权重和词性权重。
            # 这里更偏向 df，是因为检索里“跨文档区分能力”通常比原始词频更稳定。
            wts = (0.3 * idf1 + 0.7 * idf2) * \
                  np.array([ner(t) * postag(t) for t in tks])
            wts = [s for s in wts]
            tw = list(zip(tks, wts))
        else:
            # preprocess=True 时，输入可能还是句子/片段级文本，
            # 所以先做预切分和 token merge，再对拆出来的 token 分别计算权重。
            for tk in tks:
                # pretoken 先做基础切分，token_merge 再把应该合并的片段拼回去，
                # 目的是尽量得到更适合检索的 token 粒度。
                tt = self.token_merge(self.pretoken(tk, True))
                idf1 = np.array([idf(freq(t), 10000000) for t in tt])
                idf2 = np.array([idf(df(t), 1000000000) for t in tt])
                wts = (0.3 * idf1 + 0.7 * idf2) * \
                      np.array([ner(t) * postag(t) for t in tt])
                wts = [s for s in wts]
                tw.extend(zip(tt, wts))

        # 把当前这批 token 的原始分数求和，
        # 后面会归一化成相对权重，方便上层直接拿去做 boost。
        S = np.sum([s for _, s in tw])
        # 返回“token -> 归一化权重”。
        # 这样不管输入长短，权重总和都约等于 1，更适合后续拼查询表达式。
        return [(t, s / S) for t, s in tw]
