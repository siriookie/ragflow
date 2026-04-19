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

from io import BytesIO

from pypdf import PdfReader as pdf2_read

from rag.nlp import find_codec


def get_text(fnm: str, binary=None) -> str:
    txt = ""
    if binary is not None:
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
    else:
        with open(fnm, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                txt += line
    return txt


def extract_pdf_outlines(source):
    # 提取 PDF 的目录/书签大纲。
    # 这里的 source 既可能是文件路径，也可能是 PDF 二进制内容。
    # 返回结果统一是一个扁平列表，每个元素形如：
    # (标题, 层级深度, 起始页号)
    # 这样后续 parser 就不需要再关心 PDF 原始 outline 的树结构。
    try:
        # 已读 pdf2_read(...) 的调用方式：它负责把文件路径或 BytesIO 交给 PDF 读取器打开。
        # 如果 source 是字符串，就按文件路径打开；
        # 否则认为它是二进制内容，先包装成 BytesIO。
        # 这样做是为了兼容“直接传文件”和“传内存中的 PDF 数据”两种输入来源。
        with pdf2_read(source if isinstance(source, str) else BytesIO(source)) as pdf:
            # outlines 用来收集最终拍平后的目录项。
            outlines = []

            def dfs(nodes, depth):
                # 递归遍历 PDF outline 树。
                # PDF 目录天然是树结构：章节下面可能挂子章节，所以这里必须用 DFS 递归展开。
                for node in nodes:
                    # 如果当前 node 是 list，说明这是一个子目录列表，而不是具体目录项。
                    # depth + 1 表示进入下一层层级。
                    if isinstance(node, list):
                        dfs(node, depth + 1)
                    else:
                        # 当前 node 是具体目录项。
                        # /Title 是 PDF 目录标题；
                        # pdf.get_destination_page_number(node) 会解析该目录项实际跳转到哪一页；
                        # +1 是因为 PDF 内部页号通常按 0-based 返回，而对用户展示时希望是 1-based。
                        # 最终把它拍平成 (标题, 深度, 页号) 三元组。
                        outlines.append((node["/Title"], depth, pdf.get_destination_page_number(node) + 1))

            # 从 PDF 的根 outline 开始，初始深度为 0。
            dfs(pdf.outline, 0)
            # 返回拍平后的大纲结果。
            return outlines
    except Exception:
        # 任何异常都返回空列表。
        # 这是一个典型的“元信息 best-effort”策略：
        # 大纲是增强信息，提取失败不应该拖垮正文解析主流程。
        return []
