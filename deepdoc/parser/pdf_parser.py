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

import asyncio
import logging
import math
import os
import random
import re
import sys
import threading
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from io import BytesIO
from timeit import default_timer as timer

import numpy as np
import pdfplumber
import xgboost as xgb
from huggingface_hub import snapshot_download
from PIL import Image
from pypdf import PdfReader as pdf2_read
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common.file_utils import get_project_base_directory
from deepdoc.vision import OCR, AscendLayoutRecognizer, LayoutRecognizer, Recognizer, TableStructureRecognizer
from rag.nlp import rag_tokenizer
from rag.prompts.generator import vision_llm_describe_prompt
from deepdoc.parser.utils import extract_pdf_outlines
from common import settings



from common.misc_utils import thread_pool_exec

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


class RAGFlowPdfParser:
    def __init__(self, **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """

        self.ocr = OCR()
        self.parallel_limiter = None
        if settings.PARALLEL_DEVICES > 1:
            self.parallel_limiter = [asyncio.Semaphore(1) for _ in range(settings.PARALLEL_DEVICES)]

        layout_recognizer_type = os.getenv("LAYOUT_RECOGNIZER_TYPE", "onnx").lower()
        if layout_recognizer_type not in ["onnx", "ascend"]:
            raise RuntimeError("Unsupported layout recognizer type.")

        if hasattr(self, "model_speciess"):
            recognizer_domain = "layout." + self.model_speciess
        else:
            recognizer_domain = "layout"

        if layout_recognizer_type == "ascend":
            logging.debug("Using Ascend LayoutRecognizer")
            self.layouter = AscendLayoutRecognizer(recognizer_domain)
        else:  # onnx
            logging.debug("Using Onnx LayoutRecognizer")
            self.layouter = LayoutRecognizer(recognizer_domain)
        self.tbl_det = TableStructureRecognizer()

        self.updown_cnt_mdl = xgb.Booster()
        # xgboost model is very small; using CPU explicitly
        self.updown_cnt_mdl.set_param({"device": "cpu"})
        logging.info("updown_cnt_mdl initialized on CPU")
        try:
            model_dir = os.path.join(get_project_base_directory(), "rag/res/deepdoc")
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))
        except Exception:
            model_dir = snapshot_download(repo_id="InfiniFlow/text_concat_xgb_v1.0", local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"), local_dir_use_symlinks=False)
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))

        self.page_from = 0
        self.column_num = 1

    def __char_width(self, c):
        return (c["x1"] - c["x0"]) // max(len(c["text"]), 1)

    def __height(self, c):
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]), abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def _y_dis(self, a, b):
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def _match_proj(self, b):
        proj_patt = [
            r"第[零一二三四五六七八九十百]+章",
            r"第[零一二三四五六七八九十百]+[条节]",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[）\)]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]
        return any([re.match(p, b["text"]) for p in proj_patt])

    def _updown_concat_features(self, up, down):
        w = max(self.__char_width(up), self.__char_width(down))
        h = max(self.__height(up), self.__height(down))
        y_dis = self._y_dis(up, down)
        LEN = 6
        tks_down = rag_tokenizer.tokenize(down["text"][:LEN]).split()
        tks_up = rag_tokenizer.tokenize(up["text"][-LEN:]).split()
        tks_all = up["text"][-LEN:].strip() + (" " if re.match(r"[a-zA-Z0-9]+", up["text"][-1] + down["text"][0]) else "") + down["text"][:LEN].strip()
        tks_all = rag_tokenizer.tokenize(tks_all).split()
        fea = [
            up.get("R", -1) == down.get("R", -1),
            y_dis / h,
            down["page_number"] - up["page_number"],
            up["layout_type"] == down["layout_type"],
            up["layout_type"] == "text",
            down["layout_type"] == "text",
            up["layout_type"] == "table",
            down["layout_type"] == "table",
            True if re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"]) else False,
            True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
            True if re.search(r"(^.?[/,?;:\]，。；：’”？！》】）-])", down["text"]) else False,
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[\(（][^\)）]+$", up["text"]) and re.search(r"[\)）]", down["text"]) else False,
            self._match_proj(down),
            True if re.match(r"[A-Z]", down["text"]) else False,
            True if re.match(r"[A-Z]", up["text"][-1]) else False,
            True if re.match(r"[a-z0-9]", up["text"][-1]) else False,
            True if re.match(r"[0-9.%,-]+$", down["text"]) else False,
            up["text"].strip()[-2:] == down["text"].strip()[-2:] if len(up["text"].strip()) > 1 and len(down["text"].strip()) > 1 else False,
            up["x0"] > down["x1"],
            abs(self.__height(up) - self.__height(down)) / min(self.__height(up), self.__height(down)),
            self._x_dis(up, down) / max(w, 0.000001),
            (len(up["text"]) - len(down["text"])) / max(len(up["text"]), len(down["text"])),
            len(tks_all) - len(tks_up) - len(tks_down),
            len(tks_down) - len(tks_up),
            tks_down[-1] == tks_up[-1] if tks_down and tks_up else False,
            max(down["in_row"], up["in_row"]),
            abs(down["in_row"] - up["in_row"]),
            len(tks_down) == 1 and rag_tokenizer.tag(tks_down[0]).find("n") >= 0,
            len(tks_up) == 1 and rag_tokenizer.tag(tks_up[0]).find("n") >= 0,
        ]
        return fea

    @staticmethod
    def sort_X_by_page(arr, threshold):
        # sort using y1 first and then x1
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threshold and arr[j + 1]["top"] < arr[j]["top"] and arr[j + 1]["page_number"] == arr[j]["page_number"]:
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    def _has_color(self, o):
        if o.get("ncs", "") == "DeviceGray":
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and o["non_stroking_color"][0] == 1:
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True

    # CID pattern regex for unmapped font characters from pdfminer
    _CID_PATTERN = re.compile(r"\(cid\s*:\s*\d+\s*\)")

    @staticmethod
    def _is_garbled_char(ch):
        """Check if a single character is garbled (unmappable from PDF font encoding).

        A character is considered garbled if it falls into Unicode Private Use Areas
        or certain replacement/control character ranges that typically indicate
        pdfminer failed to map a CID to a valid Unicode codepoint.
        """
        if not ch:
            return False
        cp = ord(ch)
        if 0xE000 <= cp <= 0xF8FF:
            return True
        if 0xF0000 <= cp <= 0xFFFFF:
            return True
        if 0x100000 <= cp <= 0x10FFFF:
            return True
        if cp == 0xFFFD:
            return True
        if cp < 0x20 and ch not in ('\t', '\n', '\r'):
            return True
        if 0x80 <= cp <= 0x9F:
            return True
        cat = unicodedata.category(ch)
        if cat in ("Cn", "Cs"):
            return True
        return False

    @staticmethod
    def _is_garbled_text(text, threshold=0.5):
        """Check if a text string contains too many garbled characters.

        Examines each character and determines if the overall proportion
        of garbled characters exceeds the given threshold. Also detects
        pdfminer's CID placeholder patterns like '(cid:123)'.
        """
        if not text or not text.strip():
            return False
        if RAGFlowPdfParser._CID_PATTERN.search(text):
            return True
        garbled_count = 0
        total = 0
        for ch in text:
            if ch.isspace():
                continue
            total += 1
            if RAGFlowPdfParser._is_garbled_char(ch):
                garbled_count += 1
        if total == 0:
            return False
        return garbled_count / total >= threshold

    @staticmethod
    def _has_subset_font_prefix(fontname):
        """Check if a font name has a subset prefix (e.g. 'DY1+ZLQDm1-1').

        PDF subset fonts use a 6-letter uppercase tag followed by '+' before
        the actual font name. Some tools use shorter tags (e.g. 'DY1+').
        """
        if not fontname:
            return False
        return bool(re.match(r"^[A-Z0-9]{2,6}\+", fontname))

    @staticmethod
    def _is_garbled_by_font_encoding(page_chars, min_chars=20):
        """Detect garbled text caused by broken font encoding mappings.

        Some PDFs (especially older Chinese standards) embed custom fonts that
        map CJK glyphs to ASCII codepoints. The extracted text appears as
        random ASCII punctuation/symbols instead of actual CJK characters.

        Detection strategy: if a significant proportion of characters come from
        subset-embedded fonts and the page produces overwhelmingly ASCII
        (punctuation, digits, symbols) with virtually no CJK/Hangul/Kana
        characters, the page is likely garbled due to broken font encoding.
        """
        if not page_chars or len(page_chars) < min_chars:
            return False

        subset_font_count = 0
        total_non_space = 0
        ascii_punct_sym = 0
        cjk_like = 0

        for c in page_chars:
            text = c.get("text", "")
            fontname = c.get("fontname", "")
            if not text or text.isspace():
                continue
            total_non_space += 1

            if RAGFlowPdfParser._has_subset_font_prefix(fontname):
                subset_font_count += 1

            cp = ord(text[0])
            if (0x2E80 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF
                    or 0x20000 <= cp <= 0x2FA1F
                    or 0xAC00 <= cp <= 0xD7AF
                    or 0x3040 <= cp <= 0x30FF):
                cjk_like += 1
            elif (0x21 <= cp <= 0x2F or 0x3A <= cp <= 0x40
                    or 0x5B <= cp <= 0x60 or 0x7B <= cp <= 0x7E):
                ascii_punct_sym += 1

        if total_non_space < min_chars:
            return False

        subset_ratio = subset_font_count / total_non_space
        if subset_ratio < 0.3:
            return False

        cjk_ratio = cjk_like / total_non_space
        punct_ratio = ascii_punct_sym / total_non_space
        if cjk_ratio < 0.05 and punct_ratio > 0.4:
            return True

        return False

    def _evaluate_table_orientation(self, table_img, sample_ratio=0.3):
        """
        Evaluate the best rotation orientation for a table image.

        Tests 4 rotation angles (0°, 90°, 180°, 270°) and uses OCR
        confidence scores to determine the best orientation.

        Args:
            table_img: PIL Image object of the table region
            sample_ratio: Sampling ratio for quick evaluation

        Returns:
            tuple: (best_angle, best_img, confidence_scores)
                - best_angle: Best rotation angle (0, 90, 180, 270)
                - best_img: Image rotated to best orientation
                - confidence_scores: Dict of scores for each angle
        """

        rotations = [
            (0, "original"),
            (90, "rotate_90"),  # clockwise 90°
            (180, "rotate_180"),  # 180°
            (270, "rotate_270"),  # clockwise 270° (counter-clockwise 90°)
        ]

        results = {}
        best_score = -1
        best_angle = 0
        best_img = table_img
        score_0 = None

        for angle, name in rotations:
            # Rotate image
            if angle == 0:
                rotated_img = table_img
            else:
                # PIL's rotate is counter-clockwise, use negative angle for clockwise
                rotated_img = table_img.rotate(-angle, expand=True)

            # Convert to numpy array for OCR
            img_array = np.array(rotated_img)

            # Perform OCR detection and recognition
            try:
                ocr_results = self.ocr(img_array)

                if ocr_results:
                    # Calculate average confidence
                    scores = [conf for _, (_, conf) in ocr_results]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    total_regions = len(scores)

                    # Combined score: considers both average confidence and number of regions
                    # More regions + higher confidence = better orientation
                    combined_score = avg_score * (1 + 0.1 * min(total_regions, 50) / 50)
                else:
                    avg_score = 0
                    total_regions = 0
                    combined_score = 0

            except Exception as e:
                logging.warning(f"OCR failed for angle {angle}: {e}")
                avg_score = 0
                total_regions = 0
                combined_score = 0

            results[angle] = {"avg_confidence": avg_score, "total_regions": total_regions, "combined_score": combined_score}
            if angle == 0:
                score_0 = combined_score

            logging.debug(f"Table orientation {angle}°: avg_conf={avg_score:.4f}, regions={total_regions}, combined={combined_score:.4f}")

            if combined_score > best_score:
                best_score = combined_score
                best_angle = angle
                best_img = rotated_img

        # Absolute threshold rule:
        # Only choose non-0° if it exceeds 0° by more than 0.2 and 0° score is below 0.8.
        if best_angle != 0 and score_0 is not None:
            if not (best_score - score_0 > 0.2 and score_0 < 0.8):
                best_angle = 0
                best_img = table_img
                best_score = score_0

        results[best_angle] = results.get(best_angle, {"avg_confidence": 0, "total_regions": 0, "combined_score": 0})

        logging.info(f"Best table orientation: {best_angle}° (score={best_score:.4f})")

        return best_angle, best_img, results

    def _table_transformer_job(self, ZM, auto_rotate=True):
        """
        Process table structure recognition.

        When auto_rotate=True, the complete workflow:
        1. Evaluate table orientation and select the best rotation angle
        2. Use rotated image for table structure recognition (TSR)
        3. Re-OCR the rotated image
        4. Match new OCR results with TSR cell coordinates

        Args:
            ZM: Zoom factor
            auto_rotate: Whether to enable auto orientation correction
        """
        # 进入表格结构识别阶段。
        # 前面的 _layouts_rec() 只知道“这里是一个 table 区域”，
        # 但还不知道表格内部的行、列、表头、跨行跨列单元格。
        # 这一阶段就是把“表格区域”进一步拆成可重建 HTML/Markdown 表格的结构信息。
        logging.debug("Table processing...")
        # imgs 保存裁出来的每个表格小图，后面统一交给 TableStructureRecognizer 做 TSR。
        # pos 保存每张表格图在原 PDF 页里的位置和索引，方便把 TSR 的局部坐标映射回全局坐标。
        imgs, pos = [], []
        # tbcnt 记录每页有多少个表格。
        # 初始放 0，后面会做 np.cumsum，形成按页切分 recos 的前缀和。
        tbcnt = [0]
        # 裁表格图时额外向四周扩 10 像素。
        # 这样做是为了避免版面检测框刚好贴边，导致表格边线、首尾文字或表头被裁掉。
        MARGIN = 10
        # tb_cpns 存放 Table Structure Recognition 识别出的表格组件。
        # 组件包括 table row、table column、table column header、table spanning cell 等。
        self.tb_cpns = []
        # 记录每个表格最终采用的旋转角度和评估分数。
        # 表格方向不正会严重影响行列检测和 OCR，所以这里把旋转信息保存下来供后续重 OCR 和调试使用。
        self.table_rotations = {}  # Store rotation info for each table
        # 保存参与 TSR 的表格图，可能是原图，也可能是自动纠正方向后的旋转图。
        # 后续 _ocr_rotated_tables() 会复用这些图重新 OCR。
        self.rotated_table_imgs = {}  # Store rotated table images

        # page_layout 来自 _layouts_rec()，page_images 来自 __images__()。
        # 两者必须按页一一对应，否则“第 p 页的表格区域”会裁到错误的页图上。
        assert len(self.page_layout) == len(self.page_images)

        # 收集所有表格的版面信息。
        # 这里保留 page、table_index、原 layout 和裁剪坐标，
        # 是为了后面把旋转 OCR 结果、TSR 结果、原页面坐标重新对齐。
        table_layouts = []  # [(page, table_layout, left, top, right, bott), ...]

        # table_index 是跨页递增的全局表格编号。
        # 不能只用页内 j，因为后面 table_rotations、rotated_table_imgs 都需要唯一 key。
        table_index = 0
        for p, tbls in enumerate(self.page_layout):  # for page
            # 每页 layout 里可能有 text/title/figure/header 等多种区域，
            # 表格结构识别只处理 type == "table" 的区域。
            tbls = [f for f in tbls if f["type"] == "table"]
            # 记录当前页表格数量，后面用前缀和把全局 recos 切回“每页的表格列表”。
            tbcnt.append(len(tbls))
            # 当前页没有表格就直接跳过，避免无意义裁图和 TSR 推理。
            if not tbls:
                continue
            for tb in tbls:  # for table
                # 在版面模型给出的 table 框外扩 MARGIN。
                # tb 的坐标仍是 PDF 逻辑坐标，而裁剪 page_images 需要放大后的像素坐标，
                # 所以下面会再乘以 ZM。
                left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, tb["x1"] + MARGIN, tb["bottom"] + MARGIN
                # 乘回 zoomin 后的图像像素坐标。
                # __images__ 把 PDF 页按 72 * ZM 渲染成图，所以 page_images 上的裁剪必须使用像素尺度。
                left *= ZM
                top *= ZM
                right *= ZM
                bott *= ZM
                # 记录表格图左上角、所在页、全局表格编号。
                # TSR 输出的是相对“表格小图”的坐标，后面要靠这些信息还原到页坐标。
                pos.append((left, top, p, table_index))  # Add page and table_index

                # 保存更完整的表格 layout 信息。
                # _ocr_rotated_tables() 需要知道原始 table layout 和裁剪坐标，
                # 才能删掉旧 OCR 框、插入旋转后重新识别的新框。
                table_layouts.append({"page": p, "table_index": table_index, "layout": tb, "coords": (left, top, right, bott)})

                # 从整页图里裁出表格区域，减少 TSR 模型输入范围。
                # 只识别表格小图比整页识别更快，也能降低正文、图片等区域干扰表格结构检测。
                table_img = self.page_images[p].crop((left, top, right, bott))

                if auto_rotate:
                    # 自动评估表格方向。
                    # 已读 _evaluate_table_orientation()：它会尝试 0/90/180/270 四个角度，
                    # 对每个角度跑 OCR，用平均置信度和识别区域数计算综合分，
                    # 只有非 0 度明显优于 0 度时才真正旋转，避免过度纠正。
                    logging.debug(f"Evaluating orientation for table {table_index} on page {p}")
                    best_angle, rotated_img, rotation_scores = self._evaluate_table_orientation(table_img)

                    # 保存方向评估结果。
                    # 这些信息既用于后续重 OCR，也方便日志排查“为什么某个表被旋转了”。
                    self.table_rotations[table_index] = {
                        "page": p,
                        "original_pos": (left, top, right, bott),
                        "best_angle": best_angle,
                        "scores": rotation_scores,
                        "rotated_size": rotated_img.size,  # (width, height)
                    }

                    # 后续 TSR 使用纠正方向后的表格图。
                    # 这样行列结构检测是在“正常阅读方向”上完成的，尤其对横向/竖向旋转表格更稳。
                    self.rotated_table_imgs[table_index] = rotated_img
                    imgs.append(rotated_img)

                else:
                    # 关闭自动旋转时，直接使用原始裁剪图。
                    # 同时仍然写入 rotation 信息，保持后续流程读取字段时结构一致。
                    imgs.append(table_img)
                    self.table_rotations[table_index] = {"page": p, "original_pos": (left, top, right, bott), "best_angle": 0, "scores": {}, "rotated_size": table_img.size}
                    self.rotated_table_imgs[table_index] = table_img

                # 下一个表格使用新的全局编号。
                table_index += 1

        # tbcnt 初始有一个 0，之后每页 append 一次表格数，
        # 所以长度应该等于 page_images 数量 + 1。
        assert len(self.page_images) == len(tbcnt) - 1
        # 如果整份文档没有检测到任何表格，就不用继续跑 TSR。
        if not imgs:
            return

        # 执行表格结构识别 TSR。
        # 已读 TableStructureRecognizer.__call__()：
        # 它会对每张表格图检测 table row、table column、table column header、spanning cell 等结构框；
        # 然后对行框左右边界、列框上下边界做对齐修正，让结构框更规整。
        # 输出 recos 与 imgs 一一对应：每个表格图对应一组结构组件。
        recos = self.tbl_det(imgs)

        # 如果启用了自动旋转，就需要对被旋转后的表格重新 OCR。
        # 原因是：原来的 self.boxes 是在整页原方向上 OCR 得到的，
        # 如果某个表格被旋转后才适合识别结构，那么表格内文字框也应该基于旋转后的正向图重新识别，
        # 否则文字框和 TSR 行列坐标会对不上。
        if auto_rotate:
            self._ocr_rotated_tables(ZM, table_layouts, recos, tbcnt)

        # 把 TSR 结果整理进 self.tb_cpns。
        # 这里处理的是“结构组件框”，不是正文 OCR 文本框；
        # 之后会用这些组件框去给 self.boxes 里的 table 文本框打 R/H/C/SP 标签。
        tbcnt = np.cumsum(tbcnt)
        for i in range(len(tbcnt) - 1):  # for page
            # pg 临时保存当前页所有表格组件。
            pg = []
            for j, tb_items in enumerate(recos[tbcnt[i] : tbcnt[i + 1]]):  # for table
                # poss 是当前页所有表格的原始裁剪位置和编号。
                # j 是页内第几个表格，用它和 tb_items 对齐。
                poss = pos[tbcnt[i] : tbcnt[i + 1]]
                for it in tb_items:  # for table components
                    # TSR 坐标是相对“表格小图”的坐标，且在自动旋转场景下可能是相对旋转图。
                    # 这里先额外保存一份 rotated 坐标，后面对列排序时可以使用更贴近 TSR 输入图的 x 坐标。
                    it["x0_rotated"] = it["x0"]
                    it["x1_rotated"] = it["x1"]
                    it["top_rotated"] = it["top"]
                    it["bottom_rotated"] = it["bottom"]

                    # 给结构组件补上页号。
                    # poss[j][2] 是该表格所在的页下标，后面按页匹配 table 文本框时会用到。
                    it["pn"] = poss[j][2]  # page number
                    # layoutno 记录页内第几个表格，用于区分同一页上的多个表格结构。
                    it["layoutno"] = j
                    # table_index 是跨页全局表格编号，用于和 rotation 信息、重 OCR 结果对齐。
                    it["table_index"] = poss[j][3]  # table index
                    pg.append(it)
            # 把当前页的表格结构组件加入全局组件列表。
            self.tb_cpns.extend(pg)

        def gather(kwd, fzy=10, ption=0.6):
            # 从所有 TSR 组件中筛选 label 匹配 kwd 的组件，比如 header、row、spanning。
            # 先按 Y 方向排序，是为了让行/表头的编号符合从上到下的阅读顺序。
            eles = Recognizer.sort_Y_firstly([r for r in self.tb_cpns if re.match(kwd, r["label"])], fzy)
            # 清理和 OCR 文本框明显不匹配的结构区域。
            # layouts_cleanup 会利用现有 self.boxes 做几何对齐过滤，减少 TSR 假阳性。
            eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
            # 再做一次严格排序，保证后续索引 ii 是稳定的视觉顺序。
            return Recognizer.sort_Y_firstly(eles, 0)

        # 收集不同类型的表格结构组件。
        # headers 用来给文本框标记表头区域 H；
        # rows 用来标记所在行 R；
        # spans 用来标记跨行/跨列单元格 SP。
        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        # 单独收集列组件。
        # 列的排序主要按页、表格编号、x 坐标进行；
        # 如果保存了 x0_rotated，就优先用旋转图坐标，因为它更贴近 TSR 模型看到的表格方向。
        clmns = sorted([r for r in self.tb_cpns if re.match(r"table column$", r["label"])], key=lambda x: (x["pn"], x["layoutno"], x["x0_rotated"] if "x0_rotated" in x else x["x0"]))
        # 对列组件也做一次几何清理，减少错误列框影响后续表格重建。
        clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)

        # 遍历 OCR/文本框，把落在表格布局里的文本框和 TSR 结构组件关联起来。
        # 这一轮会给文本框添加 R/H/C/SP 等字段；
        # 后面的 construct_table() 会利用这些字段重建行列矩阵。
        for b in self.boxes:
            # 只处理版面类型为 table 的文本框。
            # 正文、标题、图片说明等不应该参与表格行列结构匹配。
            if b.get("layout_type", "") != "table":
                continue
            # 找到当前文本框重叠最多的行组件。
            # 命中后写入 R、R_top、R_bott，表示它属于哪一行以及该行的上下边界。
            ii = Recognizer.find_overlapped_with_threshold(b, rows, thr=0.3)
            if ii is not None:
                b["R"] = ii
                b["R_top"] = rows[ii]["top"]
                b["R_bott"] = rows[ii]["bottom"]

            # 找到当前文本框是否落在表头组件里。
            # 表头信息比普通行更重要，后续构造表格时可以帮助识别 header cell。
            ii = Recognizer.find_overlapped_with_threshold(b, headers, thr=0.3)
            if ii is not None:
                b["H_top"] = headers[ii]["top"]
                b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]
                b["H_right"] = headers[ii]["x1"]
                b["H"] = ii

            # 找到水平方向上最贴合的列组件。
            # 列匹配更关注 x 方向范围，因此不用普通 overlap，而用 horizontally tightest fit。
            ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None:
                b["C"] = ii
                b["C_left"] = clmns[ii]["x0"]
                b["C_right"] = clmns[ii]["x1"]

            # 检查当前文本框是否属于跨行/跨列单元格。
            # 命中时写入 SP，同时复用 H_* 边界字段记录这个 spanning cell 的范围。
            ii = Recognizer.find_overlapped_with_threshold(b, spans, thr=0.3)
            if ii is not None:
                b["H_top"] = spans[ii]["top"]
                b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]
                b["H_right"] = spans[ii]["x1"]
                b["SP"] = ii

    def _ocr_rotated_tables(self, ZM, table_layouts, tsr_results, tbcnt):
        """
        Re-OCR rotated table images and update self.boxes.

        Args:
            ZM: Zoom factor
            table_layouts: List of table layout info
            tsr_results: TSR recognition results
            tbcnt: Cumulative table count per page
        """
        tbcnt = np.cumsum(tbcnt)

        def _table_region(layout, page_index):
            table_x0 = layout["x0"]
            table_top = layout["top"]
            table_x1 = layout["x1"]
            table_bottom = layout["bottom"]
            table_top_cum = table_top + self.page_cum_height[page_index]
            table_bottom_cum = table_bottom + self.page_cum_height[page_index]
            return table_x0, table_top, table_x1, table_bottom, table_top_cum, table_bottom_cum

        def _collect_table_boxes(page_index, table_x0, table_x1, table_top_cum, table_bottom_cum):
            indices = [
                i
                for i, b in enumerate(self.boxes)
                if (
                    b.get("page_number") == page_index + self.page_from
                    and b.get("layout_type") == "table"
                    and b["x0"] >= table_x0 - 5
                    and b["x1"] <= table_x1 + 5
                    and b["top"] >= table_top_cum - 5
                    and b["bottom"] <= table_bottom_cum + 5
                )
            ]
            original_boxes = [self.boxes[i] for i in indices]
            insert_at = indices[0] if indices else len(self.boxes)
            for i in reversed(indices):
                self.boxes.pop(i)
            return original_boxes, insert_at

        def _restore_boxes(original_boxes, insert_at):
            for b in original_boxes:
                self.boxes.insert(insert_at, b)
                insert_at += 1
            return insert_at

        def _map_rotated_point(x, y, angle, width, height):
            # Map a point from rotated image coords back to original image coords.
            if angle == 0:
                return x, y
            if angle == 90:
                # clockwise 90: original->rotated (x', y') = (y, width - x)
                # inverse:
                return width - y, x
            if angle == 180:
                return width - x, height - y
            if angle == 270:
                # clockwise 270: original->rotated (x', y') = (height - y, x)
                # inverse:
                return y, height - x
            return x, y

        def _insert_ocr_boxes(ocr_results, page_index, table_x0, table_top, insert_at, table_index, best_angle, table_w_px, table_h_px):
            added = 0
            for bbox, (text, conf) in ocr_results:
                if conf < 0.5:
                    continue
                mapped = [_map_rotated_point(p[0], p[1], best_angle, table_w_px, table_h_px) for p in bbox]
                x_coords = [p[0] for p in mapped]
                y_coords = [p[1] for p in mapped]
                box_x0 = min(x_coords) / ZM
                box_x1 = max(x_coords) / ZM
                box_top = min(y_coords) / ZM
                box_bottom = max(y_coords) / ZM
                new_box = {
                    "text": text,
                    "x0": box_x0 + table_x0,
                    "x1": box_x1 + table_x0,
                    "top": box_top + table_top + self.page_cum_height[page_index],
                    "bottom": box_bottom + table_top + self.page_cum_height[page_index],
                    "page_number": page_index + self.page_from,
                    "layout_type": "table",
                    "layoutno": f"table-{table_index}",
                    "_rotated": True,
                    "_rotation_angle": best_angle,
                    "_table_index": table_index,
                    "_rotated_x0": box_x0,
                    "_rotated_x1": box_x1,
                    "_rotated_top": box_top,
                    "_rotated_bottom": box_bottom,
                }
                self.boxes.insert(insert_at, new_box)
                insert_at += 1
                added += 1
            return added

        for tbl_info in table_layouts:
            table_index = tbl_info["table_index"]
            page = tbl_info["page"]
            layout = tbl_info["layout"]
            left, top, right, bott = tbl_info["coords"]

            rotation_info = self.table_rotations.get(table_index, {})
            best_angle = rotation_info.get("best_angle", 0)

            # Get the rotated table image
            rotated_img = self.rotated_table_imgs.get(table_index)
            if rotated_img is None:
                continue

            # If no rotation, keep original OCR boxes untouched.
            if best_angle == 0:
                continue

            # Table region is defined by layout's x0, top, x1, bottom (page-local coords)
            table_x0, table_top, table_x1, table_bottom, table_top_cum, table_bottom_cum = _table_region(layout, page)
            original_boxes, insert_at = _collect_table_boxes(page, table_x0, table_x1, table_top_cum, table_bottom_cum)

            logging.info(f"Re-OCR table {table_index} on page {page} with rotation {best_angle}°")

            # Perform OCR on rotated image
            img_array = np.array(rotated_img)
            ocr_results = self.ocr(img_array)

            if not ocr_results:
                logging.warning(f"No OCR results for rotated table {table_index}, restoring originals")
                _restore_boxes(original_boxes, insert_at)
                continue

            # Add new OCR results to self.boxes
            # OCR coordinates are relative to rotated image, map back to original table coords
            table_w_px = right - left
            table_h_px = bott - top
            added = _insert_ocr_boxes(
                ocr_results,
                page,
                table_x0,
                table_top,
                insert_at,
                table_index,
                best_angle,
                table_w_px,
                table_h_px,
            )

            logging.info(f"Added {added} OCR results from rotated table {table_index}")

    def __ocr(self, pagenum, img, chars, ZM=3, device_id: int | None = None):
        # 记录 OCR 检测阶段耗时，便于排查 PDF 页图过大、模型推理慢等性能问题。
        start = timer()
        # 先把 PIL Image 转成 numpy 数组交给 OCR 检测器。
        # 已读子函数实现：OCR.detect() 内部调用 TextDetector，返回检测到的文本框；
        # 它只做“文字区域检测”，不做真正识别，返回值里识别文本先是空字符串。
        # 这样拆成 detect + recognize 两段，是为了后面能优先复用 PDF 原生文字，
        # 只有原生文字缺失或乱码时才裁图识别，减少 OCR 成本和误差。
        bxs = self.ocr.detect(np.array(img), device_id)
        logging.info(f"__ocr detecting boxes of an image cost ({timer() - start}s)")

        # 重新计时，下面进入“检测框整理 + PDF 原生字符合并”阶段。
        start = timer()
        # 如果 OCR 没检测到任何文字框，也要给当前页追加一个空列表。
        # self.boxes 是按页对齐的，后续 layout 识别会假设 page_images 和 boxes 数量一致。
        if not bxs:
            self.boxes.append([])
            return
        # OCR.detect() 返回的是 zip(box, ("", score)) 形式；
        # 这里保留检测框 line[0] 和识别文本 line[1][0]。
        # detect 阶段文本通常为空，但字段先保留，后面统一删掉 txt。
        bxs = [(line[0], line[1][0]) for line in bxs]
        # 把 OCR 检测框转换成 RAGFlow 内部统一 bbox 字典。
        # OCR 框坐标来自放大后的页图，所以要除以 ZM 还原到 PDF 逻辑坐标系。
        # 已读 Recognizer.sort_Y_firstly()：它按 top 排序，如果 top 差小于阈值，就按 x0 排序；
        # 这符合常见阅读顺序“先从上到下，同一行从左到右”。
        # 阈值使用 mean_height / 3，是为了容忍同一行字符/框的微小垂直抖动。
        bxs = Recognizer.sort_Y_firstly(
            [
                {"x0": b[0][0] / ZM, "x1": b[1][0] / ZM, "top": b[0][1] / ZM, "text": "", "txt": t, "bottom": b[-1][1] / ZM, "chars": [], "page_number": pagenum}
                for b, t in bxs
                # 过滤掉坐标反向的异常框，避免后面计算宽高、重叠面积时出错。
                if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
            ],
            self.mean_height[pagenum - 1] / 3,
        )

        # merge chars in the same rect
        # chars 是 pdfplumber/pdfminer 从 PDF 文本层抽出来的原生字符。
        # 这一段的策略是：先把原生字符塞回它所在的 OCR 检测框里。
        # 原理上 PDF 原生文字通常比 OCR 更准、更快；OCR 主要作为扫描件或乱码文本的兜底。
        for c in chars:
            # 已读 Recognizer.find_overlapped()：它利用按 Y 排好序的 boxes 做一个近似二分收窄，
            # 再计算重叠面积，返回和当前字符重叠最多的检测框下标。
            # 这样比对所有框全量扫描更省，也能把 PDF 字符归并到对应文字行/区域。
            ii = Recognizer.find_overlapped(c, bxs)
            # 找不到任何重叠框的字符先放到 lefted_chars。
            # 这些字符可能是检测漏掉的文字、页眉页脚碎片，或者 PDF 坐标异常的残留。
            if ii is None:
                self.lefted_chars.append(c)
                continue
            # 计算 PDF 字符高度和 OCR 检测框高度。
            # 如果两者差异过大，说明这个字符虽然重叠，但很可能不属于这个框。
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            # 高度差比例超过 0.7 且不是空格时，认为匹配不可信。
            # 保留空格的原因是空格本身没有稳定视觉高度，但对英文/数字间隔很重要。
            if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
                self.lefted_chars.append(c)
                continue
            # 匹配可信时，把这个 PDF 原生字符归到对应 OCR 框里。
            bxs[ii]["chars"].append(c)

        # 现在每个 OCR 框里可能已经挂了一批 PDF 原生字符。
        # 这一段尝试用原生字符拼出框文本，只有无法拼出或判断为乱码时才回退到 OCR 识别。
        for b in bxs:
            # 如果这个检测框没有任何 PDF 原生字符，就保留空 text，后面会裁图走 OCR。
            if not b["chars"]:
                del b["chars"]
                continue
            # 保存当前框内字符，后面乱码判断需要看字符本身和字体信息。
            box_chars = b["chars"]
            # 用框内字符平均高度作为排序阈值。
            # 已读 sort_Y_firstly() 的排序策略：高度接近时按 x 排，同一文本行会更自然。
            m_ht = np.mean([c["height"] for c in box_chars])
            # garbled_count / total_count 用来判断 PDF 原生文本是否乱码。
            # 这是为了避免“PDF 有文本层但字体映射坏了”时错误信任原生文本。
            garbled_count = 0
            total_count = 0
            for c in Recognizer.sort_Y_firstly(box_chars, m_ht):
                # 对空格做特殊处理：只有当前框已有文本，并且前一个字符像英文、数字或标点时才补空格。
                # 这样能保留英文单词/数字之间的分隔，又避免中文文本被无意义空格打碎。
                if c["text"] == " " and b["text"]:
                    if re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", b["text"][-1]):
                        b["text"] += " "
                else:
                    # 非空格字符直接追加到当前框文本中。
                    b["text"] += c["text"]
                    for ch in c["text"]:
                        # 统计非空白字符数量；空白字符不参与乱码比例判断。
                        if not ch.isspace():
                            total_count += 1
                            # 已读 _is_garbled_char()：它把 Unicode 私用区、替换符、非法/代理类别、
                            # 控制字符等视为乱码。这些通常表示 pdfminer 无法把 CID 映射成真实 Unicode。
                            if self._is_garbled_char(ch):
                                garbled_count += 1
            # 字符已经拼进 text，临时 chars 字段不再需要，避免后续结构膨胀。
            del b["chars"]
            # If the majority of characters from pdfplumber are garbled,
            # clear the text so OCR recognition will be used as fallback.
            # Strategy 1: PUA / unmapped CID characters
            # 策略 1：如果原生字符里超过一半是私用区/CID 映射失败等明显乱码，
            # 就清空 text，强制后续用 OCR 图像识别替代原生文本。
            # 取 0.5 是一个偏保守阈值：少量异常字符可以容忍，多数异常才判定整框不可信。
            if total_count > 0 and garbled_count / total_count >= 0.5:
                logging.info(
                    "Page %d: detected garbled pdfplumber text (garbled=%d/%d), falling back to OCR for box at (%.1f, %.1f)",
                    pagenum, garbled_count, total_count, b["x0"], b["top"],
                )
                b["text"] = ""
                continue
            # Strategy 2: font-encoding garbling — all chars are ASCII
            # punctuation from subset fonts (no CJK output)
            # 策略 2：处理“字体编码映射错但字符不是私用区”的情况。
            # 已读 _is_garbled_by_font_encoding()：它检查 subset font 前缀比例、CJK 字符比例、
            # ASCII 标点符号比例；如果大量字符来自子集字体，且几乎没有 CJK，却有很多 ASCII 标点，
            # 就认为是中文 glyph 被错误映射成符号。这类问题必须回退到 OCR。
            if total_count > 0 and self._is_garbled_by_font_encoding(box_chars, min_chars=5):
                logging.info(
                    "Page %d: detected font-encoding garbled text (%d chars), falling back to OCR for box at (%.1f, %.1f)",
                    pagenum, total_count, b["x0"], b["top"],
                )
                b["text"] = ""

        # 记录 PDF 原生字符归并和乱码检测的耗时。
        logging.info(f"__ocr sorting {len(chars)} chars cost {timer() - start}s")
        # 重新计时，下面进入“对缺文本框做 OCR 识别”的阶段。
        start = timer()
        # boxes_to_reg 只收集需要 OCR 识别的框。
        # 这样能避免对已经有可信 PDF 原生文本的框重复 OCR，提高速度并降低识别误差。
        boxes_to_reg = []
        # 复用整页图像的 numpy 形式，后面裁剪每个待识别框。
        img_np = np.array(img)
        for b in bxs:
            # 只有 text 为空的框才需要走图像 OCR。
            # text 为空通常有三种情况：纯扫描件、PDF 原生字符没覆盖到、原生字符被判定为乱码。
            if not b["text"]:
                # 把逻辑坐标乘回 ZM，恢复到放大页图上的像素坐标。
                left, right, top, bott = b["x0"] * ZM, b["x1"] * ZM, b["top"] * ZM, b["bottom"] * ZM
                # 已读 OCR.get_rotate_crop_image()：它用四点透视变换把检测框裁成正向小图；
                # 如果裁出来是高瘦竖图，还会尝试原图、顺时针 90 度、逆时针 90 度，
                # 用识别器置信度选最佳方向。这样能提高竖排/旋转文字的识别稳定性。
                b["box_image"] = self.ocr.get_rotate_crop_image(img_np, np.array([[left, top], [right, top], [right, bott], [left, bott]], dtype=np.float32))
                # 收集起来批量识别，批处理通常比逐框调用模型更高效。
                boxes_to_reg.append(b)
            # txt 是 detect 阶段留下的临时字段，后面不再使用，删除以保持 box 结构干净。
            del b["txt"]
        # 已读 OCR.recognize_batch()：它调用 TextRecognizer 批量识别裁剪图，
        # 并用 drop_score=0.5 过滤低置信度结果，低分会置为空字符串。
        # 这里批量识别能减少模型调用开销，也保证所有待补文本框走同一识别阈值。
        texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)
        for i in range(len(boxes_to_reg)):
            # 把 OCR 识别结果写回对应检测框。
            boxes_to_reg[i]["text"] = texts[i]
            # 裁剪图只是识别中间产物，识别结束后删除，避免占用大量内存。
            del boxes_to_reg[i]["box_image"]
        # 记录 OCR 识别阶段耗时。日志里 len(bxs) 是总检测框数，不只是 boxes_to_reg 数。
        logging.info(f"__ocr recognize {len(bxs)} boxes cost {timer() - start}s")
        # 过滤掉最终仍然没有文本的框。
        # 这些框可能是误检、低置信度 OCR 结果，或图片/线条区域。
        bxs = [b for b in bxs if b["text"]]
        # 如果当前页还没有平均行高，就用最终文本框高度的中位数补一个。
        # 中位数比均值更抗异常框，后续排序、合并、布局判断都会依赖 mean_height。
        if self.mean_height[pagenum - 1] == 0:
            self.mean_height[pagenum - 1] = np.median([b["bottom"] - b["top"] for b in bxs])
        # 把当前页的 OCR/文本层融合结果追加到 self.boxes。
        # 后续 layout 识别、列判断、文本合并都会基于这些 box 继续处理。
        self.boxes.append(bxs)

    def _layouts_rec(self, ZM, drop=True):
        # 进入版面识别前，先确认“页图数量”和“每页 OCR/文本框结果数量”是一一对应的。
        # 因为 LayoutRecognizer 会按页并行处理 image_list 和 ocr_res，
        # 如果两边页数不一致，后面把版面区域和文本框对齐时就会整页错位。
        assert len(self.page_images) == len(self.boxes)
        # self.layouter 在 __init__() 里按环境被绑定成 LayoutRecognizer 或 AscendLayoutRecognizer。
        # 它的职责不是重新做 OCR，而是：
        # 1) 对整页图像做版面目标检测，识别 text/title/table/figure/header/footer/reference 等区域；
        # 2) 把这些版面区域和现有 OCR 文本框按重叠关系对齐；
        # 3) 给文本框补上 layout_type / layoutno；
        # 4) 在 drop=True 时丢弃页眉、页脚、参考文献等被视为噪声的块；
        # 5) 对没有文本框覆盖到的 figure/equation，补出一个空文本框占位。
        # 这里把 OCR 结果和版面结果合成，目的是让后续文本合并、表格处理、图文抽取都基于“带语义标签”的 box 工作。
        # ZM 传进去是因为版面模型看的坐标来自放大后的页图，需要再按 scale_factor 映射回 PDF 逻辑坐标。
        self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)
        # 上一步返回的 self.boxes 里的 top/bottom 仍然是“页内局部坐标”：
        # 第 1 页和第 2 页各自都从 y=0 开始。
        # 但后续很多逻辑，比如全文排序、跨页拼接、统一定位，更适合使用“整篇累计坐标”。
        # 所以这里把每个 box 的纵坐标整体加上该页之前所有页的累计高度。
        # 这样做完后，不同页上的框就能被放进同一个全局坐标系里比较。
        # 这一步只平移 Y，不改 X，因为跨页串接主要依赖的是纵向阅读顺序。
        # cumulative Y
        for i in range(len(self.boxes)):
            # page_number 是从 1 开始记的，因此访问 page_cum_height 时要减 1。
            # 比如第 1 页加 0，第 2 页加第 1 页高度，第 3 页加前两页高度之和。
            self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
            # bottom 同样要一起平移，保持框的高度和相对位置不变。
            self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]

    def _assign_column(self, boxes, zoomin=3):
        if not boxes:
            return boxes
        if all("col_id" in b for b in boxes):
            return boxes

        by_page = defaultdict(list)
        for b in boxes:
            by_page[b["page_number"]].append(b)

        page_cols = {}

        for pg, bxs in by_page.items():
            if not bxs:
                page_cols[pg] = 1
                continue

            x0s_raw = np.array([b["x0"] for b in bxs], dtype=float)

            min_x0 = np.min(x0s_raw)
            max_x1 = np.max([b["x1"] for b in bxs])
            width = max_x1 - min_x0

            INDENT_TOL = width * 0.12
            x0s = []
            for x in x0s_raw:
                if abs(x - min_x0) < INDENT_TOL:
                    x0s.append([min_x0])
                else:
                    x0s.append([x])
            x0s = np.array(x0s, dtype=float)

            max_try = min(4, len(bxs))
            if max_try < 2:
                max_try = 1
            best_k = 1
            best_score = -1

            for k in range(1, max_try + 1):
                km = KMeans(n_clusters=k, n_init="auto")
                labels = km.fit_predict(x0s)

                centers = np.sort(km.cluster_centers_.flatten())
                if len(centers) > 1:
                    try:
                        score = silhouette_score(x0s, labels)
                    except ValueError:
                        continue
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_k = k

            page_cols[pg] = best_k
            logging.info(f"[Page {pg}] best_score={best_score:.2f}, best_k={best_k}")

        global_cols = Counter(page_cols.values()).most_common(1)[0][0]
        logging.info(f"Global column_num decided by majority: {global_cols}")

        for pg, bxs in by_page.items():
            if not bxs:
                continue
            k = page_cols[pg]
            if len(bxs) < k:
                k = 1
            x0s = np.array([[b["x0"]] for b in bxs], dtype=float)
            km = KMeans(n_clusters=k, n_init="auto")
            labels = km.fit_predict(x0s)

            centers = km.cluster_centers_.flatten()
            order = np.argsort(centers)

            remap = {orig: new for new, orig in enumerate(order)}

            for b, lb in zip(bxs, labels):
                b["col_id"] = remap[lb]

            grouped = defaultdict(list)
            for b in bxs:
                grouped[b["col_id"]].append(b)

        return boxes

    def _text_merge(self, zoomin=3):
        # 这一阶段做“同一行内”的横向文本框合并。
        # 前面的 OCR 和 layout 识别会产出很多碎框：
        # 有时一个视觉上的完整行会被切成多个相邻 box。
        # 如果不先把这些同一行碎框合并，后续段落合并和 chunking 都会出现断裂。
        # 这里先给文本框分栏，是为了避免双栏 PDF 中左栏末尾和右栏开头被误认为同一行相邻内容。
        bxs = self._assign_column(self.boxes, zoomin)

        def end_with(b, txt):
            # 判断某个 box 的文本是否以指定字符串结尾。
            # 这个 helper 目前在当前函数里没有被使用，属于历史合并规则留下的辅助函数。
            txt = txt.strip()
            tt = b.get("text", "").strip()
            return tt and tt.find(txt) == len(tt) - len(txt)

        def start_with(b, txts):
            # 判断某个 box 的文本是否以 txts 中任意字符串开头。
            # 和 end_with 一样，这里当前未参与实际逻辑，但保留可能是为了后续扩展规则。
            tt = b.get("text", "").strip()
            return tt and any([tt.find(t.strip()) == 0 for t in txts])

        # 横向合并相邻 box。
        # 核心原则是：只有同页、同栏、同 layout 区域、且垂直位置足够接近的相邻框才合并。
        # 这样能恢复被 OCR/文本层切碎的同一行，同时尽量避免跨栏、跨段、跨表格误合并。
        i = 0
        while i < len(bxs) - 1:
            # 当前框。
            b = bxs[i]
            # 紧邻的下一个框。
            b_ = bxs[i + 1]

            # 不同页或不同栏的框不能合并。
            # 页和栏是阅读顺序里最强的边界，跨过去合并通常就是错误。
            if b["page_number"] != b_["page_number"] or b.get("col_id") != b_.get("col_id"):
                i += 1
                continue

            # 不同 layoutno 说明它们属于不同版面区域，比如不同段落、不同标题块或不同表格区域。
            # table/figure/equation 也禁止在这里合并，因为这些内容需要保留结构边界，
            # 尤其表格要交给表格结构逻辑处理，不能按普通正文拼接。
            if b.get("layoutno", "0") != b_.get("layoutno", "1") or b.get("layout_type", "") in ["table", "figure", "equation"]:
                i += 1
                continue

            # _y_dis 返回两个框中心点的纵向距离。
            # 如果纵向距离小于当前页平均字符高度的 1/3，就认为它们在同一视觉行上。
            # 用 mean_height 做尺度归一化，是为了适配不同字号、不同分辨率的 PDF。
            if abs(self._y_dis(b, b_)) < self.mean_height[bxs[i]["page_number"] - 1] / 3:
                # 合并时把右边界扩展到后一个框的右边界。
                # 因为这是横向合并，通常 b_ 在 b 右侧。
                bxs[i]["x1"] = b_["x1"]
                # top/bottom 取平均，是为了把两个轻微上下抖动的框拉回同一条水平线上。
                bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                # 文本直接拼接，不主动加空格。
                # 空格恢复在 __images__ 里对英文/数字间距已经做过一轮；
                # 这里保守拼接，避免中文文本被插入多余空格。
                bxs[i]["text"] += b_["text"]
                # 删除被合并的后一个框。
                # i 不递增，让当前合并后的框继续尝试和新的下一个框合并。
                bxs.pop(i + 1)
                continue
            i += 1
        # 用合并后的结果替换全局 boxes，后续 _concat_downward 和过滤逻辑会继续基于它处理。
        self.boxes = bxs

    def _naive_vertical_merge(self, zoomin=3):
        # bxs = self._assign_column(self.boxes, zoomin)
        bxs = self.boxes

        grouped = defaultdict(list)
        for b in bxs:
            # grouped[(b["page_number"], b.get("col_id", 0))].append(b)
            grouped[(b["page_number"], "x")].append(b)

        merged_boxes = []
        for (pg, col), bxs in grouped.items():
            bxs = sorted(bxs, key=lambda x: (x["top"], x["x0"]))
            if not bxs:
                continue

            mh = self.mean_height[pg - 1] if self.mean_height else np.median([b["bottom"] - b["top"] for b in bxs]) or 10

            i = 0
            while i + 1 < len(bxs):
                b = bxs[i]
                b_ = bxs[i + 1]

                if b["page_number"] < b_["page_number"] and re.match(r"[0-9  •一—-]+$", b["text"]):
                    bxs.pop(i)
                    continue

                if not b["text"].strip():
                    bxs.pop(i)
                    continue

                if not b["text"].strip() or b.get("layoutno") != b_.get("layoutno"):
                    i += 1
                    continue

                if b_["top"] - b["bottom"] > mh * 1.5:
                    i += 1
                    continue

                overlap = max(0, min(b["x1"], b_["x1"]) - max(b["x0"], b_["x0"]))
                if overlap / max(1, min(b["x1"] - b["x0"], b_["x1"] - b_["x0"])) < 0.3:
                    i += 1
                    continue

                concatting_feats = [
                    b["text"].strip()[-1] in ",;:'\"，、‘“；：-",
                    len(b["text"].strip()) > 1 and b["text"].strip()[-2] in ",;:'\"，‘“、；：",
                    b_["text"].strip() and b_["text"].strip()[0] in "。；？！?”）),，、：",
                ]
                # features for not concating
                feats = [
                    b.get("layoutno", 0) != b_.get("layoutno", 0),
                    b["text"].strip()[-1] in "。？！?",
                    self.is_english and b["text"].strip()[-1] in ".!?",
                    b["page_number"] == b_["page_number"] and b_["top"] - b["bottom"] > self.mean_height[b["page_number"] - 1] * 1.5,
                    b["page_number"] < b_["page_number"] and abs(b["x0"] - b_["x0"]) > self.mean_width[b["page_number"] - 1] * 4,
                ]
                # split features
                detach_feats = [b["x1"] < b_["x0"], b["x0"] > b_["x1"]]
                if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                    logging.debug(
                        "{} {} {} {}".format(
                            b["text"],
                            b_["text"],
                            any(feats),
                            any(concatting_feats),
                        )
                    )
                    i += 1
                    continue

                b["text"] = (b["text"].rstrip() + " " + b_["text"].lstrip()).strip()
                b["bottom"] = b_["bottom"]
                b["x0"] = min(b["x0"], b_["x0"])
                b["x1"] = max(b["x1"], b_["x1"])
                bxs.pop(i + 1)

            merged_boxes.extend(bxs)

        # self.boxes = sorted(merged_boxes, key=lambda x: (x["page_number"], x.get("col_id", 0), x["top"]))
        self.boxes = merged_boxes

    def _final_reading_order_merge(self, zoomin=3):
        if not self.boxes:
            return

        self.boxes = self._assign_column(self.boxes, zoomin=zoomin)

        pages = defaultdict(lambda: defaultdict(list))
        for b in self.boxes:
            pg = b["page_number"]
            col = b.get("col_id", 0)
            pages[pg][col].append(b)

        for pg in pages:
            for col in pages[pg]:
                pages[pg][col].sort(key=lambda x: (x["top"], x["x0"]))

        new_boxes = []
        for pg in sorted(pages.keys()):
            for col in sorted(pages[pg].keys()):
                new_boxes.extend(pages[pg][col])

        self.boxes = new_boxes

    def _concat_downward(self, concat_between_pages=True):
        # 当前版本的真实行为：只把所有 box 按阅读方向做一次 Y 优先排序，然后立即返回。
        # 也就是说，下面保留的大段“纵向段落合并”代码当前不会执行。
        # 这样做的工程含义是：RAGFlow 现在更倾向保留版面框粒度，
        # 把更大粒度的 chunk 合并交给后面的 naive_merge/tokenize 阶段处理，
        # 避免在 PDF 解析层过早把上下块拼错。
        # sort_Y_firstly 的规则是：先按 top 从上到下排，Y 差很小时再按 x0 从左到右排。
        self.boxes = Recognizer.sort_Y_firstly(self.boxes, 0)
        # 这里的 return 让下面旧的纵向合并逻辑变成“保留但不启用”的代码。
        # 注释仍然补在下面，是为了你读源码时知道历史设计意图。
        return

        # 以下是历史/备用纵向合并逻辑，当前不会执行。
        # 它原本的思路是：先统计每个 box 附近同一行的框数量，作为后续机器学习模型判断上下拼接的特征之一。
        for i in range(len(self.boxes)):
            # 当前 box 所在页的平均字符高度，用来把纵向距离归一化。
            mh = self.mean_height[self.boxes[i]["page_number"] - 1]
            # in_row 记录当前 box 附近同一视觉行上的邻居数量。
            # 值越大，越可能是表格/多栏/复杂排版，而不是普通单段落。
            self.boxes[i]["in_row"] = 0
            # 只看当前位置前后最多 12 个 box，避免全量 O(n^2) 比较。
            j = max(0, i - 12)
            while j < min(i + 12, len(self.boxes)):
                # 跳过自己。
                if j == i:
                    j += 1
                    continue
                # 用中心点纵向距离除以平均行高，判断是否在同一行附近。
                ydis = self._y_dis(self.boxes[i], self.boxes[j]) / mh
                if abs(ydis) < 1:
                    # 纵向距离小于一个行高，认为在同一行附近。
                    self.boxes[i]["in_row"] += 1
                elif ydis > 0:
                    # 因为 boxes 已按 Y 排序，如果后面的框已经明显在下方，就可以提前停止。
                    break
                j += 1

        # 下面尝试把上下相邻的 box 串成段落块。
        # 逻辑会复制一份 boxes，避免在 DFS 过程中直接破坏原列表导致遍历混乱。
        boxes = deepcopy(self.boxes)
        # blocks 保存一组组被认为应该纵向拼接的 box。
        blocks = []
        while boxes:
            # chunks 是从当前起点 DFS 找到的一条向下拼接链。
            chunks = []

            def dfs(up, dp):
                # 把当前上方 box 放入拼接链。
                chunks.append(up)
                # 从 dp 开始向后找可能接在 up 下方的 box。
                i = dp
                while i < min(dp + 12, len(boxes)):
                    # 计算 up 和候选 down 的纵向中心距离。
                    ydis = self._y_dis(up, boxes[i])
                    # 判断两者是否在同一页。
                    smpg = up["page_number"] == boxes[i]["page_number"]
                    # 取 up 所在页的平均字符高度和宽度，作为距离阈值尺度。
                    mh = self.mean_height[up["page_number"] - 1]
                    mw = self.mean_width[up["page_number"] - 1]
                    # 同页情况下，如果候选框离得超过 4 个行高，就认为已经不是同一段。
                    if smpg and ydis > mh * 4:
                        break
                    # 跨页情况下容忍更大的纵向距离，因为 page_cum_height 已经把页高累计进来了。
                    if not smpg and ydis > mh * 16:
                        break
                    down = boxes[i]
                    # 如果调用方禁止跨页拼接，遇到下一页就停止。
                    if not concat_between_pages and down["page_number"] > up["page_number"]:
                        break

                    # 表格行号 R 不同且上文不是逗号结尾时，不拼。
                    # 这是为了避免把表格不同行硬拼成一句话。
                    if up.get("R", "") != down.get("R", "") and up["text"][-1] != "，":
                        i += 1
                        continue

                    # 跳过类似页码/编号格式和空文本。
                    # 这类内容通常不应参与正文段落拼接。
                    if re.match(r"[0-9]{2,3}/[0-9]{3}$", up["text"]) or re.match(r"[0-9]{2,3}/[0-9]{3}$", down["text"]) or not down["text"].strip():
                        i += 1
                        continue

                    # 任一侧为空文本都不拼。
                    if not down["text"].strip() or not up["text"].strip():
                        i += 1
                        continue

                    # 如果两个框在水平方向相距太远，说明可能是不同栏、不同表格列或不同区域。
                    if up["x1"] < down["x0"] - 10 * mw or up["x0"] > down["x1"] + 10 * mw:
                        i += 1
                        continue

                    # 对普通 text layout，如果很近的候选框属于同一个 layoutno，就直接拼接。
                    # 这是一个强规则，优先于下面的 ML 模型。
                    if i - dp < 5 and up.get("layout_type") == "text":
                        if up.get("layoutno", "1") == down.get("layoutno", "2"):
                            # 递归继续向下找下一个可拼框。
                            dfs(down, i + 1)
                            # 从候选列表中移除已经被吸收的框。
                            boxes.pop(i)
                            return
                        i += 1
                        continue

                    # 对不满足强规则的候选，抽取几何、版面、标点、词性等特征。
                    # 已读 _updown_concat_features()：它会生成一组特征给 XGBoost 模型判断上下框是否应该合并。
                    fea = self._updown_concat_features(up, down)
                    # 模型分数 <= 0.5 时认为不应拼接。
                    if self.updown_cnt_mdl.predict(xgb.DMatrix([fea]))[0] <= 0.5:
                        i += 1
                        continue
                    # 模型认为可拼接，则继续 DFS 向下扩展链。
                    dfs(down, i + 1)
                    boxes.pop(i)
                    return

            # 从当前列表第一个 box 开始尝试构造一条纵向拼接链。
            dfs(boxes[0], 1)
            # 起点 box 已经处理完，从候选列表移除。
            boxes.pop(0)
            if chunks:
                # 保存当前拼接链。
                blocks.append(chunks)

        # 把每个 block 内的多个 box 真正合并成一个大 box。
        boxes = []
        for b in blocks:
            # 只有一个 box 的 block 不需要合并。
            if len(b) == 1:
                boxes.append(b[0])
                continue
            # t 作为合并目标，逐个吸收后续 box。
            t = b[0]
            for c in b[1:]:
                # 去掉首尾空白，避免拼接时产生意外空格。
                t["text"] = t["text"].strip()
                c["text"] = c["text"].strip()
                if not c["text"]:
                    continue
                # 如果前后都是英文/数字类字符，插入一个空格，避免词粘连。
                if t["text"] and re.match(r"[0-9\.a-zA-Z]+$", t["text"][-1] + c["text"][-1]):
                    t["text"] += " "
                # 拼接文本。
                t["text"] += c["text"]
                # 合并后的框覆盖所有子框的横向范围。
                t["x0"] = min(t["x0"], c["x0"])
                t["x1"] = max(t["x1"], c["x1"])
                # 跨页拼接时，page_number 保留最早的页号。
                t["page_number"] = min(t["page_number"], c["page_number"])
                # bottom 延伸到最后一个被合并框。
                t["bottom"] = c["bottom"]
                # 如果原框没有 layout_type，而子框有，就继承子框类型。
                if not t["layout_type"] and c["layout_type"]:
                    t["layout_type"] = c["layout_type"]
            boxes.append(t)

        # 合并完成后重新按阅读顺序排序。
        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)

    def _filter_forpages(self):
        # 这一阶段做页级/目录级清洗。
        # 目标不是一般文本清洗，而是去掉 PDF 里经常会混入正文的目录页、致谢页、乱码脏页等。
        if not self.boxes:
            return
        # findit 表示是否找到了明确的目录/致谢入口。
        # 一旦找到并处理，就不再走后面的脏页检测分支。
        findit = False
        # 使用 while + 手动 i，是因为过程中会不断 pop 删除 box。
        i = 0
        while i < len(self.boxes):
            # 查找目录/目次/table of contents/致谢/acknowledge 这类页面标题。
            # 先去掉普通空格和全角空格，再 lower，是为了兼容 OCR 和 PDF 文本层的不同空白形式。
            if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$", re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())):
                i += 1
                continue
            # 找到目录/致谢入口。
            findit = True
            # 判断当前标题是否更像英文。
            # 英文目录行通常用前几个单词作为重复前缀，中文则用前几个字符。
            eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
            # 删除目录/致谢标题本身。
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            # 取标题后第一条有效内容的前缀。
            # 中文取前三个字符，英文取前两个词，用来判断目录列表到哪里结束。
            prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            # 如果标题后面有空文本框，连续删除，直到拿到一个可用于匹配的 prefix。
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            # 删除第一条目录项。
            # 后面会向下寻找同样 prefix 再次出现的位置，用来估计这一整段目录块的边界。
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            # 在后续最多 128 个框里寻找相同前缀。
            # 目录页常见模式是若干条目录项有相似开头或重复结构；
            # 找到重复后，把中间部分作为目录块删除。
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                # 删除 i 到 j 之间的目录内容。
                # 注意每次都 pop(i)，因为列表左移后下一个待删元素仍在 i。
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        # 如果已经基于明确目录/致谢标志做过过滤，就直接结束。
        # 避免再用“脏页启发式”误删正常页面。
        if findit:
            return

        # 如果没有发现目录/致谢，就再做一种页级乱码检测。
        # page_dirty 按页计数，统计每页出现疑似目录点线/乱码点串的次数。
        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            # 这里匹配连续点状符号。
            # 这类符号常见于目录页的引导线，也可能来自 OCR/PDF 抽取中的脏字符。
            if re.search(r"(··|··|··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        # 如果某页出现超过 3 次点状脏符号，就把整页标为 dirty。
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        # 没有脏页则不做任何删除。
        if not page_dirty:
            return
        # 删除所有属于 dirty 页的 box。
        # 这是比较激进的页级过滤，所以只有在重复脏符号足够多时才触发。
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        i = 0
        while i + 1 < len(self.boxes):
            b = self.boxes[i]
            b_ = self.boxes[i + 1]
            if not b["text"].strip():
                self.boxes.pop(i)
                continue
            if not b_["text"].strip():
                self.boxes.pop(i + 1)
                continue

            if (
                b["text"].strip()[0] != b_["text"].strip()[0]
                or b["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm")
                or rag_tokenizer.is_chinese(b["text"].strip()[0])
                or b["top"] > b_["bottom"]
            ):
                i += 1
                continue
            b_["text"] = b["text"] + "\n" + b_["text"]
            b_["x0"] = min(b["x0"], b_["x0"])
            b_["x1"] = max(b["x1"], b_["x1"])
            b_["top"] = b["top"]
            self.boxes.pop(i)

    def _extract_table_figure(self, need_image, ZM, return_html, need_position, separate_tables_figures=False):
        # 这一阶段把 self.boxes 里的“普通正文框”和“表格/图片相关框”正式分流。
        # 前面的 _table_transformer_job() 已经给表格文本框打上了 R/H/C/SP 等结构标签，
        # 这里要做的是：
        # 1) 从整体 boxes 中抽出 table / figure 区域；
        # 2) 尝试把 caption 归并到最近的表格或图片；
        # 3) 从 page_images 中裁出对应图像；
        # 4) 对表格调用 construct_table() 组装成 HTML 或文字描述；
        # 5) 返回“图像 + 结构化内容”的结果，同时把这些区域从正文流里移走。
        tables = {}
        figures = {}
        # 第一轮扫描：把已经被 layout 识别为 table / figure 的 box 从 self.boxes 里摘出来。
        # tables / figures 的 key 使用 page_number-layoutno，
        # 表示“同一页里的同一个布局区域”。
        i = 0
        # lst_lout_no 记录上一个处理到的 layout 编号。
        # 后面如果遇到 caption/title/reference，会把这个布局编号加入 nomerge_lout_no，
        # 用来阻止某些跨页表格被错误合并。
        lst_lout_no = ""
        # 记录不应该执行跨页合并的布局编号。
        # 典型场景是：某个 table/figure 附近已经出现 caption、title、reference，
        # 说明它的边界更明确，继续和下一页同类区域硬拼的风险更高。
        nomerge_lout_no = []
        while i < len(self.boxes):
            # 没有 layoutno 的框通常不是明确的版面块，不参与表格/图片抽取。
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            # layout 唯一键：页号 + layout 编号。
            lout_no = str(self.boxes[i]["page_number"]) + "-" + str(self.boxes[i]["layoutno"])
            # caption / title / figure caption / reference 这些块一旦出现，
            # 通常意味着前一个 table/figure 布局边界已经足够明确，不应随便跨页续接。
            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in ["table caption", "title", "figure caption", "reference"]:
                nomerge_lout_no.append(lst_lout_no)
            # 处理表格框。
            if self.boxes[i]["layout_type"] == "table":
                # “数据来源/资料来源/图表来源”这类来源说明通常不当作表格正文内容。
                # 如果保留，会污染表格单元格文本，所以这里直接丢掉。
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                # 同一 layout 的多个文本框归入同一张表。
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                # 被抽出的表格框从正文 boxes 中移除，避免后面又被当普通正文输出。
                self.boxes.pop(i)
                # 记录最近一次处理到的布局编号。
                lst_lout_no = lout_no
                continue
            # 只有 need_image=True 时才抽取 figure。
            # 某些调用方只关心文本，不需要额外返回图片区域。
            if need_image and self.boxes[i]["layout_type"] == "figure":
                # 图片来源说明同样不作为图片正文描述保留。
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                # 同一 figure layout 的多个框归到同一张图。
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                # 从正文流里移走。
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1

        # 第二步：尝试把跨页表格合并起来。
        # 很多 PDF 表格会被硬切在分页符两侧，第一页底部半张表、第二页顶部续表。
        # 如果不在这里合并，后面的 construct_table() 会把它们当成两张独立表。
        nomerge_lout_no = set(nomerge_lout_no)
        # 按出现顺序排序，便于从后往前比较相邻表格是否属于同一张跨页表。
        tbls = sorted([(k, bxs) for k, bxs in tables.items()], key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))

        # 从后往前看相邻两张表，尝试把后一张并入前一张。
        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            # 如果前一张表附近有 caption/title/reference 等边界信号，就不跨页合并。
            if k0 in nomerge_lout_no:
                continue
            # 同页的两张表不是“跨页续表”，不能合并。
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            # 相差超过一页通常不可能是同一张表的直接续页。
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            # 用后一页的平均行高作为纵向距离尺度。
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            # 如果前一张表的最后一个框和后一张表的第一个框在累计 Y 坐标上相距过远，
            # 就认为它们不是同一张表的上下两截。
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            # 满足条件时，把后一张表的所有框并到前一张表，删除后者。
            tables[k0].extend(tables[k])
            del tables[k]

        def x_overlapped(a, b):
            # 判断两个框在水平方向是否有重叠。
            # caption 寻找最近 table/figure 时，如果 x 方向已有覆盖，就把横向距离当作 0。
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        # 第三步：从剩余正文框里查找 caption，并把它们挂到最近的表格或图片上。
        # caption 之所以单独后处理，是因为 OCR/layout 阶段不一定总能稳定地把 caption 和主体放到同一组里。
        i = 0
        while i < len(self.boxes):
            # c 是候选 caption 框。
            c = self.boxes[i]
            # mh = self.mean_height[c["page_number"]-1]
            # 只有被规则识别为 caption 的框才参与下面的最近邻归并。
            if not TableStructureRecognizer.is_caption(c):
                i += 1
                continue

            # 在 tables 或 figures 中寻找离当前 caption 最近的那个布局。
            # 距离定义是 y_dis^2 + x_dis^2：
            # 纵向距离和横向距离共同决定归属，若 x 方向有重叠则只看纵向距离。
            def nearest(tbls):
                nonlocal c
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        # 已经是 caption 的框不作为主体候选，避免 caption 之间互相吸附。
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        y_dis = self._y_dis(c, b)
                        x_dis = self._x_dis(c, b) if not x_overlapped(c, b) else 0
                        dis = y_dis * y_dis + x_dis * x_dis
                        if dis < minv:
                            mink = k
                            minv = dis
                return mink, minv

            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            # if min(tv, fv) > 2000:
            #    i += 1
            #    continue
            # 如果最近的表比最近的图更近，就把 caption 插到表格内容前面。
            # insert(0, c) 的意义是让 caption 在后续拼装输出时排在主体前面。
            if tv < fv and tk:
                tables[tk].insert(0, c)
                logging.debug("TABLE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            # 否则如果存在最近的 figure，就挂到图片前面。
            elif fk:
                figures[fk].insert(0, c)
                logging.debug("FIGURE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            # 无论挂给谁，caption 本身都要从正文框里移除。
            self.boxes.pop(i)

        def cropout(bxs, ltype, poss):
            # 根据一组 table/figure 相关框，裁出对应的图像区域。
            # 这里既支持单页，也支持跨页：跨页时会逐页裁出后再上下拼接成一张长图。
            nonlocal ZM
            max_page_index = len(self.page_images) - 1

            def local_page_index(page_number):
                # 把 box 里的 page_number 映射到 page_images 的局部下标。
                # page_number 在某些场景下可能是全局页号，而当前 parser 只加载了部分页，
                # 所以当 page_from > 0 时要尝试做一次偏移修正。
                idx = page_number - 1 if page_number > 0 else 0
                if idx > max_page_index and self.page_from:
                    idx = page_number - 1 - self.page_from
                return idx

            # 先收集这组框分布在哪些页。
            pn = set()
            for b in bxs:
                idx = local_page_index(b["page_number"])
                if 0 <= idx <= max_page_index:
                    pn.add(idx)
                else:
                    # 某些异常页号无法映射回已加载页图时，只记日志并跳过。
                    logging.warning(
                        "Skip out-of-range page_number %s (page_from=%s, pages=%s)",
                        b.get("page_number"),
                        self.page_from,
                        len(self.page_images),
                    )

            # 没有任何有效页可裁，就返回 None。
            if not pn:
                return None

            # 单页表格/图片的裁图路径。
            if len(pn) < 2:
                pn = list(pn)[0]
                # 当前页在累计坐标系里的起始高度。
                ht = self.page_cum_height[pn]
                # 用所有框的外接矩形估算 table/figure 的整体范围。
                # 注意 top/bottom 要减去累计页高，还原到当前页局部坐标。
                b = {"x0": np.min([b["x0"] for b in bxs]), "top": np.min([b["top"] for b in bxs]) - ht, "x1": np.max([b["x1"] for b in bxs]), "bottom": np.max([b["bottom"] for b in bxs]) - ht}
                # 如果当前页 layout 里存在更精确的 table/figure 区域，就优先使用 layout 框裁图。
                # 这样通常比单纯用文本框外接矩形更完整，能保留边线、空白边距等。
                louts = [layout for layout in self.page_layout[pn] if layout["type"] == ltype]
                ii = Recognizer.find_overlapped(b, louts, naive=True)
                if ii is not None:
                    b = louts[ii]
                else:
                    # 找不到 layout 匹配时退回到文本框外接矩形，并记日志。
                    logging.warning(f"Missing layout match: {pn + 1},%s" % (bxs[0].get("layoutno", "")))

                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                # 极端异常情况下 right 可能小于 left，这里强制修正，避免 crop 崩溃。
                if right < left:
                    right = left + 1
                # 记录该裁图在原文档中的位置信息，供 need_position=True 的调用方使用。
                poss.append((pn + self.page_from, left, right, top, bott))
                # 真正从 page_images 裁图时要乘回 ZM，因为 page_images 是放大后的像素坐标。
                return self.page_images[pn].crop((left * ZM, top * ZM, right * ZM, bott * ZM))
            # 多页表格/图片：先按页分桶。
            pn = {}
            for b in bxs:
                p = local_page_index(b["page_number"])
                if 0 <= p <= max_page_index:
                    if p not in pn:
                        pn[p] = []
                    pn[p].append(b)
            # 按页号排序，保证拼接顺序从前到后。
            pn = sorted(pn.items(), key=lambda x: x[0])
            # 对每一页递归调用 cropout，得到每页自己的局部裁图。
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            imgs = [img for img in imgs if img is not None]
            if not imgs:
                return None
            # 多页时，把各页裁图上下拼成一张长图。
            # 这样下游无论是展示还是进一步处理，都可以把跨页表/图看成一个整体。
            pic = Image.new("RGB", (int(np.max([i.size[0] for i in imgs])), int(np.sum([m.size[1] for m in imgs]))), (245, 245, 245))
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        # res/positions 用于 tables 或“表图混合返回”模式；
        # figure_results/figure_positions 只在 separate_tables_figures=True 时单独返回 figure。
        res = []
        positions = []
        figure_results = []
        figure_positions = []
        # 先处理 figures。
        # figure 的文本内容通常就是若干说明框 + caption 拼起来的描述文本。
        for k, bxs in figures.items():
            # 用换行把 figure 相关文本串起来，保留说明文字的层次感。
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue

            poss = []

            # separate_tables_figures=True 时，figure 和 table 分开返回；
            # 否则都塞进统一的 res 列表。
            if separate_tables_figures:
                img = cropout(bxs, "figure", poss)
                if img is None:
                    continue
                # figure 的结构化文本这里只是一个字符串列表，不做像表格那样的行列重建。
                figure_results.append((img, [txt]))
                figure_positions.append(poss)
            else:
                img = cropout(bxs, "figure", poss)
                if img is None:
                    continue
                res.append((img, [txt]))
                positions.append(poss)

        # 再处理 tables。
        for k, bxs in tables.items():
            if not bxs:
                continue
            # 先按 Y 优先排序。
            # 阈值用平均半行高，是为了同一行内按 X 排、跨行时按 Y 排，更符合表格阅读顺序。
            bxs = Recognizer.sort_Y_firstly(bxs, np.mean([(b["bottom"] - b["top"]) / 2 for b in bxs]))

            poss = []

            img = cropout(bxs, "table", poss)
            if img is None:
                continue
            # 已读 TableStructureRecognizer.construct_table()：
            # 它会先移出 caption，再根据前面打好的 R/H/C/SP 标签重建行、列、表头、跨行跨列关系，
            # 最终输出 HTML 或文字描述。
            res.append((img, self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
            positions.append(poss)

        # 统一整理返回格式。
        if separate_tables_figures:
            # 保证“结果数量”和“位置数量”始终一一对应。
            assert len(positions) + len(figure_positions) == len(res) + len(figure_results)
            if need_position:
                # 返回 ((内容, 位置)) 的形式，tables 和 figures 分开给调用方。
                return list(zip(res, positions)), list(zip(figure_results, figure_positions))
            else:
                return res, figure_results
        else:
            assert len(positions) == len(res)
            if need_position:
                # 混合模式下，返回每个元素及其位置。
                return list(zip(res, positions))
            else:
                return res

    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12),
        ]:
            if re.match(p, line):
                return j
        return

    def _line_tag(self, bx, ZM):
        pn = [bx["page_number"]]
        top = bx["top"] - self.page_cum_height[pn[0] - 1]
        bott = bx["bottom"] - self.page_cum_height[pn[0] - 1]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt:
            return ""
        while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / ZM
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return ""

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), bx["x0"], bx["x1"], top, bott)

    def __filterout_scraps(self, boxes, ZM):
        def width(b):
            return b["x1"] - b["x0"]

        def height(b):
            return b["bottom"] - b["top"]

        def usefull(b):
            if b.get("layout_type"):
                return True
            if width(b) > self.page_images[b["page_number"] - 1].size[0] / ZM / 3:
                return True
            if b["bottom"] - b["top"] > self.mean_height[b["page_number"] - 1]:
                return True
            return False

        res = []
        while boxes:
            lines = []
            widths = []
            pw = self.page_images[boxes[0]["page_number"] - 1].size[0] / ZM
            mh = self.mean_height[boxes[0]["page_number"] - 1]
            mj = self.proj_match(boxes[0]["text"]) or boxes[0].get("layout_type", "") == "title"

            def dfs(line, st):
                nonlocal mh, pw, lines, widths
                lines.append(line)
                widths.append(width(line))
                mmj = self.proj_match(line["text"]) or line.get("layout_type", "") == "title"
                for i in range(st + 1, min(st + 20, len(boxes))):
                    if (boxes[i]["page_number"] - line["page_number"]) > 0:
                        break
                    if not mmj and self._y_dis(line, boxes[i]) >= 3 * mh and height(line) < 1.5 * mh:
                        break

                    if not usefull(boxes[i]):
                        continue
                    if mmj or (self._x_dis(boxes[i], line) < pw / 10):
                        # and abs(width(boxes[i])-width_mean)/max(width(boxes[i]),width_mean)<0.5):
                        # concat following
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if usefull(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logging.debug("WASTE: " + boxes[0]["text"])
            except Exception:
                pass
            boxes.pop(0)
            mw = np.mean(widths)
            if mj or mw / pw >= 0.35 or mw > 200:
                res.append("\n".join([c["text"] + self._line_tag(c, ZM) for c in lines]))
            else:
                logging.debug("REMOVED: " + "<<".join([c["text"] for c in lines]))

        return "\n\n".join(res)

    @staticmethod
    def total_page_number(fnm, binary=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                pdf = pdfplumber.open(fnm) if not binary else pdfplumber.open(BytesIO(binary))
            total_page = len(pdf.pages)
            pdf.close()
            return total_page
        except Exception:
            logging.exception("total_page_number")

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        # lefted_chars 用来存放后续流程里没有被成功并入文本框的残余字符。
        # 这里先清空，确保每次解析新 PDF 或新页范围时状态是干净的，避免上一次解析的数据串进来。
        self.lefted_chars = []
        # mean_height 记录每页字符/文本框的平均高度近似值。
        # 后面的排序、同行判断、文本合并都会依赖这个尺度信息，所以在 OCR 前先为每页准备容器。
        self.mean_height = []
        # mean_width 记录每页字符的典型宽度。
        # 它常被拿来做横向间距阈值，尤其在英文/数字间空格恢复时很有用。
        self.mean_width = []
        # boxes 保存每页最终产出的文本框。
        # __ocr() 会逐页 append，所以这里必须先重置为空列表。
        self.boxes = []
        # garbages 用来记录疑似噪声框、乱码框或后续清洗阶段要剔除的内容。
        # 先初始化为空，便于本次解析单独维护自己的“脏数据”集合。
        self.garbages = {}
        # page_cum_height 记录页高累计值，初始放一个 0 作为前缀和起点。
        # 后面布局分析会把“页内坐标”转成“整篇文档累计坐标”，这样跨页排序和定位会更统一。
        self.page_cum_height = [0]
        # page_layout 保存每页版面识别结果，后续 _layouts_rec() 会填充。
        self.page_layout = []
        # 记录这次处理从哪一页开始。
        # 后续日志、页号换算、异常提示都需要保留这个原始页偏移。
        self.page_from = page_from
        # 记录整个 __images__ 阶段耗时，便于看“读取 PDF + 转图片 + 抽文字层”花了多久。
        start = timer()
        try:
            # pdfplumber 在一些底层对象上不是完全线程安全的，所以这里通过全局锁串行打开 PDF。
            # 这是稳定性优先的工程取舍：牺牲一点并发，避免多文档并发时出现诡异崩溃或句柄冲突。
            with sys.modules[LOCK_KEY_pdfplumber]:
                # fnm 既可能是文件路径，也可能是二进制内容。
                # 统一在这里兼容两种输入，减少上层调用分支。
                with pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm)) as pdf:
                    # 保留 pdf 句柄到实例上，方便同一轮解析里别的子流程继续访问页对象和元数据。
                    self.pdf = pdf
                    # 把目标页范围转成图像。
                    # resolution=72 * zoomin 表示按 PDF 默认 72 DPI 的倍数放大；
                    # 放大的原因是 OCR 对小字和细线更敏感，分辨率过低会明显损伤检测召回。
                    # antialias=True 则是为了减轻锯齿，让字符边界更平滑，通常有利于检测和识别。
                    self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]

                    try:
                        # 优先从 PDF 文本层抽取原生字符，而不是一上来就全量 OCR。
                        # dedupe_chars() 会去掉重复字符，避免同一文字在 PDF 内部被多次绘制造成重复文本。
                        # 再用 _has_color(c) 过滤，是为了尽量排除不可见字符、装饰字符或异常颜色层带来的噪声。
                        # 这体现的是“能用文本层就先用文本层”的策略：更快、通常也比 OCR 更准。
                        self.page_chars = [[c for c in page.dedupe_chars().chars if self._has_color(c)] for page in self.pdf.pages[page_from:page_to]]
                    except Exception as e:
                        # 文本层抽取失败不能让整个 PDF 解析直接中断。
                        # 因为扫描件、本身编码异常的 PDF、本地解析兼容性问题，都可能让这里失败，
                        # 但只要页图还在，后面仍然可以退化成纯 OCR 路线完成解析。
                        logging.warning(f"Failed to extract characters for pages {page_from}-{page_to}: {str(e)}")
                        self.page_chars = [[] for _ in range(page_to - page_from)]  # If failed to extract, using empty list instead.

                    # 对每页抽出来的文本层做“乱码体检”。
                    # 一旦判定为乱码，就主动清空这一页的 page_chars，强制后续走 OCR 补救。
                    # 这里的核心思想是：错误的文本层往往比没有文本层更危险，
                    # 因为它会让系统误以为“已经抽到了文字”，结果把错误字符带进索引。
                    # 当前用了两种策略：
                    # 1) PUA / 未映射 CID 字符比例过高，常见于私有区乱码或字库映射失败；
                    # 2) 字体编码乱码，例如子集字体把中文错误映射成 ASCII 乱码。
                    for pi, page_ch in enumerate(self.page_chars):
                        # 空页或没抽到字符的页直接跳过，后续自然会走 OCR。
                        if not page_ch:
                            continue
                        # 策略 1：先抽样判断是否存在大量私有区字符、CID 残片这类“明显乱码”。
                        # 只采样前 200 个字符，是准确性和性能的折中；
                        # 对于乱码页，前面一段通常已经足够暴露问题，没必要整页全扫。
                        sample = page_ch if len(page_ch) <= 200 else page_ch[:200]
                        # 把字符对象里的 text 拼成一个样本文本，交给 _is_garbled_text() 做比例检测。
                        sample_text = "".join(c.get("text", "") for c in sample)
                        if self._is_garbled_text(sample_text, threshold=0.3):
                            logging.warning(
                                "Page %d: pdfplumber extracted mostly garbled characters (%d chars), "
                                "clearing to use OCR fallback.",
                                page_from + pi + 1, len(page_ch),
                            )
                            # 一旦发现文本层大概率不可用，就整页清空，让 __ocr() 不再尝试复用这些字符。
                            self.page_chars[pi] = []
                            continue
                        # 策略 2：检查是否是字体编码层面的乱码。
                        # 这类问题不一定表现为私有区字符，而是“看起来像普通 ASCII，
                        # 但其实应该是中文/日文/韩文被错误映射出来”的情况。
                        if self._is_garbled_by_font_encoding(page_ch):
                            logging.warning(
                                "Page %d: detected font-encoding garbled text "
                                "(subset fonts with no CJK output, %d chars), "
                                "clearing to use OCR fallback.",
                                page_from + pi + 1, len(page_ch),
                            )
                            # 同样直接清空该页原生字符，避免后续把错误文本并入 OCR 框。
                            self.page_chars[pi] = []

                    # total_page 记录整个 PDF 的总页数，不受 page_from/page_to 裁剪影响。
                    # 后续 UI 回调、页号展示和其它分页逻辑会用到这个全局页数。
                    self.total_page = len(self.pdf.pages)

        except Exception as e:
            # 这里兜底整个“打开 PDF + 转图片 + 抽文本层”的大流程异常。
            # 记录完整堆栈，方便定位是 PDF 损坏、渲染失败还是字符抽取失败。
            logging.exception(f"RAGFlowPdfParser __images__, exception: {e}")
        # 输出到这里为止的总耗时，覆盖文本层抽取和页图准备阶段。
        logging.info(f"__images__ dedupe_chars cost {timer() - start}s")

        # 说明页图已经准备完毕，后续可以正式进入 OCR。
        logging.debug("Images converted.")
        # 用抽样字符粗判整份 PDF 是否以英文为主。
        # 这里不是语言识别模型，而是一个轻量启发式判断：如果长串里英文/数字/英文标点占优，
        # 就把文档视作英文文档。
        # 这样做的原因是英文 PDF 往往更依赖空格和原生文本层，而中文扫描件更常直接走 OCR 补救。
        self.is_english = [
            re.search(r"[ a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i])))))
            for i in range(len(self.page_chars))
        ]
        # 如果超过半数页面都像英文页，就把整份文档标成英文。
        # 用“多数页投票”而不是“任意一页命中”，是为了降低目录页、封面页、表格页带来的误判。
        if sum([1 if e else 0 for e in self.is_english]) > len(self.page_images) / 2:
            self.is_english = True
        else:
            self.is_english = False

        async def __img_ocr(i, id, img, chars, limiter):
            # 在进入 __ocr() 前，先尝试恢复英文/数字之间可能丢失的空格。
            # PDF 文本层经常把相邻英文字符拆成独立字符对象，却不保留视觉上的空白。
            # 如果两个相邻字符都是拉丁文本，且间距足够大，就手动补一个空格，
            # 这样后面合并文本时更接近人眼看到的原始单词边界。
            j = 0
            while j + 1 < len(chars):
                if (
                    chars[j]["text"]
                    and chars[j + 1]["text"]
                    and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
                    and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                ):
                    chars[j]["text"] += " "
                j += 1

            # 如果配置了并发限制器，就用 semaphore 控制同时跑 OCR 的页数。
            # 这样做是为了在多 GPU / 多 worker 场景下避免瞬时把显存、CPU 线程、IO 一起打满。
            if limiter:
                async with limiter:
                    await thread_pool_exec(self.__ocr, i + 1, img, chars, zoomin, id)
            else:
                # 没有限流器时直接同步调用当前页的 __ocr()。
                self.__ocr(i + 1, img, chars, zoomin, id)

            # 每处理完 6 页回调一次进度。
            # 不按每页都回调，是为了减少 UI 更新过于频繁带来的额外开销。
            if callback and i % 6 == 5:
                callback((i + 1) * 0.6 / len(self.page_images))

        async def __img_ocr_launcher():
            def __ocr_preprocess():
                # 如果整份文档被判定为英文，这里故意不把 page_chars 传给 __ocr()。
                # 这是一个策略性取舍：英文 PDF 文本层常常存在粘连、断词、编码不稳等问题，
                # 某些情况下直接让 OCR 主导会更一致；而非英文文档更值得优先复用文本层。
                chars = self.page_chars[i] if not self.is_english else []
                # 用当前页字符高度的中位数估计“典型行高”。
                # 选中位数而不是均值，是因为它对特别大的标题字、小角标这类异常值更稳。
                self.mean_height.append(np.median(sorted([c["height"] for c in chars])) if chars else 0)
                # 同理估计典型字符宽度；如果当前页没有字符，就先给一个经验默认值 8。
                # 这个值不是绝对精确，只是给后续距离阈值一个保底尺度。
                self.mean_width.append(np.median(sorted([c["width"] for c in chars])) if chars else 8)
                # 记录当前页在 PDF 逻辑坐标里的高度。
                # 注意这里要除以 zoomin，因为 img.size 是放大后的像素尺寸，
                # 而后续版面框坐标希望统一回到 PDF 逻辑尺度。
                self.page_cum_height.append(img.size[1] / zoomin)
                return chars

            if self.parallel_limiter:
                # 有并行限制器时，说明当前环境允许多设备/多任务并行 OCR。
                # 这里先收集所有页任务，再统一 await，提高整体吞吐。
                tasks = []

                for i, img in enumerate(self.page_images):
                    # 先做当前页 OCR 前的统计准备，避免在真正启动任务后再修改共享状态。
                    chars = __ocr_preprocess()

                    # 通过取模把页面均匀分发到多个设备槽位上。
                    # 这不是严格调度器，但实现简单，通常足以把多页任务摊平到多个 GPU。
                    semaphore = self.parallel_limiter[i % settings.PARALLEL_DEVICES]

                    async def wrapper(i=i, img=img, chars=chars, semaphore=semaphore):
                        await __img_ocr(
                            i,
                            i % settings.PARALLEL_DEVICES,
                            img,
                            chars,
                            semaphore,
                        )

                    # 把每页 OCR 包成独立异步任务，后面统一并发执行。
                    tasks.append(asyncio.create_task(wrapper()))
                    # 主动让出一次事件循环，避免单次 for 循环长时间霸占调度。
                    await asyncio.sleep(0)

                try:
                    # 等待所有页 OCR 完成；一旦其中一个抛错，直接进入异常分支统一收尾。
                    await asyncio.gather(*tasks, return_exceptions=False)
                except Exception as e:
                    # 某页 OCR 失败时，记录错误并取消所有未完成任务。
                    # 这样可以避免部分任务继续跑，造成状态半成功半失败、难以推断。
                    logging.error(f"Error in OCR: {e}")
                    for t in tasks:
                        t.cancel()
                    # 再次 gather 是为了把取消后的任务都清理干净，避免悬空协程。
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

            else:
                # 没开启多设备并发时，按页顺序串行执行 OCR。
                # 这样最简单，也更容易复现和排查问题。
                for i, img in enumerate(self.page_images):
                    chars = __ocr_preprocess()
                    await __img_ocr(i, 0, img, chars, None)

        # 从这一刻开始计时纯 OCR 阶段，不再和前面的 PDF 读取/转图混在一起。
        start = timer()

        # 启动整份文档的逐页 OCR 调度器。
        asyncio.run(__img_ocr_launcher())

        # 记录 OCR 阶段耗时，方便区分瓶颈是在“取页图”还是“识别文本”。
        logging.info(f"__images__ {len(self.page_images)} pages cost {timer() - start}s")

        # 如果前面没能从文本层判断语言，而且 page_chars 基本为空，
        # 就退化成从 OCR 结果里再猜一次是否为英文文档。
        # 这样能覆盖扫描件 PDF：它没有文本层，只能依赖识别后的文字来判断语言倾向。
        if not self.is_english and not any([c for c in self.page_chars]) and self.boxes:
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(r"[ \na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join([b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]))

        # 打印最终的英文判定结果，方便观察策略是否生效。
        logging.debug(f"Is it English: {self.is_english}")

        # 把逐页高度列表转成前缀和。
        # 后面只要知道 box 在第几页，就能 O(1) 算出它映射到整篇文档后的累计 Y 坐标。
        self.page_cum_height = np.cumsum(self.page_cum_height)
        # page_cum_height 应该总是比 page_images 多一个前缀 0。
        assert len(self.page_cum_height) == len(self.page_images) + 1
        # 如果一个框都没识别出来，并且当前分辨率还不算太高，就自动放大 3 倍重试。
        # 这是典型的召回优先兜底策略：有些细小文字在低分辨率下完全检不出，
        # 提高渲染分辨率后检测器往往能恢复。
        # 上限 zoomin < 9 是为了防止无节制放大导致内存和耗时爆炸。
        if len(self.boxes) == 0 and zoomin < 9:
            self.__images__(fnm, zoomin * 3, page_from, page_to, callback)

    def __call__(self, fnm, need_image=True, zoomin=3, return_html=False, auto_rotate_tables=None):
        """
        Parse a PDF file.

        Args:
            fnm: PDF file path or binary content
            need_image: Whether to extract images
            zoomin: Zoom factor
            return_html: Whether to return tables in HTML format
            auto_rotate_tables: Whether to enable auto orientation correction for tables.
                               None: Use TABLE_AUTO_ROTATE env var setting (default: True)
                               True: Enable auto orientation correction
                               False: Disable auto orientation correction
        """
        if auto_rotate_tables is None:
            auto_rotate_tables = os.getenv("TABLE_AUTO_ROTATE", "true").lower() in ("true", "1", "yes")

        self.outlines = extract_pdf_outlines(fnm)
        self.__images__(fnm, zoomin)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin, auto_rotate=auto_rotate_tables)
        self._text_merge()
        self._concat_downward()
        self._filter_forpages()
        tbls = self._extract_table_figure(need_image, zoomin, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls

    def parse_into_bboxes(self, fnm, callback=None, zoomin=3):
        start = timer()
        self.outlines = extract_pdf_outlines(fnm)
        self.__images__(fnm, zoomin, callback=callback)
        if callback:
            callback(0.40, "OCR finished ({:.2f}s)".format(timer() - start))

        start = timer()
        self._layouts_rec(zoomin)
        if callback:
            callback(0.63, "Layout analysis ({:.2f}s)".format(timer() - start))

        # Read table auto-rotation setting from environment variable
        auto_rotate_tables = os.getenv("TABLE_AUTO_ROTATE", "true").lower() in ("true", "1", "yes")

        start = timer()
        self._table_transformer_job(zoomin, auto_rotate=auto_rotate_tables)
        if callback:
            callback(0.83, "Table analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._text_merge()
        self._concat_downward()
        self._naive_vertical_merge(zoomin)
        if callback:
            callback(0.92, "Text merged ({:.2f}s)".format(timer() - start))

        start = timer()
        tbls, figs = self._extract_table_figure(True, zoomin, True, True, True)

        def insert_table_figures(tbls_or_figs, layout_type):
            def min_rectangle_distance(rect1, rect2):
                pn1, left1, right1, top1, bottom1 = rect1
                pn2, left2, right2, top2, bottom2 = rect2
                if right1 >= left2 and right2 >= left1 and bottom1 >= top2 and bottom2 >= top1:
                    return 0
                if right1 < left2:
                    dx = left2 - right1
                elif right2 < left1:
                    dx = left1 - right2
                else:
                    dx = 0
                if bottom1 < top2:
                    dy = top2 - bottom1
                elif bottom2 < top1:
                    dy = top1 - bottom2
                else:
                    dy = 0
                return math.sqrt(dx * dx + dy * dy)  # + (pn2-pn1)*10000

            for (img, txt), poss in tbls_or_figs:
                # Positions coming from _extract_table_figure carry absolute 0-based page
                # indices (page_from offset). Convert back to chunk-local indices so we
                # stay consistent with self.boxes/page_cum_height, which are all relative
                # to the current parsing window.
                local_poss = []
                for pn, left, right, top, bott in poss:
                    local_pn = pn - self.page_from
                    if 0 <= local_pn < len(self.page_cum_height) - 1:
                        local_poss.append((local_pn, left, right, top, bott))
                    else:
                        logging.debug(f"Skip out-of-range table/figure position pn={pn}, page_from={self.page_from}")
                if not local_poss:
                    logging.debug("No valid local positions for table/figure; skip insertion.")
                    continue

                if isinstance(txt, list):
                    txt = "\n".join(txt)
                pn, left, right, top, bott = local_poss[0]
                insert_at = len(self.boxes)
                bboxes = [(i, (b["page_number"], b["x0"], b["x1"], b["top"], b["bottom"])) for i, b in enumerate(self.boxes)]
                if bboxes:
                    dists = [
                        (min_rectangle_distance((cand_pn, cand_left, cand_right, cand_top + self.page_cum_height[cand_pn], cand_bott + self.page_cum_height[cand_pn]), rect), i)
                        for i, rect in bboxes
                        for cand_pn, cand_left, cand_right, cand_top, cand_bott in local_poss
                    ]
                    if dists:
                        nearest_bbox_idx = int(np.argmin([dist for dist, _ in dists]))
                        insert_at, _ = bboxes[dists[nearest_bbox_idx][-1]]
                        if self.boxes[insert_at]["bottom"] < top + self.page_cum_height[pn]:
                            insert_at += 1
                else:
                    logging.debug("No text boxes available; append %s block directly.", layout_type)
                self.boxes.insert(
                    insert_at,
                    {
                        "page_number": pn + 1,
                        "x0": left,
                        "x1": right,
                        "top": top + self.page_cum_height[pn],
                        "bottom": bott + self.page_cum_height[pn],
                        "layout_type": layout_type,
                        "text": txt,
                        "image": img,
                        "positions": [[pn + 1, int(left), int(right), int(top), int(bott)]],
                    },
                )

        for b in self.boxes:
            b["position_tag"] = self._line_tag(b, zoomin)
            b["image"] = self.crop(b["position_tag"], zoomin)
            b["positions"] = [[pos[0][-1] + 1, *pos[1:]] for pos in RAGFlowPdfParser.extract_positions(b["position_tag"])]

        insert_table_figures(tbls, "table")
        insert_table_figures(figs, "figure")
        if callback:
            callback(1, "Structured ({:.2f}s)".format(timer() - start))
        return deepcopy(self.boxes)

    @staticmethod
    def remove_tag(txt):
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

    @staticmethod
    def extract_positions(txt):
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text, ZM=3, need_position=False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            if need_position:
                return None, None
            return

        if not getattr(self, "page_images", None):
            logging.warning("crop called without page images; skipping image generation.")
            if need_position:
                return None, None
            return

        page_count = len(self.page_images)

        filtered_poss = []
        for pns, left, right, top, bottom in poss:
            if not pns:
                logging.warning("Empty page index list in crop; skipping this position.")
                continue
            valid_pns = [p for p in pns if 0 <= p < page_count]
            if not valid_pns:
                logging.warning(f"All page indices {pns} out of range for {page_count} pages; skipping.")
                continue
            filtered_poss.append((valid_pns, left, right, top, bottom))

        poss = filtered_poss
        if not poss:
            logging.warning("No valid positions after filtering; skip cropping.")
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        first_page_idx = pos[0][0]
        poss.insert(0, ([first_page_idx], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        last_page_idx = pos[0][-1]
        if not (0 <= last_page_idx < page_count):
            logging.warning(f"Last page index {last_page_idx} out of range for {page_count} pages; skipping crop.")
            if need_position:
                return None, None
            return
        last_page_height = self.page_images[last_page_idx].size[1] / ZM
        poss.append(
            (
                [last_page_idx],
                pos[1],
                pos[2],
                min(last_page_height, pos[4] + GAP),
                min(last_page_height, pos[4] + 120),
            )
        )

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            if 0 < ii < len(poss) - 1:
                right = max(left + 10, right)
            else:
                right = left + max_width
            bottom *= ZM
            for pn in pns[1:]:
                if 0 <= pn - 1 < page_count:
                    bottom += self.page_images[pn - 1].size[1]
                else:
                    logging.warning(f"Page index {pn}-1 out of range for {page_count} pages during crop; skipping height accumulation.")

            if not (0 <= pns[0] < page_count):
                logging.warning(f"Base page index {pns[0]} out of range for {page_count} pages during crop; skipping this segment.")
                continue

            imgs.append(self.page_images[pns[0]].crop((left * ZM, top * ZM, right * ZM, min(bottom, self.page_images[pns[0]].size[1]))))
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, left, right, top, min(bottom, self.page_images[pns[0]].size[1]) / ZM))
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                if not (0 <= pn < page_count):
                    logging.warning(f"Page index {pn} out of range for {page_count} pages during crop; skipping this page.")
                    continue
                imgs.append(self.page_images[pn].crop((left * ZM, 0, right * ZM, min(bottom, self.page_images[pn].size[1]))))
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, left, right, 0, min(bottom, self.page_images[pn].size[1]) / ZM))
                bottom -= self.page_images[pn].size[1]

        if not imgs:
            if need_position:
                return None, None
            return
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    def get_position(self, bx, ZM):
        poss = []
        pn = bx["page_number"]
        top = bx["top"] - self.page_cum_height[pn - 1]
        bott = bx["bottom"] - self.page_cum_height[pn - 1]
        poss.append((pn, bx["x0"], bx["x1"], top, min(bott, self.page_images[pn - 1].size[1] / ZM)))
        while bott * ZM > self.page_images[pn - 1].size[1]:
            bott -= self.page_images[pn - 1].size[1] / ZM
            top = 0
            pn += 1
            poss.append((pn, bx["x0"], bx["x1"], top, min(bott, self.page_images[pn - 1].size[1] / ZM)))
        return poss


class PlainParser:
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        lines = []
        try:
            self.pdf = pdf2_read(filename if isinstance(filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])
        except Exception:
            logging.exception("Outlines exception")
        self.outlines = extract_pdf_outlines(filename)

        return [(line, "") for line in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


class VisionParser(RAGFlowPdfParser):
    def __init__(self, vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model
        self.outlines = []

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                self.pdf = pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm))
                self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]
                self.total_page = len(self.pdf.pages)
        except Exception:
            self.page_images = None
            self.total_page = 0
            logging.exception("VisionParser __images__")

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)
        zoomin = kwargs.get("zoomin", 3)
        self.__images__(fnm=filename, zoomin=zoomin, page_from=from_page, page_to=to_page, callback=callback)

        total_pdf_pages = self.total_page

        start_page = max(0, from_page)
        end_page = min(to_page, total_pdf_pages)

        all_docs = []

        for idx, img_binary in enumerate(self.page_images or []):
            pdf_page_num = idx  # 0-based
            if pdf_page_num < start_page or pdf_page_num >= end_page:
                continue

            from rag.app.picture import vision_llm_chunk as picture_vision_llm_chunk

            text = picture_vision_llm_chunk(
                binary=img_binary,
                vision_model=self.vision_model,
                prompt=vision_llm_describe_prompt(page=pdf_page_num + 1),
                callback=callback,
            )

            if kwargs.get("callback"):
                kwargs["callback"](idx * 1.0 / len(self.page_images), f"Processed: {idx + 1}/{len(self.page_images)}")

            if text:
                width, height = self.page_images[idx].size
                all_docs.append((text, f"@@{pdf_page_num + 1}\t{0.0:.1f}\t{width / zoomin:.1f}\t{0.0:.1f}\t{height / zoomin:.1f}##"))
        return all_docs, []


if __name__ == "__main__":
    pass
