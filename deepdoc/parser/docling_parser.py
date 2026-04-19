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
from __future__ import annotations

import logging
import re
import base64
import os
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pdfplumber
import requests
from PIL import Image

try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None  

try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
except Exception:
    class RAGFlowPdfParser:  
        pass

from deepdoc.parser.utils import extract_pdf_outlines


class DoclingContentType(str, Enum):
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    EQUATION = "equation"


@dataclass
class _BBox:
    page_no: int  
    x0: float
    y0: float
    x1: float
    y1: float


def _extract_bbox_from_prov(item, prov_attr: str = "prov") -> Optional[_BBox]:
    prov = getattr(item, prov_attr, None)
    if not prov:
        return None
    
    prov_item = prov[0] if isinstance(prov, list) else prov
    pn = getattr(prov_item, "page_no", None)
    bb = getattr(prov_item, "bbox", None)
    if pn is None or bb is None:
        return None
    
    coords = [getattr(bb, attr) for attr in ("l", "t", "r", "b")]
    if None in coords:
        return None
    
    return _BBox(page_no=int(pn), x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])


class DoclingParser(RAGFlowPdfParser):
    def __init__(self, docling_server_url: str = "", request_timeout: int = 600):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.page_images: list[Image.Image] = []
        self.page_from = 0
        self.page_to = 10_000
        self.outlines = []
        self.docling_server_url = (docling_server_url or "").rstrip("/")
        self.request_timeout = request_timeout

    def _effective_server_url(self, docling_server_url: Optional[str] = None) -> str:
        return (docling_server_url or self.docling_server_url or "").rstrip("/") or (
            os.environ.get("DOCLING_SERVER_URL", "").rstrip("/")
        )

    @staticmethod
    def _is_http_endpoint_valid(url: str, timeout: int = 5) -> bool:
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return response.status_code in [200, 301, 302, 307, 308]
        except Exception:
            try:
                response = requests.get(url, timeout=timeout, allow_redirects=True)
                return response.status_code in [200, 301, 302, 307, 308]
            except Exception:
                return False

    def check_installation(self, docling_server_url: Optional[str] = None) -> bool:
        server_url = self._effective_server_url(docling_server_url)
        if server_url:
            for path in ("/openapi.json", "/docs", "/v1/convert/source"):
                if self._is_http_endpoint_valid(f"{server_url}{path}", timeout=5):
                    return True
            self.logger.warning(f"[Docling] external server not reachable: {server_url}")
            return False

        if DocumentConverter is None:
            self.logger.warning("[Docling] 'docling' is not importable, please: pip install docling")
            return False
        try:
            _ = DocumentConverter()
            return True
        except Exception as e:
            self.logger.error(f"[Docling] init DocumentConverter failed: {e}")
            return False

    def __images__(self, fnm, zoomin: int = 1, page_from=0, page_to=600, callback=None):
        self.page_from = page_from
        self.page_to = page_to
        bytes_io = None
        try:
            if not isinstance(fnm, (str, PathLike)):
                bytes_io = BytesIO(fnm)

            opener = pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(bytes_io)
            with opener as pdf:
                pages = pdf.pages[page_from:page_to]
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for p in pages]
        except Exception as e:
            self.page_images = []
            self.logger.exception(e)
        finally:
            if bytes_io:
                bytes_io.close()

    def _make_line_tag(self,bbox: _BBox) -> str:
        if bbox is None:
            return ""
        x0,x1, top, bott = bbox.x0, bbox.x1, bbox.y0, bbox.y1
        if hasattr(self, "page_images") and self.page_images and len(self.page_images) >= bbox.page_no:
            _, page_height = self.page_images[bbox.page_no-1].size
            top, bott = page_height-top ,page_height-bott
        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(
            bbox.page_no, x0,x1, top, bott
        )

    @staticmethod
    def extract_positions(txt: str) -> list[tuple[list[int], float, float, float, float]]:
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text: str, ZM: int = 1, need_position: bool = False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            return (None, None) if need_position else None

        GAP = 6
        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1], pos[4] + GAP), min(self.page_images[pos[0][-1]].size[1], pos[4] + 120)))
        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            if bottom <= top:
                bottom = top + 4
            img0 = self.page_images[pns[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))
            
            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss)-1:
                positions.append((pns[0] + self.page_from, x0, x1, y0, y1))
            remain_bottom = bottom - img0.size[1]
            for pn in pns[1:]:
                if remain_bottom <= 0:
                    break
                page = self.page_images[pn]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(remain_bottom, page.size[1]))
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, x0, x1, y0, y1))
                remain_bottom -= page.size[1]

        if not imgs:
            return (None, None) if need_position else None

        height = sum(i.size[1] + GAP for i in imgs)
        width = max(i.size[0] for i in imgs)
        pic = Image.new("RGB", (width, int(height)), (245, 245, 245))
        h = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(h)))
            h += img.size[1] + GAP

        return (pic, positions) if need_position else pic

    def _iter_doc_items(self, doc) -> Iterable[tuple[str, Any, Optional[_BBox]]]:
        for t in getattr(doc, "texts", []):
            parent = getattr(t, "parent", "")
            ref = getattr(parent, "cref", "")
            label = getattr(t, "label", "")
            if (label in ("section_header", "text") and ref in ("#/body",)) or label in ("list_item",):
                text = getattr(t, "text", "") or ""
                bbox = _extract_bbox_from_prov(t)
                yield (DoclingContentType.TEXT.value, text, bbox)

        for item in getattr(doc, "texts", []):
            if getattr(item, "label", "") in ("FORMULA",):
                text = getattr(item, "text", "") or ""
                bbox = _extract_bbox_from_prov(item)
                yield (DoclingContentType.EQUATION.value, text, bbox)

    def _transfer_to_sections(self, doc, parse_method: str) -> list[tuple[str, ...]]:
        sections: list[tuple[str, ...]] = []
        for typ, payload, bbox in self._iter_doc_items(doc):
            if typ == DoclingContentType.TEXT.value:
                section = payload.strip()
                if not section:
                    continue
            elif typ == DoclingContentType.EQUATION.value:
                section = payload.strip()
            else:
                continue
            
            tag = self._make_line_tag(bbox) if isinstance(bbox,_BBox) else ""
            if parse_method in {"manual", "pipeline"}:
                sections.append((section, typ, tag))
            elif parse_method == "paper":
                sections.append((section + tag, typ))
            else:
                sections.append((section, tag))
        return sections

    def cropout_docling_table(self, page_no: int, bbox: tuple[float, float, float, float], zoomin: int = 1):
        if not getattr(self, "page_images", None):
            return None, ""

        idx = (page_no - 1) - getattr(self, "page_from", 0)
        if idx < 0 or idx >= len(self.page_images):
            return None, ""

        page_img = self.page_images[idx]
        W, H = page_img.size
        left, top, right, bott = bbox

        x0 = float(left)
        y0 = float(H-top)
        x1 = float(right)
        y1 = float(H-bott)

        x0, y0 = max(0.0, min(x0, W - 1)), max(0.0, min(y0, H - 1))
        x1, y1 = max(x0 + 1.0, min(x1, W)), max(y0 + 1.0, min(y1, H))

        try:
            crop = page_img.crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
        except Exception:
            return None, ""

        pos = (page_no-1 if page_no>0 else 0, x0, x1, y0, y1)
        return crop, [pos]

    def _transfer_to_tables(self, doc):
        tables = []
        for tab in getattr(doc, "tables", []):
            img = None
            positions = ""
            bbox = _extract_bbox_from_prov(tab)
            if bbox:
                img, positions = self.cropout_docling_table(bbox.page_no, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
            html = ""
            try:
                html = tab.export_to_html(doc=doc)
            except Exception:
                pass
            tables.append(((img, html), positions if positions else ""))
        for pic in getattr(doc, "pictures", []):
            img = None
            positions = ""
            bbox = _extract_bbox_from_prov(pic)
            if bbox:
                img, positions = self.cropout_docling_table(bbox.page_no, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
            captions = ""
            try:
                captions = pic.caption_text(doc=doc)
            except Exception:
                pass
            tables.append(((img, [captions]), positions if positions else ""))
        return tables

    @staticmethod
    def _sections_from_remote_text(text: str, parse_method: str) -> list[tuple[str, ...]]:
        txt = (text or "").strip()
        if not txt:
            return []
        if parse_method in {"manual", "pipeline"}:
            return [(txt, DoclingContentType.TEXT.value, "")]
        if parse_method == "paper":
            return [(txt, DoclingContentType.TEXT.value)]
        return [(txt, "")]

    @staticmethod
    def _extract_remote_document_entries(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("document"), dict):
            return [payload["document"]]
        if isinstance(payload.get("documents"), list):
            return [d for d in payload["documents"] if isinstance(d, dict)]
        if isinstance(payload.get("results"), list):
            docs = []
            for it in payload["results"]:
                if isinstance(it, dict):
                    if isinstance(it.get("document"), dict):
                        docs.append(it["document"])
                    elif isinstance(it.get("result"), dict):
                        docs.append(it["result"])
                    else:
                        docs.append(it)
            return docs
        return []

    def _parse_pdf_remote(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        parse_method: str = "raw",
        docling_server_url: Optional[str] = None,
        request_timeout: Optional[int] = None,
    ):
        # 解析最终生效的 Docling Server 地址。
        # 已读 _effective_server_url()：优先级是“调用参数 > 实例字段 > 环境变量 DOCLING_SERVER_URL”。
        # 远程模式必须先把服务地址确定下来，否则后面的 HTTP 请求无从发起。
        server_url = self._effective_server_url(docling_server_url)
        # 没有服务地址时直接报错。
        # 这里不做本地回退，是因为进入这个函数本身就说明调用方明确要走远程 Docling 路线。
        if not server_url:
            raise RuntimeError("[Docling] DOCLING_SERVER_URL is not configured.")

        # 解析请求超时时间。
        # request_timeout 是本次调用的显式参数，优先级高于实例默认值 self.request_timeout。
        # 工业上这类网络参数一般都允许“按请求覆写默认值”，便于不同任务按文档大小动态调参。
        timeout = request_timeout or self.request_timeout
        # 统一准备要发给远程服务的 PDF 原始字节。
        # 远程接口最终只认 bytes，不关心调用方最初传的是路径、bytes 还是 BytesIO。
        if binary is not None:
            if isinstance(binary, (bytes, bytearray)):
                # 直接传入 bytes / bytearray 时，转成标准 bytes 即可。
                pdf_bytes = bytes(binary)
            else:
                # BytesIO 这类对象走 getbuffer()，避免多余的逐字节拷贝逻辑散落在外面。
                pdf_bytes = bytes(binary.getbuffer())
        else:
            # 如果没有直接传 binary，就从 filepath 读取本地 PDF 文件。
            src_path = Path(filepath)
            # 文件不存在时立即抛错。
            # 远程解析虽然最终走网络，但本地输入文件还是必须先准备好。
            if not src_path.exists():
                raise FileNotFoundError(f"PDF not found: {src_path}")
            with open(src_path, "rb") as f:
                pdf_bytes = f.read()

        if callback:
            # 通知上层：已经进入远程请求阶段。
            # 0.2 是阶段型进度，不是严格网络进度；目的是让 UI/任务系统知道已经开始出网调用。
            callback(0.2, f"[Docling] Requesting external server: {server_url}")

        # filename 用于放进请求体，便于远程服务做日志、调试和文件名关联。
        # 没有合法名字时退化成 input.pdf，保证协议字段完整。
        filename = Path(filepath).name or "input.pdf"
        # 把 PDF 字节流转成 base64 字符串。
        # 原因是当前远程接口走 JSON body，JSON 不适合直接承载二进制，base64 是最常见的工业做法。
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        # 第一套请求体，对应当前优先尝试的 /v1/convert/source 接口。
        # 这里显式声明 from_formats / to_formats，是为了告诉 Docling 服务：
        # 输入是 pdf，输出希望拿到 json / md / text 三种表示。
        v1_payload = {
            "options": {
                "from_formats": ["pdf"],
                "to_formats": ["json", "md", "text"],
            },
            "sources": [
                {
                    "kind": "file",
                    "filename": filename,
                    "base64_string": b64,
                }
            ],
        }
        # 第二套请求体，对应历史/兼容接口 /v1alpha/convert/source。
        # 和 v1 的核心差异是字段名从 sources 变成了 file_sources。
        # 这段代码的本质是在做 API 版本兼容，避免服务端升级不同步时客户端完全不可用。
        v1alpha_payload = {
            "options": {
                "from_formats": ["pdf"],
                "to_formats": ["json", "md", "text"],
            },
            "file_sources": [
                {
                    "filename": filename,
                    "base64_string": b64,
                }
            ],
        }
        # errors 用来累计多个 endpoint 的失败信息。
        # 这样如果两个版本接口都失败，最终报错信息会更完整，方便排障。
        errors = []
        # response_json 最终保存成功接口返回的 JSON。
        # 先置 None，后面作为“是否已有成功响应”的判定标志。
        response_json = None
        # 依次尝试当前版 v1 接口和兼容版 v1alpha 接口。
        # 工业上这种“新接口优先，旧接口兜底”的策略很常见，
        # 能在服务升级过渡期显著降低客户端兼容成本。
        for endpoint, payload in (
            ("/v1/convert/source", v1_payload),
            ("/v1alpha/convert/source", v1alpha_payload),
        ):
            try:
                # 向 Docling Server 发起 JSON POST 请求。
                # timeout 控制整个请求生命周期，防止大文件或服务异常时线程长期挂死。
                resp = requests.post(
                    f"{server_url}{endpoint}",
                    json=payload,
                    timeout=timeout,
                )
                # 2xx/3xx 之前这里只把“<300”视为成功。
                # 一旦成功，就直接解析 JSON 并停止继续尝试后续 endpoint。
                if resp.status_code < 300:
                    response_json = resp.json()
                    break
                # 非成功状态码时，把 endpoint、状态码和部分响应正文都收集起来。
                # 只截前 300 个字符，是为了保留关键信息同时避免错误信息过长。
                errors.append(f"{endpoint}: HTTP {resp.status_code} {resp.text[:300]}")
            except Exception as exc:
                # 网络异常、超时、JSON 传输错误等都在这里收集。
                # 注意这里只记录并继续尝试另一个兼容 endpoint，而不是立刻终止。
                errors.append(f"{endpoint}: {exc}")

        # 如果两个 endpoint 都失败，统一抛出聚合错误。
        # 这样调用方能一次看到所有尝试路径的失败原因，而不是只看到最后一次错误。
        if response_json is None:
            raise RuntimeError("[Docling] remote convert failed: " + " | ".join(errors))

        # 从远程 JSON 里提取“文档对象列表”。
        # 已读 _extract_remote_document_entries()：
        # - 兼容 document / documents / results 三种外层结构；
        # - results 内部又兼容 document / result / 直接对象三种形态。
        # 这一步本质是在做远程协议宽松解析，降低服务端返回 shape 演进对客户端的破坏。
        docs = self._extract_remote_document_entries(response_json)
        # 没提取到任何文档对象时直接报错。
        # 说明服务虽然返回了 JSON，但并不是当前客户端理解的有效解析结果。
        if not docs:
            raise RuntimeError("[Docling] remote response does not contain parsed documents.")

        # sections 保存最终线性文本结果。
        # 远程模式当前不构造结构化 tables，所以 tables 先固定为空列表。
        sections: list[tuple[str, ...]] = []
        tables = []
        # 逐个处理远程返回的文档对象。
        for doc in docs:
            # 远程模式优先取 md_content。
            # 原因是 Markdown 比纯 text 更保留文档结构，比如标题、列表、简单表格等。
            md = doc.get("md_content")
            # text_content 是 markdown 不可用时的次级回退。
            txt = doc.get("text_content")
            if isinstance(md, str) and md.strip():
                # 把 Markdown 文本转换成和本地模式尽量兼容的 section 结构。
                # 已读 _sections_from_remote_text()：它会根据 parse_method 组装不同 shape，
                # 但因为远程模式通常没有可靠 bbox，所以位置字段会是空字符串。
                sections.extend(self._sections_from_remote_text(md, parse_method=parse_method))
            elif isinstance(txt, str) and txt.strip():
                # 没有 markdown 时，退化成纯文本 sections。
                sections.extend(self._sections_from_remote_text(txt, parse_method=parse_method))

            # 某些服务版本会把更完整内容包进 json_content。
            # 这里再做一次 fallback，但只在主流程还没产出 sections 时才启用，
            # 避免 md_content/text_content 与 json_content 重复拼接导致重复内容。
            json_content = doc.get("json_content")
            if isinstance(json_content, dict):
                # 当前只从 json_content 里再兜底取 md_content。
                md_fallback = json_content.get("md_content")
                if isinstance(md_fallback, str) and md_fallback.strip() and not sections:
                    sections.extend(self._sections_from_remote_text(md_fallback, parse_method=parse_method))

        if callback:
            # 通知上层远程解析最终抽到了多少 sections。
            callback(0.95, f"[Docling] Remote sections: {len(sections)}")
        # 返回远程解析结果。
        # 目前远程模式只稳定输出 sections，tables 保持空列表。
        # 这是一种典型的工业折中：先把主文本链路做稳定，再逐步补表格/图片等增强结构。
        return sections, tables

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None, 
        lang: Optional[str] = None,        
        method: str = "auto",             
        delete_output: bool = True,
        parse_method: str = "raw",
        docling_server_url: Optional[str] = None,
        request_timeout: Optional[int] = None,
    ):
        # 先提取 PDF 大纲。
        # 已读 extract_pdf_outlines()：它会尽量从 PDF 目录/书签结构里抽出章节信息，
        # 这是文档级元数据，不依赖 Docling 本身的正文解析结果，所以应尽早提取。
        # 这里如果传了 binary，就优先用 binary；否则回退到 filepath。
        # 这样做的原因是调用方可能只给了内存中的 PDF 数据，没有稳定落盘文件。
        self.outlines = extract_pdf_outlines(binary if binary is not None else filepath)

        # 检查当前 Docling 是否可用。
        # 已读 check_installation()：
        # 1) 如果存在 docling_server_url / self.docling_server_url / 环境变量 DOCLING_SERVER_URL，
        #    就优先探测远程 Docling Server 是否可达；
        # 2) 否则检查本地 docling 包是否成功导入，并尝试实例化 DocumentConverter()。
        # 这样设计的原理是把“远程服务模式”和“本地库模式”统一收敛到一个可用性检查入口，
        # 避免主流程里到处散落环境判断。
        # 工业上这是很常见的做法：先做 fail-fast 探测，避免后面半路才因为依赖缺失报更隐蔽的错误。
        if not self.check_installation(docling_server_url=docling_server_url):
            # 这里直接抛错而不是静默降级，是因为当前 parser 本身就是 Docling parser。
            # 一旦 Docling 不可用，再继续往下执行只会产生误导性结果或更深层异常。
            raise RuntimeError("Docling not available, please install `docling`")

        # 解析最终生效的 Docling Server 地址。
        # 已读 _effective_server_url()：优先级是
        # 调用参数 > 实例字段 > 环境变量 DOCLING_SERVER_URL。
        # 这么做是为了让显式调用参数拥有最高控制权，符合运维和调试直觉。
        server_url = self._effective_server_url(docling_server_url)
        # 如果配置了远程 Docling Server，就直接走远程解析分支。
        # 已读 _parse_pdf_remote()：
        # - 它会把 PDF 转成 base64 塞进 JSON 请求体；
        # - 依次尝试 /v1/convert/source 与 /v1alpha/convert/source 两套接口；
        # - 再从返回 JSON 中提取 md_content / text_content 并转换成 sections。
        # 为什么远程优先：一旦显式配置了 server_url，通常就表示部署者希望统一用服务侧资源，
        # 不应该再偷偷回退到本地 DocumentConverter。
        # 工业考量是“显式配置优先”，否则会让环境行为不可预测。
        if server_url:
            return self._parse_pdf_remote(
                filepath=filepath,
                binary=binary,
                callback=callback,
                parse_method=parse_method,
                docling_server_url=server_url,
                request_timeout=request_timeout,
            )

        # 走到这里说明没有启用远程服务，只能走本地 DocumentConverter 模式。
        if binary is not None:
            # 如果输入是内存中的 PDF 二进制，而不是现成文件路径，
            # 这里需要先把它落成一个临时 PDF 文件。
            # 原因是当前本地 Docling 这条链最终调用的是 conv.convert(str(src_path))，
            # 它要求的是文件路径，而不是 BytesIO。
            # output_dir 允许调用方指定临时输出目录；否则默认落在当前目录下的 .docling_tmp。
            # 这是一个典型的工程取舍：落盘会多一次 IO，但兼容性最好、实现也最稳定。
            tmpdir = Path(output_dir) if output_dir else Path.cwd() / ".docling_tmp"
            # 确保临时目录存在。
            # parents=True 允许递归创建父目录，exist_ok=True 保证重复调用时不报错。
            tmpdir.mkdir(parents=True, exist_ok=True)
            # 临时文件名尽量复用原 filepath 的文件名，便于排障和日志定位；
            # 如果 filepath 没有名字，就退化成 input.pdf。
            name = Path(filepath).name or "input.pdf"
            tmp_pdf = tmpdir / name
            # 把 binary 写入临时 PDF。
            with open(tmp_pdf, "wb") as f:
                if isinstance(binary, (bytes, bytearray)):
                    # bytes / bytearray 可以直接写。
                    f.write(binary)
                else:
                    # 如果是 BytesIO 一类对象，就从 buffer 里取出底层字节。
                    f.write(binary.getbuffer())
            # 后续所有本地解析都统一使用这个落盘后的 src_path。
            src_path = tmp_pdf
        else:
            # 调用方本来就传了文件路径，就直接使用该路径。
            src_path = Path(filepath)
            # 本地文件不存在时立即报错。
            # 这里不做静默兜底，因为解析器无法凭空构造输入文件。
            if not src_path.exists():
                raise FileNotFoundError(f"PDF not found: {src_path}")

        if callback:
            # 通知上层“开始转换”。
            # 0.1 不是严格的真实进度，而是一个阶段性进度标记，
            # 用于让 UI / 任务系统知道已经进入 Docling 正文转换阶段。
            callback(0.1, f"[Docling] Converting: {src_path}")

        try:
            # 预渲染 PDF 页图。
            # 已读 __images__()：它会用 pdfplumber 把 page_from:page_to 范围内的页面渲染成 PIL 图像，
            # 存入 self.page_images。
            # 这些页图不是 Docling 转文本必须的，但后续：
            # - _make_line_tag() 需要页高做坐标翻转；
            # - crop() / cropout_docling_table() 需要裁图；
            # 因此这里尽量先准备好页面图。
            # zoomin=1 的考量是：DoclingParser 这里主要为了位置与裁图，不追求 OCR 那种高倍放大。
            self.__images__(str(src_path), zoomin=1)
        except Exception as e:
            # 页图渲染失败只记 warning，不中断主流程。
            # 原因是 Docling 的正文解析和页图裁图是两条相对独立的能力链：
            # 即便裁图增强失败，文本解析仍然可能成功。
            # 工业上这类“增强能力失败不拖垮主能力”的容错非常常见。
            self.logger.warning(f"[Docling] render pages failed: {e}")

        # 初始化本地 Docling 转换器。
        # 这里在 check_installation() 之后再次实例化，是因为真正解析仍然需要一个工作中的 converter 实例。
        conv = DocumentConverter()  
        # 调用 Docling 把 PDF 文件路径转换成文档对象。
        # 这是本地模式的核心解析动作。
        conv_res = conv.convert(str(src_path))
        # 从转换结果里取出 Docling document。
        # 后面 sections 和 tables 的提取都会围绕这个 document 展开。
        doc = conv_res.document
        if callback:
            # 通知上层“Docling 已经完成主解析”。
            # getattr(doc, 'num_pages', 'n/a') 是防御式写法：
            # 某些版本/实现的 document 可能没有 num_pages 属性，不能因此让进度回调崩掉。
            callback(0.7, f"[Docling] Parsed doc: {getattr(doc, 'num_pages', 'n/a')} pages")

        # 把 Docling document 转成 RAGFlow 的 sections。
        # 已读 _transfer_to_sections()：
        # - 它内部会调用 _iter_doc_items()，筛选正文 text / section_header / list_item / FORMULA；
        # - 对每个 item 提取 text 和 bbox；
        # - 再根据 parse_method 组装成不同 shape 的 section 元组。
        # 为什么单独做这一层转换：RAGFlow 下游并不想依赖 Docling 原生对象结构，
        # 而是需要稳定统一的 section 表示。
        sections = self._transfer_to_sections(doc, parse_method=parse_method)
        # 把 Docling document 里的 tables 和 pictures 转成 RAGFlow 的 tables 结果。
        # 已读 _transfer_to_tables()：
        # - 对表格会尝试 export_to_html(doc=doc)；
        # - 对表格和图片都会提取 provenance bbox，并从 page_images 中裁图；
        # - 图片还会额外抽 caption_text(doc=doc)。
        # 这里统一命名为 tables，是因为在 RAGFlow 下游，表格和图片都属于“富内容块”。
        tables = self._transfer_to_tables(doc)

        if callback:
            # 通知上层 sections / tables 数量，便于日志、监控和 UI 展示。
            callback(0.95, f"[Docling] Sections: {len(sections)}, Tables: {len(tables)}")

        if binary is not None and delete_output:
            # 只有“输入原本来自 binary 且允许删除”时，才清理临时 PDF。
            # 这样不会误删用户原始文件，同时默认也不会让 .docling_tmp 无限堆积。
            # missing_ok=True 允许文件已不存在时仍然安静返回，更适合清理型代码。
            try:
                Path(src_path).unlink(missing_ok=True)
            except Exception:
                # 清理失败不影响主结果返回。
                # 这是典型的 best-effort 清理策略：资源回收重要，但不能盖过主成功路径。
                pass

        if callback:
            # 最终完成回调。
            callback(1.0, "[Docling] Done.")
        # 返回统一格式的解析结果：
        # - sections: 文本/公式等线性内容
        # - tables: 表格与图片等富内容块
        return sections, tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = DoclingParser()
    print("Docling available:", parser.check_installation())
    sections, tables = parser.parse_pdf(filepath="test_docling/toc.pdf", binary=None)
    print(len(sections), len(tables))
