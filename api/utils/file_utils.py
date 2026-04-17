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


# Standard library imports
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from io import BytesIO

import pdfplumber
from PIL import Image

# Local imports
from api.constants import FILE_NAME_LEN_LIMIT, IMG_BASE64_PREFIX
from api.db import FileType

# Robustness and resource limits: reject oversized inputs to avoid DoS and OOM.
MAX_BLOB_SIZE_THUMBNAIL = 50 * 1024 * 1024  # 50 MiB for thumbnail generation
MAX_BLOB_SIZE_PDF = 100 * 1024 * 1024  # 100 MiB for PDF repair / read
GHOSTSCRIPT_TIMEOUT_SEC = 120  # Timeout for Ghostscript subprocess

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


def _normalize_filename_for_type(filename):
    """Extract a safe basename for type detection. Returns (normalized_str, True) or ("", False)."""
    if filename is None:
        return "", False
    if not isinstance(filename, str):
        return "", False
    base = os.path.basename(filename).strip()
    if not base or len(base) > FILE_NAME_LEN_LIMIT:
        return "", False
    return base.lower(), True


def filename_type(filename):
    """Return file type from extension. Handles None, empty, path-only, and oversized names."""
    normalized, ok = _normalize_filename_for_type(filename)
    if not ok:
        return FileType.OTHER.value
    filename = normalized
    if re.match(r".*\.pdf$", filename):
        return FileType.PDF.value

    if re.match(
        r".*\.(msg|eml|doc|docx|ppt|pptx|yml|xml|htm|json|jsonl|ldjson|csv|txt|ini|xls|xlsx|wps|rtf|hlp|pages|numbers|key|md|mdx|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|html|sql|epub)$", filename
    ):
        return FileType.DOC.value

    if re.match(r".*\.(wav|flac|ape|alac|wavpack|wv|mp3|aac|ogg|vorbis|opus)$", filename):
        return FileType.AURAL.value

    if re.match(
        r".*\.(jpg|jpeg|png|tif|gif|pcx|tga|exif|fpx|svg|psd|cdr|pcd|dxf|ufo|eps|ai|raw|WMF|webp|avif|apng|icon|ico|mpg|mpeg|avi|rm|rmvb|mov|wmv|asf|dat|asx|wvx|mpe|mpa|mp4|avi|mkv)$", filename
    ):
        return FileType.VISUAL.value

    return FileType.OTHER.value


def thumbnail_img(filename, blob):
    """
    Generate thumbnail image bytes for PDF, image, or PPT. MySQL LongText max length is 65535.

    Robustness and edge cases:
    - Rejects None, empty, or oversized blob to avoid DoS/OOM.
    - Uses basename for type detection (handles paths like "a/b/c.pdf").
    - Catches corrupt or malformed files and returns None instead of raising.
    - Normalizes PIL image mode (e.g. RGBA -> RGB) for safe PNG export.
    """
    # 没有文件内容时无法生成缩略图，直接返回 None。
    if blob is None:
        return None
    try:
        # 先尝试获取二进制大小，用于后续空文件和超大文件判断。
        blob_len = len(blob)
    except TypeError:
        # 非 bytes-like 输入无法处理，直接返回 None。
        return None
    # 空文件或超出缩略图处理上限的文件直接放弃，避免资源消耗过大。
    if blob_len == 0 or blob_len > MAX_BLOB_SIZE_THUMBNAIL:
        return None

    # 规范化文件名，只保留安全 basename，并统一为小写，便于后续按后缀判断类型。
    normalized, ok = _normalize_filename_for_type(filename)
    # 文件名不合法时无法可靠判断类型，直接返回 None。
    if not ok:
        return None
    # 使用规范化后的文件名进行类型判断。
    filename = normalized

    # PDF 类型：渲染第一页生成缩略图。
    if re.match(r".*\.pdf$", filename):
        try:
            # pdfplumber 使用全局锁，避免并发打开 PDF 时出现线程安全问题。
            with sys.modules[LOCK_KEY_pdfplumber]:
                # 用 BytesIO 包装二进制内容，再交给 pdfplumber 打开。
                pdf = pdfplumber.open(BytesIO(blob))
                # 没有页面的 PDF 无法生成缩略图，关闭后返回 None。
                if not pdf.pages:
                    pdf.close()
                    return None
                # 用内存缓冲区接收最终生成的 PNG。
                buffered = BytesIO()
                # 初始渲染分辨率。
                resolution = 32
                # 保存最终生成的图片字节。
                img = None
                # 最多尝试 10 次；如果图片太大，就逐步降低分辨率。
                for _ in range(10):
                    # 把 PDF 第一页渲染成图片并写入内存。
                    pdf.pages[0].to_image(resolution=resolution).annotated.save(buffered, format="png")
                    # 读取当前 PNG 字节结果。
                    img = buffered.getvalue()
                    # 如果图片仍然过大，并且分辨率还能继续降，就减半分辨率重试。
                    if len(img) >= 64000 and resolution >= 2:
                        resolution = resolution / 2
                        buffered = BytesIO()
                    else:
                        # 图片大小已经可接受，或分辨率不能再降，停止循环。
                        break
                # 手动关闭 PDF 句柄。
                pdf.close()
                # 返回生成好的 PDF 缩略图字节。
                return img
        except Exception:
            # PDF 损坏或渲染失败时，统一返回 None，不向上抛异常。
            return None

    # 普通图片类型：直接压缩生成缩略图。
    if re.match(r".*\.(jpg|jpeg|png|tif|gif|icon|ico|webp)$", filename):
        try:
            # 读取图片内容。
            image = Image.open(BytesIO(blob))
            # 强制加载像素数据，尽早暴露损坏图片异常。
            image.load()
            # 某些模式不能安全导出为 PNG，先统一转成 RGB。
            if image.mode in ("RGBA", "P", "LA"):
                image = image.convert("RGB")
            # 原地缩放到不超过 30x30 的缩略图尺寸。
            image.thumbnail((30, 30))
            # 用内存缓冲区接收输出 PNG。
            buffered = BytesIO()
            # 保存为 PNG 格式。
            image.save(buffered, format="png")
            # 返回缩略图字节。
            return buffered.getvalue()
        except Exception:
            # 图片损坏或 PIL 处理失败时，统一返回 None。
            return None

    # PPT/PPTX thumbnail would require a licensed library; skip and return None.
    # PPT/PPTX 当前没有启用缩略图方案，直接返回 None。
    if re.match(r".*\.(ppt|pptx)$", filename):
        return None

    # 其余不支持的类型统一返回 None。
    return None


def thumbnail(filename, blob):
    img = thumbnail_img(filename, blob)
    if img is not None:
        return IMG_BASE64_PREFIX + base64.b64encode(img).decode("utf-8")
    else:
        return ""


def repair_pdf_with_ghostscript(input_bytes):
    """Attempt to repair corrupt PDF bytes via Ghostscript. Returns original bytes on failure or timeout."""
    # 没有输入内容或输入为空时，不做修复，直接返回原值或空字节串。
    if input_bytes is None or len(input_bytes) == 0:
        return input_bytes if input_bytes is not None else b""
    # 超过 PDF 修复大小上限时，跳过修复，避免高成本处理和潜在 DoS 风险。
    if len(input_bytes) > MAX_BLOB_SIZE_PDF:
        return input_bytes

    # 如果系统里没有安装 Ghostscript 可执行程序 `gs`，就无法修复，直接返回原始内容。
    if shutil.which("gs") is None:
        return input_bytes

    # 创建临时输入/输出文件，把 PDF 字节先落盘给 Ghostscript 使用。
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_in, tempfile.NamedTemporaryFile(suffix=".pdf") as temp_out:
        # 把原始 PDF 内容写入临时输入文件。
        temp_in.write(input_bytes)
        # 刷新缓冲区，确保 Ghostscript 能读到完整输入文件。
        temp_in.flush()

        # 组装 Ghostscript 命令：
        # - 输出到 temp_out
        # - 使用 pdfwrite 重新生成 PDF
        # - 使用 /prepress 质量设置尽量保真
        cmd = [
            "gs",
            "-o",
            temp_out.name,
            "-sDEVICE=pdfwrite",
            "-dPDFSETTINGS=/prepress",
            temp_in.name,
        ]
        try:
            # 调用 Ghostscript 执行 PDF 重写/修复。
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=GHOSTSCRIPT_TIMEOUT_SEC,
            )
            # Ghostscript 返回非 0 状态码时，视为修复失败，退回原始 PDF。
            if proc.returncode != 0:
                return input_bytes
            # 将输出文件指针移回开头，准备读取修复后的 PDF 内容。
            temp_out.seek(0)
            # 读取修复后的 PDF 字节流。
            repaired_bytes = temp_out.read()
            # 如果输出文件为空，说明修复结果无效，仍退回原始内容。
            if not repaired_bytes:
                return input_bytes
            # 修复成功，返回修复后的 PDF 字节。
            return repaired_bytes
        except subprocess.TimeoutExpired:
            # Ghostscript 超时则放弃修复，返回原始 PDF。
            return input_bytes
        except Exception:
            # 任何其他异常都不向上抛出，统一退回原始 PDF。
            return input_bytes


def read_potential_broken_pdf(blob):
    """
    Return PDF bytes, optionally repaired via Ghostscript if initially unreadable.

    Edge cases and robustness:
    - None blob returns b"" to avoid callers receiving None.
    - Empty blob returned as-is.
    - Oversized blob (> MAX_BLOB_SIZE_PDF) returned as-is without repair to avoid DoS.
    """
    # 没有传入任何二进制内容时，返回空字节串，避免调用方拿到 None。
    if blob is None:
        return b""
    try:
        # 先尝试获取 PDF 二进制长度，用于后续空文件和超大文件判断。
        blob_len = len(blob)
    except TypeError:
        # 非 bytes-like 对象无法取长度时，按无效输入处理，返回空字节串。
        return b""
    # 空文件直接原样返回，不再做额外处理。
    if blob_len == 0:
        return blob

    # 内部探测函数：尝试用 pdfplumber 打开 PDF，能正常打开并且至少有一页就视为可读。
    def try_open(data):
        try:
            # 用 BytesIO 包装原始字节流，交给 pdfplumber 解析。
            with pdfplumber.open(BytesIO(data)) as pdf:
                # 只要能解析出页面，就认为这个 PDF 基本可用。
                if pdf.pages:
                    return True
        except Exception:
            # 任意解析异常都说明当前 PDF 可能损坏或不兼容。
            return False
        # 能打开但没有页面时，也视为不可用。
        return False

    # 如果原始 PDF 已经可以正常打开，就直接返回原始内容，不做修复。
    if try_open(blob):
        return blob

    # 超过修复大小阈值的 PDF 不尝试修复，避免高成本处理和潜在 DoS 风险。
    if blob_len > MAX_BLOB_SIZE_PDF:
        return blob

    # 走到这里说明 PDF 初始不可读，尝试用 Ghostscript 做一次修复。
    repaired = repair_pdf_with_ghostscript(blob)
    # 如果修复后的 PDF 能正常打开，就返回修复后的内容。
    if try_open(repaired):
        return repaired

    # 修复失败或修复后仍不可读，则退回原始 PDF 字节，交由上层后续处理。
    return blob


def sanitize_path(raw_path: str | None) -> str:
    """Normalize and sanitize a user-provided path segment.

    - Converts backslashes to forward slashes
    - Strips leading/trailing slashes
    - Removes '.' and '..' segments
    - Restricts characters to A-Za-z0-9, underscore, dash, and '/'
    - Returns "" for None, empty, or non-string input (robustness).
    """
    if raw_path is None or not isinstance(raw_path, str):
        return ""
    raw_path = raw_path.strip()
    if not raw_path:
        return ""
    backslash_re = re.compile(r"[\\]+")
    unsafe_re = re.compile(r"[^A-Za-z0-9_\-/]")
    normalized = backslash_re.sub("/", raw_path)
    normalized = normalized.strip("/")
    parts = [seg for seg in normalized.split("/") if seg and seg not in (".", "..")]
    sanitized = "/".join(parts)
    sanitized = unsafe_re.sub("", sanitized)
    return sanitized
