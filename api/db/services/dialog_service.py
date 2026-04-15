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
import binascii
import logging
import re
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from timeit import default_timer as timer
from langfuse import Langfuse
from peewee import fn
from api.db.services.file_service import FileService
from common.constants import LLMType, ParserType, StatusEnum
from api.db.db_models import DB, Dialog
from api.db.services.common_service import CommonService
from api.db.services.doc_metadata_service import DocMetadataService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.langfuse_service import TenantLangfuseService
from api.db.services.llm_service import LLMBundle
from common.metadata_utils import apply_meta_data_filter
from api.db.services.tenant_llm_service import TenantLLMService
from api.db.joint_services.tenant_model_service import get_model_config_by_id, get_model_config_by_type_and_name, get_tenant_default_model_by_type
from common.time_utils import current_timestamp, datetime_format
from common.text_utils import normalize_arabic_digits
from rag.graphrag.general.mind_map_extractor import MindMapExtractor
from rag.advanced_rag import DeepResearcher
from rag.app.tag import label_question
from rag.nlp.search import index_name
from rag.prompts.generator import chunks_format, citation_prompt, cross_languages, full_question, kb_prompt, keyword_extraction, message_fit_in, \
    PROMPT_JINJA_ENV, ASK_SUMMARY
from common.token_utils import num_tokens_from_string
from rag.utils.tavily_conn import Tavily
from common.string_utils import remove_redundant_spaces
from common import settings


class DialogService(CommonService):
    model = Dialog

    @classmethod
    def save(cls, **kwargs):
        """Save a new record to database.

        This method creates a new record in the database with the provided field values,
        forcing an insert operation rather than an update.

        Args:
            **kwargs: Record field values as keyword arguments.

        Returns:
            Model instance: The created record object.
        """
        sample_obj = cls.model(**kwargs).save(force_insert=True)
        return sample_obj

    @classmethod
    def update_many_by_id(cls, data_list):
        """Update multiple records by their IDs.

        This method updates multiple records in the database, identified by their IDs.
        It automatically updates the update_time and update_date fields for each record.

        Args:
            data_list (list): List of dictionaries containing record data to update.
                             Each dictionary must include an 'id' field.
        """
        with DB.atomic():
            for data in data_list:
                data["update_time"] = current_timestamp()
                data["update_date"] = datetime_format(datetime.now())
                cls.model.update(data).where(cls.model.id == data["id"]).execute()

    @classmethod
    @DB.connection_context()
    def get_list(cls, tenant_id, page_number, items_per_page, orderby, desc, id, name):
        chats = cls.model.select()
        if id:
            chats = chats.where(cls.model.id == id)
        if name:
            chats = chats.where(cls.model.name == name)
        chats = chats.where((cls.model.tenant_id == tenant_id) & (cls.model.status == StatusEnum.VALID.value))
        if desc:
            chats = chats.order_by(cls.model.getter_by(orderby).desc())
        else:
            chats = chats.order_by(cls.model.getter_by(orderby).asc())

        chats = chats.paginate(page_number, items_per_page)

        return list(chats.dicts())

    @classmethod
    @DB.connection_context()
    def get_by_tenant_ids(
        cls,
        joined_tenant_ids,
        user_id,
        page_number,
        items_per_page,
        orderby,
        desc,
        keywords,
        id=None,
        name=None,
    ):
        from api.db.db_models import User

        fields = [
            cls.model.id,
            cls.model.tenant_id,
            cls.model.name,
            cls.model.description,
            cls.model.language,
            cls.model.llm_id,
            cls.model.llm_setting,
            cls.model.prompt_type,
            cls.model.prompt_config,
            cls.model.similarity_threshold,
            cls.model.vector_similarity_weight,
            cls.model.top_n,
            cls.model.top_k,
            cls.model.do_refer,
            cls.model.rerank_id,
            cls.model.kb_ids,
            cls.model.icon,
            cls.model.status,
            User.nickname,
            User.avatar.alias("tenant_avatar"),
            cls.model.update_time,
            cls.model.create_time,
        ]
        dialogs = (
            cls.model.select(*fields)
            .join(User, on=(cls.model.tenant_id == User.id))
            .where(
                (cls.model.tenant_id.in_(joined_tenant_ids) | (cls.model.tenant_id == user_id))
                & (cls.model.status == StatusEnum.VALID.value),
            )
        )
        if id:
            dialogs = dialogs.where(cls.model.id == id)
        if name:
            dialogs = dialogs.where(cls.model.name == name)
        if keywords:
            dialogs = dialogs.where(fn.LOWER(cls.model.name).contains(keywords.lower()))
        if desc:
            dialogs = dialogs.order_by(cls.model.getter_by(orderby).desc())
        else:
            dialogs = dialogs.order_by(cls.model.getter_by(orderby).asc())

        count = dialogs.count()

        if page_number and items_per_page:
            dialogs = dialogs.paginate(page_number, items_per_page)

        return list(dialogs.dicts()), count

    @classmethod
    @DB.connection_context()
    def get_all_dialogs_by_tenant_id(cls, tenant_id):
        fields = [cls.model.id]
        dialogs = cls.model.select(*fields).where(cls.model.tenant_id == tenant_id)
        dialogs.order_by(cls.model.create_time.asc())
        offset, limit = 0, 100
        res = []
        while True:
            d_batch = dialogs.offset(offset).limit(limit)
            _temp = list(d_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += limit
        return res

    @classmethod
    @DB.connection_context()
    def get_null_tenant_llm_id_row(cls):
        fields = [
            cls.model.id,
            cls.model.tenant_id,
            cls.model.llm_id
        ]
        objs = cls.model.select(*fields).where(cls.model.tenant_llm_id.is_null())
        return list(objs)

    @classmethod
    @DB.connection_context()
    def get_null_tenant_rerank_id_row(cls):
        fields = [
            cls.model.id,
            cls.model.tenant_id,
            cls.model.rerank_id
        ]
        objs = cls.model.select(*fields).where(cls.model.tenant_rerank_id.is_null())
        return list(objs)


async def async_chat_solo(dialog, messages, stream=True):
    # 根据对话绑定的 llm_id 推断模型类型。
    # 这样做是为了区分普通文本聊天模型和图像理解模型，后面附件处理和模型调用方式都依赖这个分支。
    llm_type = TenantLLMService.llm_id2llm_type(dialog.llm_id)
    # `attachments` 用来承接文本附件内容，最终会拼到用户最后一条消息里。
    attachments = ""
    # 文本聊天模型场景下，图片会作为多模态附件单独保存。
    image_attachments = []
    # 图像理解模型场景下，保留原始图片文件对象，后面直接传给模型接口。
    image_files = []
    # 如果最后一条用户消息携带了文件，需要先把文件拆分成文本和图片两类。
    # 这样做是为了兼容“同一轮消息里同时带文本、图片、文档”的输入形式。
    if "files" in messages[-1]:
        if llm_type == "chat":
            # 普通聊天模型下，把文件拆成文本内容和 data URI 形式的图片附件。
            text_attachments, image_attachments = split_file_attachments(messages[-1]["files"])
        else:
            # 图像理解模型下，文本内容仍转成字符串，但图片保留原始对象，便于直接走多模态接口。
            text_attachments, image_files = split_file_attachments(messages[-1]["files"], raw=True)
        # 多个文本附件之间用空行拼接。
        # 这样做是为了给模型一个相对清晰的边界，避免不同文件内容直接黏连。
        attachments = "\n\n".join(text_attachments)
    # 根据 tenant_llm_id 读取模型配置。
    # 这里用租户绑定后的模型配置，而不是裸 llm_id，是为了拿到实际可调用的供应商参数和密钥配置。
    model_config = get_model_config_by_id(dialog.tenant_llm_id)
    # 构造聊天模型对象，后续统一通过这个 bundle 发起生成请求。
    chat_mdl = LLMBundle(dialog.tenant_id, model_config)
    # 模型工厂信息后面用于多模态消息格式转换。
    factory = model_config.get("llm_factory", "") if model_config else ""

    # 读取对话的 prompt 配置。
    prompt_config = dialog.prompt_config
    # 默认不启用 TTS，只有 prompt 明确要求时才初始化。
    tts_mdl = None
    if prompt_config.get("tts"):
        # TTS 使用租户默认语音模型。
        # 这样做是为了让“文本生成”和“语音合成”解耦，不强绑定到同一个聊天模型。
        default_tts_model = get_tenant_default_model_by_type(dialog.tenant_id, LLMType.TTS)
        tts_mdl = LLMBundle(dialog.tenant_id, default_tts_model)
    # 组装要发给模型的消息列表，并过滤掉 system 消息。
    # 因为 system prompt 会作为单独参数传入，所以这里不应重复放进消息历史。
    msg = [{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if m["role"] != "system"]
    # 如果有文本附件，就直接追加到最后一条消息内容后面。
    # 这样做是为了把“用户输入 + 附件文本”合并成模型可以直接理解的一次提问。
    if attachments and msg:
        msg[-1]["content"] += attachments
    # 普通聊天模型如果带了图片附件，需要把最后一条用户消息转成多模态格式。
    # 这样做是为了复用文本聊天模型的多模态能力，而不是额外走 image2text 分支。
    if llm_type == "chat" and image_attachments:
        convert_last_user_msg_to_multimodal(msg, image_attachments, factory)
    # 流式模式下，逐片把模型输出转发给上游。
    if stream:
        if llm_type == "chat":
            # 文本聊天模型的流式接口。
            stream_iter = chat_mdl.async_chat_streamly_delta(prompt_config.get("system", ""), msg, dialog.llm_setting)
        else:
            # 图像理解模型在流式生成时额外传入图片文件。
            stream_iter = chat_mdl.async_chat_streamly_delta(prompt_config.get("system", ""), msg, dialog.llm_setting, images=image_files)
        # `_stream_with_think_delta` 会把普通文本增量和 think 标记拆开。
        async for kind, value, state in _stream_with_think_delta(stream_iter):
            if kind == "marker":
                # think 起止标记作为控制信号单独上抛，避免和正文混在一起。
                flags = {"start_to_think": True} if value == "<think>" else {"end_to_think": True}
                yield {"answer": "", "reference": {}, "audio_binary": None, "prompt": "", "created_at": time.time(), "final": False, **flags}
                continue
            # 普通文本增量除了返回文字，也同步生成分片语音。
            # 这样做是为了让前端在流式展示文本时，具备边播边出的能力。
            yield {"answer": value, "reference": {}, "audio_binary": tts(tts_mdl, value), "prompt": "", "created_at": time.time(), "final": False}
    else:
        # 非流式模式下一次性拿完整答案。
        if llm_type == "chat":
            answer = await chat_mdl.async_chat(prompt_config.get("system", ""), msg, dialog.llm_setting)
        else:
            answer = await chat_mdl.async_chat(prompt_config.get("system", ""), msg, dialog.llm_setting, images=image_files)
        # 记录最终用户输入和模型回答，便于后续排查问题。
        user_content = msg[-1].get("content", "[content not available]")
        logging.debug("User: {}|Assistant: {}".format(user_content, answer))
        # 非流式模式下直接返回完整答案和整段 TTS 结果。
        yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, answer), "prompt": "", "created_at": time.time()}


def get_models(dialog):
    embd_mdl, chat_mdl, rerank_mdl, tts_mdl = None, None, None, None
    kbs = KnowledgebaseService.get_by_ids(dialog.kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))
    if len(embedding_list) > 1:
        raise Exception("**ERROR**: Knowledge bases use different embedding models.")

    if embedding_list:
        embd_owner_tenant_id = kbs[0].tenant_id
        embd_model_config = get_model_config_by_type_and_name(embd_owner_tenant_id, LLMType.EMBEDDING, embedding_list[0])
        embd_mdl = LLMBundle(embd_owner_tenant_id, embd_model_config)
        if not embd_mdl:
            raise LookupError("Embedding model(%s) not found" % embedding_list[0])

    if dialog.tenant_llm_id:
        chat_model_config = get_model_config_by_id(dialog.tenant_llm_id)
    elif dialog.llm_id:
        chat_model_config = get_model_config_by_type_and_name(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)
    else:
        chat_model_config = get_tenant_default_model_by_type(dialog.tenant_id, LLMType.CHAT)

    chat_mdl = LLMBundle(dialog.tenant_id, chat_model_config)

    if dialog.rerank_id:
        rerank_model_config = get_model_config_by_type_and_name(dialog.tenant_id, LLMType.RERANK, dialog.rerank_id)
        rerank_mdl = LLMBundle(dialog.tenant_id, rerank_model_config)

    if dialog.prompt_config.get("tts"):
        default_tts_model_config = get_tenant_default_model_by_type(dialog.tenant_id, LLMType.TTS)
        tts_mdl = LLMBundle(dialog.tenant_id, default_tts_model_config)
    return kbs, embd_mdl, rerank_mdl, chat_mdl, tts_mdl


def split_file_attachments(files: list[dict] | None, raw: bool = False) -> tuple[list[str], list[str] | list[dict]]:
    # 没有文件时直接返回两个空列表。
    # 这样做是为了让调用方始终收到稳定的二元返回结构，避免额外判空分支。
    if not files:
        return [], []

    # `text_attachments` 用来收集可直接拼进 prompt 的文本内容。
    text_attachments = []
    # `raw=True` 表示调用方希望保留图片的原始对象，而不是转成 data URI 字符串。
    # 这个模式主要给图像理解模型使用，因为它们通常需要原始图片输入。
    if raw:
        # 一次性取回文件内容和图片文件对象。
        # 这样做是为了避免重复读取文件源，同时把“文本内容”和“图片对象”同步拆分出来。
        file_contents, image_files = FileService.get_files(files, raw=True)
        for content in file_contents:
            # 非字符串内容统一转成字符串。
            # 这样做是为了让下游拼接 prompt 时不必处理多种 Python 类型。
            if not isinstance(content, str):
                content = str(content)
            text_attachments.append(content)
        # 原样返回文本附件和图片文件对象。
        return text_attachments, image_files

    # `raw=False` 时，图片会以 data URI 形式单独返回给文本聊天模型使用。
    image_attachments = []
    # 非 raw 模式下，统一从文件服务取回可直接消费的内容。
    for content in FileService.get_files(files, raw=False):
        # 同样先做字符串归一化，保证后续判断和拼接稳定。
        if not isinstance(content, str):
            content = str(content)
        # 以 `data:` 开头的内容视为图片 data URI。
        # 这样做是为了把图片和纯文本文件内容分开，便于后续构造多模态消息。
        if content.strip().startswith("data:"):
            image_attachments.append(content.strip())
            continue
        # 其余内容都按普通文本附件处理。
        text_attachments.append(content)
    # 返回文本附件和图片附件两类结果。
    return text_attachments, image_attachments


_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>[A-Za-z0-9+/=\s]+)$")


def _parse_data_uri_or_b64(s: str, default_mime: str = "image/png") -> tuple[str, str]:
    s = (s or "").strip()
    match = _DATA_URI_RE.match(s)
    if match:
        mime = match.group("mime").strip()
        b64 = match.group("b64").strip()
        return mime, b64
    return default_mime, s


def _normalize_text_from_content(content) -> str:
    # 空内容统一归一化为空字符串。
    # 这样做是为了让调用方不必额外处理 None，后续字符串拼接逻辑也会更稳定。
    if content is None:
        return ""
    # 如果本来就是普通字符串，直接返回。
    # 这样做是为了优先走最常见路径，避免对简单文本做不必要的结构解析。
    if isinstance(content, str):
        return content
    # 列表通常表示多模态块内容，这里只提取其中可视为文本的部分。
    # 这样做是为了从复杂消息结构中恢复出“用户的文字说明”，供后续兼容处理。
    if isinstance(content, list):
        texts = []
        for blk in content:
            # 只有字典块才按标准多模态消息结构继续解析。
            if isinstance(blk, dict):
                # 标准文本块通常以 `type=text` 或 `type=input_text` 表示。
                # 这样做是为了兼容不同模型厂商或 SDK 的文本块命名差异。
                if blk.get("type") in {"text", "input_text"}:
                    txt = blk.get("text")
                    # 只收集非空文本，避免把空块也拼进去。
                    if txt:
                        texts.append(str(txt))
                # 有些块未必带标准 `type`，但只要存在可转文本的 `text` 字段，也一并兜底收集。
                # 这样做是为了提升兼容性，避免遗漏弱约定格式里的文本内容。
                elif "text" in blk and isinstance(blk.get("text"), (str, int, float)):
                    texts.append(str(blk["text"]))
        # 多个文本块之间用换行连接，并去掉首尾空白。
        # 这样做是为了尽量保留块之间的边界感，同时输出干净的纯文本结果。
        return "\n".join(texts).strip()
    # 其它非列表、非字符串内容统一转成字符串返回。
    # 这样做是为了保证函数输出类型稳定，调用方始终能得到 `str`。
    return str(content)


def convert_last_user_msg_to_multimodal(msg: list[dict], image_data_uris: list[str], factory: str) -> None:
    # 没有消息或没有图片时直接返回。
    # 这样做是为了避免无意义地改写消息结构，也防止后面对空列表做倒序遍历。
    if not msg or not image_data_uris:
        return

    # 统一归一化模型工厂名，便于后续按厂商分支判断。
    # 这样做是为了兼容大小写、前后空格等输入差异，减少分支判断失败。
    factory_norm = (factory or "").strip().lower()

    # 从后往前找最后一条 user 消息。
    # 这样做是因为多模态附件只应该绑定到“当前这轮最后一次用户输入”，而不是更早的历史消息。
    for idx in range(len(msg) - 1, -1, -1):
        # 只处理 user 消息，跳过 assistant / system 等其它角色。
        if msg[idx].get("role") != "user":
            continue

        # 取出原始内容，并统一抽取其中的纯文本部分。
        # 这样做是为了兼容 `content` 可能已经是字符串、块列表或其它结构的情况。
        original_content = msg[idx].get("content", "")
        text = _normalize_text_from_content(original_content)

        # Gemini 需要使用其专有的 parts 结构。
        if factory_norm == "gemini":
            parts = []
            # 文本存在时先放文本块，保持“文字说明在前、图片在后”的顺序。
            if text:
                parts.append({"text": text})
            for image in image_data_uris:
                # Gemini 要求把图片拆成 mime 和纯 base64 数据。
                # 这样做是为了适配 Gemini 接口对 inline_data 的格式要求。
                mime, b64 = _parse_data_uri_or_b64(str(image), default_mime="image/png")
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            # 用 Gemini 规范的多模态结构替换最后一条 user 消息内容。
            msg[idx]["content"] = parts
            return

        # Anthropic 也有自己的内容块格式，与 Gemini/OpenAI 系列不同。
        if factory_norm == "anthropic":
            blocks = []
            # 文本块优先放前面，保证模型先看到用户文字说明。
            if text:
                blocks.append({"type": "text", "text": text})
            for image in image_data_uris:
                # Anthropic 的图片块要求 `source.type=base64` 并显式带上 media_type。
                mime, b64 = _parse_data_uri_or_b64(str(image), default_mime="image/png")
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    }
                )
            # 用 Anthropic 规范的块结构覆盖原消息内容。
            msg[idx]["content"] = blocks
            return

        # 默认分支走更通用的 OpenAI 风格多模态结构。
        multimodal_content = []
        # 如果原内容本身已经是块列表，先深拷贝一份，避免直接原地复用造成共享引用副作用。
        if isinstance(original_content, list):
            multimodal_content = deepcopy(original_content)
        else:
            # 否则把原始内容转成文本块，作为多模态消息的起始部分。
            text_content = "" if original_content is None else str(original_content)
            if text_content:
                multimodal_content.append({"type": "text", "text": text_content})

        for data_uri in image_data_uris:
            image_url = data_uri
            # 非字符串图片内容统一转成字符串，保证后续 URL 处理一致。
            if not isinstance(image_url, str):
                image_url = str(image_url)
            # 如果不是完整 data URI，就按 png 的 base64 数据补成标准 data URI。
            # 这样做是为了兼容上游有时只传纯 base64 字符串的情况。
            if not image_url.startswith("data:"):
                image_url = f"data:image/png;base64,{image_url}"
            # 追加标准的 image_url 块。
            multimodal_content.append({"type": "image_url", "image_url": {"url": image_url}})

        # 用通用多模态结构替换最后一条用户消息内容。
        msg[idx]["content"] = multimodal_content
        return


BAD_CITATION_PATTERNS = [
    re.compile(r"\(\s*ID\s*[: ]*\s*(\d+)\s*\)"),  # (ID: 12)
    re.compile(r"\[\s*ID\s*[: ]*\s*(\d+)\s*\]"),  # [ID: 12]
    re.compile(r"【\s*ID\s*[: ]*\s*(\d+)\s*】"),  # 【ID: 12】
    re.compile(r"ref\s*(\d+)", flags=re.IGNORECASE),  # ref12、REF 12
]
CITATION_MARKER_PATTERN = re.compile(r"\[(?:ID:)?([0-9\u0660-\u0669\u06F0-\u06F9]+)\]")


def repair_bad_citation_formats(answer: str, kbinfos: dict, idx: set):
    max_index = len(kbinfos["chunks"])
    normalized_answer = normalize_arabic_digits(answer) or ""

    def safe_add(i):
        if 0 <= i < max_index:
            idx.add(i)
            return True
        return False

    def find_and_replace(pattern, group_index=1, repl=lambda digits: f"ID:{digits}"):
        nonlocal answer
        nonlocal normalized_answer

        matches = list(pattern.finditer(normalized_answer))
        if not matches:
            return

        parts = []
        last_idx = 0
        for match in matches:
            parts.append(answer[last_idx:match.start()])
            try:
                i = int(match.group(group_index))
            except Exception:
                parts.append(answer[match.start():match.end()])
                last_idx = match.end()
                continue

            if safe_add(i):
                digit_start, digit_end = match.span(group_index)
                digits_original = answer[digit_start:digit_end]
                parts.append(f"[{repl(digits_original)}]")
            else:
                parts.append(answer[match.start():match.end()])
            last_idx = match.end()

        parts.append(answer[last_idx:])
        answer = "".join(parts)
        normalized_answer = normalize_arabic_digits(answer) or ""

    for pattern in BAD_CITATION_PATTERNS:
        find_and_replace(pattern)

    return answer, idx


async def async_chat(dialog, messages, stream=True, **kwargs):
    # 记录函数入口，便于排查异步链路是否真正进入聊天主流程。
    logging.debug("Begin async_chat")
    # 约束最后一条消息必须来自用户。
    # 这样做是为了保证模型始终在“回答用户当前问题”，而不是错误地接着 assistant/system 消息继续生成。
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    # 如果既没有绑定知识库，也没有启用联网搜索，则直接走纯大模型对话模式。
    # 这样做是为了在最简单场景下绕过复杂的检索增强流程，减少额外开销和依赖。
    if not dialog.kb_ids and not dialog.prompt_config.get("tavily_api_key"):
        async for ans in async_chat_solo(dialog, messages, stream):
            yield ans
        return

    # 记录本轮聊天开始时间，后面会统计整条链路各阶段耗时。
    chat_start_ts = timer()
    # 根据 llm_id 推断模型类型，用于区分普通聊天模型与图像理解模型。
    llm_type = TenantLLMService.llm_id2llm_type(dialog.llm_id)
    # 图像理解模型和普通聊天模型走的是不同的配置通道。
    # 这样做是为了让同一个对话框架能适配不同模态的模型参数。
    if llm_type == "image2text":
        llm_model_config = TenantLLMService.get_model_config(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        llm_model_config = TenantLLMService.get_model_config(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    # 读取模型工厂信息，后面处理多模态消息时会按不同厂商格式转换。
    factory = llm_model_config.get("llm_factory", "") if llm_model_config else ""
    # 获取模型允许的最大 token 数，并在后续裁剪上下文时作为上限依据。
    max_tokens = llm_model_config.get("max_tokens", 8192)

    # 记录“模型配置检查完成”的时间点，供性能分析使用。
    check_llm_ts = timer()

    # Langfuse 用于链路 tracing；默认关闭，只有租户配置了可用 key 才启用。
    langfuse_tracer = None
    trace_context = {}
    langfuse_keys = TenantLangfuseService.filter_by_tenant(tenant_id=dialog.tenant_id)
    if langfuse_keys:
        langfuse = Langfuse(public_key=langfuse_keys.public_key, secret_key=langfuse_keys.secret_key, host=langfuse_keys.host)
        try:
            # 先校验 Langfuse 连通性。
            # 这样做是为了避免 tracing 服务异常时影响主聊天链路，观测系统不能反向拖垮业务。
            if langfuse.auth_check():
                langfuse_tracer = langfuse
                trace_id = langfuse_tracer.create_trace_id()
                trace_context = {"trace_id": trace_id}
        except Exception:
            # Langfuse 不可用时直接跳过。
            # 这样做是为了把 tracing 视为“可选增强能力”，而不是必需依赖。
            pass

    # 记录 tracing 检查完成时间点。
    check_langfuse_tracer_ts = timer()
    # 绑定本轮聊天会用到的各类模型：知识库、向量、重排、聊天、语音。
    # 这样做是为了把租户级模型选择统一解析成可直接调用的对象。
    kbs, embd_mdl, rerank_mdl, chat_mdl, tts_mdl = get_models(dialog)
    # 可选的工具调用会话和工具清单。
    # 如果传入，就把工具绑定到 chat model，让模型可以发起 tool call。
    toolcall_session, tools = kwargs.get("toolcall_session"), kwargs.get("tools")
    if toolcall_session and tools:
        chat_mdl.bind_tools(toolcall_session, tools)
    # 记录模型绑定结束时间点。
    bind_models_ts = timer()

    # 检索器是全局配置能力，下面的知识召回都通过它完成。
    retriever = settings.retriever
    # 只截取最近三条用户问题，用于多轮改写、检索和关键词增强。
    # 这样做是为了兼顾上下文连续性和成本，避免把很久以前的问题无差别带入当前检索。
    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    # `attachments` 代表要限制召回范围的文档 ID。
    attachments = None
    # 支持通过 kwargs 直接传 doc_ids。
    if "doc_ids" in kwargs:
        attachments = [doc_id for doc_id in kwargs["doc_ids"].split(",") if doc_id]
    # `attachments_` 是拼进 system prompt 的文本附件内容。
    attachments_= ""
    # 文本模型下的图片附件，以及图像模型下的原始图片文件列表。
    image_attachments = []
    image_files = []
    # 也支持从最后一条用户消息里读取 doc_ids。
    # 这样做是为了让“消息自带附件”和“接口参数附件”两种调用方式都能被统一消费。
    if "doc_ids" in messages[-1]:
        attachments = [doc_id for doc_id in messages[-1]["doc_ids"] if doc_id]
    # 处理最后一条用户消息附带的文件。
    # 文本模型把文件拆成“文本内容 + 图片附件”，图像模型则保留原始图片文件给多模态接口使用。
    if "files" in messages[-1]:
        if llm_type == "chat":
            text_attachments, image_attachments = split_file_attachments(messages[-1]["files"])
        else:
            text_attachments, image_files = split_file_attachments(messages[-1]["files"], raw=True)
        # 文本附件直接拼成字符串追加到系统提示词后面。
        # 这样做是为了让模型在没有专门文件解析协议时，也能把附件文本纳入本轮理解。
        attachments_ = "\n\n".join(text_attachments)

    # 读取 prompt 配置；后续的参数检查、问题改写、引用开关都来自这里。
    prompt_config = dialog.prompt_config
    # 从知识库里抽取字段映射。
    # 如果字段信息足够完整，可以尝试直接把问题翻译为 SQL，优先走结构化查询。
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    logging.debug(f"field_map retrieved: {field_map}")
    # 字段映射存在时，优先尝试 SQL 检索。
    # 这样做是因为结构化问题用 SQL 往往比向量检索更准确，尤其是聚合、统计和精确过滤类问题。
    if field_map:
        logging.debug("Use SQL to retrieval:{}".format(questions[-1]))
        ans = await use_sql(questions[-1], field_map, dialog.tenant_id, chat_mdl, prompt_config.get("quote", True), dialog.kb_ids)
        # 聚合类查询即使没有 chunk，也可能已经拿到了正确答案，因此这里用“有答案或有引用”作为成功条件。
        if ans and (ans.get("reference", {}).get("chunks") or ans.get("answer")):
            yield ans
            return
        else:
            # SQL 失败后自动回退到向量/混合检索，避免一条路走死。
            logging.debug("SQL failed or returned no results, falling back to vector search")

    # 取出 prompt 中要求的动态参数名，后面会校验必填项。
    param_keys = [p["key"] for p in prompt_config.get("parameters", [])]
    logging.debug(f"attachments={attachments}, param_keys={param_keys}, embd_mdl={embd_mdl}")

    # 校验 prompt 参数。
    # 除 `knowledge` 外，其余参数如果是必填但未传入，则直接报错；如果是可选但未传入，则把占位符替换为空白。
    # 这样做是为了避免把 `{foo}` 这样的未展开模板字符串直接发给模型。
    for p in prompt_config["parameters"]:
        if p["key"] == "knowledge":
            continue
        if p["key"] not in kwargs and not p["optional"]:
            raise KeyError("Miss parameter: " + p["key"])
        if p["key"] not in kwargs:
            prompt_config["system"] = prompt_config["system"].replace("{%s}" % p["key"], " ")

    # 多轮场景下可选地把最近几轮用户问题改写成一个完整问题。
    # 这样做是为了提升检索召回质量，避免只用最后一句省略式提问导致语义缺失。
    if len(questions) > 1 and prompt_config.get("refine_multiturn"):
        questions = [await full_question(dialog.tenant_id, dialog.llm_id, messages)]
    else:
        # 否则只保留当前轮最后一个问题，减少无关上下文对检索的干扰。
        questions = questions[-1:]

    # 按配置做跨语言改写。
    # 这样做通常是为了让查询语言与知识库语言更匹配，提升检索和生成质量。
    if prompt_config.get("cross_languages"):
        questions = [await cross_languages(dialog.tenant_id, dialog.llm_id, questions[0], prompt_config["cross_languages"])]

    # 如果配置了元数据过滤，则先用元数据规则缩小候选文档范围。
    # 这样做是为了在真正召回内容前就把明显不相关的文档排除掉。
    if dialog.meta_data_filter:
        metas = DocMetadataService.get_flatted_meta_by_kbs(dialog.kb_ids)
        attachments = await apply_meta_data_filter(
            dialog.meta_data_filter,
            metas,
            questions[-1],
            chat_mdl,
            attachments,
        )

    # 可选的关键词提取增强。
    # 这样做是为了给检索问题附加更有判别力的关键词，提升召回覆盖率。
    if prompt_config.get("keyword", False):
        questions[-1] += await keyword_extraction(chat_mdl, questions[-1])

    # 记录“问题改写 / 预处理”完成时间点。
    refine_question_ts = timer()

    # `thought` 预留给思维链或中间思考文本；`kbinfos` 保存本轮全部召回结果；`knowledges` 是最终拼 prompt 的知识片段。
    thought = ""
    kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}
    knowledges = []

    # 只有 prompt 参数里显式声明了 `knowledge`，才进入知识检索链路。
    # 这样做是为了支持“同一套对话框架下，有些 prompt 依赖知识库，有些完全不依赖”的场景。
    if "knowledge" in param_keys:
        logging.debug("Proceeding with retrieval")
        # 把涉及的知识库租户 ID 去重。
        # 有些场景下一个对话可能挂了多个 KB，这里需要统一成检索器能消费的 tenant 列表。
        tenant_ids = list(set([kb.tenant_id for kb in kbs]))
        knowledges = []
        # 开启 reasoning 时，走 DeepResearcher 深度研究模式，而不是一次性普通召回。
        # 这样做是为了支持多步检索、思考和逐步输出研究过程。
        if prompt_config.get("reasoning", False) or kwargs.get("reasoning"):
            reasoner = DeepResearcher(
                chat_mdl,
                prompt_config,
                partial(
                    retriever.retrieval,
                    embd_mdl=embd_mdl,
                    tenant_ids=tenant_ids,
                    kb_ids=dialog.kb_ids,
                    page=1,
                    page_size=dialog.top_n,
                    similarity_threshold=0.2,
                    vector_similarity_weight=0.3,
                    doc_ids=attachments,
                ),
            )
            # 用异步队列承接研究过程中的中间输出，再流式转发给前端。
            queue = asyncio.Queue()
            async def callback(msg:str):
                nonlocal queue
                await queue.put(msg + "<br/>")

            # 先发送开始标记，让前端切换到“思考中”展示模式。
            await callback("<START_DEEP_RESEARCH>")
            task = asyncio.create_task(reasoner.research(kbinfos, questions[-1], questions[-1], callback=callback))
            while True:
                msg = await queue.get()
                if msg.find("<START_DEEP_RESEARCH>") == 0:
                    # 向上游发出 think 开始信号，而不是普通文本。
                    yield {"answer": "", "reference": {}, "audio_binary": None, "final": False, "start_to_think": True}
                elif msg.find("<END_DEEP_RESEARCH>") == 0:
                    # 思考结束后发出结束信号，并跳出研究过程流式循环。
                    yield {"answer": "", "reference": {}, "audio_binary": None, "final": False, "end_to_think": True}
                    break
                else:
                    # 普通研究过程文本按增量答案向上游转发。
                    yield {"answer": msg, "reference": {}, "audio_binary": None, "final": False}

            # 等待后台研究任务真正结束，确保 `kbinfos` 已经填充完整。
            await task

        else:
            # 常规 RAG 路径：先做向量召回，再按配置做 TOC 增强、子块展开、联网检索、知识图谱增强。
            if embd_mdl:
                kbinfos = await retriever.retrieval(
                    " ".join(questions),
                    embd_mdl,
                    tenant_ids,
                    dialog.kb_ids,
                    1,
                    dialog.top_n,
                    dialog.similarity_threshold,
                    dialog.vector_similarity_weight,
                    doc_ids=attachments,
                    top=dialog.top_k,
                    aggs=True,
                    rerank_mdl=rerank_mdl,
                    rank_feature=label_question(" ".join(questions), kbs),
                )
                if prompt_config.get("toc_enhance"):
                    # 目录增强用于在已召回 chunk 的基础上再做一层结构化补强，提升长文档定位能力。
                    cks = await retriever.retrieval_by_toc(" ".join(questions), kbinfos["chunks"], tenant_ids, chat_mdl, dialog.top_n)
                    if cks:
                        kbinfos["chunks"] = cks
                # 子块召回用于把父块命中的上下文继续展开到更细粒度内容。
                kbinfos["chunks"] = retriever.retrieval_by_children(kbinfos["chunks"], tenant_ids)
            if prompt_config.get("tavily_api_key"):
                # 开启 Tavily 时，把联网搜索结果并入知识结果。
                # 这样做是为了补充知识库外的实时信息。
                tav = Tavily(prompt_config["tavily_api_key"])
                tav_res = tav.retrieve_chunks(" ".join(questions))
                kbinfos["chunks"].extend(tav_res["chunks"])
                kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
            if prompt_config.get("use_kg"):
                # 知识图谱召回适合补充实体关系类信息。
                default_chat_model = get_tenant_default_model_by_type(dialog.tenant_id, LLMType.CHAT)
                ck = await settings.kg_retriever.retrieval(" ".join(questions), tenant_ids, dialog.kb_ids, embd_mdl,
                                                       LLMBundle(dialog.tenant_id, default_chat_model))
                if ck["content_with_weight"]:
                    # KG 结果优先插到前面，让它在知识拼接中有更高显著性。
                    kbinfos["chunks"].insert(0, ck)

    # 把召回结果裁剪并拼成真正会注入 prompt 的 knowledge 文本。
    knowledges = kb_prompt(kbinfos, max_tokens)
    logging.debug("{}->{}".format(" ".join(questions), "\n->".join(knowledges)))

    # 记录检索阶段结束时间。
    retrieval_ts = timer()
    # 如果完全没有召回到知识，并且 prompt 配置了空结果兜底回复，则直接返回预设文案。
    # 这样做是为了在“明确无知识可答”场景下给出稳定回复，而不是让模型胡编。
    if not knowledges and prompt_config.get("empty_response"):
        empty_res = prompt_config["empty_response"]
        yield {"answer": empty_res, "reference": kbinfos, "prompt": "\n\n### Query:\n%s" % " ".join(questions),
               "audio_binary": tts(tts_mdl, empty_res), "final": True}
        return

    # 把知识块拼进 kwargs，让 system prompt 的 `{knowledge}` 占位符可以展开。
    kwargs["knowledge"] = "\n------\n" + "\n\n------\n\n".join(knowledges)
    # 读取模型生成参数，例如 temperature、top_p、max_tokens 等。
    gen_conf = dialog.llm_setting

    # 构造发给模型的完整消息。
    # 第一条固定是 system prompt，并把附件文本直接追加进去。
    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)+attachments_}]
    # `prompt4citation` 是额外附加的“请按引用格式回答”提示。
    prompt4citation = ""
    if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
        prompt4citation = citation_prompt()
    # 把历史消息拼接进来，同时移除已有的内部引用占位标记，避免污染新一轮生成。
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if m["role"] != "system"])
    # 按 token 上限裁剪消息，避免超出模型窗口。
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.95))
    # 文本模型如果带了图片附件，需要把最后一条用户消息转成多模态格式。
    if llm_type == "chat" and image_attachments:
        convert_last_user_msg_to_multimodal(msg, image_attachments, factory)
    # 至少应当保留 system + 一条用户消息，否则说明消息裁剪有 bug。
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    # 单独保存 system prompt 内容，后面会用于 Langfuse 和调试信息输出。
    prompt = msg[0]["content"]

    # 如果生成参数里也带了 max_tokens，则再按模型窗口剩余容量做一次上限收缩。
    # 这样做是为了避免“输入 token + 输出 token”总和超窗。
    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(gen_conf["max_tokens"], max_tokens - used_token_count)

    def decorate_answer(answer):
        # 这里会闭包使用外层状态，对最终答案做引用修复、调试信息追加和 tracing 收尾。
        nonlocal embd_mdl, prompt_config, knowledges, kwargs, kbinfos, prompt, retrieval_ts, questions, langfuse_tracer

        # `refs` 是最终返回给前端/上层的引用信息。
        refs = []
        # 如果答案里包含思考块，把 think 内容和最终回答拆开处理。
        # 这样做是为了既保留完整回答文本，又让引用修复主要作用在最终可见答案部分。
        ans = answer.split("</think>")
        think = ""
        if len(ans) == 2:
            think = ans[0] + "</think>"
            answer = ans[1]

        # 只有存在知识且启用了引用时，才做引用插入/修复。
        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            idx = set([])
            normalized_answer = normalize_arabic_digits(answer) or ""
            # 如果模型没有自己打引用标记，就基于向量和词法相似度自动补引用。
            if embd_mdl and not CITATION_MARKER_PATTERN.search(normalized_answer):
                answer, idx = retriever.insert_citations(
                    answer,
                    [ck["content_ltks"] for ck in kbinfos["chunks"]],
                    [ck["vector"] for ck in kbinfos["chunks"]],
                    embd_mdl,
                    tkweight=1 - dialog.vector_similarity_weight,
                    vtweight=dialog.vector_similarity_weight,
                )
            else:
                # 如果模型已经输出了引用标记，则直接解析已有标记。
                for match in CITATION_MARKER_PATTERN.finditer(normalized_answer):
                    i = int(match.group(1))
                    if i < len(kbinfos["chunks"]):
                        idx.add(i)

            # 对坏格式引用做修复，兼容模型输出不规范的情况。
            answer, idx = repair_bad_citation_formats(answer, kbinfos, idx)

            # 先把 chunk 索引映射为 doc_id，再据此过滤文档聚合结果。
            # 这样做是为了最终返回的引用文档只保留真正被答案命中的部分。
            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
            recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            if not recall_docs:
                # 如果没有命中过滤结果，则退回全部 doc_aggs，避免返回空引用。
                recall_docs = kbinfos["doc_aggs"]
            kbinfos["doc_aggs"] = recall_docs

            # 深拷贝一份引用结果，避免后续清理字段时污染原始召回对象。
            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    # 向量字段只用于内部计算，不需要返回给前端，删除可减少响应体积。
                    del c["vector"]

        # 对常见 API Key 配置错误补充更明确的引导。
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model providers -> API-Key'"
        # 记录生成结束时间，并计算各阶段耗时。
        finish_chat_ts = timer()

        total_time_cost = (finish_chat_ts - chat_start_ts) * 1000
        check_llm_time_cost = (check_llm_ts - chat_start_ts) * 1000
        check_langfuse_tracer_cost = (check_langfuse_tracer_ts - check_llm_ts) * 1000
        bind_embedding_time_cost = (bind_models_ts - check_langfuse_tracer_ts) * 1000
        refine_question_time_cost = (refine_question_ts - bind_models_ts) * 1000
        retrieval_time_cost = (retrieval_ts - refine_question_ts) * 1000
        generate_result_time_cost = (finish_chat_ts - retrieval_ts) * 1000

        # 统计生成 token 数，并把查询、耗时、速率等调试信息附加到 prompt 里。
        # 这样做是为了便于上层展示、排查和离线分析。
        tk_num = num_tokens_from_string(think + answer)
        prompt += "\n\n### Query:\n%s" % " ".join(questions)
        prompt = (
            f"{prompt}\n\n"
            "## Time elapsed:\n"
            f"  - Total: {total_time_cost:.1f}ms\n"
            f"  - Check LLM: {check_llm_time_cost:.1f}ms\n"
            f"  - Check Langfuse tracer: {check_langfuse_tracer_cost:.1f}ms\n"
            f"  - Bind models: {bind_embedding_time_cost:.1f}ms\n"
            f"  - Query refinement(LLM): {refine_question_time_cost:.1f}ms\n"
            f"  - Retrieval: {retrieval_time_cost:.1f}ms\n"
            f"  - Generate answer: {generate_result_time_cost:.1f}ms\n\n"
            "## Token usage:\n"
            f"  - Generated tokens(approximately): {tk_num}\n"
            f"  - Token speed: {int(tk_num / (generate_result_time_cost / 1000.0))}/s"
        )

        # 如果开启了 Langfuse，则把整理后的输出写回 tracing，并结束 generation span。
        if langfuse_tracer and "langfuse_generation" in locals():
            langfuse_output = "\n" + re.sub(r"^.*?(### Query:.*)", r"\1", prompt, flags=re.DOTALL)
            langfuse_output = {"time_elapsed:": re.sub(r"\n", "  \n", langfuse_output), "created_at": time.time()}
            langfuse_generation.update(output=langfuse_output)
            langfuse_generation.end()

        # 返回统一结构的最终答案对象。
        return {"answer": think + answer, "reference": refs, "prompt": re.sub(r"\n", "  \n", prompt), "created_at": time.time()}

    # 在真正调用模型前开启 Langfuse generation span。
    if langfuse_tracer:
        langfuse_generation = langfuse_tracer.start_generation(
            trace_context=trace_context, name="chat", model=llm_model_config["llm_name"],
            input={"prompt": prompt, "prompt4citation": prompt4citation, "messages": msg}
        )

    # 流式模式：逐片返回 delta，并在结束后补一条 final 消息携带完整引用和调试信息。
    if stream:
        if llm_type == "chat":
            stream_iter = chat_mdl.async_chat_streamly_delta(prompt + prompt4citation, msg[1:], gen_conf)
        else:
            stream_iter = chat_mdl.async_chat_streamly_delta(prompt + prompt4citation, msg[1:], gen_conf, images=image_files)
        # `last_state` 保存整个流式生成的累计状态，最后需要从中取出完整文本做引用修饰。
        last_state = None
        async for kind, value, state in _stream_with_think_delta(stream_iter):
            last_state = state
            if kind == "marker":
                # think 起止标记单独作为控制信号上抛，而不是混在普通文本里。
                flags = {"start_to_think": True} if value == "<think>" else {"end_to_think": True}
                yield {"answer": "", "reference": {}, "audio_binary": None, "final": False, **flags}
                continue
            # 普通文本分片直接增量返回，并按分片实时生成 TTS 音频。
            yield {"answer": value, "reference": {}, "audio_binary": tts(tts_mdl, value), "final": False}
        full_answer = last_state.full_text if last_state else ""
        if full_answer:
            # 最后一条 final 消息不再携带正文文本，而是主要携带引用、prompt 和最终态标记。
            # 这样做是为了避免前端把完整答案再重复追加一遍。
            final = decorate_answer(thought + full_answer)
            final["final"] = True
            final["audio_binary"] = None
            final["answer"] = ""
            yield final
    else:
        # 非流式模式：一次性拿到完整答案，再统一做引用修饰和 TTS。
        if llm_type == "chat":
            answer = await chat_mdl.async_chat(prompt + prompt4citation, msg[1:], gen_conf)
        else:
            answer = await chat_mdl.async_chat(prompt + prompt4citation, msg[1:], gen_conf, images=image_files)
        # 记录用户输入和最终回答，便于问题定位。
        user_content = msg[-1].get("content", "[content not available]")
        logging.debug("User: {}|Assistant: {}".format(user_content, answer))
        res = decorate_answer(answer)
        res["audio_binary"] = tts(tts_mdl, answer)
        yield res

    # 生成器显式返回，结束本轮异步聊天。
    return


async def use_sql(question, field_map, tenant_id, chat_mdl, quota=True, kb_ids=None):
    # SQL 分支入口。
    # 当字段映射足够完整时，会优先尝试把自然语言问题翻译成 SQL 来查，而不是直接走向量检索。
    logging.debug(f"use_sql: Question: {question}")

    # 识别当前底层文档引擎。
    # 不同引擎的表名规则、JSON 字段提取方式和文档名字段都不完全一样，必须先分流。
    if settings.DOC_ENGINE_INFINITY:
        doc_engine = "infinity"
    elif settings.DOC_ENGINE_OCEANBASE:
        doc_engine = "oceanbase"
    else:
        doc_engine = "es"

    # 构造要查询的底层表/索引名。
    # ES/OS 一般是租户级索引；Infinity 单知识库场景则会把 kb_id 拼进表名。
    base_table = index_name(tenant_id)
    if doc_engine == "infinity" and kb_ids and len(kb_ids) == 1:
        # Infinity 单 KB 模式是一库一表，因此需要把 kb_id 编进表名。
        table_name = f"{base_table}_{kb_ids[0]}"
        logging.debug(f"use_sql: Using Infinity table name: {table_name}")
    else:
        # 其它模式下直接使用租户级基础表名。
        table_name = base_table
        logging.debug(f"use_sql: Using ES/OS table name: {table_name}")

    # 不同引擎文档名字段不同，后面构造引用时会用到。
    expected_doc_name_column = "docnm" if doc_engine == "infinity" else "docnm_kwd"

    def has_source_columns(columns):
        # 检查查询结果里是否已经包含 citation 所需的源字段。
        # 至少需要 `doc_id` 和文档名字段，否则很难构造标准 reference。
        normalized_names = {str(col.get("name", "")).lower() for col in columns}
        return "doc_id" in normalized_names and bool({"docnm_kwd", "docnm"} & normalized_names)

    def is_aggregate_sql(sql_text):
        # 粗略判断是否为聚合查询。
        # 聚合查询和普通明细查询在“是否必须带 source columns”这件事上的要求不同。
        return bool(re.search(r"(count|sum|avg|max|min|distinct)\s*\(", (sql_text or "").lower()))

    def normalize_sql(sql):
        # 清洗 LLM 生成的 SQL 文本。
        # 目标是把可能混有思考内容、Markdown 围栏、结尾分号的文本规范成可执行 SQL。
        logging.debug(f"use_sql: Raw SQL from LLM: {repr(sql[:500])}")
        # 去掉思考块。
        sql = re.sub(r"</think>\n.*?\n\s*", "", sql, flags=re.DOTALL)
        sql = re.sub(r"思考\n.*?\n", "", sql, flags=re.DOTALL)
        # 去掉 Markdown 代码块包装。
        sql = re.sub(r"```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*$", "", sql, flags=re.IGNORECASE)
        # 去掉尾部分号，避免部分解析器不兼容。
        return sql.rstrip().rstrip(';').strip()

    def add_kb_filter(sql):
        # 自动为 SQL 补充 kb_id 过滤条件。
        # Infinity 单库模式已经通过表名限定了范围，所以这里只给 ES/OS 等模式补 WHERE 条件。
        if doc_engine == "infinity" or not kb_ids:
            return sql

        # 单知识库直接等值过滤，多知识库则用 OR 拼接。
        if len(kb_ids) == 1:
            kb_filter = f"kb_id = '{kb_ids[0]}'"
        else:
            kb_filter = "(" + " OR ".join([f"kb_id = '{kb_id}'" for kb_id in kb_ids]) + ")"

        # 尽量在不破坏原查询结构的情况下补进 kb 过滤。
        if "where " not in sql.lower():
            o = sql.lower().split("order by")
            if len(o) > 1:
                sql = o[0] + f" WHERE {kb_filter}  order by " + o[1]
            else:
                sql += f" WHERE {kb_filter}"
        elif "kb_id =" not in sql.lower() and "kb_id=" not in sql.lower():
            sql = re.sub(r"\bwhere\b ", f"where {kb_filter} and ", sql, flags=re.IGNORECASE)
        return sql

    def is_row_count_question(q: str) -> bool:
        # 特判“整张表/数据集有多少行”这一类问题。
        # 这类问题最稳定的答案通常就是 `COUNT(*)`，不需要让模型自由生成复杂 SQL。
        q = (q or "").lower()
        if not re.search(r"\bhow many rows\b|\bnumber of rows\b|\brow count\b", q):
            return False
        return bool(re.search(r"\bdataset\b|\btable\b|\bspreadsheet\b|\bexcel\b", q))

    # 按后端引擎生成不同的 SQL prompt。
    # 本质上是在把“自然语言问题 + 字段映射”约束成当前引擎真正能执行的 SQL。
    if doc_engine == "infinity":
        # Infinity 的字段主要放在 `chunk_data` JSON 中，因此要明确告诉模型如何用 `json_extract_string` 取值。
        json_field_names = list(field_map.keys())
        row_count_override = (
            f"SELECT COUNT(*) AS rows FROM {table_name}"
            if is_row_count_question(question)
            else None
        )
        # system prompt 负责定义角色、语法约束和输出边界。
        sys_prompt = """你是一个数据库管理员。请为一个包含 JSON 类型字段 chunk_data 的表编写 SQL。

JSON 字段提取：
json_extract_string(chunk_data, '$.FieldName')

数值转换：
CAST(json_extract_string(chunk_data, '$.FieldName') AS INTEGER/FLOAT)

NULL 判断：
json_extract_isnull(chunk_data, '$.FieldName') == false

规则：
使用下面列表中**完全一致（区分大小写）**的字段名
对于 SELECT：必须包含 doc_id、docnm，以及用 json_extract_string() 提取的字段
对于 COUNT：使用 COUNT(*) 或 COUNT(DISTINCT json_extract_string(...))
为提取的字段添加 AS 别名
不要选择 content 字段
仅在以下情况在 WHERE 子句中添加 NULL 判断（json_extract_isnull() == false）：
问题中要求“show me”或“display”某些列
问题提到“not null”或“excluding null”
对某个特定列做计数
❗对于 COUNT(*) 查询，不要添加 NULL 判断（因为 COUNT(*) 会统计所有行，包括 NULL）
只输出 SQL，不要附加任何解释"""
        # user prompt 提供表名、字段清单和具体问题。
        user_prompt = """Table: {}
Fields (EXACT case): {}
{}
Question: {}
Write SQL using json_extract_string() with exact field names. Include doc_id, docnm for data queries. Only SQL.""".format(
            table_name,
            ", ".join(json_field_names),
            "\n".join([f"  - {field}" for field in json_field_names]),
            question
        )
    elif doc_engine == "oceanbase":
        # OceanBase 和 Infinity 类似，也需要 JSON 字段提取语义，只是源字段命名略有差异。
        json_field_names = list(field_map.keys())
        row_count_override = (
            f"SELECT COUNT(*) AS rows FROM {table_name}"
            if is_row_count_question(question)
            else None
        )
        sys_prompt = """You are a Database Administrator. Write SQL for a table with JSON 'chunk_data' column.

JSON Extraction: json_extract_string(chunk_data, '$.FieldName')
Numeric Cast: CAST(json_extract_string(chunk_data, '$.FieldName') AS INTEGER/FLOAT)
NULL Check: json_extract_isnull(chunk_data, '$.FieldName') == false

RULES:
1. Use EXACT field names (case-sensitive) from the list below
2. For SELECT: include doc_id, docnm_kwd, and json_extract_string() for requested fields
3. For COUNT: use COUNT(*) or COUNT(DISTINCT json_extract_string(...))
4. Add AS alias for extracted field names
5. DO NOT select 'content' field
6. Only add NULL check (json_extract_isnull() == false) in WHERE clause when:
   - Question asks to "show me" or "display" specific columns
   - Question mentions "not null" or "excluding null"
   - Add NULL check for count specific column
   - DO NOT add NULL check for COUNT(*) queries (COUNT(*) counts all rows including nulls)
7. Output ONLY the SQL, no explanations"""
        user_prompt = """Table: {}
Fields (EXACT case): {}
{}
Question: {}
Write SQL using json_extract_string() with exact field names. Include doc_id, docnm_kwd for data queries. Only SQL.""".format(
            table_name,
            ", ".join(json_field_names),
            "\n".join([f"  - {field}" for field in json_field_names]),
            question
        )
    else:
        # Build ES/OS prompts with direct field access
        row_count_override = None
        sys_prompt = """你是一名数据库管理员。编写 SQL 查询。
规则：


使用下面模式中的精确字段名（例如：product_tks，而不是 product）


对以数字开头的字段名加引号："123_field"


当问题要求“show me”或“display”某些列时，在 WHERE 子句中添加 IS NOT NULL


在非聚合查询中包含 doc_id / docnm


只输出 SQL，不要包含任何解释"""
        user_prompt = """Table: {}
Available fields:
{}
Question: {}
Write SQL using exact field names above. Include doc_id, docnm_kwd for data queries. Only SQL.""".format(
            table_name,
            "\n".join([f"  - {k} ({v})" for k, v in field_map.items()]),
            question
        )

    # 记录 SQL 执行尝试次数，便于日志定位是首次执行还是重试/修复执行。
    tried_times = 0

    async def get_table(custom_user_prompt=None):
        # 内部执行器：让模型写 SQL，再调用底层 SQL 检索执行。
        # 提取成内部函数是为了复用到首次执行、报错重试和补 source columns 三种场景。
        nonlocal sys_prompt, user_prompt, question, tried_times, row_count_override
        # 对特定“总行数”问题优先使用固定 SQL，避免模型在简单 COUNT 场景下生成不稳定语句。
        if row_count_override and custom_user_prompt is None:
            sql = row_count_override
        else:
            # 其余场景交给聊天模型把自然语言问题翻译成 SQL。
            prompt = custom_user_prompt if custom_user_prompt is not None else user_prompt
            sql = await chat_mdl.async_chat(sys_prompt, [{"role": "user", "content": prompt}], {"temperature": 0.06})
        # 先做 SQL 规范化，再补知识库过滤条件。
        sql = normalize_sql(sql)
        sql = add_kb_filter(sql)

        logging.debug(f"{question} get SQL(refined): {sql}")
        tried_times += 1
        logging.debug(f"use_sql: Executing SQL retrieval (attempt {tried_times})")
        # 真正执行 SQL，返回统一 JSON 结构的表格结果。
        tbl = settings.retriever.sql_retrieval(sql, format="json")
        if tbl is None:
            logging.debug("use_sql: SQL retrieval returned None")
            return None, sql
        logging.debug(f"use_sql: SQL retrieval completed, got {len(tbl.get('rows', []))} rows")
        return tbl, sql

    async def repair_table_for_missing_source_columns(previous_sql):
        # 当查询结果缺少 `doc_id` / 文档名字段时，专门提示模型在不改变查询意图的前提下补这些列。
        # 这样做是为了让 SQL 结果也能走统一 citation/reference 结构。
        if doc_engine in ("infinity", "oceanbase"):
            json_field_names = list(field_map.keys())
            repair_prompt = """Table name: {};
JSON fields available in 'chunk_data' column (use exact names):
{}

Question: {}
Previous SQL:
{}

The previous SQL result is missing required source columns for citations.
Rewrite SQL to keep the same query intent and include doc_id and {} in the SELECT list.
For extracted JSON fields, use json_extract_string(chunk_data, '$.field_name').
Return ONLY SQL.""".format(
                table_name,
                "\n".join([f"  - {field}" for field in json_field_names]),
                question,
                previous_sql,
                expected_doc_name_column
            )
        else:
            repair_prompt = """Table name: {}
Available fields:
{}

Question: {}
Previous SQL:
{}

The previous SQL result is missing required source columns for citations.
Rewrite SQL to keep the same query intent and include doc_id and docnm_kwd in the SELECT list.
Return ONLY SQL.""".format(
                table_name,
                "\n".join([f"  - {k} ({v})" for k, v in field_map.items()]),
                question,
                previous_sql
            )
        return await get_table(custom_user_prompt=repair_prompt)

    try:
        # 第一次尝试：正常生成并执行 SQL。
        tbl, sql = await get_table()
        logging.debug(f"use_sql: Initial SQL execution SUCCESS. SQL: {sql}")
        logging.debug(f"use_sql: Retrieved {len(tbl.get('rows', []))} rows, columns: {[c['name'] for c in tbl.get('columns', [])]}")
    except Exception as e:
        logging.warning(f"use_sql: Initial SQL execution FAILED with error: {e}")
        # 首次失败后，把错误信息回喂给模型再重试一次。
        # 这样做是为了利用 LLM 的自修复能力，提高 SQL 翻译成功率。
        if doc_engine in ("infinity", "oceanbase"):
            # JSON 字段引擎的重试提示词会再次强调 `json_extract_string` 的正确语法。
            json_field_names = list(field_map.keys())
            user_prompt = """
表名：{}
在 chunk_data 列中可用的 JSON 字段（请使用这些精确字段名，通过 json_extract_string）：
{}

问题：{}

请使用如下语法编写 SQL：
json_extract_string(chunk_data, '$.field_name')，字段名必须来自上述列表。
只输出 SQL，不要任何解释。

你上一次提供的 SQL 报错如下：
{}

请修正该错误，并重新编写 SQL，使用
json_extract_string(chunk_data, '$.field_name') 语法，并确保字段名正确。
只输出 SQL，不要任何解释。
""".format(table_name, "\n".join([f"  - {field}" for field in json_field_names]), question, e)
        else:
            # ES/OS 模式的重试提示词直接基于普通字段名修正 SQL。
            user_prompt = """
        Table name: {};
        Table of database fields are as follows (use the field names directly in SQL):
        {}

        Question are as follows:
        {}
        Please write the SQL using the exact field names above, only SQL, without any other explanations or text.


        The SQL error you provided last time is as follows:
        {}

        Please correct the error and write SQL again using the exact field names above, only SQL, without any other explanations or text.
        """.format(table_name, "\n".join([f"{k} ({v})" for k, v in field_map.items()]), question, e)
        try:
            # 第二次尝试。
            tbl, sql = await get_table()
            logging.debug(f"use_sql: Retry SQL execution SUCCESS. SQL: {sql}")
            logging.debug(f"use_sql: Retrieved {len(tbl.get('rows', []))} rows on retry")
        except Exception:
            # 两次都失败时，放弃 SQL 分支，让外层逻辑自行回退到检索链路。
            logging.error("use_sql: Retry SQL execution also FAILED, returning None")
            return

    # 执行成功但 0 行结果时，也视为 SQL 路径不可用。
    # 外层会据此回退到普通向量检索。
    if len(tbl["rows"]) == 0:
        logging.warning(f"use_sql: No rows returned from SQL query, returning None. SQL: {sql}")
        return None

    # 非聚合 SQL 如果缺少 source columns，就再做一次“补源字段”的 SQL 修复。
    if not is_aggregate_sql(sql) and not has_source_columns(tbl.get("columns", [])):
        logging.warning(f"use_sql: Non-aggregate SQL missing required source columns; retrying once. SQL: {sql}")
        try:
            repaired_tbl, repaired_sql = await repair_table_for_missing_source_columns(sql)
            if (
                repaired_tbl
                and len(repaired_tbl.get("rows", [])) > 0
                and has_source_columns(repaired_tbl.get("columns", []))
            ):
                tbl, sql = repaired_tbl, repaired_sql
                logging.info(f"use_sql: Source-column SQL repair succeeded. SQL: {sql}")
            else:
                logging.warning(f"use_sql: Source-column SQL repair did not provide required columns. Repaired SQL: {repaired_sql}")
        except Exception as e:
            # 这里不再中断，而是允许带着“最佳努力结果”继续构造 answer。
            logging.warning(f"use_sql: Source-column SQL repair failed, returning best-effort answer. Error: {e}")

    # 走到这里说明已经拿到了可以转成最终答案的表格数据。
    logging.debug(f"use_sql: Proceeding with {len(tbl['rows'])} rows to build answer")

    # 识别结果里哪些列是 citation 所需的源字段。
    docid_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"].lower() == "doc_id"])
    doc_name_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"].lower() in ["docnm_kwd", "docnm"]])

    logging.debug(f"use_sql: All columns: {[(i, c['name']) for i, c in enumerate(tbl['columns'])]}")
    logging.debug(f"use_sql: docid_idx={docid_idx}, doc_name_idx={doc_name_idx}")

    # 真正用于展示的数据列，不包括 `doc_id` / 文档名字段。
    column_idx = [ii for ii in range(len(tbl["columns"])) if ii not in (docid_idx | doc_name_idx)]

    logging.debug(f"use_sql: column_idx={column_idx}")
    logging.debug(f"use_sql: field_map={field_map}")

    # 把底层字段名映射成更适合展示的列名。
    # 这样做是为了把内部字段或表达式转换成用户更容易读懂的表头。
    def map_column_name(col_name):
        if col_name.lower() == "count(star)":
            return "COUNT(*)"

        # First, try to extract AS alias from any expression (aggregate functions, json_extract_string, etc.)
        # Pattern: anything AS alias_name
        as_match = re.search(r'\s+AS\s+([^\s,)]+)', col_name, re.IGNORECASE)
        if as_match:
            alias = as_match.group(1).strip('"\'')

            # Use the alias for display name lookup
            if alias in field_map:
                display = field_map[alias]
                return re.sub(r"(/.*|（[^（）]+）)", "", display)
            # If alias not in field_map, try to match case-insensitively
            for field_key, display_value in field_map.items():
                if field_key.lower() == alias.lower():
                    return re.sub(r"(/.*|（[^（）]+）)", "", display_value)
            # Return alias as-is if no mapping found
            return alias

        # Try direct mapping first (for simple column names)
        if col_name in field_map:
            display = field_map[col_name]
            # Clean up any suffix patterns
            return re.sub(r"(/.*|（[^（）]+）)", "", display)

        # Try case-insensitive match for simple column names
        col_lower = col_name.lower()
        for field_key, display_value in field_map.items():
            if field_key.lower() == col_lower:
                return re.sub(r"(/.*|（[^（）]+）)", "", display_value)

        # For aggregate expressions or complex expressions without AS alias,
        # try to replace field names with display names
        result = col_name
        for field_name, display_name in field_map.items():
            # Replace field_name with display_name in the expression
            result = result.replace(field_name, display_name)

        # Clean up any suffix patterns
        result = re.sub(r"(/.*|（[^（）]+）)", "", result)
        return result

    # 先构造 Markdown 表头。
    # 如果存在 source columns，就额外放一个 `Source` 列，用来承载内部 citation 标记。
    columns = (
            "|" + "|".join(
        [map_column_name(tbl["columns"][i]["name"]) for i in column_idx]) + (
                "|Source|" if docid_idx and doc_name_idx else "|")
    )

    # Markdown 表头分隔线。
    line = "|" + "|".join(["------" for _ in range(len(column_idx))]) + ("|------|" if docid_idx and docid_idx else "")

    # 逐行构造 Markdown 表格内容。
    # 这里先把一行转换成“列名 -> 值”的字典，是为了避免 SQL 返回列顺序变化导致取值错位。
    rows = []
    for row_idx, r in enumerate(tbl["rows"]):
        row_dict = {tbl["columns"][i]["name"]: r[i] for i in range(len(tbl["columns"])) if i < len(r)}
        if row_idx == 0:
            logging.debug(f"use_sql: First row data: {row_dict}")
        row_values = []
        for col_idx in column_idx:
            col_name = tbl["columns"][col_idx]["name"]
            value = row_dict.get(col_name, " ")
            # 统一清理冗余空格，并把 None 展示为空。
            row_values.append(remove_redundant_spaces(str(value)).replace("None", " "))
        # 如果有 source columns，就插入内部引用占位符。
        # 后续回答链路可以据此把表格行和来源文档关联起来。
        if docid_idx and doc_name_idx:
            row_values.append(f" ##{row_idx}$$")
        row_str = "|" + "|".join(row_values) + "|"
        if re.sub(r"[ |]+", "", row_str):
            rows.append(row_str)
    if quota:
        rows = "\n".join(rows)
    else:
        rows = "\n".join(rows)
    # 清理时间字段里不适合直接展示的 ISO 时间部分。
    rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

    # 如果结果里仍然没有 source columns，就只能走降级逻辑。
    if not docid_idx or not doc_name_idx:
        logging.warning(f"use_sql: SQL missing required doc_id or docnm_kwd field. docid_idx={docid_idx}, doc_name_idx={doc_name_idx}. SQL: {sql}")
        # 聚合查询比较特殊：表格答案本身是有效的，但仍希望尽量补出来源引用。
        if is_aggregate_sql(sql):
            # 先保留原始表格答案。
            answer = "\n".join([columns, line, rows])

            # 再尝试从原 SQL 的 WHERE 条件反查来源文档。
            # 这样做是为了在不破坏聚合结果格式的前提下补齐 chunks/doc_aggs。
            where_match = re.search(r"\bwhere\b(.+?)(?:\bgroup by\b|\border by\b|\blimit\b|$)", sql, re.IGNORECASE)
            if where_match:
                where_clause = where_match.group(1).strip()
                # 用同样的过滤条件查询来源文档。
                chunks_sql = f"select doc_id, docnm_kwd from {table_name} where {where_clause}"
                # 加 LIMIT，避免补引用时查出过多来源。
                if "limit" not in chunks_sql.lower():
                    chunks_sql += " limit 20"
                logging.debug(f"use_sql: Fetching chunks with SQL: {chunks_sql}")
                try:
                    chunks_tbl = settings.retriever.sql_retrieval(chunks_sql, format="json")
                    if chunks_tbl.get("rows") and len(chunks_tbl["rows"]) > 0:
                        # 构造补引用所需的 chunks，列名匹配使用大小写不敏感方式。
                        chunks_did_idx = next((i for i, c in enumerate(chunks_tbl["columns"]) if c["name"].lower() == "doc_id"), None)
                        chunks_dn_idx = next((i for i, c in enumerate(chunks_tbl["columns"]) if c["name"].lower() in ["docnm_kwd", "docnm"]), None)
                        if chunks_did_idx is not None and chunks_dn_idx is not None:
                            chunks = [{"doc_id": r[chunks_did_idx], "docnm_kwd": r[chunks_dn_idx]} for r in chunks_tbl["rows"]]
                            # 按文档聚合出现次数，生成 doc_aggs。
                            doc_aggs = {}
                            for r in chunks_tbl["rows"]:
                                doc_id = r[chunks_did_idx]
                                doc_name = r[chunks_dn_idx]
                                if doc_id not in doc_aggs:
                                    doc_aggs[doc_id] = {"doc_name": doc_name, "count": 0}
                                doc_aggs[doc_id]["count"] += 1
                            doc_aggs_list = [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in doc_aggs.items()]
                            logging.debug(f"use_sql: Returning aggregate answer with {len(chunks)} chunks from {len(doc_aggs)} documents")
                            return {"answer": answer, "reference": {"chunks": chunks, "doc_aggs": doc_aggs_list}, "prompt": sys_prompt}
                except Exception as e:
                    # 补引用失败不影响聚合答案本身。
                    logging.warning(f"use_sql: Failed to fetch chunks: {e}")
            # 实在补不出来源，就返回无 chunks 的聚合答案。
            return {"answer": answer, "reference": {"chunks": [], "doc_aggs": []}, "prompt": sys_prompt}
        # 非聚合查询也降级返回表格答案，只是没有标准 reference。
        return {"answer": "\n".join([columns, line, rows]), "reference": {"chunks": [], "doc_aggs": []}, "prompt": sys_prompt}

    # 有标准 source columns 时，正式构造 doc_aggs。
    docid_idx = list(docid_idx)[0]
    doc_name_idx = list(doc_name_idx)[0]
    doc_aggs = {}
    for r in tbl["rows"]:
        # 统计每个文档在结果表中出现了多少次。
        if r[docid_idx] not in doc_aggs:
            doc_aggs[r[docid_idx]] = {"doc_name": r[doc_name_idx], "count": 0}
        doc_aggs[r[docid_idx]]["count"] += 1

    # 最终返回 Markdown 表格答案、逐行来源 chunks、按文档聚合的 doc_aggs，以及这次使用的 SQL prompt。
    result = {
        "answer": "\n".join([columns, line, rows]),
        "reference": {
            "chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[doc_name_idx]} for r in tbl["rows"]],
            "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in doc_aggs.items()],
        },
        "prompt": sys_prompt,
    }
    logging.debug(f"use_sql: Returning answer with {len(result['reference']['chunks'])} chunks from {len(doc_aggs)} documents")
    return result

def clean_tts_text(text: str) -> str:
    # 空文本直接返回空字符串。
    # 这样做是为了让调用方在没有可朗读内容时尽早结束，避免后续正则和编码处理的无效开销。
    if not text:
        return ""

    # 先做一次 UTF-8 容错编解码，丢弃无法正常表示的非法字节。
    # 这样做是为了降低 TTS 服务在遇到脏数据、异常编码字符时的失败概率。
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # 移除不可见控制字符，但保留常见可用空白。
    # 这样做是为了避免控制字符干扰语音合成引擎，同时不破坏正常文本的基本分隔结构。
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

    # 构造 emoji 过滤正则。
    # 这样做是因为大多数 TTS 模型对 emoji 的朗读效果很差，甚至会直接报错或输出异常停顿。
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U0001FAD0-\U0001FAFF]+",
        flags=re.UNICODE
    )
    # 去掉 emoji 和部分符号表情字符。
    text = emoji_pattern.sub("", text)

    # 把连续空白折叠成单个空格，并去掉首尾空白。
    # 这样做是为了让朗读节奏更稳定，避免模型把多余换行或重复空格读成异常停顿。
    text = re.sub(r"\s+", " ", text).strip()

    # 限制送入 TTS 的最大文本长度。
    # 这样做是为了控制单次合成成本和延迟，也避免超长文本触发底层模型或接口限制。
    MAX_LEN = 500
    if len(text) > MAX_LEN:
        # 超长时直接截断，保留前 500 个字符。
        text = text[:MAX_LEN]

    # 返回清洗后的可朗读文本。
    return text

def tts(tts_mdl, text):
    # 没有 TTS 模型或没有文本时直接返回空。
    # 这样做是为了把语音合成视为可选增强能力，避免因为缺少 TTS 条件而影响主回答流程。
    if not tts_mdl or not text:
        return None
    # 先对文本做适合语音合成的清洗。
    # 这样做是为了去掉不适合朗读的符号、过长内容或噪声，降低 TTS 失败率并提升朗读效果。
    text = clean_tts_text(text)
    # 清洗后如果文本已经为空，就不再继续调用 TTS。
    # 这样做是为了避免把无意义文本送进语音模型，浪费计算并可能触发异常。
    if not text:
        return None
    # 用二进制缓冲区累计语音流返回的所有音频分片。
    bin = b""
    try:
        # 逐片拉取 TTS 模型输出的音频数据并拼接。
        # 这样做是为了兼容底层按流方式返回音频字节，而不是一次性返回完整文件。
        for chunk in tts_mdl.tts(text):
            bin += chunk
    except Exception as e:
        # TTS 失败时只记录日志并返回空，不让异常打断主对话链路。
        # 这样做是因为语音输出通常是附加能力，失败不应该影响文本回答本身。
        logging.error(f"TTS failed: {e}, text={text!r}")
        return None
    # 把完整音频字节转成十六进制字符串返回，便于后续通过文本协议传输。
    return binascii.hexlify(bin).decode("utf-8")


class _ThinkStreamState:
    def __init__(self) -> None:
        self.full_text = ""
        self.last_idx = 0
        self.endswith_think = False
        self.last_full = ""
        self.last_model_full = ""
        self.in_think = False
        self.buffer = ""


def _next_think_delta(state: _ThinkStreamState) -> str:
    full_text = state.full_text
    if full_text == state.last_full:
        return ""
    state.last_full = full_text
    delta_ans = full_text[state.last_idx:]

    if delta_ans.find("<think>") == 0:
        state.last_idx += len("<think>")
        return "<think>"
    if delta_ans.find("<think>") > 0:
        delta_text = full_text[state.last_idx:state.last_idx + delta_ans.find("<think>")]
        state.last_idx += delta_ans.find("<think>")
        return delta_text
    if delta_ans.endswith("</think>"):
        state.endswith_think = True
    elif state.endswith_think:
        state.endswith_think = False
        return "</think>"

    state.last_idx = len(full_text)
    if full_text.endswith("</think>"):
        state.last_idx -= len("</think>")
    return re.sub(r"(<think>|</think>)", "", delta_ans)


async def _stream_with_think_delta(stream_iter, min_tokens: int = 16):
    # 用状态对象维护完整文本、上次已消费位置、think 边界和缓冲区。
    # 这样做是为了把底层模型原始流式输出整理成更稳定的“文本分片 + think 标记”事件流。
    state = _ThinkStreamState()
    # 逐片消费底层流式输出。
    async for chunk in stream_iter:
        # 空分片直接跳过，避免无意义处理。
        if not chunk:
            continue
        # 如果当前分片以前一个完整模型输出为前缀，说明底层返回的是“累计全文”模式。
        # 这时只截取新增部分，转换成真正的 delta。
        if chunk.startswith(state.last_model_full):
            new_part = chunk[len(state.last_model_full):]
            state.last_model_full = chunk
        else:
            # 否则把当前 chunk 视为天然增量，并追加到完整模型输出里。
            # 这样做是为了同时兼容“累计全文流”和“天然 delta 流”两种底层实现。
            new_part = chunk
            state.last_model_full += chunk
        # 如果没有新增内容，就继续读下一个分片。
        if not new_part:
            continue
        # 把新增文本并入完整答案文本。
        state.full_text += new_part
        # 从完整文本里提取下一段对上游有意义的 delta。
        # 这个 delta 可能是普通文本，也可能是 `<think>` / `</think>` 标记。
        delta = _next_think_delta(state)
        # 当前还没有可安全输出的内容时，继续累积。
        if not delta:
            continue
        # 对 think 起止标记做专门处理。
        if delta in ("<think>", "</think>"):
            # 连续重复的 `<think>` 直接忽略，避免重复进入思考态。
            if delta == "<think>" and state.in_think:
                continue
            # 未进入思考态时收到 `</think>` 也忽略，避免状态错乱。
            if delta == "</think>" and not state.in_think:
                continue
            # 输出标记前，先把已缓冲的普通文本吐出去，保证文本和控制信号边界清晰。
            if state.buffer:
                yield ("text", state.buffer, state)
                state.buffer = ""
            # 更新当前是否处于 think 区间的状态。
            state.in_think = delta == "<think>"
            # 把 think 标记作为单独事件上抛，而不是混在文本里。
            yield ("marker", delta, state)
            continue
        # 普通文本先进入缓冲区，不立刻逐字符输出。
        state.buffer += delta
        # 只有当缓冲区达到最小 token 数时才输出。
        # 这样做是为了减少分片过碎造成的前端抖动和上下游处理开销。
        if num_tokens_from_string(state.buffer) < min_tokens:
            continue
        # 达到阈值后，把缓冲文本作为一个文本事件输出。
        yield ("text", state.buffer, state)
        state.buffer = ""

    # 流结束后，如果还有残留文本缓冲，补发最后一段。
    if state.buffer:
        yield ("text", state.buffer, state)
        state.buffer = ""
    # 如果流结束时仍挂着一个待补发的 `</think>`，这里补一个结束标记。
    # 这样做是为了保证 think 区间在事件层面总是成对闭合。
    if state.endswith_think:
        yield ("marker", "</think>", state)

async def async_ask(question, kb_ids, tenant_id, chat_llm_name=None, search_config={}):
    doc_ids = search_config.get("doc_ids", [])
    rerank_mdl = None
    kb_ids = search_config.get("kb_ids", kb_ids)
    chat_llm_name = search_config.get("chat_id", chat_llm_name)
    rerank_id = search_config.get("rerank_id", "")
    meta_data_filter = search_config.get("meta_data_filter")

    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))

    is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
    retriever = settings.retriever if not is_knowledge_graph else settings.kg_retriever
    embd_owner_tenant_id = kbs[0].tenant_id
    embd_model_config = get_model_config_by_type_and_name(embd_owner_tenant_id, LLMType.EMBEDDING, embedding_list[0])
    embd_mdl = LLMBundle(embd_owner_tenant_id, embd_model_config)
    chat_model_config = get_model_config_by_type_and_name(tenant_id, LLMType.CHAT, chat_llm_name)
    chat_mdl = LLMBundle(tenant_id, chat_model_config)
    if rerank_id:
        rerank_model_config = get_model_config_by_type_and_name(tenant_id, LLMType.RERANK, rerank_id)
        rerank_mdl = LLMBundle(tenant_id, rerank_model_config)
    max_tokens = chat_mdl.max_length
    tenant_ids = list(set([kb.tenant_id for kb in kbs]))

    if meta_data_filter:
        metas = DocMetadataService.get_flatted_meta_by_kbs(kb_ids)
        doc_ids = await apply_meta_data_filter(meta_data_filter, metas, question, chat_mdl, doc_ids)

    kbinfos = await retriever.retrieval(
        question=question,
        embd_mdl=embd_mdl,
        tenant_ids=tenant_ids,
        kb_ids=kb_ids,
        page=1,
        page_size=12,
        similarity_threshold=search_config.get("similarity_threshold", 0.1),
        vector_similarity_weight=search_config.get("vector_similarity_weight", 0.3),
        top=search_config.get("top_k", 1024),
        doc_ids=doc_ids,
        aggs=True,
        rerank_mdl=rerank_mdl,
        rank_feature=label_question(question, kbs)
    )

    knowledges = kb_prompt(kbinfos, max_tokens)
    sys_prompt = PROMPT_JINJA_ENV.from_string(ASK_SUMMARY).render(knowledge="\n".join(knowledges))

    msg = [{"role": "user", "content": question}]

    def decorate_answer(answer):
        nonlocal knowledges, kbinfos, sys_prompt
        answer, idx = retriever.insert_citations(answer, [ck["content_ltks"] for ck in kbinfos["chunks"]], [ck["vector"] for ck in kbinfos["chunks"]],
                                                 embd_mdl, tkweight=0.7, vtweight=0.3)
        idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
        recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
        if not recall_docs:
            recall_docs = kbinfos["doc_aggs"]
        kbinfos["doc_aggs"] = recall_docs
        refs = deepcopy(kbinfos)
        for c in refs["chunks"]:
            if c.get("vector"):
                del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        refs["chunks"] = chunks_format(refs)
        return {"answer": answer, "reference": refs}

    stream_iter = chat_mdl.async_chat_streamly_delta(sys_prompt, msg, {"temperature": 0.1})
    last_state = None
    async for kind, value, state in _stream_with_think_delta(stream_iter):
        last_state = state
        if kind == "marker":
            flags = {"start_to_think": True} if value == "<think>" else {"end_to_think": True}
            yield {"answer": "", "reference": {}, "final": False, **flags}
            continue
        yield {"answer": value, "reference": {}, "final": False}
    full_answer = last_state.full_text if last_state else ""
    final = decorate_answer(full_answer)
    final["final"] = True
    final["answer"] = ""
    yield final


async def gen_mindmap(question, kb_ids, tenant_id, search_config={}):
    meta_data_filter = search_config.get("meta_data_filter", {})
    doc_ids = search_config.get("doc_ids", [])
    rerank_id = search_config.get("rerank_id", "")
    rerank_mdl = None
    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    if not kbs:
        return {"error": "No KB selected"}
    tenant_embedding_list = list(set([kb.tenant_embd_id for kb in kbs]))
    tenant_ids = list(set([kb.tenant_id for kb in kbs]))
    if tenant_embedding_list[0]:
        embd_model_config = get_model_config_by_id(tenant_embedding_list[0])
        embd_owner_tenant_id = kbs[0].tenant_id
    else:
        embd_owner_tenant_id = kbs[0].tenant_id
        embd_model_config = get_model_config_by_type_and_name(embd_owner_tenant_id, LLMType.EMBEDDING, kbs[0].embd_id)
    embd_mdl = LLMBundle(embd_owner_tenant_id, embd_model_config)
    chat_id = search_config.get("chat_id", "")
    if chat_id:
        chat_model_config = get_model_config_by_type_and_name(tenant_id, LLMType.CHAT, chat_id)
    else:
        chat_model_config = get_tenant_default_model_by_type(tenant_id, LLMType.CHAT)
    chat_mdl = LLMBundle(tenant_id, chat_model_config)
    if rerank_id:
        rerank_model_config = get_model_config_by_type_and_name(tenant_id, LLMType.RERANK, rerank_id)
        rerank_mdl = LLMBundle(tenant_id, rerank_model_config)

    if meta_data_filter:
        metas = DocMetadataService.get_flatted_meta_by_kbs(kb_ids)
        doc_ids = await apply_meta_data_filter(meta_data_filter, metas, question, chat_mdl, doc_ids)

    ranks = await settings.retriever.retrieval(
        question=question,
        embd_mdl=embd_mdl,
        tenant_ids=tenant_ids,
        kb_ids=kb_ids,
        page=1,
        page_size=12,
        similarity_threshold=search_config.get("similarity_threshold", 0.2),
        vector_similarity_weight=search_config.get("vector_similarity_weight", 0.3),
        top=search_config.get("top_k", 1024),
        doc_ids=doc_ids,
        aggs=False,
        rerank_mdl=rerank_mdl,
        rank_feature=label_question(question, kbs),
    )
    mindmap = MindMapExtractor(chat_mdl)
    mind_map = await mindmap([c["content_with_weight"] for c in ranks["chunks"]])
    return mind_map.output
