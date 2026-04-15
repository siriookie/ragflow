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
import time
from uuid import uuid4
from common.constants import StatusEnum
from api.db.db_models import Conversation, DB
from api.db.services.api_service import API4ConversationService
from api.db.services.common_service import CommonService
from api.db.services.dialog_service import DialogService, async_chat
from common.misc_utils import get_uuid
import json

from rag.prompts.generator import chunks_format


class ConversationService(CommonService):
    model = Conversation

    @classmethod
    @DB.connection_context()
    def get_list(cls, dialog_id, page_number, items_per_page, orderby, desc, id, name, user_id=None):
        sessions = cls.model.select().where(cls.model.dialog_id == dialog_id)
        if id:
            sessions = sessions.where(cls.model.id == id)
        if name:
            sessions = sessions.where(cls.model.name == name)
        if user_id:
            sessions = sessions.where(cls.model.user_id == user_id)
        if desc:
            sessions = sessions.order_by(cls.model.getter_by(orderby).desc())
        else:
            sessions = sessions.order_by(cls.model.getter_by(orderby).asc())

        if items_per_page > 0:
            sessions = sessions.paginate(page_number, items_per_page)

        return list(sessions.dicts())

    @classmethod
    @DB.connection_context()
    def get_all_conversation_by_dialog_ids(cls, dialog_ids):
        sessions = cls.model.select().where(cls.model.dialog_id.in_(dialog_ids))
        sessions.order_by(cls.model.create_time.asc())
        offset, limit = 0, 100
        res = []
        while True:
            s_batch = sessions.offset(offset).limit(limit)
            _temp = list(s_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += limit
        return res


def structure_answer(conv, ans, message_id, session_id):
    # 从下游返回结果里取出 reference 字段。
    # 这样做是为了统一处理检索引用信息，后续无论是返回给前端还是写回会话，都基于同一份引用数据。
    reference = ans["reference"]
    # 如果 reference 不是字典，强制兜底为空字典。
    # 这样做是为了保证后续对 `reference["chunks"]`、`reference.get(...)` 的访问不会因为类型异常而报错。
    if not isinstance(reference, dict):
        reference = {}
        ans["reference"] = {}
    # 判断当前答案是否为最终态；默认按最终态处理。
    # 默认 True 的原因是很多非流式调用只会返回一次完整结果，这时不需要显式传 final 也应正常落入最终分支。
    is_final = ans.get("final", True)

    # 将原始 reference 结构格式化成前端和持久化逻辑更易消费的 chunk 列表。
    # 这样做是为了把底层检索结果标准化，避免上层依赖下游实现细节。
    chunk_list = chunks_format(reference)

    # 用标准化后的 chunk 列表覆盖 reference。
    # 这样做是为了让后续所有消费者读取到的都是统一结构，而不是混杂原始和格式化后的两套数据。
    reference["chunks"] = chunk_list
    # 把消息 ID 绑定到当前答案上。
    # 这样做是为了让前端在流式过程中能稳定识别“这几个分片属于同一条回答”。
    ans["id"] = message_id
    # 把 session_id 一并写回答案。
    # 这样做是为了让调用方在消费答案时无需额外维护上下文，也能直接知道答案归属哪个会话。
    ans["session_id"] = session_id

    # 如果没有传入会话对象，只做答案结构整理，不做会话内存态更新。
    # 这样做是为了让这个函数既能服务“纯返回值整形”，也能服务“整形 + 回写会话”两种场景。
    if not conv:
        return ans

    # 确保会话消息列表存在。
    # 这样做是为了让后面的追加、覆盖逻辑始终运行在统一的数据结构上。
    if not conv.message:
        conv.message = []
    # 默认使用答案正文作为本次要写入 assistant 消息的内容。
    content = ans["answer"]
    # 如果模型开始输出思维链标记，则把内容替换成 `<think>`。
    # 这样做是为了在流式展示时保留“思考开始”的边界，而不是把控制信号误当普通文本拼接。
    if ans.get("start_to_think"):
        content = "<think>"
    # 如果模型结束思维链标记，则把内容替换成 `</think>`。
    # 这样做可以与开始标记成对出现，便于前端或后续处理识别 think 区段。
    elif ans.get("end_to_think"):
        content = "</think>"

    # 如果当前会话最后一条不是 assistant，说明还没有可供更新的回复占位，需要新插入一条 assistant 消息。
    # 这样做是为了兼容某些未预插入 assistant 占位消息的调用路径。
    if not conv.message or conv.message[-1].get("role", "") != "assistant":
        conv.message.append({"role": "assistant", "content": content, "created_at": time.time(), "id": message_id})
    else:
        # 走到这里说明最后一条已经是 assistant，需要更新这条已有消息。
        if is_final:
            # 最终态且有完整 answer 时，直接用最终答案整体覆盖最后一条 assistant 消息。
            # 这样做是为了避免流式拼接过程中残留中间态内容，确保落库结果是干净、完整的最终回答。
            if ans.get("answer"):
                conv.message[-1] = {"role": "assistant", "content": ans["answer"], "created_at": time.time(), "id": message_id}
            else:
                # 最终态但 answer 为空时，只刷新时间和 ID，不覆盖原内容。
                # 这样做是为了兼容“最终分片不带正文、只带结束信号”的场景，避免把已有内容清空。
                conv.message[-1]["created_at"] = time.time()
                conv.message[-1]["id"] = message_id
        else:
            # 非最终态时，把本次增量内容追加到已有 assistant 文本后面。
            # 这样做是为了支持流式输出的逐片累积，让会话对象始终反映当前已生成到哪里。
            conv.message[-1]["content"] = (conv.message[-1].get("content") or "") + content
            # 每次收到新分片都刷新更新时间，方便前端排序和后端审计看到真实活跃时间。
            conv.message[-1]["created_at"] = time.time()
            # 同步写入消息 ID，确保流式过程中消息标识保持稳定。
            conv.message[-1]["id"] = message_id
    # 只有会话上存在 reference 容器时，才尝试更新引用信息。
    # 这样做是为了兼容没有初始化引用槽位的调用场景，避免越界或空对象访问。
    if conv.reference:
        # 最终态时一定更新引用；非最终态只有在已有 chunks 或 doc_aggs 时才更新。
        # 这样做是为了避免用“空引用中间态”覆盖掉前面已经积累出的有效引用信息。
        should_update_reference = is_final or bool(reference.get("chunks")) or bool(reference.get("doc_aggs"))
        if should_update_reference:
            # 把当前轮的最后一个 reference 槽位替换为最新引用结果。
            # 这里使用覆盖而不是追加，是因为同一轮回答只应对应一份最终引用视图。
            conv.reference[-1] = reference
    # 返回整理后的答案给上游调用方。
    return ans


async def async_completion(tenant_id, chat_id, question, name="New session", session_id=None, stream=True, **kwargs):
    # 会话名称不能为空。
    # 这样做是为了保证新建会话时具备最基本的可展示元数据，避免后续列表展示、检索或审计时出现空标题。
    assert name, "`name` can not be empty."
    # 按租户和对话 ID 查询对话配置，并且只允许命中有效状态的数据。
    # 这样做是为了同时完成“资源存在性校验”和“租户隔离校验”，防止越权访问其他租户的聊天配置。
    dia = DialogService.query(id=chat_id, tenant_id=tenant_id, status=StatusEnum.VALID.value)
    # 如果查不到对话，直接中断。
    # 这样做是为了在真正进入推理流程前尽早失败，避免无效请求继续消耗模型和检索资源。
    assert dia, "You do not own the chat."

    # 如果没有传入 session_id，说明这是一个全新的会话，需要初始化会话记录。
    if not session_id:
        # 生成新的会话 ID。
        # 这样做是为了让本次对话从一开始就有稳定主键，后续流式输出和持久化都可以引用同一个会话。
        session_id = get_uuid()
        # 构造会话初始数据。
        # 这里预先写入 assistant 的开场白，是因为前端通常需要在第一次进入会话时立即看到系统预设的欢迎语或引导语。
        conv = {
            "id": session_id,
            "dialog_id": chat_id,
            "name": name,
            # 将对话配置中的 prologue 作为第一条 assistant 消息落库。
            # 这样做可以把“系统预设开场白”也纳入正式会话历史，保证前端显示、历史回放和上下文重建一致。
            "message": [{"role": "assistant", "content": dia[0].prompt_config.get("prologue"), "created_at": time.time()}],
            # 记录发起者 user_id；拿不到时兜底为空串。
            # 这样做是为了兼容匿名场景，同时为后续审计、过滤和权限控制保留用户维度。
            "user_id": kwargs.get("user_id", "")
        }
        # 持久化新会话。
        # 先保存再返回，是为了确保前端一旦拿到 session_id，就能立刻通过该 ID 查询到真实存在的会话。
        ConversationService.save(**conv)
        # 根据是否流式输出，决定返回 SSE 事件流还是普通单次结果。
        if stream:
            # 先返回一条包含开场白的 SSE 数据。
            # 这样做是为了让前端在“新建会话”的第一时间就能渲染欢迎语，而不是空白等待。
            yield "data:" + json.dumps({"code": 0, "message": "",
                                        "data": {
                                            "answer": conv["message"][0]["content"],
                                            "reference": {},
                                            "audio_binary": None,
                                            "id": None,
                                        "session_id": session_id
                                        }},
                                    ensure_ascii=False) + "\n\n"
            # 再返回一条结束标记 True。
            # 这样做是为了复用统一的 SSE 消费协议，让前端知道这一轮输出已经完整结束。
            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"
            # 初始化会话完成后直接返回，不再进入后续问答逻辑。
            return
        else:
            # 非流式模式下，直接组织成普通字典结果返回。
            # 这样做是为了兼容一次性拿完整结果的调用方，不强迫所有客户端都走 SSE。
            answer = {
                "answer": conv["message"][0]["content"],
                "reference": {},
                "audio_binary": None,
                "id": None,
                "session_id": session_id
            }
            # 通过异步生成器产出一次结果。
            # 保持 `async_completion` 的统一生成器接口，调用方无论流式或非流式都可以用同一种迭代方式消费。
            yield answer
            # 首次会话初始化到此结束。
            return

    # 走到这里说明是已有会话，需要把用户问题追加到历史中继续对话。
    conv = ConversationService.query(id=session_id, dialog_id=chat_id)
    # 会话不存在时抛出明确异常，而不是静默创建。
    # 这样做是为了避免客户端传错 session_id 时污染数据，也避免把“续聊”误处理成“新会话”。
    if not conv:
        raise LookupError("Session does not exist")

    # 查询接口返回列表，这里取实际会话对象。
    conv = conv[0]
    # `msg` 是传给下游模型/检索链路的消息列表，后面会从完整会话中裁剪出适合推理的内容。
    msg = []
    # 把本轮用户输入封装为标准消息结构。
    # 增加唯一 ID 是为了让这一轮问答在流式更新、消息替换和前端渲染时有稳定锚点。
    question = {
        "content": question,
        "role": "user",
        "id": str(uuid4())
    }

    # 透传本次请求携带的文件附件。
    # 这样做是为了让下游聊天流程能够解析文件内容，把“用户文本问题 + 附件内容”作为一个完整输入来处理。
    if isinstance(kwargs.get("files"), list) and kwargs["files"]:
        question["files"] = kwargs["files"]

    # 先把用户问题追加到完整会话历史中。
    # 这样后面无论是落库还是构造上下文，都会基于同一份最新会话状态，避免内存态和持久态不一致。
    conv.message.append(question)
    # 从完整消息历史中提炼出真正发送给模型的消息序列。
    # 这里不会直接把 `conv.message` 原样传下去，因为部分消息只适合存档，不适合参与本轮生成。
    for m in conv.message:
        # 跳过 system 消息。
        # 这样做通常是因为系统设定已经体现在 dialog 配置里，重复发送可能造成提示词冗余甚至行为偏移。
        if m["role"] == "system":
            continue
        # 如果开头第一条就是 assistant，则跳过。
        # 这样做是为了避免把“开场白”当成用户问题之前的有效上下文，减少无意义 token 消耗和回答干扰。
        if m["role"] == "assistant" and not msg:
            continue
        # 保留其余消息，形成真正参与模型推理的上下文。
        msg.append(m)
    # 取最后一条消息的 ID，也就是当前用户问题的 ID。
    # 后续生成出来的 assistant 回复会复用这个 message_id，从而把一问一答绑定在同一轮交互上。
    message_id = msg[-1].get("id")
    # 再次根据会话中的 dialog_id 取回完整对话对象。
    # 这样做是为了拿到标准对象形态的对话配置，供下游 `async_chat` 使用。
    e, dia = DialogService.get_by_id(conv.dialog_id)

    # 获取调用方额外指定的知识库 ID；默认空列表。
    kb_ids = kwargs.get("kb_ids",[])
    # 将对话原有知识库和本次请求附加知识库合并去重。
    # 这样做是为了支持“会话级默认知识库 + 请求级临时知识库”共同参与检索，同时避免重复检索同一个库。
    dia.kb_ids = list(set(dia.kb_ids + kb_ids))
    # 如果还没有 reference 容器，则初始化为空列表。
    # reference 与 message 是按轮次对齐存储的，后续每一轮回答都会占用一个 reference 槽位。
    if not conv.reference:
        conv.reference = []
    # 先插入一个空的 assistant 占位消息。
    # 这样做是为了在流式生成过程中可以持续把增量答案写回同一条 assistant 记录，而不是不断新增新消息。
    conv.message.append({"role": "assistant", "content": "", "id": message_id})
    # 为这一轮回答预先插入空引用结构。
    # 这样做是为了让引用信息和 assistant 回复天然一一对应，后续 chunk/doc 聚合结果可以原地补全。
    conv.reference.append({"chunks": [], "doc_aggs": []})

    # 流式模式下，通过 SSE 持续向调用方推送增量结果。
    if stream:
        try:
            # 调用下游异步聊天流程，并以流模式接收分片答案。
            async for ans in async_chat(dia, msg, True, **kwargs):
                # 把下游原始答案结构整合进当前会话对象，包括累计文本、引用和会话 ID。
                # 这样做是为了把“模型输出格式”转换成“系统内部统一的会话状态格式”。
                ans = structure_answer(conv, ans, message_id, session_id)
                # 每收到一个分片就立即推送给前端，降低首字延迟并提升交互体验。
                yield "data:" + json.dumps({"code": 0, "data": ans}, ensure_ascii=False) + "\n\n"
            # 流式输出完成后再统一落库最终会话状态。
            # 这样做可以减少流式过程中频繁写数据库的开销，同时保存完整最终答案。
            ConversationService.update_by_id(conv.id, conv.to_dict())
        except Exception as e:
            # 把异常转换成前端可识别的 SSE 错误消息，而不是让连接直接断掉。
            # 这样做是为了让客户端能正常结束当前轮次并展示错误信息，避免只能看到莫名中断。
            yield "data:" + json.dumps({"code": 500, "message": str(e),
                                        "data": {"answer": "**ERROR**: " + str(e), "reference": []}},
                                       ensure_ascii=False) + "\n\n"
        # 无论成功还是失败，最后都发送结束标记。
        # 这样做是为了保证前端消费协议稳定，始终可以依赖同一个“结束信号”收尾。
        yield "data:" + json.dumps({"code": 0, "data": True}, ensure_ascii=False) + "\n\n"

    else:
        # 非流式模式下，只保留最终答案对象。
        answer = None
        # 这里虽然也调用异步生成器，但会在拿到第一条完整答案后立即退出。
        # 这样做是为了复用同一套下游聊天接口，避免为同步/异步分别维护两套实现。
        async for ans in async_chat(dia, msg, False, **kwargs):
            # 将最终答案整理到会话对象中。
            answer = structure_answer(conv, ans, message_id, session_id)
            # 非流式模式在拿到完整答案后立即落库。
            # 因为不会再有后续增量分片，所以此时持久化就是这一轮的最终状态。
            ConversationService.update_by_id(conv.id, conv.to_dict())
            # 只取第一条结果后退出循环。
            # 这隐含约束了 `async_chat(..., False)` 在非流式场景下应当产出“单个完整答案”。
            break
        # 产出最终答案给调用方。
        yield answer

async def async_iframe_completion(dialog_id, question, session_id=None, stream=True, **kwargs):
    e, dia = DialogService.get_by_id(dialog_id)
    assert e, "Dialog not found"
    if not session_id:
        session_id = get_uuid()
        conv = {
            "id": session_id,
            "dialog_id": dialog_id,
            "user_id": kwargs.get("user_id", ""),
            "message": [{"role": "assistant", "content": dia.prompt_config["prologue"], "created_at": time.time()}]
        }
        API4ConversationService.save(**conv)
        yield "data:" + json.dumps({"code": 0, "message": "",
                                    "data": {
                                        "answer": conv["message"][0]["content"],
                                        "reference": {},
                                        "audio_binary": None,
                                        "id": None,
                                        "session_id": session_id
                                    }},
                                   ensure_ascii=False) + "\n\n"
        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"
        return
    else:
        session_id = session_id
        e, conv = API4ConversationService.get_by_id(session_id)
        assert e, "Session not found!"

    if not conv.message:
        conv.message = []
    messages = conv.message
    question = {
        "role": "user",
        "content": question,
        "id": str(uuid4())
    }
    messages.append(question)

    msg = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue
        msg.append(m)
    if not msg[-1].get("id"):
        msg[-1]["id"] = get_uuid()
    message_id = msg[-1]["id"]

    if not conv.reference:
        conv.reference = []
    conv.reference.append({"chunks": [], "doc_aggs": []})

    if stream:
        try:
            async for ans in async_chat(dia, msg, True, **kwargs):
                ans = structure_answer(conv, ans, message_id, session_id)
                yield "data:" + json.dumps({"code": 0, "message": "", "data": ans},
                                           ensure_ascii=False) + "\n\n"
            API4ConversationService.append_message(conv.id, conv.to_dict())
        except Exception as e:
            yield "data:" + json.dumps({"code": 500, "message": str(e),
                                        "data": {"answer": "**ERROR**: " + str(e), "reference": []}},
                                       ensure_ascii=False) + "\n\n"
        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

    else:
        answer = None
        async for ans in async_chat(dia, msg, False, **kwargs):
            answer = structure_answer(conv, ans, message_id, session_id)
            API4ConversationService.append_message(conv.id, conv.to_dict())
            break
        yield answer
