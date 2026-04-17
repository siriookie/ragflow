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
import datetime
import json
import logging
import re
from copy import deepcopy
from typing import Tuple
import jinja2
import json_repair
from common.misc_utils import hash_str2int
from rag.nlp import rag_tokenizer
from rag.prompts.template import load_prompt
from common.constants import TAG_FLD
from common.token_utils import encoder, num_tokens_from_string

STOP_TOKEN = "<|STOP|>"
COMPLETE_TASK = "complete_task"
INPUT_UTILIZATION = 0.5


def get_value(d, k1, k2):
    return d.get(k1, d.get(k2))


def chunks_format(reference):
    if not reference or not isinstance(reference, dict):
        return []
    raw_chunks = reference.get("chunks", [])
    if not isinstance(raw_chunks, list):
        return []
    return [
        {
            "id": get_value(chunk, "chunk_id", "id"),
            "content": get_value(chunk, "content", "content_with_weight"),
            "document_id": get_value(chunk, "doc_id", "document_id"),
            "document_name": get_value(chunk, "docnm_kwd", "document_name"),
            "dataset_id": get_value(chunk, "kb_id", "dataset_id"),
            "image_id": get_value(chunk, "image_id", "img_id"),
            "positions": get_value(chunk, "positions", "position_int"),
            "url": chunk.get("url"),
            "similarity": chunk.get("similarity"),
            "vector_similarity": chunk.get("vector_similarity"),
            "term_similarity": chunk.get("term_similarity"),
            "row_id": chunk.get("row_id"),
            "doc_type": get_value(chunk, "doc_type_kwd", "doc_type"),
        }
        for chunk in raw_chunks
        if isinstance(chunk, dict)
    ]


def message_fit_in(msg, max_length=4000):
    # 统计当前消息列表的总 token 数。
    def count():
        # 这里会在外层对 msg 重新赋值，所以用 nonlocal 读取最新的 msg。
        nonlocal msg
        # 保存每条消息的角色和 token 数，便于后面累计总长度。
        tks_cnts = []
        # 逐条统计消息内容的 token 数。
        for m in msg:
            tks_cnts.append({"role": m["role"], "count": num_tokens_from_string(m["content"])})
        # 汇总所有消息的 token 数。
        total = 0
        # 把每条消息的 token 数累加成总长度。
        for m in tks_cnts:
            total += m["count"]
        # 返回当前消息列表的总 token 数。
        return total

    # 先计算完整消息列表的 token 数。
    c = count()
    # 如果完整消息本来就没超限，直接返回，不做任何裁剪。
    if c < max_length:
        return c, msg

    # 超限时，优先保留所有 system 消息，因为它们通常承载规则、提示词和约束。
    msg_ = [m for m in msg if m["role"] == "system"]
    # 如果原始消息不止一条，再额外保留最后一条消息。
    # 通常最后一条是最新 user 输入，对当前请求最关键。
    if len(msg) > 1:
        msg_.append(msg[-1])
    # 用精简后的消息列表替换原 msg，后续都基于这个子集继续裁剪。
    msg = msg_
    # 重新计算裁剪后的 token 数。
    c = count()
    # 如果此时已经能放进上下文窗口，就直接返回。
    if c < max_length:
        return c, msg

    # 再进一步裁剪前，分别计算第一条和最后一条消息的 token 数。
    # 这里通常分别对应 system prompt 和最后一条关键消息。
    ll = num_tokens_from_string(msg_[0]["content"])
    ll2 = num_tokens_from_string(msg_[-1]["content"])
    # 如果第一条消息占比超过 80%，说明 system prompt 过长，
    # 优先截断第一条消息，尽量完整保留最后一条输入。
    if ll / (ll + ll2) > 0.8:
        # 取出第一条消息内容准备裁剪。
        m = msg_[0]["content"]
        # 先按 tokenizer 编码，再截到“最大长度 - 最后一条消息长度”，最后解码回字符串。
        m = encoder.decode(encoder.encode(m)[: max_length - ll2])
        # 把裁剪后的内容写回第一条消息。
        msg[0]["content"] = m
        # 返回裁剪后的消息列表，并把长度视为正好占满 max_length。
        return max_length, msg

    # 否则说明最后一条消息更适合被截断，或者至少不需要优先截断 system prompt。
    m = msg_[-1]["content"]
    # 按同样方式裁剪最后一条消息内容。
    m = encoder.decode(encoder.encode(m)[: max_length - ll2])
    # 把裁剪后的内容写回最后一条消息。
    msg[-1]["content"] = m
    # 返回裁剪后的消息列表，并把长度视为正好占满 max_length。
    return max_length, msg


def kb_prompt(kbinfos, max_tokens, hash_id=False):
    # 延迟导入文档服务，避免模块加载阶段产生不必要依赖。
    from api.db.services.document_service import DocumentService
    # 延迟导入元数据服务，后面需要为每个文档补充 metadata。
    from api.db.services.doc_metadata_service import DocMetadataService

    # 先把每个检索 chunk 的正文内容取出来，兼容 content / content_with_weight 两种字段名。
    knowledges = [get_value(ck, "content", "content_with_weight") for ck in kbinfos["chunks"]]
    # 记录原始候选 chunk 总数，便于后面日志输出。
    kwlg_len = len(knowledges)
    # 累计已经放入 prompt 的 token 数。
    used_token_count = 0
    # 记录最终允许放进 prompt 的 chunk 数量。
    chunks_num = 0
    # 逐条检查检索结果，决定能装进 prompt 的 chunk 上限。
    for i, c in enumerate(knowledges):
        # 空内容直接跳过，不占 token 预算。
        if not c:
            continue
        # 把当前 chunk 的内容 token 数累加到总预算中。
        used_token_count += num_tokens_from_string(c)
        # 当前 chunk 可参与计数。
        chunks_num += 1
        # 预留一点余量，只使用 max_tokens 的 97%，避免 prompt 太满。
        if max_tokens * 0.97 < used_token_count:
            # 超预算时只保留前面已经能装下的内容。
            knowledges = knowledges[:i]
            # 记录有多少检索结果因为 token 限制没有放进 prompt。
            logging.warning(f"Not all the retrieval into prompt: {len(knowledges)}/{kwlg_len}")
            break
    # 会先从这些 chunk 里抽文档 ID：
    #
    # ["doc_100", "doc_100", "doc_200"]
    # 然后批量去查文档对象。
    #
    # 假设查回来是：
    #
    # [
    #     Document(id="doc_100", name="三国人物志"),
    #     Document(id="doc_200", name="蜀汉政权史")
    # ]
    # 取出将被放进 prompt 的这些 chunk 所属文档 ID，并批量查文档对象。
    docs = DocumentService.get_by_ids([get_value(ck, "doc_id", "document_id") for ck in kbinfos["chunks"][:chunks_num]])
    # doc_100 的 metadata 是
    # {"author": "张三", "category": "人物"}
    # doc_200 的 metadata 是
    # {"author": "李四", "category": "政权", "year": "2024"}
    # 最后 docs 就变成：
    #
    # {
    #     "doc_100": {"author": "张三", "category": "人物"},
    #     "doc_200": {"author": "李四", "category": "政权", "year": "2024"}
    # }
    # docs_with_meta 用来构建 doc_id -> metadata 的映射。
    docs_with_meta = {}
    # 为每个文档补查 metadata，后面一起拼进 prompt。
    for d in docs:
        meta = DocMetadataService.get_document_metadata(d.id)
        # 没有 metadata 时用空 dict 占位，避免后面遍历报错。
        docs_with_meta[d.id] = meta if meta else {}
    # 后面只需要 doc_id 到 metadata 的映射，所以直接覆盖 docs 变量。
    docs = docs_with_meta

    # 把单个字段渲染成 prompt 里的树状节点文本。
    def draw_node(k, line):
        # 非字符串值统一转成字符串，方便拼接显示。
        if line is not None and not isinstance(line, str):
            line = str(line)
        # 空值不渲染，避免产生噪声。
        if not line:
            return ""
        # 把多行文本压平成单行，并加上树状前缀。
        return f"\n├── {k}: " + re.sub(r"\n+", " ", line, flags=re.DOTALL)

    # 重新用 knowledges 变量承载最终格式化后的 prompt 片段列表。
    knowledges = []
    # 遍历最终允许进入 prompt 的 chunk，逐个格式化。
    for i, ck in enumerate(kbinfos["chunks"][:chunks_num]):
        # 先写入 chunk 的展示 ID；如启用 hash_id，则用稳定哈希值代替原始 chunk_id。
        cnt = "\nID: {}".format(i if not hash_id else hash_str2int(get_value(ck, "id", "chunk_id"), 500))
        # 追加文档标题。
        cnt += draw_node("Title", get_value(ck, "docnm_kwd", "document_name"))
        # 如果有 URL，就把来源链接也拼进去。
        cnt += draw_node("URL", ck['url']) if "url" in ck else ""
        # 再把当前文档的 metadata 字段逐个追加到 prompt 中。
        for k, v in docs.get(get_value(ck, "doc_id", "document_id"), {}).items():
            cnt += draw_node(k, v)
        # 最后追加正文内容标题。
        cnt += "\n└── Content:\n"
        # 拼接当前 chunk 的正文内容。
        cnt += get_value(ck, "content", "content_with_weight")
        # 把格式化好的单条知识片段放入结果列表。
        knowledges.append(cnt)

    # 返回最终可直接拼到 LLM prompt 里的知识片段列表。
    return knowledges


def memory_prompt(message_list, max_tokens):
    used_token_count = 0
    content_list = []
    for message in message_list:
        current_content_tokens = num_tokens_from_string(message["content"])
        if used_token_count + current_content_tokens > max_tokens * 0.97:
            logging.warning(f"Not all the retrieval into prompt: {len(content_list)}/{len(message_list)}")
            break
        content_list.append(message["content"])
        used_token_count += current_content_tokens
    return content_list


CITATION_PROMPT_TEMPLATE = load_prompt("citation_prompt")
CITATION_PLUS_TEMPLATE = load_prompt("citation_plus")
CONTENT_TAGGING_PROMPT_TEMPLATE = load_prompt("content_tagging_prompt")
CROSS_LANGUAGES_SYS_PROMPT_TEMPLATE = load_prompt("cross_languages_sys_prompt")
CROSS_LANGUAGES_USER_PROMPT_TEMPLATE = load_prompt("cross_languages_user_prompt")
FULL_QUESTION_PROMPT_TEMPLATE = load_prompt("full_question_prompt")
KEYWORD_PROMPT_TEMPLATE = load_prompt("keyword_prompt")
QUESTION_PROMPT_TEMPLATE = load_prompt("question_prompt")
VISION_LLM_DESCRIBE_PROMPT = load_prompt("vision_llm_describe_prompt")
VISION_LLM_FIGURE_DESCRIBE_PROMPT = load_prompt("vision_llm_figure_describe_prompt")
VISION_LLM_FIGURE_DESCRIBE_PROMPT_WITH_CONTEXT = load_prompt("vision_llm_figure_describe_prompt_with_context")
STRUCTURED_OUTPUT_PROMPT = load_prompt("structured_output_prompt")

ANALYZE_TASK_SYSTEM = load_prompt("analyze_task_system")
ANALYZE_TASK_USER = load_prompt("analyze_task_user")
NEXT_STEP = load_prompt("next_step")
REFLECT = load_prompt("reflect")
SUMMARY4MEMORY = load_prompt("summary4memory")
RANK_MEMORY = load_prompt("rank_memory")
META_FILTER = load_prompt("meta_filter")
ASK_SUMMARY = load_prompt("ask_summary")

PROMPT_JINJA_ENV = jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)


def citation_prompt(user_defined_prompts: dict = {}) -> str:
    template = PROMPT_JINJA_ENV.from_string(user_defined_prompts.get("citation_guidelines", CITATION_PROMPT_TEMPLATE))
    return template.render()


def citation_plus(sources: str) -> str:
    template = PROMPT_JINJA_ENV.from_string(CITATION_PLUS_TEMPLATE)
    return template.render(example=citation_prompt(), sources=sources)


async def keyword_extraction(chat_mdl, content, topn=3):
    template = PROMPT_JINJA_ENV.from_string(KEYWORD_PROMPT_TEMPLATE)
    rendered_prompt = template.render(content=content, topn=topn)

    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def question_proposal(chat_mdl, content, topn=3):
    template = PROMPT_JINJA_ENV.from_string(QUESTION_PROMPT_TEMPLATE)
    rendered_prompt = template.render(content=content, topn=topn)

    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def full_question(tenant_id=None, llm_id=None, messages=[], language=None, chat_mdl=None):
    # 延迟导入相关依赖。
    # 这样做是为了避免模块加载时形成不必要的循环依赖，也让这个工具函数在被调用时才解析模型相关对象。
    from common.constants import LLMType
    from api.db.services.llm_service import LLMBundle
    from api.db.services.tenant_llm_service import TenantLLMService
    from api.db.joint_services.tenant_model_service import get_model_config_by_type_and_name

    # 如果调用方没有直接传入可用的 chat_mdl，就在这里按 tenant 和 llm_id 临时构造一个。
    # 这样做是为了让 `full_question` 既能复用外部已绑定模型，也能独立工作。
    if not chat_mdl:
        # 图片理解模型和普通聊天模型的配置通道不同，这里要先按 llm 类型分流。
        if TenantLLMService.llm_id2llm_type(llm_id) == "image2text":
            chat_model_config = get_model_config_by_type_and_name(tenant_id, LLMType.IMAGE2TEXT, llm_id)
        else:
            chat_model_config = get_model_config_by_type_and_name(tenant_id, LLMType.CHAT, llm_id)
        # 构造一个可直接调用的模型 bundle。
        chat_mdl = LLMBundle(tenant_id, chat_model_config)
    # `conv` 用来收集会进入“问题补全”提示词的对话历史。
    conv = []
    for m in messages:
        # 只保留 user / assistant 两类消息。
        # 这样做是为了让问题改写只聚焦真正的对话内容，而不把 system 等控制消息混进去。
        if m["role"] not in ["user", "assistant"]:
            continue
        # 把每条消息规范成 `ROLE: content` 的纯文本形式。
        # 这样做是为了给 LLM 一个清晰、简单且厂商无关的多轮对话上下文。
        conv.append("{}: {}".format(m["role"].upper(), m["content"]))
    # 把多轮消息拼成一段连续对话文本。
    conversation = "\n".join(conv)
    # 生成今天、昨天、明天的绝对日期。
    # 这样做是为了帮助模型把“今天/昨天/明天”这类相对时间补全成更明确的查询意图。
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    # 用模板生成“把当前多轮对话补全成完整问题”的 system prompt。
    template = PROMPT_JINJA_ENV.from_string(FULL_QUESTION_PROMPT_TEMPLATE)
    rendered_prompt = template.render(
        today=today,
        yesterday=yesterday,
        tomorrow=tomorrow,
        conversation=conversation,
        language=language,
    )

    # 调用模型生成补全后的完整问题。
    # 这里 user 输入固定为 `Output: `，意味着真正的任务说明都在 rendered_prompt 里。
    ans = await chat_mdl.async_chat(rendered_prompt, [{"role": "user", "content": "Output: "}])
    # 去掉可能夹带的思考内容，只保留最终输出的问题文本。
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    # 如果模型没有报错，就返回补全后的问题；否则回退到原始最后一条用户消息。
    # 这样做是为了保证问题改写失败时不会阻断主检索链路。
    return ans if ans.find("**ERROR**") < 0 else messages[-1]["content"]


async def cross_languages(tenant_id, llm_id, query, languages=[]):
    from common.constants import LLMType
    from api.db.services.llm_service import LLMBundle
    from api.db.services.tenant_llm_service import TenantLLMService
    from api.db.joint_services.tenant_model_service import get_model_config_by_type_and_name, get_tenant_default_model_by_type

    if llm_id and TenantLLMService.llm_id2llm_type(llm_id) == "image2text":
        chat_model_config = get_model_config_by_type_and_name(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        if not llm_id:
            chat_model_config = get_tenant_default_model_by_type(tenant_id, LLMType.CHAT)
        else:
            chat_model_config = get_model_config_by_type_and_name(tenant_id, LLMType.CHAT, llm_id)
    chat_mdl = LLMBundle(tenant_id, chat_model_config)
    rendered_sys_prompt = PROMPT_JINJA_ENV.from_string(CROSS_LANGUAGES_SYS_PROMPT_TEMPLATE).render()
    rendered_user_prompt = PROMPT_JINJA_ENV.from_string(CROSS_LANGUAGES_USER_PROMPT_TEMPLATE).render(query=query,
                                                                                                     languages=languages)

    ans = await chat_mdl.async_chat(rendered_sys_prompt, [{"role": "user", "content": rendered_user_prompt}],
                                    {"temperature": 0.2})
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    if ans.find("**ERROR**") >= 0:
        return query
    return "\n".join([a for a in re.sub(r"(^Output:|\n+)", "", ans, flags=re.DOTALL).split("===") if a.strip()])


async def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
    template = PROMPT_JINJA_ENV.from_string(CONTENT_TAGGING_PROMPT_TEMPLATE)

    for ex in examples:
        ex["tags_json"] = json.dumps(ex[TAG_FLD], indent=2, ensure_ascii=False)

    rendered_prompt = template.render(
        topn=topn,
        all_tags=all_tags,
        examples=examples,
        content=content,
    )

    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.5})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        raise Exception(kwd)

    try:
        obj = json_repair.loads(kwd)
    except json_repair.JSONDecodeError:
        try:
            result = kwd.replace(rendered_prompt[:-1], "").replace("user", "").replace("model", "").strip()
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            obj = json_repair.loads(result)
        except Exception as e:
            logging.exception(f"JSON parsing error: {result} -> {e}")
            raise e
    res = {}
    for k, v in obj.items():
        try:
            if int(v) > 0:
                res[str(k)] = int(v)
        except Exception:
            pass
    return res


def vision_llm_describe_prompt(page=None) -> str:
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_DESCRIBE_PROMPT)

    return template.render(page=page)


def vision_llm_figure_describe_prompt() -> str:
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_FIGURE_DESCRIBE_PROMPT)
    return template.render()


def vision_llm_figure_describe_prompt_with_context(context_above: str, context_below: str) -> str:
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_FIGURE_DESCRIBE_PROMPT_WITH_CONTEXT)
    return template.render(context_above=context_above, context_below=context_below)


def tool_schema(tools_description: list[dict], complete_task=False):
    if not tools_description:
        return ""
    desc = {}
    if complete_task:
        desc[COMPLETE_TASK] = {
            "type": "function",
            "function": {
                "name": COMPLETE_TASK,
                "description": "When you have the final answer and are ready to complete the task, call this function with your answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "The final answer to the user's question"}},
                    "required": ["answer"]
                }
            }
        }
    for idx, tool in enumerate(tools_description):
        name = tool["function"]["name"]
        desc[name] = tool

    return "\n\n".join([f"## {i + 1}. {fnm}\n{json.dumps(des, ensure_ascii=False, indent=4)}" for i, (fnm, des) in
                        enumerate(desc.items())])


def form_history(history, limit=-6):
    context = ""
    for h in history[limit:]:
        if h["role"] == "system":
            continue
        role = "USER"
        if h["role"].upper() != role:
            role = "AGENT"
        context += f"\n{role}: {h['content'][:2048] + ('...' if len(h['content']) > 2048 else '')}"
    return context


async def analyze_task_async(chat_mdl, prompt, task_name, tools_description: list[dict],
                             user_defined_prompts: dict = {}):
    tools_desc = tool_schema(tools_description)
    context = ""

    if user_defined_prompts.get("task_analysis"):
        template = PROMPT_JINJA_ENV.from_string(user_defined_prompts["task_analysis"])
    else:
        template = PROMPT_JINJA_ENV.from_string(ANALYZE_TASK_SYSTEM + "\n\n" + ANALYZE_TASK_USER)
    context = template.render(task=task_name, context=context, agent_prompt=prompt, tools_desc=tools_desc)
    kwd = await chat_mdl.async_chat(context, [{"role": "user", "content": "Please analyze it."}])
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def next_step_async(chat_mdl, history: list, tools_description: list[dict], task_desc,
                          user_defined_prompts: dict = {}):
    if not tools_description:
        return "", 0
    desc = tool_schema(tools_description)
    template = PROMPT_JINJA_ENV.from_string(user_defined_prompts.get("plan_generation", NEXT_STEP))
    user_prompt = "\nWhat's the next tool to call? If ready OR IMPOSSIBLE TO BE READY, then call `complete_task`."
    hist = deepcopy(history)
    if hist[-1]["role"] == "user":
        hist[-1]["content"] += user_prompt
    else:
        hist.append({"role": "user", "content": user_prompt})
    json_str = await chat_mdl.async_chat(
        template.render(task_analysis=task_desc, desc=desc, today=datetime.datetime.now().strftime("%Y-%m-%d")),
        hist[1:],
        stop=["<|stop|>"],
    )
    tk_cnt = num_tokens_from_string(json_str)
    json_str = re.sub(r"^.*</think>", "", json_str, flags=re.DOTALL)
    return json_str, tk_cnt


async def reflect_async(chat_mdl, history: list[dict], tool_call_res: list[Tuple], user_defined_prompts: dict = {}):
    tool_calls = [{"name": p[0], "result": p[1]} for p in tool_call_res]
    goal = history[1]["content"]
    template = PROMPT_JINJA_ENV.from_string(user_defined_prompts.get("reflection", REFLECT))
    user_prompt = template.render(goal=goal, tool_calls=tool_calls)
    hist = deepcopy(history)
    if hist[-1]["role"] == "user":
        hist[-1]["content"] += user_prompt
    else:
        hist.append({"role": "user", "content": user_prompt})
    _, msg = message_fit_in(hist, chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    return """
**Observation**
{}

**Reflection**
{}
    """.format(json.dumps(tool_calls, ensure_ascii=False, indent=2), ans)


def form_message(system_prompt, user_prompt):
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def structured_output_prompt(schema=None) -> str:
    template = PROMPT_JINJA_ENV.from_string(STRUCTURED_OUTPUT_PROMPT)
    return template.render(schema=schema)


async def tool_call_summary(chat_mdl, name: str, params: dict, result: str, user_defined_prompts: dict = {}) -> str:
    template = PROMPT_JINJA_ENV.from_string(SUMMARY4MEMORY)
    system_prompt = template.render(name=name,
                                    params=json.dumps(params, ensure_ascii=False, indent=2),
                                    result=result)
    user_prompt = "→ Summary: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


async def rank_memories_async(chat_mdl, goal: str, sub_goal: str, tool_call_summaries: list[str],
                              user_defined_prompts: dict = {}):
    template = PROMPT_JINJA_ENV.from_string(RANK_MEMORY)
    system_prompt = template.render(goal=goal, sub_goal=sub_goal,
                                    results=[{"i": i, "content": s} for i, s in enumerate(tool_call_summaries)])
    user_prompt = " → rank: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:], stop="<|stop|>")
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


async def gen_meta_filter(chat_mdl, meta_data: dict, query: str, constraints: dict = None) -> dict:
    meta_data_structure = {}
    for key, values in meta_data.items():
        meta_data_structure[key] = list(values.keys()) if isinstance(values, dict) else values

    sys_prompt = PROMPT_JINJA_ENV.from_string(META_FILTER).render(
        current_date=datetime.datetime.today().strftime('%Y-%m-%d'),
        metadata_keys=json.dumps(meta_data_structure),
        user_question=query,
        constraints=json.dumps(constraints) if constraints else None
    )
    user_prompt = "Generate filters:"
    ans = await chat_mdl.async_chat(sys_prompt, [{"role": "user", "content": user_prompt}])
    ans = re.sub(r"(^.*</think>|```json\n|```\n*$)", "", ans, flags=re.DOTALL)
    try:
        ans = json_repair.loads(ans)
        assert isinstance(ans, dict), ans
        assert "conditions" in ans and isinstance(ans["conditions"], list), ans
        return ans
    except Exception:
        logging.exception(f"Loading json failure: {ans}")

    return {"conditions": []}


async def gen_json(system_prompt: str, user_prompt: str, chat_mdl, gen_conf={}, max_retry=2):
    # 延迟导入 LLM 缓存工具，避免模块加载阶段引入不必要依赖。
    from rag.graphrag.utils import get_llm_cache, set_llm_cache
    # 先按“模型名 + system prompt + user prompt + 生成参数”尝试命中缓存，
    # 相同输入下直接复用历史结果，减少重复调用模型。
    cached = get_llm_cache(chat_mdl.llm_name, system_prompt, user_prompt, gen_conf)
    # 如果缓存存在，直接修复并解析成 JSON 返回。
    if cached:
        return json_repair.loads(cached)
    # 组装 system/user 消息，并在超长时自动裁剪到模型上下文窗口内。
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    # err 用来记录上一次 JSON 解析失败的异常信息，便于下一轮提示模型自我修正。
    err = ""
    # ans 保存模型最近一次返回的原始文本结果。
    ans = ""
    # 最多重试 max_retry 次，给模型机会修正不合法的 JSON。
    for _ in range(max_retry):
        # 如果上一轮已经生成了内容，但解析失败，就把“错误结果 + 异常信息”附加给模型，
        # 让它基于失败样例重新输出合法 JSON。
        if ans and err:
            msg[-1]["content"] += f"\nGenerated JSON is as following:\n{ans}\nBut exception while loading:\n{err}\nPlease reconsider and correct it."
        # 调用聊天模型生成 JSON 风格输出。
        ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:], gen_conf=gen_conf)
        # 去掉可能出现的思维链前缀和 Markdown ```json 代码块包裹，只保留 JSON 主体。
        ans = re.sub(r"(^.*</think>|```json\n|```\n*$)", "", ans, flags=re.DOTALL)
        try:
            # 用 json_repair 容错解析，允许模型输出带少量格式问题的“近似 JSON”。
            res = json_repair.loads(ans)
            # 解析成功后写入缓存，后续相同请求可直接复用。
            set_llm_cache(chat_mdl.llm_name, system_prompt, ans, user_prompt, gen_conf)
            # 返回解析后的 Python 对象（通常是 dict 或 list）。
            return res
        except Exception as e:
            # 解析失败时记录日志，方便排查模型输出格式问题。
            logging.exception(f"Loading json failure: {ans}")
            # 把本轮异常消息累计到 err，供下一次重试时反馈给模型。
            err += str(e)


TOC_DETECTION = load_prompt("toc_detection")


async def detect_table_of_contents(page_1024: list[str], chat_mdl):
    toc_secs = []
    for i, sec in enumerate(page_1024[:22]):
        ans = await gen_json(PROMPT_JINJA_ENV.from_string(TOC_DETECTION).render(page_txt=sec), "Only JSON please.",
                             chat_mdl)
        if toc_secs and not ans["exists"]:
            break
        toc_secs.append(sec)
    return toc_secs


TOC_EXTRACTION = load_prompt("toc_extraction")
TOC_EXTRACTION_CONTINUE = load_prompt("toc_extraction_continue")


async def extract_table_of_contents(toc_pages, chat_mdl):
    if not toc_pages:
        return []

    return await gen_json(PROMPT_JINJA_ENV.from_string(TOC_EXTRACTION).render(toc_page="\n".join(toc_pages)),
                          "Only JSON please.", chat_mdl)


async def toc_index_extractor(toc: list[dict], content: str, chat_mdl):
    tob_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the title of the section are not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = tob_extractor_prompt + '\nTable of contents:\n' + json.dumps(toc, ensure_ascii=False,
                                                                          indent=2) + '\nDocument pages:\n' + content
    return await gen_json(prompt, "Only JSON please.", chat_mdl)


TOC_INDEX = load_prompt("toc_index")


async def table_of_contents_index(toc_arr: list[dict], sections: list[str], chat_mdl):
    if not toc_arr or not sections:
        return []

    toc_map = {}
    for i, it in enumerate(toc_arr):
        k1 = (it["structure"] + it["title"]).replace(" ", "")
        k2 = it["title"].strip()
        if k1 not in toc_map:
            toc_map[k1] = []
        if k2 not in toc_map:
            toc_map[k2] = []
        toc_map[k1].append(i)
        toc_map[k2].append(i)

    for it in toc_arr:
        it["indices"] = []
    for i, sec in enumerate(sections):
        sec = sec.strip()
        if sec.replace(" ", "") in toc_map:
            for j in toc_map[sec.replace(" ", "")]:
                toc_arr[j]["indices"].append(i)

    all_pathes = []

    def dfs(start, path):
        nonlocal all_pathes
        if start >= len(toc_arr):
            if path:
                all_pathes.append(path)
            return
        if not toc_arr[start]["indices"]:
            dfs(start + 1, path)
            return
        added = False
        for j in toc_arr[start]["indices"]:
            if path and j < path[-1][0]:
                continue
            _path = deepcopy(path)
            _path.append((j, start))
            added = True
            dfs(start + 1, _path)
        if not added and path:
            all_pathes.append(path)

    dfs(0, [])
    path = max(all_pathes, key=lambda x: len(x))
    for it in toc_arr:
        it["indices"] = []
    for j, i in path:
        toc_arr[i]["indices"] = [j]
    print(json.dumps(toc_arr, ensure_ascii=False, indent=2))

    i = 0
    while i < len(toc_arr):
        it = toc_arr[i]
        if it["indices"]:
            i += 1
            continue

        if i > 0 and toc_arr[i - 1]["indices"]:
            st_i = toc_arr[i - 1]["indices"][-1]
        else:
            st_i = 0
        e = i + 1
        while e < len(toc_arr) and not toc_arr[e]["indices"]:
            e += 1
        if e >= len(toc_arr):
            e = len(sections)
        else:
            e = toc_arr[e]["indices"][0]

        for j in range(st_i, min(e + 1, len(sections))):
            ans = await gen_json(PROMPT_JINJA_ENV.from_string(TOC_INDEX).render(
                structure=it["structure"],
                title=it["title"],
                text=sections[j]), "Only JSON please.", chat_mdl)
            if ans["exist"] == "yes":
                it["indices"].append(j)
                break

        i += 1

    return toc_arr


async def check_if_toc_transformation_is_complete(content, toc, chat_mdl):
    prompt = """
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
    response = await gen_json(prompt, "Only JSON please.", chat_mdl)
    return response['completed']


async def toc_transformer(toc_pages, chat_mdl):
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    The `structure` is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.
    The `title` is a short phrase or a several-words term.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>
        },
        ...
    ],
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    toc_content = "\n".join(toc_pages)
    prompt = init_prompt + '\n Given table of contents\n:' + toc_content

    def clean_toc(arr):
        for a in arr:
            a["title"] = re.sub(r"[.·….]{2,}", "", a["title"])

    last_complete = await gen_json(prompt, "Only JSON please.", chat_mdl)
    if_complete = await check_if_toc_transformation_is_complete(toc_content,
                                                                json.dumps(last_complete, ensure_ascii=False, indent=2),
                                                                chat_mdl)
    clean_toc(last_complete)
    if if_complete == "yes":
        return last_complete

    while not (if_complete == "yes"):
        prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
        The response should be in the following JSON format:

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {json.dumps(last_complete[-24:], ensure_ascii=False, indent=2)}

        Please continue the json structure, directly output the remaining part of the json structure."""
        new_complete = await gen_json(prompt, "Only JSON please.", chat_mdl)
        if not new_complete or str(last_complete).find(str(new_complete)) >= 0:
            break
        clean_toc(new_complete)
        last_complete.extend(new_complete)
        if_complete = await check_if_toc_transformation_is_complete(toc_content,
                                                                    json.dumps(last_complete, ensure_ascii=False,
                                                                               indent=2), chat_mdl)

    return last_complete


TOC_LEVELS = load_prompt("assign_toc_levels")


async def assign_toc_levels(toc_secs, chat_mdl, gen_conf={"temperature": 0.2}):
    if not toc_secs:
        return []
    return await gen_json(
        PROMPT_JINJA_ENV.from_string(TOC_LEVELS).render(),
        str(toc_secs),
        chat_mdl,
        gen_conf
    )


TOC_FROM_TEXT_SYSTEM = load_prompt("toc_from_text_system")
TOC_FROM_TEXT_USER = load_prompt("toc_from_text_user")


# Generate TOC from text chunks with text llms
async def gen_toc_from_text(txt_info: dict, chat_mdl, callback=None):
    if callback:
        callback(msg="")
    try:
        ans = await gen_json(
            PROMPT_JINJA_ENV.from_string(TOC_FROM_TEXT_SYSTEM).render(),
            PROMPT_JINJA_ENV.from_string(TOC_FROM_TEXT_USER).render(
                text="\n".join([json.dumps(d, ensure_ascii=False) for d in txt_info["chunks"]])),
            chat_mdl,
            gen_conf={"temperature": 0.0, "top_p": 0.9}
        )
        txt_info["toc"] = ans if ans and not isinstance(ans, str) else []
    except Exception as e:
        logging.exception(e)


def split_chunks(chunks, max_length: int):
    """
    Pack chunks into batches according to max_length, returning [{"id": idx, "text": chunk_text}, ...].
    Do not split a single chunk, even if it exceeds max_length.
    """

    result = []
    batch, batch_tokens = [], 0

    for idx, chunk in enumerate(chunks):
        t = num_tokens_from_string(chunk)
        if batch_tokens + t > max_length:
            result.append(batch)
            batch, batch_tokens = [], 0
        batch.append({idx: chunk})
        batch_tokens += t
    if batch:
        result.append(batch)
    return result


async def run_toc_from_text(chunks, chat_mdl, callback=None):
    input_budget = int(chat_mdl.max_length * INPUT_UTILIZATION) - num_tokens_from_string(
        TOC_FROM_TEXT_USER + TOC_FROM_TEXT_SYSTEM
    )

    input_budget = 1024 if input_budget > 1024 else input_budget
    chunk_sections = split_chunks(chunks, input_budget)
    titles = []

    chunks_res = []
    tasks = []
    for i, chunk in enumerate(chunk_sections):
        if not chunk:
            continue
        chunks_res.append({"chunks": chunk})
        tasks.append(asyncio.create_task(gen_toc_from_text(chunks_res[-1], chat_mdl, callback)))
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        logging.error(f"Error generating TOC: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    for chunk in chunks_res:
        titles.extend(chunk.get("toc", []))

    # Filter out entries with title == -1
    prune = len(titles) > 512
    max_len = 12 if prune else 22
    filtered = []
    for x in titles:
        if not isinstance(x, dict) or not x.get("title") or x["title"] == "-1":
            continue
        if len(rag_tokenizer.tokenize(x["title"]).split(" ")) > max_len:
            continue
        if re.match(r"[0-9,.()/ -]+$", x["title"]):
            continue
        filtered.append(x)

    logging.info(f"\n\nFiltered TOC sections:\n{filtered}")
    if not filtered:
        return []

    # Generate initial level (level/title)
    raw_structure = [x.get("title", "") for x in filtered]

    # Assign hierarchy levels using LLM
    toc_with_levels = await assign_toc_levels(raw_structure, chat_mdl, {"temperature": 0.0, "top_p": 0.9})
    if not toc_with_levels:
        return []

    # Merge structure and content (by index)
    prune = len(toc_with_levels) > 512
    max_lvl = "0"
    sorted_list = sorted([t.get("level", "0") for t in toc_with_levels if isinstance(t, dict)])
    if sorted_list:
        max_lvl = sorted_list[-1]
    merged = []
    for _, (toc_item, src_item) in enumerate(zip(toc_with_levels, filtered)):
        if prune and toc_item.get("level", "0") >= max_lvl:
            continue
        merged.append({
            "level": toc_item.get("level", "0"),
            "title": toc_item.get("title", ""),
            "chunk_id": src_item.get("chunk_id", ""),
        })

    return merged


TOC_RELEVANCE_SYSTEM = load_prompt("toc_relevance_system")
TOC_RELEVANCE_USER = load_prompt("toc_relevance_user")
async def relevant_chunks_with_toc(query: str, toc: list[dict], chat_mdl, topn: int = 6):
    import numpy as np
    try:
        ans = await gen_json(
            PROMPT_JINJA_ENV.from_string(TOC_RELEVANCE_SYSTEM).render(),
            PROMPT_JINJA_ENV.from_string(TOC_RELEVANCE_USER).render(query=query, toc_json="[\n%s\n]\n" % "\n".join(
                [json.dumps({"level": d["level"], "title": d["title"]}, ensure_ascii=False) for d in toc])),
            chat_mdl,
            gen_conf={"temperature": 0.0, "top_p": 0.9}
        )
        id2score = {}
        for ti, sc in zip(toc, ans):
            if not isinstance(sc, dict) or sc.get("score", -1) < 1:
                continue
            for id in ti.get("ids", []):
                if id not in id2score:
                    id2score[id] = []
                id2score[id].append(sc["score"] / 5.)
        for id in id2score.keys():
            id2score[id] = np.mean(id2score[id])
        return [(id, sc) for id, sc in list(id2score.items()) if sc >= 0.3][:topn]
    except Exception as e:
        logging.exception(e)
    return []


META_DATA = load_prompt("meta_data")
async def gen_metadata(chat_mdl, schema: dict, content: str):
    template = PROMPT_JINJA_ENV.from_string(META_DATA)
    for k, desc in schema["properties"].items():
        if "enum" in desc and not desc.get("enum"):
            del desc["enum"]
        if desc.get("enum"):
            desc["description"] += "\n** Extracted values must strictly match the given list specified by `enum`. **"
    system_prompt = template.render(content=content, schema=schema)
    user_prompt = "Output: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


SUFFICIENCY_CHECK = load_prompt("sufficiency_check")
async def sufficiency_check(chat_mdl, question: str, ret_content: str):
    try:
        return await gen_json(
            PROMPT_JINJA_ENV.from_string(SUFFICIENCY_CHECK).render(question=question, retrieved_docs=ret_content),
            "Output:\n",
            chat_mdl
        )
    except Exception as e:
        logging.exception(e)
    return {}


MULTI_QUERIES_GEN = load_prompt("multi_queries_gen")
async def multi_queries_gen(chat_mdl, question: str, query:str, missing_infos:list[str], ret_content: str):
    try:
        return await gen_json(
            PROMPT_JINJA_ENV.from_string(MULTI_QUERIES_GEN).render(
                original_question=question,
                original_query=query,
                missing_info="\n - ".join(missing_infos),
                retrieved_docs=ret_content
            ),
            "Output:\n",
            chat_mdl
        )
    except Exception as e:
        logging.exception(e)
    return {}
