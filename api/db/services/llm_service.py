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
import inspect
import logging
import queue
import re
import threading
from functools import partial
from typing import Generator

from api.db.db_models import LLM
from api.db.services.common_service import CommonService
from api.db.services.tenant_llm_service import LLM4Tenant, TenantLLMService
from common.constants import LLMType
from common.token_utils import num_tokens_from_string


class LLMService(CommonService):
    model = LLM


def get_init_tenant_llm(user_id):
    from common import settings

    tenant_llm = []

    model_configs = {
        LLMType.CHAT: settings.CHAT_CFG,
        LLMType.EMBEDDING: settings.EMBEDDING_CFG,
        LLMType.SPEECH2TEXT: settings.ASR_CFG,
        LLMType.IMAGE2TEXT: settings.IMAGE2TEXT_CFG,
        LLMType.RERANK: settings.RERANK_CFG,
    }

    seen = set()
    factory_configs = []
    for factory_config in [
        settings.CHAT_CFG,
        settings.EMBEDDING_CFG,
        settings.ASR_CFG,
        settings.IMAGE2TEXT_CFG,
        settings.RERANK_CFG,
    ]:
        factory_name = factory_config["factory"]
        if factory_name not in seen:
            seen.add(factory_name)
            factory_configs.append(factory_config)

    for factory_config in factory_configs:
        for llm in LLMService.query(fid=factory_config["factory"]):
            tenant_llm.append(
                {
                    "tenant_id": user_id,
                    "llm_factory": factory_config["factory"],
                    "llm_name": llm.llm_name,
                    "model_type": llm.model_type,
                    "api_key": model_configs.get(llm.model_type, {}).get("api_key", factory_config["api_key"]),
                    "api_base": model_configs.get(llm.model_type, {}).get("base_url", factory_config["base_url"]),
                    "max_tokens": llm.max_tokens if llm.max_tokens else 8192,
                }
            )

    unique = {}
    for item in tenant_llm:
        key = (item["tenant_id"], item["llm_factory"], item["llm_name"])
        if key not in unique:
            unique[key] = item
    return list(unique.values())


class LLMBundle(LLM4Tenant):
    def __init__(self, tenant_id: str, model_config: dict, lang="Chinese", **kwargs):
        super().__init__(tenant_id, model_config, lang, **kwargs)

    def bind_tools(self, toolcall_session, tools):
        if not self.is_tools:
            logging.warning(f"Model {self.model_config['llm_name']} does not support tool call, but you have assigned one or more tools to it!")
            return
        self.mdl.bind_tools(toolcall_session, tools)

    def encode(self, texts: list):
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="encode", model=self.model_config["llm_name"], input={"texts": texts})

        safe_texts = []
        for text in texts:
            token_size = num_tokens_from_string(text)
            if token_size > self.max_length:
                target_len = int(self.max_length * 0.95)
                safe_texts.append(text[:target_len])
            else:
                safe_texts.append(text)

        embeddings, used_tokens = self.mdl.encode(safe_texts)
        if self.model_config["llm_factory"] == "Builtin":
            logging.info("LLMBundle.encode_queries query: {}, emd len: {}, used_tokens: {}. Builtin model don't need to update token usage".format(texts, len(embeddings), used_tokens))
        elif not TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens):
            logging.error("LLMBundle.encode can't update token usage for <tenant redacted>/EMBEDDING used_tokens: {}".format(used_tokens))

        if self.langfuse:
            generation.update(usage_details={"total_tokens": used_tokens})
            generation.end()

        return embeddings, used_tokens

    def encode_queries(self, query: str):
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="encode_queries", model=self.model_config["llm_name"], input={"query": query})

        emd, used_tokens = self.mdl.encode_queries(query)
        if self.model_config["llm_factory"] == "Builtin":
            logging.info("LLMBundle.encode_queries query: {}, emd len: {}, used_tokens: {}. Builtin model don't need to update token usage".format(query, len(emd), used_tokens))
        elif not TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens):
            logging.error("LLMBundle.encode_queries can't update token usage for <tenant redacted>/EMBEDDING used_tokens: {}".format(used_tokens))

        if self.langfuse:
            generation.update(usage_details={"total_tokens": used_tokens})
            generation.end()

        return emd, used_tokens

    def similarity(self, query: str, texts: list):
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="similarity", model=self.model_config["llm_name"], input={"query": query, "texts": texts})

        sim, used_tokens = self.mdl.similarity(query, texts)
        if not TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens):
            logging.error("LLMBundle.similarity can't update token usage for {}/RERANK used_tokens: {}".format(self.tenant_id, used_tokens))

        if self.langfuse:
            generation.update(usage_details={"total_tokens": used_tokens})
            generation.end()

        return sim, used_tokens

    def describe(self, image, max_tokens=300):
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="describe", metadata={"model": self.model_config["llm_name"]})

        txt, used_tokens = self.mdl.describe(image)
        if not TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens):
            logging.error("LLMBundle.describe can't update token usage for {}/IMAGE2TEXT used_tokens: {}".format(self.tenant_id, used_tokens))

        if self.langfuse:
            generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
            generation.end()

        return txt

    def describe_with_prompt(self, image, prompt):
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="describe_with_prompt", metadata={"model": self.model_config["llm_name"], "prompt": prompt})

        txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)
        if not TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens):
            logging.error("LLMBundle.describe can't update token usage for {}/IMAGE2TEXT used_tokens: {}".format(self.tenant_id, used_tokens))

        if self.langfuse:
            generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
            generation.end()

        return txt

    def transcription(self, audio):
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="transcription", metadata={"model": self.model_config["llm_name"]})

        txt, used_tokens = self.mdl.transcription(audio)
        if not TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens):
            logging.error("LLMBundle.transcription can't update token usage for {}/SEQUENCE2TXT used_tokens: {}".format(self.tenant_id, used_tokens))

        if self.langfuse:
            generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
            generation.end()

        return txt

    def stream_transcription(self, audio):
        mdl = self.mdl
        supports_stream = hasattr(mdl, "stream_transcription") and callable(getattr(mdl, "stream_transcription"))
        if supports_stream:
            if self.langfuse:
                generation = self.langfuse.start_generation(
                    trace_context=self.trace_context,
                    name="stream_transcription",
                    metadata={"model": self.model_config["llm_name"]},
                )
            final_text = ""
            used_tokens = 0

            try:
                for evt in mdl.stream_transcription(audio):
                    if evt.get("event") == "final":
                        final_text = evt.get("text", "")

                    yield evt

            except Exception as e:
                err = {"event": "error", "text": str(e)}
                yield err
                final_text = final_text or ""
            finally:
                if final_text:
                    used_tokens = num_tokens_from_string(final_text)
                    TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens)

                if self.langfuse:
                    generation.update(
                        output={"output": final_text},
                        usage_details={"total_tokens": used_tokens},
                    )
                    generation.end()

            return

        if self.langfuse:
            generation = self.langfuse.start_generation(
                trace_context=self.trace_context,
                name="stream_transcription",
                metadata={"model": self.model_config["llm_name"]},
            )

        full_text, used_tokens = mdl.transcription(audio)
        if not TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens):
            logging.error(f"LLMBundle.stream_transcription can't update token usage for {self.tenant_id}/SEQUENCE2TXT used_tokens: {used_tokens}")

        if self.langfuse:
            generation.update(
                output={"output": full_text},
                usage_details={"total_tokens": used_tokens},
            )
            generation.end()

        yield {
            "event": "final",
            "text": full_text,
            "streaming": False,
        }

    def tts(self, text: str) -> Generator[bytes, None, None]:
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="tts", input={"text": text})

        for chunk in self.mdl.tts(text):
            if isinstance(chunk, int):
                if not TenantLLMService.increase_usage_by_id(self.model_config["id"], chunk):
                    logging.error("LLMBundle.tts can't update token usage for {}/TTS".format(self.tenant_id))
                return
            yield chunk

        if self.langfuse:
            generation.end()

    def _remove_reasoning_content(self, txt: str) -> str:
        if txt is None:
            return None
        first_think_start = txt.find("<think>")
        if first_think_start == -1:
            return txt

        last_think_end = txt.rfind("</think>")
        if last_think_end == -1:
            return txt

        if last_think_end < first_think_start:
            return txt

        return txt[last_think_end + len("</think>") :]

    @staticmethod
    def _clean_param(chat_partial, **kwargs):
        func = chat_partial.func
        sig = inspect.signature(func)
        support_var_args = False
        allowed_params = set()

        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                support_var_args = True
            elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                allowed_params.add(param.name)
        if support_var_args:
            return kwargs
        else:
            return {k: v for k, v in kwargs.items() if k in allowed_params}

    def _run_coroutine_sync(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result_queue: queue.Queue = queue.Queue()

        def runner():
            try:
                result_queue.put((True, asyncio.run(coro)))
            except Exception as e:
                result_queue.put((False, e))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        success, value = result_queue.get_nowait()
        if success:
            return value
        raise value

    def _sync_from_async_stream(self, async_gen_fn, *args, **kwargs):
        result_queue: queue.Queue = queue.Queue()

        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def consume():
                try:
                    async for item in async_gen_fn(*args, **kwargs):
                        result_queue.put(item)
                except Exception as e:
                    result_queue.put(e)
                finally:
                    result_queue.put(StopIteration)

            loop.run_until_complete(consume())
            loop.close()

        threading.Thread(target=runner, daemon=True).start()

        while True:
            item = result_queue.get()
            if item is StopIteration:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _bridge_sync_stream(self, gen):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                for item in gen:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, StopAsyncIteration)

        threading.Thread(target=worker, daemon=True).start()
        return queue

    async def async_chat(self, system: str, history: list, gen_conf: dict = {}, **kwargs):
        # 如果当前 bundle 开启了工具调用，并且底层模型也支持带 tools 的异步聊天接口，则优先走带工具版本。
        # 这样做是为了在支持 tool call 的模型上保留完整能力，而不是退化成普通聊天。
        if self.is_tools and getattr(self.mdl, "is_tools", False) and hasattr(self.mdl, "async_chat_with_tools"):
            base_fn = self.mdl.async_chat_with_tools
        # 否则回退到普通异步聊天接口。
        elif hasattr(self.mdl, "async_chat"):
            base_fn = self.mdl.async_chat
        else:
            # 两种接口都没有时直接报错。
            # 这样做是为了尽早暴露模型实现不完整的问题，避免调用方在更后面的位置才看到模糊错误。
            raise RuntimeError(f"Model {self.mdl} does not implement async_chat or async_chat_with_tools")

        # Langfuse generation 对象，默认不开启。
        generation = None
        # 如果配置了 Langfuse，就为这次调用创建一条 tracing 记录。
        # 这样做是为了把 system prompt、历史消息、输出和 token 用量统一纳入观测。
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="chat", model=self.model_config["llm_name"], input={"system": system, "history": history})

        # 先把固定参数预绑定到目标函数上。
        # 这样后面只需要处理当前模型真正支持的可选参数。
        chat_partial = partial(base_fn, system, history, gen_conf)
        # 按底层函数签名过滤 kwargs。
        # 这样做是为了避免把不被具体模型实现支持的参数直接传进去导致调用失败。
        use_kwargs = self._clean_param(chat_partial, **kwargs)

        try:
            # 执行一次非流式异步聊天，返回完整文本和 token 用量。
            txt, used_tokens = await chat_partial(**use_kwargs)
        except Exception as e:
            # 出错时把异常写入 Langfuse，并结束 tracing，再继续向上抛出。
            # 这样做是为了让观测系统能保留失败信息，同时不吞掉真实异常。
            if generation:
                generation.update(output={"error": str(e)})
                generation.end()
            raise

        # 去掉内部 reasoning 内容，只保留对外展示的回答部分。
        # 这样做是为了避免把模型内部思考过程直接暴露给上层调用方。
        txt = self._remove_reasoning_content(txt)
        # 如果未开启详细工具调用显示，则去掉工具调用协议文本。
        # 这样做是为了让最终回答更干净，不把内部 `<tool_call>` 结构直接暴露给用户界面。
        if not self.verbose_tool_use:
            txt = re.sub(r"<tool_call>.*?</tool_call>", "", txt, flags=re.DOTALL)

        # 如果拿到了 token 用量，就回写模型使用量统计。
        if used_tokens and not TenantLLMService.increase_usage_by_id(self.model_config["id"], used_tokens):
            logging.error("LLMBundle.async_chat can't update token usage for {}/CHAT llm_name: {}, used_tokens: {}".format(self.tenant_id, self.model_config["llm_name"], used_tokens))

        # 把最终输出和 token 用量写入 Langfuse，并结束这次 generation。
        if generation:
            generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
            generation.end()

        # 返回清洗后的最终文本结果。
        return txt

    async def async_chat_streamly(self, system: str, history: list, gen_conf: dict = {}, **kwargs):
        total_tokens = 0
        ans = ""
        _bundle_is_tools = self.is_tools
        _mdl_is_tools = getattr(self.mdl, "is_tools", False)
        _has_with_tools = hasattr(self.mdl, "async_chat_streamly_with_tools")
        if _bundle_is_tools and _mdl_is_tools and _has_with_tools:
            stream_fn = getattr(self.mdl, "async_chat_streamly_with_tools", None)
        elif hasattr(self.mdl, "async_chat_streamly"):
            stream_fn = getattr(self.mdl, "async_chat_streamly", None)
        else:
            raise RuntimeError(f"Model {self.mdl} does not implement async_chat or async_chat_with_tools")

        generation = None
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="chat_streamly", model=self.model_config["llm_name"], input={"system": system, "history": history})

        if stream_fn:
            chat_partial = partial(stream_fn, system, history, gen_conf)
            use_kwargs = self._clean_param(chat_partial, **kwargs)
            try:
                async for txt in chat_partial(**use_kwargs):
                    if isinstance(txt, int):
                        total_tokens = txt
                        break

                    if txt.endswith("</think>") and ans.endswith("</think>"):
                        ans = ans[: -len("</think>")]

                    if not self.verbose_tool_use:
                        txt = re.sub(r"<tool_call>.*?</tool_call>", "", txt, flags=re.DOTALL)

                    ans += txt
                    yield ans
            except Exception as e:
                if generation:
                    generation.update(output={"error": str(e)})
                    generation.end()
                raise
            if total_tokens and not TenantLLMService.increase_usage_by_id(self.model_config["id"], total_tokens):
                logging.error("LLMBundle.async_chat_streamly can't update token usage for {}/CHAT llm_name: {}, used_tokens: {}".format(self.tenant_id, self.model_config["llm_name"], total_tokens))
            if generation:
                generation.update(output={"output": ans}, usage_details={"total_tokens": total_tokens})
                generation.end()
            return

    async def async_chat_streamly_delta(self, system: str, history: list, gen_conf: dict = {}, **kwargs):
        # 记录本次流式生成最终消耗的 token 数。
        # 底层有些模型会在流结束时额外返回一个整数作为 token 用量，因此这里先准备变量承接。
        total_tokens = 0
        # `ans` 用来累计完整回答文本。
        # 虽然这个方法对外只返回 delta 分片，但内部仍需要保留完整结果，以便做 tracing 和最终统计。
        ans = ""
        # 如果当前 bundle 启用了工具调用，且底层模型也支持工具调用流式接口，则优先走带 tools 的流式方法。
        # 这样做是为了在支持 tool call 的模型上保留完整能力，而不是退化成普通文本流。
        if self.is_tools and getattr(self.mdl, "is_tools", False) and hasattr(self.mdl, "async_chat_streamly_with_tools"):
            stream_fn = getattr(self.mdl, "async_chat_streamly_with_tools", None)
        # 否则回退到普通流式聊天接口。
        elif hasattr(self.mdl, "async_chat_streamly"):
            stream_fn = getattr(self.mdl, "async_chat_streamly", None)
        else:
            # 两种流式接口都不存在时直接报错。
            # 这样做是为了尽早暴露模型能力缺失，而不是在运行到中途才出现更隐晦的问题。
            raise RuntimeError(f"Model {self.mdl} does not implement async_chat or async_chat_with_tools")

        # Langfuse generation 对象，默认不启用。
        generation = None
        # 如果配置了 Langfuse，就为本次流式对话开启一条观测记录。
        # 这样做是为了把输入、输出和 token 用量统一打到 tracing 系统里。
        if self.langfuse:
            generation = self.langfuse.start_generation(trace_context=self.trace_context, name="chat_streamly", model=self.model_config["llm_name"], input={"system": system, "history": history})

        # 只有真正拿到流式函数后，才开始构造并执行流式调用。
        if stream_fn:
            # 先把固定参数 `system/history/gen_conf` 预绑定进去，后面只需要传清洗后的可选参数。
            chat_partial = partial(stream_fn, system, history, gen_conf)
            # 按底层函数签名过滤 kwargs。
            # 这样做是为了避免把不支持的参数传给具体模型实现，引发类型错误。
            use_kwargs = self._clean_param(chat_partial, **kwargs)
            try:
                # 逐片消费底层模型输出。
                async for txt in chat_partial(**use_kwargs):
                    # 某些实现会在流尾返回 int，表示总 token 数，而不是文本分片。
                    # 这样做是为了在不额外增加返回通道的情况下，把统计信息也顺带带出来。
                    if isinstance(txt, int):
                        total_tokens = txt
                        break

                    # 避免重复累积 `</think>` 结束标记。
                    # 某些模型在 think 边界上可能重复发同一个结束片段，这里做一次去重保护。
                    if txt.endswith("</think>") and ans.endswith("</think>"):
                        ans = ans[: -len("</think>")]

                    # 如果关闭了详细工具调用显示，则把 `<tool_call>...</tool_call>` 片段从输出里去掉。
                    # 这样做是为了避免前端直接看到内部工具调用协议文本。
                    if not self.verbose_tool_use:
                        txt = re.sub(r"<tool_call>.*?</tool_call>", "", txt, flags=re.DOTALL)

                    # 内部累计完整答案，供最终 tracing 和结果归档使用。
                    ans += txt
                    # 对外只返回本次新增加的 delta，而不是累计全文。
                    # 这是它与 `async_chat_streamly` 的关键区别：前者适合增量渲染，后者适合直接替换显示。
                    yield txt
            except Exception as e:
                # 出错时把异常写入 Langfuse，再把异常继续抛出给上层处理。
                if generation:
                    generation.update(output={"error": str(e)})
                    generation.end()
                raise
            # 流结束后，如果拿到了 token 用量，就回写租户模型使用量统计。
            if total_tokens and not TenantLLMService.increase_usage_by_id(self.model_config["id"], total_tokens):
                logging.error("LLMBundle.async_chat_streamly can't update token usage for {}/CHAT llm_name: {}, used_tokens: {}".format(self.tenant_id, self.model_config["llm_name"], total_tokens))
            # 最后把完整输出和 token 用量写入 Langfuse，并结束这次 generation。
            if generation:
                generation.update(output={"output": ans}, usage_details={"total_tokens": total_tokens})
                generation.end()
            # 显式结束异步生成器。
            return
