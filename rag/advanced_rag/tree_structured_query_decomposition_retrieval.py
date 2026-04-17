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
import logging
from functools import partial
from api.db.services.llm_service import LLMBundle
from rag.prompts import kb_prompt
from rag.prompts.generator import sufficiency_check, multi_queries_gen
from rag.utils.tavily_conn import Tavily
from timeit import default_timer as timer


class TreeStructuredQueryDecompositionRetrieval:
    def __init__(self,
                 chat_mdl: LLMBundle,
                 prompt_config: dict,
                 kb_retrieve: partial = None,
                 kg_retrieve: partial = None
                 ):
        self.chat_mdl = chat_mdl
        self.prompt_config = prompt_config
        self._kb_retrieve = kb_retrieve
        self._kg_retrieve = kg_retrieve
        self._lock = asyncio.Lock()

    async def _retrieve_information(self, search_query):
        # 从不同信息源检索与当前 search_query 相关的内容。
        """Retrieve information from different sources"""
        # 1. 先做知识库检索。
        # kbinfos 约定为统一结构：{"chunks": [...], "doc_aggs": [...]}。
        kbinfos = []
        try:
            # 如果配置了知识库检索函数，就用当前 query 去查；
            # 否则兜底为空结果，保证后续结构一致。
            kbinfos = await self._kb_retrieve(question=search_query) if self._kb_retrieve else {"chunks": [], "doc_aggs": []}
        except Exception as e:
            # 知识库检索失败时记录错误，但不中断整个 Deep Research 流程。
            logging.error(f"Knowledge base retrieval error: {e}")

        # 2. 再做 Web 检索（前提是配置了 Tavily API Key）。
        try:
            # 只有 prompt_config 里配置了 tavily_api_key 才启用联网搜索。
            if self.prompt_config.get("tavily_api_key"):
                # 初始化 Tavily 客户端。
                tav = Tavily(self.prompt_config["tavily_api_key"])
                # 用当前 query 检索网页片段和文档聚合结果。
                tav_res = tav.retrieve_chunks(search_query)
                # 把网页检索出来的 chunks 合并进统一结果中。
                kbinfos["chunks"].extend(tav_res["chunks"])
                # 把网页来源的文档聚合信息也一并合并进去。
                kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
        except Exception as e:
            # Web 检索失败同样只记日志，不阻断整个研究流程。
            logging.error(f"Web retrieval error: {e}")

        # 3. 最后做知识图谱检索（前提是显式开启 use_kg 且注入了 kg_retrieve）。
        try:
            # 只有开启 use_kg 并且提供了知识图谱检索函数，才执行 KG 检索。
            if self.prompt_config.get("use_kg") and self._kg_retrieve:
                # 用当前 query 检索一条 KG 增强结果。
                ck = await self._kg_retrieve(question=search_query)
                # 只有 KG 返回的内容非空时，才把它加入结果集。
                if ck["content_with_weight"]:
                    # 插到 chunks 最前面，表示 KG 结果在后续上下文中优先级更高。
                    kbinfos["chunks"].insert(0, ck)
        except Exception as e:
            # 知识图谱检索失败也只记录日志，避免单一路径失败影响整体检索。
            logging.error(f"Knowledge graph retrieval error: {e}")

        # 返回已经合并好的多源检索结果。
        return kbinfos

    async def _async_update_chunk_info(self, chunk_info, kbinfos):
        async with self._lock:
            """Update chunk information for citations"""
            if not chunk_info["chunks"]:
                # If this is the first retrieval, use the retrieval results directly
                for k in chunk_info.keys():
                    chunk_info[k] = kbinfos[k]
            else:
                # Merge newly retrieved information, avoiding duplicates
                cids = [c["chunk_id"] for c in chunk_info["chunks"]]
                for c in kbinfos["chunks"]:
                    if c["chunk_id"] not in cids:
                        chunk_info["chunks"].append(c)

                dids = [d["doc_id"] for d in chunk_info["doc_aggs"]]
                for d in kbinfos["doc_aggs"]:
                    if d["doc_id"] not in dids:
                        chunk_info["doc_aggs"].append(d)

    async def research(self, chunk_info, question, query, depth=3, callback=None):
        if callback:
            await callback("<START_DEEP_RESEARCH>")
        await self._research(chunk_info, question, query, depth, callback)
        if callback:
            await callback("<END_DEEP_RESEARCH>")

    async def _research(self, chunk_info, question, query, depth=3, callback=None):
        # 递归深度耗尽时停止继续向下拆问题，直接返回空结果。
        if depth == 0:
            #if callback:
            #    await callback("Reach the max search depth.")
            return ""
        # 如果提供了回调，就先告诉外层当前正在用哪个 query 做检索。
        if callback:
            await callback(f"Searching by `{query}`...")
        # 记录本轮检索开始时间，用于后面输出耗时。
        st = timer()
        # 按当前 query 执行多源检索，可能包含知识库、Web 和 KG 三路结果。
        ret = await self._retrieve_information(query)
        # 通过回调输出本轮检索命中的 chunk 数和耗时。
        if callback:
            await callback("Retrieval %d results in %.1fms"%(len(ret["chunks"]), (timer()-st)*1000))
        # 把本轮检索结果合并进共享的 chunk_info，供后续统一引用和答案生成使用。
        await self._async_update_chunk_info(chunk_info, ret)
        # 把检索结果格式化成适合 LLM 判断“信息是否充分”的 prompt 片段。
        # 这里只使用模型最大上下文长度的一半，避免检查阶段占满上下文窗口。
        ret = kb_prompt(ret, self.chat_mdl.max_length*0.5)

        # 通知外层，接下来要做充分性判断。
        if callback:
            await callback("Checking the sufficiency for retrieved information.")
        # 让 LLM 判断当前检索到的信息是否足以回答原始问题。
        suff = await sufficiency_check(self.chat_mdl, question, ret)
        # 如果已经足够回答问题，就结束当前分支递归。
        if suff["is_sufficient"]:
            if callback:
                # 通过回调明确告诉外层：当前信息已经足够。
                await callback(f"Yes, the retrieved information is sufficient for '{question}'.")
            return ret

        #if callback:
        #    await callback("The retrieved information is not sufficient. Planing next steps...")
        # 如果信息不够，让 LLM 根据“原问题 + 当前查询 + 缺失信息 + 已检索内容”
        # 生成下一批要继续搜索的子问题和子查询。
        succ_question_info = await multi_queries_gen(self.chat_mdl, question, query, suff["missing_information"], ret)
        # 把下一步计划通过回调抛给外层，便于前端展示 Deep Research 的思考路径。
        if callback:
            await callback("Next step is to search for the following questions:</br> - " + "</br> - ".join(step["question"] for step in succ_question_info["questions"]))
        # 准备并发执行下一层递归搜索任务。
        steps = []
        # 为每个子问题创建一个异步任务，并把深度减一。
        for step in succ_question_info["questions"]:
            steps.append(asyncio.create_task(self._research(chunk_info, step["question"], step["query"], depth-1, callback)))
        # 并发等待所有子任务完成；即使某个分支异常，也先收集结果而不是立刻中断整个研究流程。
        results = await asyncio.gather(*steps, return_exceptions=True)
        # 把所有子分支返回结果拼接成一个字符串作为当前层的返回值。
        return "\n".join([str(r) for r in results])
