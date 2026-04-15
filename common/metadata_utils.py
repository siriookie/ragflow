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
import ast
import logging
from typing import Any, Callable, Dict

import json_repair

def convert_conditions(metadata_condition):
    if metadata_condition is None:
        metadata_condition = {}
    op_mapping = {
        "is": "=",
        "not is": "≠",
        ">=": "≥",
        "<=": "≤",
        "!=": "≠"
    }
    return [
        {
            "op": op_mapping.get(cond["comparison_operator"], cond["comparison_operator"]),
            "key": cond["name"],
            "value": cond["value"]
        }
        for cond in metadata_condition.get("conditions", [])
    ]


def meta_filter(metas: dict, filters: list[dict], logic: str = "and"):
    doc_ids = set([])

    def filter_out(v2docs, operator, value):
        ids = []
        for input, docids in v2docs.items():

            if operator in ["=", "≠", ">", "<", "≥", "≤"]:
                # Check if input is in YYYY-MM-DD date format
                input_str = str(input).strip()
                value_str = str(value).strip()

                # Strict date format detection: YYYY-MM-DD (must be 10 chars with correct format)
                is_input_date = (
                    len(input_str) == 10 and
                    input_str[4] == '-' and
                    input_str[7] == '-' and
                    input_str[:4].isdigit() and
                    input_str[5:7].isdigit() and
                    input_str[8:10].isdigit()
                )

                is_value_date = (
                    len(value_str) == 10 and
                    value_str[4] == '-' and
                    value_str[7] == '-' and
                    value_str[:4].isdigit() and
                    value_str[5:7].isdigit() and
                    value_str[8:10].isdigit()
                )

                if is_value_date:
                    # Query value is in date format
                    if is_input_date:
                        # Data is also in date format: perform date comparison
                        input = input_str
                        value = value_str
                    else:
                        # Data is not in date format: skip this record (no match)
                        continue
                else:
                    # Query value is not in date format: use original logic
                    try:
                        if isinstance(input, list):
                            input = input[0]
                        input = ast.literal_eval(input)
                        value = ast.literal_eval(value)
                    except Exception:
                        pass

                    # Convert strings to lowercase
                    if isinstance(input, str):
                        input = input.lower()
                    if isinstance(value, str):
                        value = value.lower()
            else:
                # Non-comparison operators: maintain original logic
                if isinstance(input, str):
                    input = input.lower()
                if isinstance(value, str):
                    value = value.lower()

            matched = False
            try:
                if operator == "contains":
                    matched = str(input).find(value) >= 0 if not isinstance(input, list) else any(str(i).find(value) >= 0 for i in input)
                elif operator == "not contains":
                    matched = str(input).find(value) == -1 if not isinstance(input, list) else all(str(i).find(value) == -1 for i in input)
                elif operator == "in":
                    matched = input in value if not isinstance(input, list) else all(i in value for i in input)
                elif operator == "not in":
                    matched = input not in value if not isinstance(input, list) else all(i not in value for i in input)
                elif operator == "start with":
                    matched = str(input).lower().startswith(str(value).lower()) if not isinstance(input, list) else "".join([str(i).lower() for i in input]).startswith(str(value).lower())
                elif operator == "end with":
                    matched = str(input).lower().endswith(str(value).lower()) if not isinstance(input, list) else "".join([str(i).lower() for i in input]).endswith(str(value).lower())
                elif operator == "empty":
                    matched = not input
                elif operator == "not empty":
                    matched = bool(input)
                elif operator == "=":
                    matched = input == value
                elif operator == "≠":
                    matched = input != value
                elif operator == ">":
                    matched = input > value
                elif operator == "<":
                    matched = input < value
                elif operator == "≥":
                    matched = input >= value
                elif operator == "≤":
                    matched = input <= value
            except Exception:
                pass

            if matched:
                ids.extend(docids)
        return ids

    for f in filters:
        k = f["key"]
        if k not in metas:
            # Key not found in metas: treat as no match
            ids = []
        else:
            v2docs = metas[k]
            ids = filter_out(v2docs, f["op"], f["value"])

        if not doc_ids:
            doc_ids = set(ids)
        else:
            if logic == "and":
                doc_ids = doc_ids & set(ids)
                if not doc_ids:
                    return []
            else:
                doc_ids = doc_ids | set(ids)
    return list(doc_ids)


async def apply_meta_data_filter(
    meta_data_filter: dict | None,
    metas: dict,
    question: str,
    chat_mdl: Any = None,
    base_doc_ids: list[str] | None = None,
    manual_value_resolver: Callable[[dict], dict] | None = None,
) -> list[str] | None:
    """
    Apply metadata filtering rules and return the filtered doc_ids.

    meta_data_filter supports three modes:
    - auto: generate filter conditions via LLM (gen_meta_filter)
    - semi_auto: generate conditions using selected metadata keys only
    - manual: directly filter based on provided conditions

    Returns:
        list of doc_ids, ["-999"] when manual filters yield no result, or None
        when auto/semi_auto filters return empty.
    """
    # 延迟导入 `gen_meta_filter`，避免模块级循环依赖。
    # 这样做是因为 `generator` 和 metadata 工具之间存在互相调用关系。
    from rag.prompts.generator import gen_meta_filter # move from the top of the file to avoid circular import

    # 以调用方传入的 `base_doc_ids` 作为初始候选集合。
    # 这样做是为了支持“先有一层文档范围限制，再叠加元数据过滤”的场景。
    doc_ids = list(base_doc_ids) if base_doc_ids else []

    # 没有元数据过滤配置时，直接返回当前候选集合。
    if not meta_data_filter:
        return doc_ids

    # 读取过滤模式。
    method = meta_data_filter.get("method")

    if method == "auto":
        # `auto` 模式下，让 LLM 基于问题和全部可用元数据自动生成过滤条件。
        filters: dict = await gen_meta_filter(chat_mdl, metas, question)
        # 把自动生成的条件交给 `meta_filter` 执行，并把结果并入当前 doc_ids。
        doc_ids.extend(meta_filter(metas, filters["conditions"], filters.get("logic", "and")))
        # auto 模式如果没有筛出结果，返回 None。
        # 这样做通常表示“自动推断的过滤条件没有命中”，上层可以据此走更宽松的回退逻辑。
        if not doc_ids:
            return None
    elif method == "semi_auto":
        # `semi_auto` 模式下，只允许模型在指定元数据键范围内推断过滤条件。
        # 同时还可以为某些键显式限定允许使用的操作符。
        selected_keys = []
        constraints = {}
        for item in meta_data_filter.get("semi_auto", []):
            # 纯字符串表示“只开放这个元数据键给模型使用”。
            if isinstance(item, str):
                selected_keys.append(item)
            elif isinstance(item, dict):
                # 字典形式除了指定键，还可以限定 op。
                key = item.get("key")
                op = item.get("op")
                selected_keys.append(key)
                if op:
                    constraints[key] = op

        # 只有真的选出了键，才继续做半自动过滤。
        if selected_keys:
            # 只截取被允许参与推断的那部分元数据。
            filtered_metas = {key: metas[key] for key in selected_keys if key in metas}
            if filtered_metas:
                # 让 LLM 在受限元数据集合上生成条件。
                filters: dict = await gen_meta_filter(chat_mdl, filtered_metas, question, constraints=constraints)
                # 注意这里执行过滤时仍然传原始 `metas`，因为条件键虽然受限，但最终匹配要基于完整元数据映射执行。
                doc_ids.extend(meta_filter(metas, filters["conditions"], filters.get("logic", "and")))
                # semi_auto 没命中时也返回 None，含义与 auto 一致。
                if not doc_ids:
                    return None
    elif method == "manual":
        # `manual` 模式完全不让模型推断，直接使用前端/调用方给定的过滤条件。
        filters = meta_data_filter.get("manual", [])
        if manual_value_resolver:
            # 如果提供了自定义 resolver，就先把手动条件做一次值解析。
            # 这样做是为了支持把动态占位值转换成最终过滤值。
            filters = [manual_value_resolver(flt) for flt in filters]
        # 手动条件直接交给 `meta_filter` 执行。
        doc_ids.extend(meta_filter(metas, filters, meta_data_filter.get("logic", "and")))
        # manual 模式下，如果明确给了过滤条件但没有结果，返回 `["-999"]` 而不是 None。
        # 这样做通常是为了向上层明确表达“手动筛选后结果为空”，避免被误判成“没做过滤”或触发自动回退。
        if filters and not doc_ids:
            doc_ids = ["-999"]

    # 返回最终过滤后的 doc_ids。
    return doc_ids


def dedupe_list(values: list) -> list:
    seen = set()
    deduped = []
    for item in values:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def update_metadata_to(metadata, meta):
    if not meta:
        return metadata
    if isinstance(meta, str):
        try:
            meta = json_repair.loads(meta)
        except Exception:
            logging.error("Meta data format error.")
            return metadata
    if not isinstance(meta, dict):
        return metadata

    for k, v in meta.items():
        if isinstance(v, list):
            v = [vv for vv in v if isinstance(vv, str)]
            if not v:
                continue
            v = dedupe_list(v)
        if not isinstance(v, list) and not isinstance(v, str):
            continue
        if k not in metadata:
            metadata[k] = v
            continue
        if isinstance(metadata[k], list):
            if isinstance(v, list):
                metadata[k].extend(v)
            else:
                metadata[k].append(v)
            metadata[k] = dedupe_list(metadata[k])
        else:
            metadata[k] = v

    return metadata


def metadata_schema(metadata: dict|list|None) -> Dict[str, Any]:
    if not metadata:
        return {}
    properties = {}

    for item in metadata:
        key = item.get("key")
        if not key:
            continue

        prop_schema = {
            "description": item.get("description", "")
        }
        if "enum" in item and item["enum"]:
            prop_schema["enum"] = item["enum"]
            prop_schema["type"] = "string"

        properties[key] = prop_schema

    json_schema = {
        "type": "object",
        "properties": properties,
    }

    json_schema["additionalProperties"] = False
    return json_schema


def _is_json_schema(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if "$schema" in obj:
        return True
    return obj.get("type") == "object" and isinstance(obj.get("properties"), dict)


def _is_metadata_list(obj: list) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    for item in obj:
        if not isinstance(item, dict):
            return False
        key = item.get("key")
        if not isinstance(key, str) or not key:
            return False
        if "enum" in item and not isinstance(item["enum"], list):
            return False
        if "description" in item and not isinstance(item["description"], str):
            return False
        if "descriptions" in item and not isinstance(item["descriptions"], str):
            return False
    return True


def turn2jsonschema(obj: dict | list) -> Dict[str, Any]:
    if isinstance(obj, dict) and _is_json_schema(obj):
        return obj
    if isinstance(obj, list) and _is_metadata_list(obj):
        normalized = []
        for item in obj:
            description = item.get("description", item.get("descriptions", ""))
            normalized_item = {
                "key": item.get("key"),
                "description": description,
            }
            if "enum" in item:
                normalized_item["enum"] = item["enum"]
            normalized.append(normalized_item)
        return metadata_schema(normalized)
    return {}
