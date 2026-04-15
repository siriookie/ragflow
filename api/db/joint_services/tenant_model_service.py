#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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
import os
import enum
from common import settings
from common.constants import LLMType
from api.db.services.llm_service import LLMService
from api.db.services.tenant_llm_service import TenantLLMService, TenantService


def get_model_config_by_id(tenant_model_id: int) -> dict:
    found, model_config = TenantLLMService.get_by_id(tenant_model_id)
    if not found:
        raise LookupError(f"Tenant Model with id {tenant_model_id} not found")
    config_dict = model_config.to_dict()
    llm = LLMService.query(llm_name=config_dict["llm_name"])
    if llm:
        config_dict["is_tools"] = llm[0].is_tools
    return config_dict


def get_model_config_by_type_and_name(tenant_id: str, model_type: str, model_name: str):
    if not model_name:
        raise Exception("Model Name is required")
    model_type_val = model_type.value if hasattr(model_type, "value") else model_type
    model_config = TenantLLMService.get_api_key(tenant_id, model_name, model_type_val)
    if not model_config:
        # model_name in format 'name@factory', split model_name and try again
        pure_model_name, fid = TenantLLMService.split_model_name_and_factory(model_name)
        compose_profiles = os.getenv("COMPOSE_PROFILES", "")
        is_tei_builtin_embedding = (
            model_type_val == LLMType.EMBEDDING.value
            and "tei-" in compose_profiles
            and pure_model_name == os.getenv("TEI_MODEL", "")
            and (fid == "Builtin" or fid is None)
        )
        if is_tei_builtin_embedding:
            # configured local embedding model
            embedding_cfg = settings.EMBEDDING_CFG
            config_dict = {
                "llm_factory": "Builtin",
                "api_key": embedding_cfg["api_key"],
                "llm_name": pure_model_name,
                "api_base": embedding_cfg["base_url"],
                "model_type": LLMType.EMBEDDING.value,
            }
        else:
            model_config = TenantLLMService.get_api_key(tenant_id, pure_model_name, model_type_val)
            if not model_config:
                raise LookupError(f"Tenant Model with name {model_name} and type {model_type_val} not found")
            config_dict = model_config.to_dict()
    else:
        # model_name without @factory
        config_dict = model_config.to_dict()
    config_model_type = config_dict.get("model_type")
    config_model_type = config_model_type.value if hasattr(config_model_type, "value") else config_model_type
    if config_model_type != model_type_val:
        raise LookupError(
            f"Tenant Model with name {model_name} has type {config_model_type}, expected {model_type_val}"
        )
    llm = LLMService.query(llm_name=config_dict["llm_name"])
    if llm:
        config_dict["is_tools"] = llm[0].is_tools
    return config_dict


def get_tenant_default_model_by_type(tenant_id: str, model_type: str|enum.Enum):
    # 先按租户 ID 读取租户记录。
    # 这样做是为了从租户配置里找到该租户为不同模型类型预设的默认模型名。
    exist, tenant = TenantService.get_by_id(tenant_id)
    # 租户不存在时立即报错。
    # 这样做是为了避免后续从空对象读取默认模型字段，顺便把错误定位在更准确的根因上。
    if not exist:
        raise LookupError("Tenant not found")
    # 把 `model_type` 统一归一化成字符串值。
    # 这样做是为了同时兼容传入 `LLMType.xxx` 枚举和直接传字符串两种调用方式。
    model_type_val = model_type if isinstance(model_type, str) else model_type.value
    # 先定义一个空的模型名变量，后面根据模型类型从租户配置里选出对应字段。
    model_name: str = ""
    # 按模型类型映射到租户上的默认模型字段。
    # 这样做是为了把“模型类型”与“租户配置中的具体默认模型名”解耦，调用方只需要关心类型，不需要知道字段细节。
    match model_type_val:
        case LLMType.EMBEDDING.value:
            # 向量模型默认取租户的 embedding 配置。
            model_name = tenant.embd_id
        case LLMType.SPEECH2TEXT.value:
            # 语音转文本模型默认取租户的 ASR 配置。
            model_name =  tenant.asr_id
        case LLMType.IMAGE2TEXT.value:
            # 图像转文本模型默认取租户的 img2txt 配置。
            model_name = tenant.img2txt_id
        case LLMType.CHAT.value:
            # 聊天模型默认取租户的主 LLM 配置。
            model_name = tenant.llm_id
        case LLMType.RERANK.value:
            # 重排模型默认取租户的 rerank 配置。
            model_name = tenant.rerank_id
        case LLMType.TTS.value:
            # 文本转语音模型默认取租户的 TTS 配置。
            model_name = tenant.tts_id
        case LLMType.OCR.value:
            # OCR 不从这里兜底默认模型，调用方必须显式指定。
            # 这样做通常是因为 OCR 的使用场景和模型选择更强依赖具体任务，不适合静态默认化。
            raise Exception("OCR model name is required")
        case _:
            # 未知模型类型直接报错，避免静默走错分支。
            raise Exception(f"Unknown model type {model_type}")
    # 如果该类型没有配置默认模型，也直接报错。
    # 这样做是为了阻止系统在模型配置缺失时继续运行，避免后面出现更隐蔽的失败。
    if not model_name:
        raise Exception(f"No default {model_type} model is set.")
    # 拿到默认模型名后，再解析成完整模型配置返回。
    # 这样做是为了把调用方真正需要的供应商参数、密钥、接口地址等完整信息统一取回。
    return get_model_config_by_type_and_name(tenant_id, model_type, model_name)
