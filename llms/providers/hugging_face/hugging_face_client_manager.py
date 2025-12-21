"""Handles the interaction with the Hugging Face API with all engines."""

import os
import re
from typing import Any, Dict

import torch
from transformers import AutoProcessor, AutoTokenizer  # type: ignore

from llms.generation_config import GenerationConfig
from llms.providers.client_manager import ClientManager
from llms.providers.hugging_face.constants import (
    USE_FLASH_ATTN,
    VLLM_OPENAI_KEY,
    get_api_provider_arg,
)
from llms.providers.hugging_face.hosting_utils import get_local_server_info, launch_local_server
from llms.providers.hugging_face.model_specific.model_loader import HFModelLoader
from llms.providers.openai.openai_client_manager import get_client_manager as get_openai_client_manager


class HuggingFaceClientManager(ClientManager):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            provider="huggingface",
            max_api_key_retry=float("inf"),
            max_key_process_count=float("inf"),
        )
        self.model: Any = None
        self.processor: AutoProcessor | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.model_id: str = model_id
        self.set_hf_token()
        self.openai_client: Any | None = None

    def get_device(self, gen_config: GenerationConfig) -> str:
        if gen_config.device:
            return gen_config.device

        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _is_flash_attn_available(self) -> bool:
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            return False

    def load_automodel(self, gen_config: GenerationConfig | Dict[str, Any]) -> None:
        if isinstance(gen_config, GenerationConfig):
            gen_config_dict = gen_config.to_dict()
        else:
            gen_config_dict = gen_config

        if "flash_attn" not in gen_config_dict or gen_config_dict["flash_attn"] is None:
            gen_config_dict["flash_attn"] = USE_FLASH_ATTN

        self.model = HFModelLoader.load_model(
            model_path=gen_config_dict["model_path"],
            device=gen_config_dict.get("device", ""),
            trust_remote_code=True,
            flash_attn=gen_config_dict.get("flash_attn", False),
            dtype=gen_config_dict.get("torch_dtype", ""),
            quant_bits=gen_config_dict.get("quant_bits", ""),
        )

        self.model.eval()  # type: ignore

    def get_tokenizer(self) -> AutoTokenizer | None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        return self.tokenizer

    def get_processor(self) -> AutoProcessor | None:
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        return self.processor

    def get_model(self, gen_config: GenerationConfig | Dict[str, Any], engine: str = "automodel", p_id: str = "") -> Any:
        if self.model is not None:
            return self.model

        # If no model set, load the model
        if re.match("automodel", engine, flags=re.IGNORECASE):
            self.load_automodel(gen_config)
            self.get_processor()
            self.get_tokenizer()

        elif re.match("server", engine, flags=re.IGNORECASE):
            gen_config_dict = gen_config.to_dict() if isinstance(gen_config, GenerationConfig) else gen_config
            model_path = gen_config_dict["model_path"]
            server_info = get_local_server_info(gen_config_dict)
            if server_info:
                if not model_path == server_info["model_path"]:
                    raise ValueError(f"A server is running on {server_info['endpoint']} but it is not for {model_path}; server info: {server_info}")
                else:
                    self.model = server_info["endpoint"]
            else:
                endpoint = launch_local_server(model_path, gen_config_dict)
                self.model = endpoint

        elif re.match("openai", engine, flags=re.IGNORECASE):
            metadata = gen_config.metadata if isinstance(gen_config, GenerationConfig) else gen_config.get("metadata", {})
            if not metadata:
                raise ValueError("Metadata is required for Hugging Face OpenAI engine")
            base_url = metadata["base_url"]
            provider = metadata["provider"]
            model_path = gen_config.model_path if isinstance(gen_config, GenerationConfig) else gen_config.get("model_path", "")

            openai_client_manager = get_openai_client_manager(
                model_id=self.model_id + str(p_id),
                base_url=base_url,
                provider=provider,
                max_api_key_retry=get_api_provider_arg(provider, "max_api_key_retry"),
                max_key_process_count=get_api_provider_arg(provider, "max_key_process_count"),
            )
            self.openai_client_manager = openai_client_manager

            self.openai_client = openai_client_manager.get_client()
            self.openai_aclient = openai_client_manager.get_aclient()
            self.model = model_path

        elif re.match("vllm", engine, flags=re.IGNORECASE):
            raise NotImplementedError("VLLM engine needs debugging")
            # gen_config_dict = gen_config.to_dict() if isinstance(gen_config, GenerationConfig) else gen_config
            # model_path = gen_config_dict["model_path"]
            # endpoint = launch_vllm_server(model_path, gen_config_dict)
            # self.openai_client = get_openai_client_manager(
            #     model_id=model_path,
            #     api_key=VLLM_OPENAI_KEY,
            #     base_url=endpoint,
            # ).get_client()
            # self.model = endpoint

        elif re.match("tgi", engine, flags=re.IGNORECASE):
            raise NotImplementedError("TGI engine not implemented")
        else:
            raise ValueError(f"Unsupported engine: {engine}. Please use 'automodel', 'server', or 'vllm'.")

        return self.model

    def set_hf_token(self) -> None:
        if self.api_key:
            os.environ["HF_TOKEN"] = self.api_key
            return
        try:
            self.fetch_api_key()
        except Exception:
            self.api_key = "EMPTY"
            # os.environ["HF_TOKEN"] = "EMPTY"

    def get_openai_client(self, model_id: str = "", endpoint: str = "") -> Any:
        if self.openai_client is None:
            self.openai_client = get_openai_client_manager(model_id=self.model_id, api_key=VLLM_OPENAI_KEY, base_url=endpoint).get_client()
        return self.openai_client

    def get_openai_aclient(self, model_id: str = "", endpoint: str = "") -> Any:
        if self.openai_aclient is None:
            self.openai_aclient = get_openai_client_manager(model_id=self.model_id, api_key=VLLM_OPENAI_KEY, base_url=endpoint).get_aclient()
        return self.openai_aclient


# Module-level instance (singleton)
_global_client_manager: Dict[tuple, HuggingFaceClientManager] = {}


def get_client_manager(model_id: str = "", client_manager_idx: int = 0) -> HuggingFaceClientManager:
    global _global_client_manager

    if not model_id and len(_global_client_manager) > 0:
        key = list(_global_client_manager.keys())[client_manager_idx]
        return _global_client_manager[key]

    if (model_id, client_manager_idx) not in _global_client_manager:
        _global_client_manager[(model_id, client_manager_idx)] = HuggingFaceClientManager(model_id=model_id)
    return _global_client_manager[(model_id, client_manager_idx)]


def get_client_managers() -> Dict[tuple, HuggingFaceClientManager]:
    global _global_client_manager
    return _global_client_manager
