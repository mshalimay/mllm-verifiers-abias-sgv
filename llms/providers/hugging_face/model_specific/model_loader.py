import gc
import re
from typing import Any, Dict

import torch
from transformers import (  # type:ignore
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Qwen2_5_VLForConditionalGeneration,
)

from core_utils.signal_utils import signal_manager
from llms.providers.hugging_face.setup_utils import (
    get_device_map,
    get_dtype,
    is_bitsandbytes_available,
    is_flash_attn_available,
    is_quantized_model,
)

model_references = {}


def cleanup():
    model_references.clear()
    gc.collect()
    torch.cuda.empty_cache()


signal_manager.add_cleanup_function(cleanup)


class HFModelLoader:
    """
    Encapsulates methods to load models from Hugging Face's hub.
    """

    @classmethod
    def load_model(
        cls,
        model_path: str,
        device: str = "auto",
        trust_remote_code: bool = True,
        flash_attn: bool = False,
        dtype: str = "",
        quant_bits: str = "",
        kwargs: Dict[str, Any] = {},
    ) -> Any:
        """Load a model from Hugging Face's hub.

        Args:
            model_path (str): The path to the model on Hugging Face's hub.
            device (str, optional): The device to load the model on. Defaults to "auto".
            trust_remote_code (bool, optional): Whether to trust the remote code. Defaults to True.
            flash_attn (bool, optional): Whether to use flash attention. Defaults to False.
            dtype (str, optional): The dtype to load the model on. Defaults to "".
            quant_bits (str, optional): The quant bits to load the model on. Defaults to "".
            kwargs (Dict[str, Any], optional): Additional keyword arguments. Defaults to {}.
        """
        # --------------------------------
        # Regularize params
        # --------------------------------
        # Check if bitsandbytes is installed
        quant_bits = str(quant_bits).lower()
        if quant_bits and not is_bitsandbytes_available():
            raise ValueError("Bitsandbytes is not installed. Please install it using `pip install bitsandbytes`.")

        # Update kwargs with flash attention if requested and available
        use_flash_attn = flash_attn and is_flash_attn_available() and "cpu" not in device.lower()
        if use_flash_attn:
            kwargs.update({"attn_implementation": "flash_attention_2"})
        elif flash_attn:
            print("Warning: Flash Attention requested but not available. Using default attention implementation.")

        # --------------------------------
        # Set dtype, devices
        # --------------------------------
        # Empty cache and get device map
        torch.cuda.empty_cache()
        device_map = get_device_map(device)
        kwargs.update(device_map)

        # Initial get-set dtype
        dtype = dtype or get_dtype(device, model_path)

        # --------------------------------
        # Handle quantization
        # --------------------------------
        # Check if quantized model
        is_quantized, quant_method = is_quantized_model(model_path)

        # If AWQ quantized model, adjust dtype
        if is_quantized:
            if "awq" in quant_method:
                dtype = "float16"  # HuggingFace prefers float16 for AWQ
            else:
                dtype = "auto"

        # If runtime quantization requested and not quantized already, set up quantization config
        if quant_bits and not is_quantized:
            from transformers import BitsAndBytesConfig  # type:ignore

            if "8" in quant_bits:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True, do_fuse=True)  # type: ignore
            elif "4" in quant_bits:
                bnb_config = BitsAndBytesConfig(  # type: ignore
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False,
                    do_fuse=True,
                )
            else:
                raise ValueError(f"Invalid quantize value: {quant_bits}. Please use '8bit', or '4bit'.")

            kwargs.update({"quantization_config": bnb_config})
            dtype = "auto"

        # --------------------------------
        # Load model
        # --------------------------------
        # Update kwargs with dtype
        kwargs.update({"torch_dtype": dtype})

        # Load model with model specific or default method
        if re.match(r"Qwen/Qwen2\.5-VL-.*", model_path, flags=re.IGNORECASE):
            # Qwen2.5 VL model loader
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_safetensors=True,
                **kwargs,
            )

        elif re.match(r"ByteDance-Seed/UI-TARS-.*", model_path, flags=re.IGNORECASE):
            # UI-TARS model loader
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_safetensors=True,
                **kwargs,
            )
        else:
            # Default model loader
            model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                model_path,
                trust_remote_code=trust_remote_code,
                use_safetensors=True,
                **kwargs,
            )

        global model_references
        model_references[model_path] = model

        return model
