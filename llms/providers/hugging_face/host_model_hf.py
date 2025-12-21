import argparse
import warnings
from typing import Any, Dict

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException

from core_utils.network_utils import find_free_port
from llms.providers.hugging_face.constants import USE_FLASH_ATTN
from llms.providers.hugging_face.hugging_face_client_manager import get_client_manager
from llms.providers.hugging_face.model_specific.model_processor import ModelProcessor
from llms.providers.hugging_face.parsing_utils import count_tokens, get_trim_prompt_idxs

warnings.filterwarnings("ignore")

# Usage:
# python -m llms.providers.hugging_face.host_model_hf "Qwen/Qwen2.5-VL-3B-Instruct" --host 0.0.0.0 --port 8000

# -------------------------------
# Argument parsing and device setup
# -------------------------------
parser = argparse.ArgumentParser(description="Host a Hugging Face model with FastAPI")
parser.add_argument(
    "model_path",
    # "--model_path",
    # default="Qwen/Qwen2.5-VL-3B-Instruct",
    type=str,
    help="Path to the model to use. (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct')",
)

parser.add_argument(
    "--port",
    type=int,
    default=None,
    help="Port to run the server on",
)

parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to run the server on",
)

parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Device to run the model on (e.g., 'cpu', 'cuda', 'auto', 'mps')",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="",
    help="Data type to run the model on (e.g., 'float16', 'float32', 'bfloat16', 'auto')",
)

parser.add_argument(
    "--quant-bits",
    type=str,
    default="",
    help="Quantization bits to run the model on (e.g., '8bit', '4bit')",
)

parser.add_argument(
    "--flash-attn",
    action="store_true",
    help="Use flash attention",
)
args = parser.parse_args()


if not args.port:
    args.port = find_free_port()

# -------------------------------
# Load model
# -------------------------------
kwargs = {
    "model_path": args.model_path,
    "device": args.device,
    "dtype": args.dtype,
    "quant_bits": args.quant_bits,
    "flash_attn": args.flash_attn or USE_FLASH_ATTN,
    "trust_remote_code": True,
}

client_manager = get_client_manager(args.model_path)
model = client_manager.get_model(kwargs, engine="automodel")
processor = client_manager.get_processor()


# -------------------------------
# Create FastAPI app and endpoint
# -------------------------------
app = FastAPI()


# @app.post("/{model_name:path}/generate")
@app.post("/generate")
async def generate(messages: list[list[dict[str, Any]]], gen_kwargs: dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Endpoint to perform text generation.

    The request body should have two keys:
      - "messages": containing the list of messages
      - "gen_kwargs": a dict of additional keyword arguments for the model generate method.
    """
    try:
        # Convert list inputs to torch tensors
        inputs = ModelProcessor.get_inputs(messages, args.model_path)
        params = dict(gen_kwargs)

        params.update({"return_dict_in_generate": True, "output_scores": True})

        device = model.device
        if hasattr(inputs, "shape"):
            response = model.generate(inputs.to(device), **params)
        else:
            response = model.generate(**(inputs.to(device)), **params)

        # Index to trim the prompt from the output
        trim_prompt_idxs = get_trim_prompt_idxs(inputs, len(response.sequences))

        # Decode outputs
        natural_outputs = ModelProcessor.decode_outputs(response.sequences, args.model_path, start_idxs=trim_prompt_idxs, skip_special_tokens=True)

        # Convert `response` to dict
        response_dict = {}
        response_dict["usage"] = count_tokens(response, inputs)

        # Convert scores tensors to list if present
        if "return_scores" in params and params["return_scores"]:
            if hasattr(response, "scores") and response.scores is not None:
                response_dict["scores"] = [score.detach().cpu().tolist() for score in response.scores]  # type: ignore

        return {"model_messages": natural_outputs, "api_response": response_dict}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/{model_name:path}/info")
@app.post("/info")
async def info() -> dict[str, Any]:
    return {
        "model_path": args.model_path,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "quant_bits": args.quant_bits,
        "flash_attn": args.flash_attn,
        "endpoint": f"http://{args.host}:{args.port}/",
    }


def cleanup() -> None:
    global model, processor, client_manager
    del model
    del processor
    del client_manager
    torch.cuda.empty_cache()


# -------------------------------
# Run the FastAPI server using uvicorn when executed directly
# -------------------------------
if __name__ == "__main__":
    # signal_manager.add_cleanup_function(cleanup)
    uvicorn.run(app, host=args.host, port=args.port)
