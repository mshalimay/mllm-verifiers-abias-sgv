import argparse
import io
import re
import warnings
from typing import Any, List, Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from transformers import AutoModel, AutoProcessor, Blip2ForConditionalGeneration, Blip2Processor

warnings.filterwarnings("ignore")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Host a model with FastAPI.")
parser.add_argument("--model_name", type=str, default="Salesforce/blip2-flan-t5-xl", help="Name of the model to use")
parser.add_argument("--port", type=int, default=9555, help="Port to run the server on")
parser.add_argument("--device", type=str, default="cuda", help='Device to run the model on (e.g., "cpu", "cuda")')
parser.add_argument("--dtype", type=str, default="auto", help='Data type for model weights (e.g., "float32", "float16", "auto")')
args = parser.parse_args()

app = FastAPI()


def get_device_map(device: str | dict[str, Any]) -> dict[str, Any]:
    """
    Given a string or dictionary for the device, return a dictionary to be passed as the `device_map`.

    - If a dictionary (e.g., {"layer1": "cuda:0", "layer2": "cuda:1"}) is provided, it is returned as is.
    - If a string is provided:
      - If it starts with "cuda" (e.g., "cuda:0", "cuda:1"), the entire model is mapped onto that device.
      - If it is "cpu", the device map is set as "cpu".
      - Otherwise, the device map is set as "auto" for automatic placement.
    """
    # Case 1: If device is already a dictionary, use it directly.
    device_map: str | dict[str, Any]
    if isinstance(device, dict):
        device_map = device
    # Case 2: Handle when device is provided as a string.
    elif isinstance(device, str):
        if device.lower().startswith("cuda"):
            device_map = {"": device}
        elif device.lower() == "cpu":
            device_map = "cpu"
        else:
            device_map = "auto"
    else:
        raise ValueError("The device parameter must be a string or a dictionary.")

    return {"device_map": device_map}


# Different versions of BLIP2 in HF are showing sensitivity to image resolution / fp precision
def maybe_load_model_high_precision(model_id: str, original_dtype: str, device_map) -> Any:
    torch.cuda.empty_cache()
    if "blip2" in model_id.lower():
        if original_dtype != "float32":
            try:
                model = Blip2ForConditionalGeneration.from_pretrained(model_id, dtype="float32", **device_map)
                return model
            except Exception as _:
                torch.cuda.empty_cache()
                model = Blip2ForConditionalGeneration.from_pretrained(model_id, dtype=original_dtype, **device_map)
                return model
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(model_id, dtype=original_dtype, **device_map)
            return model
    else:
        if original_dtype != "float32" and not any(dty in model_id.lower() for dty in ["fp8", "int8", "int4"]):
            try:
                model = AutoModel.from_pretrained(model_id, dtype="float32", **device_map)
                return model
            except Exception as _:
                torch.cuda.empty_cache()
                model = AutoModel.from_pretrained(model_id, dtype=original_dtype, **device_map)
                return model
        else:
            model = AutoModel.from_pretrained(model_id, dtype=original_dtype, **device_map)
            return model


# server-cuda:0 -> cuda:0 // server-cuda -> cuda // cuda -> cuda // cuda:0 -> cuda:0
args.device = re.sub(r"^server-(cuda(:\d+)?)$", r"\1", args.device)
torch.cuda.empty_cache()
device_map = get_device_map(args.device)
if "blip2" in args.model_name.lower():
    processor = Blip2Processor.from_pretrained(args.model_name)
    model = maybe_load_model_high_precision(args.model_name, args.dtype, device_map)
else:
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = maybe_load_model_high_precision(args.model_name, args.dtype, device_map)


@app.post("/")
async def root() -> dict:
    return {"message": "Hello World"}


@app.post("/caption/")
async def caption_images(files: List[UploadFile] = File(...), prompt: Optional[List[str]] = Form(None), max_new_tokens: int = Form(32)) -> dict:
    images = [Image.open(io.BytesIO(await file.read())) for file in files]

    if not prompt:  # Obs: new versions of Blip2 requires empty prompts as list of empty strings
        prompt = [""] * len(images)

    # Try to caption full batch
    try:
        inputs = processor(images=images, text=prompt, return_tensors="pt").to(args.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # # Remove unicode, spaces, and newlines
        # captions = [re.sub(r'[^\x00-\x7F]+', '', caption).strip() for caption in captions]

        for caption in captions:
            if "taiwan " in caption:
                print("stop")

        return {"captions": captions}

    except Exception as _:
        # try to caption one by one
        captions = []
        errors = []
        for i, (img, p) in enumerate(zip(images, prompt)):
            try:
                inputs = processor(images=[img], text=[p], return_tensors="pt").to(args.device, args.dtype)
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                captions.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])

            except Exception as e:
                print(f"Error: {e}")
                errors.append(e)
                captions.append("")

        for caption in captions:
            if "tv" in caption:
                print("stop")

        return {"captions": captions, "errors": errors}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
