# ===============================================================================
# Captioner
# ===============================================================================
import argparse
import io
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import requests
import torch
from benchmark_config import (
    CAPTIONER_DEFAULT_IMAGE_FORMAT,
    CAPTIONER_ENDPOINT,
    CAPTIONER_HOST_SCRIPT,
    CAPTIONER_PORT,
    CAPTIONER_SUPPORTED_MODELS,
)
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


def is_captioner_running(script=CAPTIONER_HOST_SCRIPT, endpoint=CAPTIONER_ENDPOINT, timeout=30) -> bool:
    is_running = False
    min_tries = 1  # Set a minimum of 1 in case of latency and low `timeout`

    # Check: is the script running?
    # This does not work if captioner is on another machine
    # cmd = ["pgrep", "-f", script]
    # result = subprocess.run(cmd, stdout=subprocess.PIPE)
    # is_running = result.returncode == 0
    # print(f"Script running: {is_running}")

    # Check: is the endpoint responding?
    print(f"Checking captioner endpoint: {endpoint}")
    # endpoint = re.sub(r"/caption/", "/", endpoint)

    time_start = time.time()
    while time.time() - time_start < timeout or min_tries > 0:
        min_tries -= 1
        try:
            response = requests.post(endpoint, timeout=10)
            is_running = True if response.status_code else False
            break
        except requests.exceptions.RequestException as _:
            is_running = False

    return is_running


def start_captioner(
    model_name,
    model_device,
    script=CAPTIONER_HOST_SCRIPT,
    port=CAPTIONER_PORT,
    endpoint=CAPTIONER_ENDPOINT,
    conda_env="vwebarena",
    tmux_session_name="vwa_captioner",
    max_retries=3,
) -> None:
    """
    If host_captioner.py is not running, start it in a background tmux session.
    This approach uses tmux to create a detached session for the captioner.
    """
    if is_captioner_running(script=script, endpoint=endpoint, timeout=1):
        print(f"[INFO] `{script}` running on {endpoint}")
        return

    retries, is_running = 0, False
    while True:
        if retries > max_retries:
            raise RuntimeError(f"Captioner not found. Failed to start captioner script {script} after {max_retries} retries")

        python_exe = sys.executable
        script_abs = str(Path(script).expanduser().resolve())

        # Try to start a tmux session hosting the captioner
        try:
            tmux_command = (
                f'tmux new-session -d -s "{tmux_session_name}" '
                # f'"conda init; conda activate {conda_env}; python {script} --model_name {model_name} --device {model_device} --port {port}"'
                f'"{python_exe} {script_abs} --model_name {model_name} --device {model_device} --port {port}"'
            )
            print(f"Trying to start captioner on tmux with command: `{tmux_command}`")
            _ = subprocess.run(
                tmux_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,  # Raise an error if the command fails
            )

        except subprocess.CalledProcessError as e:
            if "duplicate session" in e.stderr:
                is_running = True
                pass

            else:
                print("tmux command failed:")
                print("stdout:", e.stdout)
                print("stderr:", e.stderr)
                raise

        is_running = is_running or is_captioner_running(script=script, endpoint=endpoint)
        if is_running:
            print(f"[INFO] `{script}` running on {endpoint}")
            return
        retries += 1


def get_captioning_fn(
    device,
    dtype,
    model_name: str = "Salesforce/blip2-flan-t5-xl",
    server_url: str = CAPTIONER_ENDPOINT,
) -> callable:
    if model_name not in CAPTIONER_SUPPORTED_MODELS:
        raise NotImplementedError(f"Model {model_name} not supported")

    # If hosting model, start the server if it is not already running
    if "server" in device:
        # server-cuda:0 -> cuda:0 // server-cuda -> cuda
        model_device = re.sub(r"^server-(cuda(:\d+)?)$", r"\1", device)

        # Check if server is already running
        if is_captioner_running(timeout=15):
            print(f"Captioner receiving requests on {server_url}")
        else:
            start_captioner(model_name=model_name, model_device=model_device, endpoint=server_url)

    # If not hosting model, move to appropriate devices
    else:
        if "blip2" in model_name:
            captioning_processor = Blip2Processor.from_pretrained(model_name)
            captioning_model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
            captioning_model.to(device)

    # Create the captioning function
    if "server" in device:

        def caption_images(
            images: List[Image.Image],
            prompt: List[str] = None,
            max_new_tokens: int = 32,
            server_url: str = server_url,
        ) -> List[str]:
            format_to_mime = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "GIF": "image/gif",
            }

            image_files = []
            for i, image in enumerate(images):
                with io.BytesIO() as output:
                    # Default to PNG to not affect image quality in the transfer
                    if image.format == "JPEG":
                        image_format = CAPTIONER_DEFAULT_IMAGE_FORMAT  # e.g., "PNG"
                    else:
                        image_format = image.format if image.format in format_to_mime else CAPTIONER_DEFAULT_IMAGE_FORMAT

                    image.save(output, format=image_format)
                    image_bytes = output.getvalue()
                    mime_type = format_to_mime.get(image_format, format_to_mime[CAPTIONER_DEFAULT_IMAGE_FORMAT])
                    image_files.append(("files", (f"image_{i}.{image_format.lower()}", image_bytes, mime_type)))

            # Prepare data payload
            data = {"max_new_tokens": max_new_tokens}
            if prompt is not None:
                data["prompt"] = prompt

            # Send images and prompt to the server
            response = requests.post(server_url, files=image_files, data=data)

            # Convert response to PIL Image
            if response.status_code == 200:
                captions = response.json().get("captions", [])
            else:
                print(f"WARNING: Error captioning images: {response.status_code}")
                captions = [""] * len(images)

            return captions

    else:

        def caption_images(
            images: List[Image.Image],
            prompt: List[str] = None,
            max_new_tokens: int = 32,
        ) -> List[str]:
            if prompt is None:
                prompt = [""] * len(images)  # Obs: new checkpoints of Blip2 require empty prompts as list of empty strings

                # Perform VQA
                inputs = captioning_processor(images=images, return_tensors="pt").to(device, dtype)
                generated_ids = captioning_model.generate(**inputs, max_new_tokens=max_new_tokens)
                captions = captioning_processor.batch_decode(generated_ids, skip_special_tokens=True)
            else:
                # Regular captioning. Prompt is a list of strings, one for each image
                assert len(images) == len(prompt), "Number of images and prompts must match, got {} and {}".format(len(images), len(prompt))
                inputs = captioning_processor(images=images, text=prompt, return_tensors="pt").to(device, dtype)
                generated_ids = captioning_model.generate(**inputs, max_new_tokens=max_new_tokens)
                captions = captioning_processor.batch_decode(generated_ids, skip_special_tokens=True)
            return captions

    return caption_images


def define_captioning_fn(agent_captioning_model_device, agent_captioning_model, eval_captioning_model_device, eval_captioning_model, observation_type) -> tuple[callable, callable]:
    caption_image_fn = eval_caption_image_fn = None
    agent_device = agent_captioning_model_device

    if "server" in agent_device:
        model_device = "cuda" if "cuda" in agent_device else "cpu"
        dtype = "float32" if model_device == "cpu" else "float16"
    else:
        dtype = torch.float32 if agent_device == "cpu" else torch.float16

    # Define agent captioning function
    if agent_captioning_model and observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        caption_image_fn = get_captioning_fn(agent_device, dtype, agent_captioning_model)
    else:
        caption_image_fn = None
        agent_captioning_model = None

    # Define captioning function for VQA score
    if "server" in eval_captioning_model_device:
        model_device = "cuda" if "cuda" in eval_captioning_model_device else "cpu"
        dtype = "float32" if model_device == "cpu" else "float16"
    else:
        dtype = torch.float16 if (torch.cuda.is_available() and "cuda" in eval_captioning_model_device) else torch.float32

    # If eval captioning model is the same as the agent captioning model, use the same object for VQA scoring
    if caption_image_fn is not None and eval_captioning_model == agent_captioning_model:
        eval_captioning_model_device = agent_captioning_model_device
        eval_caption_image_fn = caption_image_fn
    else:
        eval_caption_image_fn = get_captioning_fn(
            eval_captioning_model_device,
            dtype,
            eval_captioning_model,
        )

    return caption_image_fn, eval_caption_image_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--model_device", type=str, default="cuda:0")
    parser.add_argument("--script", type=str, default=CAPTIONER_HOST_SCRIPT)
    parser.add_argument("--port", type=int, default=CAPTIONER_PORT)
    parser.add_argument("--endpoint", type=str, default=CAPTIONER_ENDPOINT)
    parser.add_argument("--conda_env", type=str, default="vwebarena")
    parser.add_argument("--tmux_session_name", type=str, default="vwa_captioner")
    args = parser.parse_args()

    start_captioner(
        model_name=args.model_name,
        model_device=args.model_device,
        script=args.script,
        port=args.port,
        endpoint=args.endpoint,
        conda_env=args.conda_env,
        tmux_session_name=args.tmux_session_name,
    )
