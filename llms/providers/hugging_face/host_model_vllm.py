import argparse
import os
import subprocess
import sys
import time
from typing import Any

import torch

from core_utils.logger_utils import logger
from core_utils.network_utils import find_free_port, is_server_running, wait_for_server
from core_utils.signal_utils import signal_manager
from llms.providers.hugging_face.constants import (
    ENDPOINT_TEMPLATE,
    VLLM_ENFORCE_EAGER,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    VLLM_QUANTIZE,
    VLLM_TENSOR_PARALLEL_SIZE,
    VLLM_USE_V1,
)

tmux_session_id = ""
popen_process: subprocess.Popen[Any] | None = None

os.environ["VLLM_USE_V1"] = str(VLLM_USE_V1)
# Usage:
# python -m llms.providers.hugging_face.host_model_vllm "Qwen/Qwen2.5-VL-3B-Instruct" --host localhost --port <port> --num-gpus <num_gpus> --max-model-len <max_model_len>


def cleanup() -> None:
    global popen_process, tmux_session_id
    if popen_process:
        popen_process.terminate()
        popen_process.wait()
    if tmux_session_id:
        subprocess.run(["tmux", "kill-session", "-t", tmux_session_id])
    torch.cuda.empty_cache()


def build_command(args: argparse.Namespace) -> list[str]:
    command = ["vllm", "serve", args.model_path]
    if args.host is not None:
        command.extend(["--host", args.host])
    if args.port is not None:
        command.extend(["--port", str(args.port)])
    if args.num_gpus is not None:
        command.extend(["--tensor-parallel-size", str(args.num_gpus)])
    if args.quantize:
        command.append("--quantize")
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])

    if args.enforce_eager:
        command.append("--enforce-eager")

    if args.gpu_memory_utilization:
        command.append("--gpu-memory-utilization")
        command.append(str(args.gpu_memory_utilization))

    if args.max_image_per_prompt or args.max_video_per_prompt:
        command.extend(["--limit-mm-per-prompt"])
    if args.max_image_per_prompt and args.max_video_per_prompt:
        command.extend([f"image={str(args.max_image_per_prompt)},video={str(args.max_video_per_prompt)}"])
    elif args.max_image_per_prompt:
        command.extend([f"image={str(args.max_image_per_prompt)}"])
    elif args.max_video_per_prompt:
        command.extend([f"video={str(args.max_video_per_prompt)}"])

    return command


def deploy_vllm(args: argparse.Namespace) -> tuple[str, subprocess.Popen[Any] | None]:
    global tmux_session_id, popen_process

    # Automatically find an available port if none was specified.
    if args.port is None:
        args.port = find_free_port()

    endpoint = ENDPOINT_TEMPLATE.format(host=args.host, port=args.port)

    if is_server_running(endpoint):
        logger.info(f"There's already a service running at {endpoint}")
        return endpoint, None

    command = build_command(args)
    try:
        if args.tmux:
            # Build the vLLM command components.
            command_str = " ".join(command)
            model_str = args.model_path.replace("/", "_")
            # Build the tmux command to run the vLLM command in a new session.
            final_command = f'tmux new-session -d -s vllm_{model_str} bash -c "{command_str}"'
            logger.info(f"Attempting to start vLLM server in tmux session with command: {final_command}")
            process = subprocess.Popen(
                final_command,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                env=os.environ.copy(),  # Pass the current environment variables
            )
            tmux_session_id = f"vllm_{model_str}"
            popen_process = process
        else:
            logger.info(f"Attempting to start vLLM server with command: {' '.join(command)}")
            process = subprocess.Popen(
                command,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                env=os.environ.copy(),  # Pass the current environment variables
            )
            popen_process = process
    except Exception as e:
        logger.error(f"Error starting vLLM server: {e}")
        sys.exit(1)

    logger.info(f"Waiting for vLLM server to start at {endpoint}")
    wait_for = 60 * 2
    is_running = wait_for_server(endpoint, max_retries=100, total_time=wait_for, sleep_for=5)
    if not is_running:
        logger.error(f"vLLM server didn't start after {wait_for} seconds at {endpoint}")
        cleanup()
        sys.exit(1)
    else:
        logger.info(f"vLLM server started at {endpoint}")

    return endpoint, process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host a model using vLLM with OpenAI API compatibility.")
    parser.add_argument(
        "model_path",
        # "--model_path",
        # default="Qwen/Qwen2.5-VL-3B-Instruct",
        # default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        type=str,
        help="The model identifier or path.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address for the vLLM server (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the vLLM server. If not provided, an available port is chosen automatically.",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Run the vLLM server in a new tmux session named 'vllm_<model_name>'.",
    )
    parser.add_argument(
        "--num-gpus",
        "--tensor-parallel-size",
        dest="num_gpus",
        type=int,
        default=VLLM_TENSOR_PARALLEL_SIZE,
        help=f"Number of tensor parallel GPUs (default: {VLLM_TENSOR_PARALLEL_SIZE}).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=VLLM_QUANTIZE,
        help="Enable quantization of the model using bitsandbytes.",
    )
    parser.add_argument(
        "--max-model-len",
        dest="max_model_len",
        type=int,
        default=VLLM_MAX_MODEL_LEN,
        help=f"Maximum model length (default: {VLLM_MAX_MODEL_LEN}).",
    )
    parser.add_argument(
        "--max-image-per-prompt",
        type=int,
        default=999,  # vllm V0 max is 16
        help="Limit the maximum number of images per prompt.",
    )

    parser.add_argument(
        "--max-video-per-prompt",
        type=int,
        default=999,  # vllm V0 max is 2
        help="Limit the maximum number of videos per prompt.",
    )

    parser.add_argument(
        "--eager",
        "--enforce-eager",
        action="store_true",
        dest="enforce_eager",
        help=f"Enforce eager execution (default: {VLLM_ENFORCE_EAGER}).",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        "--gpu-mem",
        type=float,
        dest="gpu_memory_utilization",
        default=VLLM_GPU_MEMORY_UTILIZATION,
        help="The fraction of GPU memory to use for the vLLM server.",
    )

    args = parser.parse_args()

    args.enforce_eager = args.enforce_eager or VLLM_ENFORCE_EAGER
    return args


def watch_dog() -> None:
    global popen_process
    while popen_process and popen_process.poll() is None:
        time.sleep(5)
    cleanup()


if __name__ == "__main__":
    signal_manager.add_cleanup_function(cleanup)
    torch.cuda.empty_cache()
    args = parse_args()
    deploy_vllm(args)
    watch_dog()
