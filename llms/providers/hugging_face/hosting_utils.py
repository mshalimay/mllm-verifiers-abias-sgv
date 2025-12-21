"""
Some functions to deploy and interact with local hosted models.
"""

import os
import re
import subprocess
from typing import Any, Dict

import requests

from core_utils.logger_utils import logger
from core_utils.network_utils import is_server_running, wait_for_server
from core_utils.signal_utils import signal_manager
from llms.providers.hugging_face.constants import ENDPOINT_TEMPLATE, VLLM_ENFORCE_EAGER, VLLM_GPU_MEMORY_UTILIZATION

process: subprocess.Popen[Any] | None = None

# ===============================================================
# LINK Manual hosting helpers
# ===============================================================


def build_endpoint(kwargs: Dict[str, Any]) -> tuple[str, str, str]:
    if "port" in kwargs:
        port = kwargs["port"]
        host = kwargs.get("host", "localhost")

    elif "endpoint" in kwargs:
        match = re.search(r"^(?:https?://)?([^:/]+):(\d+)", kwargs["endpoint"])
        if match and match.group(2) is not None:
            if match.group(1) is not None:
                host = match.group(1)
            else:
                host = "localhost"
            port = match.group(2)
        else:
            raise ValueError(f"{os.path.basename(__file__)}: Invalid endpoint format. Please use the format 'host:port'.")

    endpoint = f"http://{host}:{port}"
    return endpoint, host, port


def get_local_server_info(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    endpoint, _, _ = build_endpoint(kwargs)
    if is_server_running(f"{endpoint}/info"):
        response = requests.post(f"{endpoint}/info")  # type: ignore
        return response.json()  # type: ignore
    else:
        return {}


def launch_local_server(model_id: str, kwargs: Dict[str, Any]) -> str:
    deploy_kwargs = {
        "model_path": kwargs["model_path"],
        "device": kwargs.get("device", ""),
        "dtype": kwargs.get("dtype", ""),
        "quant_bits": kwargs.get("quant_bits", ""),
        "flash_attn": kwargs.get("flash_attn", False),
    }
    endpoint, host, port = build_endpoint(kwargs)

    if is_server_running(f"{endpoint}/info"):
        return endpoint

    cmd = [
        "python",
        "-m",
        "llms.providers.hugging_face.host_model_hf",
        model_id,
        "--host",
        host,
        "--port",
        port,
    ]

    if deploy_kwargs["device"]:
        cmd.append("--device")
        cmd.append(str(deploy_kwargs["device"]))

    if deploy_kwargs["dtype"]:
        cmd.append("--dtype")
        cmd.append(str(deploy_kwargs["dtype"]))

    if deploy_kwargs["quant_bits"]:
        cmd.append("--quant-bits")
        cmd.append(str(deploy_kwargs["quant_bits"]))

    if deploy_kwargs["flash_attn"]:
        cmd.append("--flash-attn")

    logger.info(f"{__file__}: Attempting to launch local server at {endpoint} with command {cmd}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    is_running = wait_for_server(
        f"{endpoint}/info",
        total_time=60 * 2,
        timeout=10,
        sleep_for=3,
        max_retries=1e5,
    )

    if not is_running:
        # Force kill the process
        process.kill()
        process.wait()
        raise ValueError(f"Unable to start server on {endpoint}")

    def cleanup_process() -> None:
        process.kill()
        process.wait()

    signal_manager.add_cleanup_function(cleanup_process)

    return endpoint


# ===============================================================
# LINK VLLM hosting helpers
# ===============================================================
def cleanup_vllm() -> None:
    global process
    if process:
        process.kill()
        process.wait()
    # Send pkill 'serve vllm'
    # subprocess.run(["pkill", "-f", "vllm serve"])


def launch_vllm_server(model_path: str, kwargs: Dict[str, Any]) -> str:
    global process
    from llms.providers.hugging_face.constants import VLLM_MAX_MODEL_LEN, VLLM_QUANTIZE, VLLM_TENSOR_PARALLEL_SIZE

    endpoint, host, port = build_endpoint(kwargs)
    endpoint = ENDPOINT_TEMPLATE.format(host=host, port=port)
    if is_server_running(endpoint):
        return endpoint

    deploy_kwargs = {
        "model_path": model_path,
        "host": host,
        "port": port,
        "tmux": kwargs.get("tmux", True),
        "num_gpus": kwargs.get("num_gpus", VLLM_TENSOR_PARALLEL_SIZE),
        "quantize": kwargs.get("quantize", VLLM_QUANTIZE),
        "max_model_len": kwargs.get("max_model_len", VLLM_MAX_MODEL_LEN),
        "enforce_eager": kwargs.get("enforce_eager", VLLM_ENFORCE_EAGER),
        "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", VLLM_GPU_MEMORY_UTILIZATION),
    }

    # Start building the command list. The model_path is passed as a positional parameter.
    cmd = [
        "python",
        "-m",
        "llms.providers.hugging_face.host_model_vllm",
        model_path,
    ]
    # Always include host and port.
    cmd.extend(["--host", host])
    cmd.extend(["--port", port])

    # Add tensor parallel size only if it's set.
    if deploy_kwargs.get("num_gpus"):
        cmd.extend(["--tensor-parallel-size", str(deploy_kwargs["num_gpus"])])

    # Only add the --quantize flag if quantization is enabled.
    if deploy_kwargs.get("quantize"):
        cmd.append("--quantize")

    # Only add max_model_len if provided.
    if deploy_kwargs.get("max_model_len"):
        cmd.extend(["--max-model-len", str(deploy_kwargs["max_model_len"])])

    # Only add the tmux flag if requested.
    if deploy_kwargs.get("tmux"):
        cmd.append("--tmux")

    if deploy_kwargs.get("enforce_eager"):
        cmd.append("--enforce-eager")

    if deploy_kwargs.get("gpu_memory_utilization"):
        cmd.append("--gpu-memory-utilization")
        cmd.append(str(deploy_kwargs["gpu_memory_utilization"]))

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait until the server is responsive.
    is_running = wait_for_server(endpoint, total_time=60 * 2, timeout=10, sleep_for=3, max_retries=1e5)
    if not is_running and process:
        cleanup_vllm()
    signal_manager.add_cleanup_function(cleanup_vllm)

    return endpoint
