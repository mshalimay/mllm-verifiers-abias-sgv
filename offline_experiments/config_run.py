import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path

from core_utils.logger_utils import logger
from offline_experiments.build_configs import build_all_first_pass_configs, build_all_verif_configs
from offline_experiments.config.eval_configs import out_dir_template
from offline_experiments.config.llm_configs import llm_configs
from offline_experiments.runners import run_batch_mode, run_sequential

warnings.filterwarnings("ignore", message="Pydantic serializer warnings:")
from typing import Any, Dict, List, Optional

parser = argparse.ArgumentParser()
parser.add_argument("--fp", action="store_true")
parser.add_argument("--v", action="store_true")
parser.add_argument("--c", type=str, default="")
args = parser.parse_args()

if sys.gettrace() is not None:
    # @debug
    args.c = "offline_experiments/config/experiment_config/gemini-25-flash.py"
    args.v = True
    args.fp = False
else:
    args.c = args.c


# ==============================================================================
# Run Experiments
# ==============================================================================
# Defaults & Declarations for static type checkers; values are populated via dynamic import from args.c
CASES: Optional[List[Dict[str, Any]]] = None
run_config: Optional[Dict[str, Any]] = None
gen_config_fp: Optional[Dict[str, Any]] = None
gen_config_v: Optional[Dict[str, Any]] = None
num_retries: Optional[int] = 1
model_fp: Optional[str] = ""
model_v: Optional[str] = None
max_api_keys: Optional[int] = None
max_batch_runners: Optional[int] = None
sort_by_config: bool = True
task_list_vwa: str = ""
task_list_osw: str = ""
task_list_agrb_vwa: str = ""
ann_k: str = ""
ann_v: str = ""
overwrite: bool = False
batch_mode: bool = True
dump_html: bool = False
build_prompt_sequential: bool = False
batch_call_llm: bool = False
sequential_min_fire_batch: int = 5

"""Dynamic experiment config loading.
"""
import importlib.util

_spec = importlib.util.spec_from_file_location("py_config", args.c)
if not _spec or not _spec.loader:
    raise RuntimeError(f"Unable to import config module from {args.c}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
logger.info(f"Loaded experiment module from {args.c}")

_required = [
    "CASES",
    "model_v",
    "max_api_keys",
    "max_batch_runners",
]
_optional = [
    "num_retries",
    "ann_k",
    "ann_v",
    "task_list_vwa",
    "task_list_osw",
    "task_list_agrb_vwa",
    "model_fp",
    # Optional overrides for top-level flags
    "sort_by_config",
    "overwrite",
    "batch_mode",
    "dump_html",
    "build_prompt_sequential",
    "batch_call_llm",
    "sequential_min_fire_batch",
]
_missing = [name for name in _required if not hasattr(_mod, name)]
if _missing:
    raise RuntimeError(f"Config file {args.c} missing required variables: {', '.join(_missing)}")

_overridden = []
for _name in _required + _optional:
    if hasattr(_mod, _name):
        globals()[_name] = getattr(_mod, _name)
        _overridden.append(_name)


run_config = {
    "sort_by_config": sort_by_config,
    "overwrite": overwrite,
    "batch_mode": batch_mode,
    "build_prompt_sequential": build_prompt_sequential,
    "max_batch_size": -1,
    "max_api_keys": max_api_keys,
    "max_batch_size_runners": max_batch_runners,
    "task_lists": {
        "vwa": task_list_vwa,
        "osw": task_list_osw,
        "agrb_vwa": task_list_agrb_vwa,
    },
    "dump_html": dump_html,
    "sequential_min_fire_batch": sequential_min_fire_batch,
    "batch_call_llm": batch_call_llm,
    # "num_processes": 2,
    # "skip_payload": 15 * 1024 * 1024,
}

# Narrow types for static analyzers
assert isinstance(CASES, list)
assert isinstance(run_config, dict)
assert isinstance(ann_k, str)
assert isinstance(ann_v, str)
assert isinstance(num_retries, int)
assert isinstance(model_fp, str)
assert isinstance(model_v, str)


gen_config_v = {**llm_configs[model_v]}


if __name__ == "__main__":
    # Log the proccess ID
    logger.info(f"Process ID: {os.getpid()}")
    num_retries = 1 if run_config.get("overwrite") else num_retries

    if args.fp:
        from offline_experiments.first_pass import build_llm_call_args as build_llm_call_args_fp

        gen_config_fp = {**llm_configs[model_fp]}
        configs_per_env = build_all_first_pass_configs(CASES, out_dir_template=out_dir_template, ann_k=ann_k)
        if len(configs_per_env) == 0:
            logger.info("No configs to run")
            exit()

        for i in range(num_retries):
            logger.info(f"Running first pass: attempt {i + 1} of {num_retries}")
            run_batch_mode(
                configs_per_env,
                run_config=run_config,
                gen_config=gen_config_fp,
                build_llm_call_args_fn=build_llm_call_args_fp,
            )

    if args.v:
        from offline_experiments.verify import build_llm_call_args as build_llm_call_args_verify

        configs_per_env = build_all_verif_configs(CASES, model_v=model_v, gen_config_v=gen_config_v, out_dir_template=out_dir_template, ann_v=ann_v)
        if len(configs_per_env) == 0:
            logger.info("No configs to run")
            exit()

        if run_config.get("batch_mode"):
            for i in range(num_retries):
                logger.info(f"Running verify: attempt {i + 1} of {num_retries}")
                run_batch_mode(
                    configs_per_env,
                    run_config=run_config,
                    gen_config=gen_config_v,
                    build_llm_call_args_fn=build_llm_call_args_verify,
                )
        else:
            asyncio.run(
                run_sequential(
                    configs_per_env,
                    run_config=run_config,
                    gen_config=gen_config_v,
                    build_llm_call_args_fn=build_llm_call_args_verify,
                )
            )
