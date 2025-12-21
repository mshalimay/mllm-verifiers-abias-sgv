import asyncio
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from core_utils.logger_utils import logger
from llms.llm_utils import batch_call_llm, call_llm

MAX_WORKERS = 30  # Max number of threads to use for building LLM call args in parallel.
MAX_RETRIES = 3  # If exception during batch processing, retry up to 3 times.


def _get_tasks_subset(filepath):
    domain_task_ids: set[str] = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            domain, task_id = line.split()
            if "task_id" in task_id:
                continue
            domain_task_ids.add(f"{domain}_{task_id}")
    return domain_task_ids


def get_files_and_task_ids_from_source_file_list(path_to_csv, domain):
    file_paths = []
    if not path_to_csv:
        raise ValueError("source_file_list is not set in config")
    df = pd.read_csv(path_to_csv)
    df = df[df["domain"] == domain]
    for _, row in df.iterrows():
        file_path = row["traj_source_path"]
        task_id = row["task_id"]
        file_paths.append((file_path, task_id))

    return file_paths


def get_files_and_task_ids(config, domain, run_config):
    """
    Given a configuration with a trace path template, finds all matching trajectory files
    Returns:
        List of tuples, each tuple is (file_path, task_id)
    """
    if config.get("source_file_list", None):
        files_with_task_ids = get_files_and_task_ids_from_source_file_list(config["source_file_list"], domain)
    else:
        raise ValueError("source_file_list must be specified in config")

    conversation_dir = f"{config['out_dir'].strip('-')}/conversation"
    files_with_task_ids_final = []

    if run_config.get("task_lists", {}).get(config["env"]):
        task_subset = _get_tasks_subset(run_config["task_lists"][config["env"]])
    else:
        task_subset = None

    for file_path, task_id in files_with_task_ids:
        # If output exists and not overwrite, skip
        full_conversation_path = f"{conversation_dir}/{task_id}.txt"
        if os.path.exists(full_conversation_path) and not run_config["overwrite"]:
            logger.info(f"Skipping {task_id} because {full_conversation_path} exists.")
            continue

        # Check in file subset if provide. If not, skip
        if task_subset is not None and f"{domain}_{task_id}" not in task_subset:
            continue
        files_with_task_ids_final.append((file_path, task_id))
    return files_with_task_ids_final


def split_jobs_into_batches(all_jobs_data, max_batch_size=80):
    # Calculate the number of batches; each batch will have at most max_batch_size items.
    total_tasks = len(all_jobs_data)
    if total_tasks == 0:
        return []

    if max_batch_size > 0:
        num_batches = (total_tasks + max_batch_size - 1) // max_batch_size
    else:
        num_batches = 1

    # Create batches: each sublist is at most max_batch_size long.
    batch_idxs = np.array_split(range(total_tasks), num_batches)
    for batch_idx in batch_idxs:
        yield [all_jobs_data[i] for i in batch_idx]


def get_all_jobs_data(configs_per_env, run_config):
    all_jobs_data_env = {}
    for env in configs_per_env:
        configs_per_case = configs_per_env[env]
        order_id = 0  # Auxiliary variable to sort the jobs by order they appear
        all_jobs_data: list[tuple[str, str, dict, int]] = []  # trace_path, task_id, config, order_id

        # For each dict_entry in CASES, we have a config for each domain.
        for case_name, configs_per_domain in configs_per_case.items():
            # Collect data to run in batch mode.
            for domain, config in configs_per_domain.items():
                for trace_path, task_id in get_files_and_task_ids(config, domain=domain, run_config=run_config):
                    all_jobs_data.append((trace_path, task_id, config, order_id))
            order_id += 1

        all_jobs_data_env[env] = all_jobs_data
    return all_jobs_data_env


def run_batch_mode(configs_per_env, run_config, gen_config, build_llm_call_args_fn):
    max_batch_size_runners = run_config["max_batch_size_runners"]

    for env in configs_per_env:
        configs_per_case = configs_per_env[env]
        order_id = 0  # Auxiliary variable to sort the jobs by order they appear
        all_jobs_data: list[tuple[str, str, dict, int]] = []  # trace_path, task_id, config, order_id

        # For each dict_entry in CASES, we have a config for each domain.
        for case_name, configs_per_domain in configs_per_case.items():
            # Collect data to run in batch mode.
            for domain, config in configs_per_domain.items():
                for trace_path, task_id in get_files_and_task_ids(config, domain=domain, run_config=run_config):
                    all_jobs_data.append((trace_path, task_id, config, order_id))
            order_id += 1

        if len(all_jobs_data) == 0:
            logger.info(f"No tasks to run for env {env}")
            continue

        # Sort the jobs_data by task_id and CASES order
        if run_config.get("sort_by_config", True):
            all_jobs_data.sort(key=lambda x: (x[3], x[1]))
        else:
            # Sort by task_id
            all_jobs_data.sort(key=lambda x: x[1])

        # Create a list of batches. Each batch is represented by a tuple: (jobs_data, retry_count)
        batches = [(jobs_data, 0) for jobs_data in list(split_jobs_into_batches(all_jobs_data, max_batch_size=max_batch_size_runners))]
        if len(batches) == 0:
            logger.info(f"No tasks to run for env {env}")
            continue

        max_retries = MAX_RETRIES
        batch_index = 0  # For logging/tracking purposes
        total_batches = len(batches)

        logger.info(f"Starting execution for env {env}: {len(all_jobs_data)} tasks broken into {total_batches} batches")
        while batches:
            batch_index += 1
            jobs_data, attempt = batches.pop(0)  # Get the next batch and its current attempt number
            logger.info(f"Processing batch {batch_index} of {total_batches} with {len(jobs_data)} items, attempt {attempt + 1}/{max_retries}")
            batch_call_llm_args = {
                "prompts": [],
                "conversation_dirs": [],
                "usage_dirs": [],
                "call_ids": [],
            }
            gc.collect()
            batch_error_occurred = False
            try:
                # Using a list to preserve order (this helps in downstream caching of images)
                if run_config.get("build_prompt_sequential", False):
                    for trace_path, task_id, config, _ in jobs_data:
                        result = build_llm_call_args_fn(trace_path, task_id, config, run_config)
                        if result is None or not result:
                            continue

                        prompts, conversation_dirs, usage_dirs, call_ids = result
                        if not prompts:
                            continue
                        batch_call_llm_args["prompts"].extend(prompts)
                        batch_call_llm_args["conversation_dirs"].extend(conversation_dirs)
                        batch_call_llm_args["usage_dirs"].extend(usage_dirs)
                        batch_call_llm_args["call_ids"].extend(call_ids)
                else:
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [executor.submit(build_llm_call_args_fn, trace_path, task_id, config, run_config) for trace_path, task_id, config, _ in jobs_data]

                        # Iterate over futures in the order they were submitted
                        for idx, future in enumerate(futures):
                            result = future.result()
                            if result is None or not result:
                                continue

                            prompts, conversation_dirs, usage_dirs, call_ids = result
                            if not prompts:
                                continue
                            batch_call_llm_args["prompts"].extend(prompts)
                            batch_call_llm_args["conversation_dirs"].extend(conversation_dirs)
                            batch_call_llm_args["usage_dirs"].extend(usage_dirs)
                            batch_call_llm_args["call_ids"].extend(call_ids)

                # Run the batch of LLM calls.
                if len(batch_call_llm_args["prompts"]) == 0:
                    logger.info(f"Skipping batch {batch_index}. No tasks to run.")
                    continue
                logger.info(f"Running {len(batch_call_llm_args['prompts'])} calls in batch mode")

                _, _ = batch_call_llm(
                    gen_kwargs=gen_config,
                    prompts=batch_call_llm_args["prompts"],
                    conversation_dirs=batch_call_llm_args["conversation_dirs"],
                    usage_dirs=batch_call_llm_args["usage_dirs"],
                    call_ids=batch_call_llm_args["call_ids"],
                    max_batch_size=-1,
                    num_workers=run_config.get("max_api_keys", 2),
                    multiprocess_mode=run_config.get("multiprocess_batch_mode", False),
                    verbose=True,
                    return_outputs=False,
                    max_api_keys=run_config.get("max_api_keys", 1),
                    dump_html=run_config.get("dump_html", False),
                    dump_txt=True,
                    # order_by_payload_size=True,
                    ovewrite_txt=True,
                )
                gc.collect()

            except Exception as e:
                logger.error(f"Error during batch call: {e}", exc_info=True)
                batch_error_occurred = True

            finally:
                batch_call_llm_args.clear()
                gc.collect()

            # If an error occurred processing this batch, and we haven't hit the max retries, re-add it.
            if batch_error_occurred:
                if attempt + 1 < max_retries:
                    logger.error(f"Batch {batch_index} encountered an error. Re-adding batch for retry (attempt {attempt + 1}/{max_retries}).")
                    batches.append((jobs_data, attempt + 1))
                else:
                    logger.error(f"Batch {batch_index} exceeded max retries. Skipping batch.")
                continue
            # end of single batch run.
        logger.info(f"Finished execution for env {env}")
        # end of env run.


async def run_sequential(configs_per_env, run_config, gen_config, build_llm_call_args_fn):
    queue = asyncio.Queue()
    logger.info(f"Run config for sequential run:{run_config}")
    start_time = asyncio.get_event_loop().time()
    # Precompute all jobs (trace_path, task_id, config, order_id) per env
    all_jobs_data_env = get_all_jobs_data(configs_per_env, run_config)
    total_jobs = sum(len(jobs) for jobs in all_jobs_data_env.values())
    if total_jobs == 0:
        logger.info("No jobs found. Exiting early.")
        return
    logger.info(f"Total jobs to process = {total_jobs}")

    def _batch_call_llm(
        gen_kwargs,
        prompts,
        conversation_dirs,
        usage_dirs,
        call_ids,
    ):
        try:
            for job in split_jobs_into_batches(
                list(
                    zip(
                        prompts,
                        conversation_dirs,
                        usage_dirs,
                        call_ids,
                    )
                ),
                max_batch_size=run_config["max_batch_size_runners"],
            ):
                p_batch, cdir_batch, udir_batch, cid_batch = zip(*job)
                logger.info(f"Consumer: Firing aggregated batch of {len(p_batch)} prompts")
                _ = batch_call_llm(
                    gen_kwargs=gen_kwargs,
                    prompts=p_batch,  # type:ignore
                    conversation_dirs=cdir_batch,  # type:ignore
                    usage_dirs=udir_batch,  # type:ignore
                    call_ids=cid_batch,  # type:ignore
                    max_batch_size=-1,
                    num_workers=run_config.get("max_api_keys", 2),
                    multiprocess_mode=run_config.get("multiprocess_batch_mode", False),
                    verbose=True,
                    return_outputs=False,
                    max_api_keys=run_config.get("max_api_keys", 1),
                    dump_html=run_config.get("dump_html", False),
                    dump_txt=True,
                    # order_by_payload_size=True,
                    ovewrite_txt=True,
                )
            return 1
        except Exception as e:
            logger.warning(f"async_batch_call_llm: Error during batch call: {e}")
            return None

    async def async_call_llm(prompt, gen_config, conversation_dir, usage_dir, call_id):
        try:
            return call_llm(
                prompt=prompt,
                gen_kwargs=gen_config,
                conversation_dir=conversation_dir,
                usage_dir=usage_dir,
                call_id=call_id,
                verbose=True,
                dump_html=run_config.get("dump_html", False),
                dump_txt=True,
                ovewrite_txt=True,
            )
        except Exception as e:
            logger.warning(f"async_call_llm: Error for call_id={call_id} conversation_dir={conversation_dir}: {e}")
            return None

    async def producer():
        items_added = 0
        for env, jobs in all_jobs_data_env.items():
            logger.info(f"Producer: Env {env} has {len(jobs)} jobs")
            for trace_path, task_id, config, order_id in jobs:
                logger.info(f"Producer: Building LLM call args for task {task_id}")
                try:
                    llm_call_args = await asyncio.to_thread(build_llm_call_args_fn, trace_path, task_id, config, run_config)
                except Exception as e:
                    logger.warning(
                        f"Producer: Exception while building call args for task {task_id}: {e}",
                        exc_info=True,
                    )
                    continue
                if not llm_call_args or not llm_call_args[0]:
                    logger.info(f"Producer: Skipping task {task_id} - empty prompts returned")
                    continue
                await queue.put(llm_call_args)
                items_added += 1
        logger.info(f"Producer: Finished enqueueing {items_added} jobs (of {total_jobs}). Sending sentinel.")
        await queue.put(None)

    async def consumer():
        processed = 0
        logger.info("Consumer: Starting to process items from queue")
        # Minimum number of aggregated prompts before firing a batch (unless sentinel or buffer flush condition)
        min_agg_prompts = run_config.get("sequential_min_fire_batch", 5)
        # Buffers for non-batch path aggregation
        buffer_prompts: list = []
        buffer_conversation_dirs: list[str] = []
        buffer_usage_dirs: list[str] = []
        buffer_call_ids: list[str] = []
        while True:
            if run_config.get("batch_call_llm", False):
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    percent = (processed / total_jobs) * 100 if total_jobs else 100.0
                    logger.info(f"Consumer: Received sentinel. Processed {processed}/{total_jobs} jobs ({percent:.2f}%). Exiting.")
                    break

                prompts, conversation_dirs, usage_dirs, call_ids = item
                for prompt, conversation_dir, usage_dir, call_id in zip(prompts, conversation_dirs, usage_dirs, call_ids):
                    await async_call_llm(
                        prompt=prompt,
                        gen_config=gen_config,
                        conversation_dir=conversation_dir,
                        usage_dir=usage_dir,
                        call_id=call_id,
                    )
                    processed += 1
                    percent = (processed / total_jobs) * 100
                    curr_time = asyncio.get_event_loop().time()
                    elapsed = curr_time - start_time
                    logger.info(f"Consumer: Progress {processed}/{total_jobs} ({percent:.2f}%). minutes elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                    jobs_remaining = total_jobs - processed
                    if processed > 0 and jobs_remaining > 0:
                        time_per_job = elapsed / processed
                        estimated_time_remaining = time_per_job * jobs_remaining
                    else:
                        estimated_time_remaining = None
                    if estimated_time_remaining:
                        logger.info(f"Estimated time remaining: {time.strftime('%H:%M:%S', time.gmtime(estimated_time_remaining))}")
            else:
                first_item = await queue.get()
                if first_item is None:
                    # Sentinel received: flush buffer if any then exit
                    queue.task_done()
                    if buffer_prompts:
                        try:
                            await asyncio.to_thread(
                                _batch_call_llm,
                                gen_kwargs=gen_config,
                                prompts=buffer_prompts,
                                conversation_dirs=buffer_conversation_dirs,
                                usage_dirs=buffer_usage_dirs,
                                call_ids=buffer_call_ids,
                            )
                        except Exception as e:
                            logger.error(f"Consumer: Exception flushing buffer on sentinel: {e}", exc_info=True)
                        buffer_prompts.clear()
                        buffer_conversation_dirs.clear()
                        buffer_usage_dirs.clear()
                        buffer_call_ids.clear()
                    break

                # Add first item to buffers
                p, c_dirs, u_dirs, c_ids = first_item
                buffer_prompts.extend(p)
                buffer_conversation_dirs.extend(c_dirs)
                buffer_usage_dirs.extend(u_dirs)
                buffer_call_ids.extend(c_ids)
                queue.task_done()

                # Drain any immediately available additional items (non-blocking)
                while not queue.empty():
                    try:
                        nxt = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if nxt is None:
                        # Put sentinel back to be processed in next iteration after buffer flush
                        await queue.put(None)
                        break
                    p2, c2_dirs, u2_dirs, c2_ids = nxt
                    buffer_prompts.extend(p2)
                    buffer_conversation_dirs.extend(c2_dirs)
                    buffer_usage_dirs.extend(u2_dirs)
                    buffer_call_ids.extend(c2_ids)
                    queue.task_done()
                    processed += 1

                # Only fire batch if threshold reached
                if len(buffer_prompts) >= min_agg_prompts:
                    try:
                        await asyncio.to_thread(
                            _batch_call_llm,
                            gen_kwargs=gen_config,
                            prompts=buffer_prompts,
                            conversation_dirs=buffer_conversation_dirs,
                            usage_dirs=buffer_usage_dirs,
                            call_ids=buffer_call_ids,
                        )
                        processed += len(buffer_prompts)
                    except Exception as e:
                        logger.error(f"Consumer: Exception during batch_call_llm aggregated batch: {e}", exc_info=True)
                    buffer_prompts.clear()
                    buffer_conversation_dirs.clear()
                    buffer_usage_dirs.clear()
                    buffer_call_ids.clear()
                    percent = (processed / total_jobs) * 100
                    curr_time = asyncio.get_event_loop().time()
                    elapsed = curr_time - start_time
                    logger.info(f"Consumer: Progress {processed}/{total_jobs} ({percent:.2f}%). minutes elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                    jobs_remaining = total_jobs - processed
                    if processed > 0 and jobs_remaining > 0:
                        time_per_job = elapsed / processed
                        estimated_time_remaining = time_per_job * jobs_remaining
                        logger.info(f"Estimated time remaining: {time.strftime('%H:%M:%S', time.gmtime(estimated_time_remaining))}")
                # Continue loop without extra queue.task_done()
                continue

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    await producer_task
    await queue.join()
    await consumer_task
    logger.info("run_sequential: Completed all LLM calls")
