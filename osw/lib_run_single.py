import datetime
import json
import logging
import os
import time
from pathlib import Path

from verifier.first_pass import get_first_pass_knowledge
from verifier.second_pass import Evaluation, get_second_pass_evaluation

logger = logging.getLogger("desktopenv.experiment")


def save_trajectory(trajectory: dict, task_results_path: Path):
    trajectory_path = task_results_path / "trajectory.json"
    with open(trajectory_path, "w", encoding="utf-8") as trajectory_file:
        json.dump(trajectory, trajectory_file, indent=2)


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(10)
    obs = env._get_obs()  # Get the initial observation
    initial_screenshot = obs["screenshot"]
    domain = Path(example_result_dir).parent.name
    task_id = Path(example_result_dir).name
    initial_screenshot_path = Path(example_result_dir) / "step_0.png"
    with open(initial_screenshot_path, "wb") as screenshot_file:
        screenshot_file.write(initial_screenshot)
    trajectory = {
        "domain": domain,
        "task_id": task_id,
        "objective": instruction,
        "initial_screenshot": initial_screenshot_path.name,
    }
    run_results_path = Path(example_result_dir).parent.parent
    if args.verifier == "two_pass" or args.give_knowledge_to_agent:
        first_pass_knowledge = get_first_pass_knowledge(objective=instruction, screenshot=initial_screenshot, run_results_path=run_results_path, domain=domain, task_id=task_id)
        trajectory["first_pass_knowledge"] = first_pass_knowledge
    else:
        first_pass_knowledge = None
    trajectory["steps"] = []
    done = False
    step_idx = 0
    # env.controller.start_recording()
    is_intermediate_feedback = False
    while not done and step_idx < max_steps:
        verifier_loop_count = 0
        if not is_intermediate_feedback:
            feedback = None
        verifier_loop = []
        while verifier_loop_count < 3:
            verifier_loop_count += 1
            knowledge = first_pass_knowledge if args.give_knowledge_to_agent else None
            response, actions = agent.predict(instruction, knowledge, obs, feedback, is_intermediate_feedback, Path(example_result_dir))
            verifier_loop.append({"agent": response})
            if args.verifier == "none":
                break
            if actions[0] in ["DONE", "FAIL"]:
                screenshots = [observation["screenshot"] for observation in agent.observations]
                prompt_type = "call_user" if args.prompt_style == "qwen2vl_user" else "normal"
                (evaluation, feedback, verifier_text) = get_second_pass_evaluation(
                    objective=instruction,
                    screenshots=screenshots,
                    thoughts=agent.thoughts,
                    first_pass_knowledge=first_pass_knowledge,
                    prompt_type=prompt_type,
                    run_results_path=run_results_path,
                    domain=domain,
                    task_id=task_id,
                )
                verifier_loop.append({"verifier": verifier_text})
                if evaluation == Evaluation.SUCCESS:
                    break
                if evaluation == Evaluation.INFEASIBLE:
                    actions = ["FAIL"]
                    break
            else:
                break
        screenshot_file_names = []
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            screenshot_path = Path(example_result_dir) / f"step_{step_idx + 1}_{action_timestamp}.png"
            with open(screenshot_path, "wb") as screenshot_file:
                screenshot_file.write(obs["screenshot"])
            screenshot_file_names.append(screenshot_path.name)
            if done:
                logger.info("The episode is done.")
                break
        trajectory["steps"].append(
            {
                "step_index": step_idx + 1,
                "generated_text": response,
                "screenshots": screenshot_file_names,
            }
        )
        if args.verifier in {"two_pass", "one_pass"} and args.verify_every_n_steps > 0 and verifier_loop_count == 1 and (step_idx + 1) % args.verify_every_n_steps == 0 and step_idx != max_steps - 1:
            screenshots = [observation["screenshot"] for observation in agent.observations]
            prompt_type = "call_user" if args.prompt_style == "qwen2vl_user" else "normal"
            _, feedback, verifier_text = get_second_pass_evaluation(
                objective=instruction,
                screenshots=screenshots,
                thoughts=agent.thoughts,
                first_pass_knowledge=first_pass_knowledge,
                prompt_type=prompt_type,
                run_results_path=run_results_path,
                domain=domain,
                task_id=task_id,
            )
            is_intermediate_feedback = True
            verifier_loop.append({"verifier": verifier_text})
        else:
            is_intermediate_feedback = False
        if len(verifier_loop) > 1:
            trajectory["steps"][-1]["verifier_loop"] = verifier_loop
        save_trajectory(trajectory, task_results_path=Path(example_result_dir))
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    if result is True:
        score = 1
    elif result is False:
        score = 0
    else:
        score = float(result)
    trajectory["score"] = score
    save_trajectory(trajectory, task_results_path=Path(example_result_dir))
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    # env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger
