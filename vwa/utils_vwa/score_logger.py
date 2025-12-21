from typing import Any

from browser_env.env_utils import wait_for_page_to_stabilize

from core_utils.file_utils import update_json
from core_utils.logger_utils import logger
from core_utils.timing_utils import timeit
from utils_vwa.utils_vwa import TrajectoryView


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


@timeit(custom_name="AGENT:hard_verify")
def hard_verify(trajectory: TrajectoryView, meta_data: dict[str, Any], intent: str) -> float:
    tmp = None
    try:
        env = meta_data["env"]
        evaluator = meta_data["evaluator"]
        cur_url = env.page.url
        x, y = env.page.evaluate("() => [window.scrollX, window.scrollY]")

        # Run evaluator on a throwaway page so we don't mutate env.page or its history
        tmp = env.context.new_page()

        tmp.goto(cur_url)
        wait_for_page_to_stabilize(
            page=tmp,
            logger=logger,
            min_num_trues=4,
            return_early=False,
            return_after=None,
            min_wait_time_seconds=env.sleep_after_execution,
        )

        # restore scroll position
        tmp.evaluate("(pos) => window.scrollTo(pos[0], pos[1])", [x, y])
        score = evaluator(trajectory=trajectory.trajectory, page=tmp)
        return float("nan") if not is_number(score) else score

    except Exception as e:
        logger.warning(f"Error scoring trajectory: {e}", exc_info=True)
        return float("nan")

    finally:
        if tmp is not None:
            try:
                tmp.close()
            except Exception as e:
                logger.warning(f"Error closing temporary page: {e}. Ignoring.", exc_info=True)


def compute_confusion(score: float, verifier_score: int) -> str:
    if score == 0 and verifier_score == 0:
        return "TN"
    elif score == 1 and verifier_score == 1:
        return "TP"
    elif score == 0 and verifier_score == 1:
        return "FP"
    elif score == 1 and verifier_score == 0:
        return "FN"
    else:
        logger.warning(f"Not able to determine confusion for verifier score: {verifier_score} and score: {score}")
        return ""


class ScoreLogger:
    def __init__(self, scores_per_round_file: str = "scores_per_round.json"):
        self.json_file = scores_per_round_file
        self.score_per_round: dict[str, Any] = {}

    def log_scores_per_round(
        self,
        trajectory: TrajectoryView,
        intent: str,
        meta_data: dict[str, Any],
        verifier_score: int | None = None,
    ) -> float:
        args = meta_data["args"]
        url = trajectory.states[-1]["info"]["page"].url
        state_idx = len(trajectory.states) - 1
        task_id = meta_data["task_id"]
        domain = meta_data.get("domain", "")
        attempt_num = meta_data["attempt_num"]
        domain_task_id = f"{domain}_{task_id}"

        # Get hard evaluation for current trajectory == score given by evaluator functions from benchmark
        score = hard_verify(trajectory, meta_data, intent)

        # Create entry for current task if not present
        if domain_task_id not in self.score_per_round:
            self.score_per_round[domain_task_id] = {}

        # Create entry for current attempt if not present
        if attempt_num not in self.score_per_round[domain_task_id]:
            self.score_per_round[domain_task_id][attempt_num] = {}
            self.score_per_round[domain_task_id][attempt_num]["domain"] = domain
            self.score_per_round[domain_task_id][attempt_num]["intent"] = intent
            self.score_per_round[domain_task_id][attempt_num]["rounds_per_state"] = {}
            self.score_per_round[domain_task_id][attempt_num]["scores"] = []
            self.score_per_round[domain_task_id][attempt_num]["attempt_num"] = attempt_num

        # Update round counts for the current state of current task
        if state_idx not in self.score_per_round[domain_task_id][attempt_num]["rounds_per_state"]:
            self.score_per_round[domain_task_id][attempt_num]["rounds_per_state"][state_idx] = 0
        else:
            self.score_per_round[domain_task_id][attempt_num]["rounds_per_state"][state_idx] += 1

        last_action = trajectory.actions[-1]
        data = {
            "state_idx": state_idx,
            "score": score,
            "url": url,
            "raw_prediction": last_action["raw_prediction"],
            "parsed_action": last_action.get("parsed_action", ""),
            "round": self.score_per_round[domain_task_id][attempt_num]["rounds_per_state"][state_idx],
        }
        if verifier_score is not None:
            data.update({"verifier_score": verifier_score})
            confusion = compute_confusion(score, verifier_score)
            data.update({"confusion": confusion})

        if "retrieved_knowledge" in trajectory.states[-1]:
            data.update({"retrieved_knowledge": trajectory.states[-1]["retrieved_knowledge"]})

        if "verifier_executor_loop_utterances" in last_action:
            data.update({"verifier_raw_response": last_action["verifier_executor_loop_utterances"]["verifier"][-1]})

        self.score_per_round[domain_task_id][attempt_num]["scores"].append(data)

        # Dump the score per round to a json file
        update_json(file_path=f"{args.result_dir}/{self.json_file}", data=self.score_per_round)

        if "scores_per_round" not in trajectory.states[state_idx]:
            trajectory.states[state_idx]["scores_per_round"] = []
        trajectory.states[state_idx]["scores_per_round"].append(data)
        return score
