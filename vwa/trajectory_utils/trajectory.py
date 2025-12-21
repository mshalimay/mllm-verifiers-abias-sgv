import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from core_utils.image_utils import any_to_path
from core_utils.logger_utils import logger
from trajectory_utils.trajectory_utils import annotate_image


def get_config(task_id: str, domain: str, raw_config_json: str = "config_files/vwa/test_vwa.raw.json") -> dict:
    config_path = Path(raw_config_json)
    if not config_path.exists():
        raise Exception(f"Config file {config_path} does not exist")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    for task_data in configs:
        if task_data["task_id"] == task_id and task_data["domain"] == domain:
            return task_data
    raise Exception(f"Task {task_id} not found in config file {config_path}")


class Trajectory:
    steps: list[dict[str, Any]]
    task_id: str | None
    cum_reward: float
    experiment_name: str | None
    datetime: str | None
    total_time_seconds: float | None
    terminated: bool | None
    truncated: bool | None
    info: dict[str, Any] = {}
    episode_completed: bool

    def __init__(
        self,
        task_id: str | None = None,
        experiment_name: str | None = None,
    ):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.cum_reward = 0.0
        self.steps = []
        self.episode_completed = False
        self.terminated = None
        self.truncated = None
        self.info = {}
        self.total_time_seconds = None

    @property
    def states(self) -> list[dict[str, Any]]:  # Return list of single state dicts
        return [step["state"] for step in self.steps]  # Access "state" (singular)

    @property
    def actions(self) -> list[list[dict[str, Any]]]:
        return [step["actions"] for step in self.steps]


class VWA_Trajectory(Trajectory):
    def __init__(
        self,
        task_id: str,
        objective_text,
        objective_imgs: list[Any] | None = None,
        domain: str = "",
        args: Optional[dict[str, Any]] | str = None,
        config: Optional[dict[str, Any]] = None,
        experiment_name: str | None = None,
        config_file: str = "",
        file_path: str = "",
        metadata: dict[str, Any] = {},
    ):
        super().__init__(task_id=task_id, experiment_name=experiment_name)
        self.task_id = task_id
        self.domain = domain
        self.args = args
        self.config = config
        self.reward = None
        self.objective = {
            "text": objective_text,
            "images": objective_imgs or [],
        }
        self.metadata = metadata
        self.episode_completed = False
        self.file_path = file_path

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert non-JSON-serializable objects to serializable forms.
        Filters out functions, argparse.Namespace, and other problematic types.
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif callable(obj):
            # Skip functions/callables - they can't be serialized
            return f"<function: {getattr(obj, '__name__', 'unknown')}>"
        elif isinstance(obj, argparse.Namespace):
            # Convert argparse.Namespace to dict, but filter out non-serializable values
            return self._make_json_serializable(vars(obj))
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                try:
                    serialized_value = self._make_json_serializable(value)
                    result[key] = serialized_value
                except (TypeError, ValueError):
                    # Skip values that can't be serialized
                    result[key] = f"<non-serializable: {type(value).__name__}>"
            return result
        elif isinstance(obj, (list, tuple)):
            result = []
            for item in obj:
                try:
                    result.append(self._make_json_serializable(item))
                except (TypeError, ValueError):
                    result.append(f"<non-serializable: {type(item).__name__}>")
            return result
        else:
            # For other types, try to convert to string or skip
            try:
                # Test if it's JSON serializable
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return f"<non-serializable: {type(obj).__name__}>"

    def add_step(self, state=None, action=None, metadata=None, reward: float | None = None):
        step = {
            "step_index": len(self.steps),
            "reward": float(reward) if reward is not None else None,
            "state": None,
            "actions": None,
        }
        if state is not None:
            step["state"] = self.add_state(state, metadata, step["step_index"], reward)
        if action is not None:
            step["actions"] = self.add_actions(action, step["step_index"], metadata)

        self.steps.append(step)

    def update_step(self, step_index=-1, state=None, action=None, metadata=None, reward: float | None = None):
        # VWA format:
        # metadata = {action_str_history: list[str], task_id: str, args: argparse.Namespace, config_file: str, evaluator: Callable}
        if step_index < 0:
            step = self.steps[-1]
        elif step_index < len(self.steps):
            step = self.steps[step_index]
        else:
            raise ValueError(f"Step index {step_index} is out of bounds.")

        if state is not None:
            step["state"] = self.add_state(state, metadata, step_index, reward)

        if action is not None:
            step["actions"] = self.add_actions(action, step_index, metadata)

        if reward is not None:
            step["reward"] = float(reward)

    def clean_empty_fields(self, dict: dict[str, Any]):
        new_dict = {}
        for k, v in dict.items():
            try:
                if v:
                    new_dict[k] = v
            except Exception:
                new_dict[k] = v
        return new_dict

    def _get_coordinates(self, action):
        try:
            if action.get("element_center", None) is not None:
                left, top = action.get("element_center")
                coords = {"x": left, "y": top, "w": 0, "h": 0}
            elif action.get("coords", None) is not None:
                left, top = action.get("coords")
                if left > 0 or top > 0:  # Heuristic for VWA; if coords are 0,0,0,0, then it's not a valid coordinate
                    coords = {"x": left, "y": top, "w": 0, "h": 0}
                else:
                    coords = None
            else:
                coords = None
        except Exception:
            coords = None
        return coords

    def add_actions(self, action, step_index, metadata):
        # VWA format: action = {action: str, metadata: dict, verifier_executor_loop_utterances: dict[str, list[str]]}
        # TODO: multiple actions; define agents at initiation and loop through them
        actions_dict = {"agents": {}, "metadata": {}}
        atomic_actions_executor = []
        atomic_actions_executor.append(
            {
                "generated_text": action.get("raw_prediction", ""),
                "parsed_action": action.get("parsed_action", ""),
                "action_str_env_parsed": action.get("action_str_env_parsed", ""),
                "invalid": action.get("invalid", False),
                "thought_summary": action.get("thought_summary", ""),
            }
        )

        coordinates = self._get_coordinates(action)
        if coordinates is not None:
            atomic_actions_executor[0]["coordinates"] = coordinates
        actions_dict["agents"]["executor"] = atomic_actions_executor

        if "verifier_executor_loop_utterances" in action:
            actions_dict["metadata"]["verifier_executor_loop_utterances"] = action["verifier_executor_loop_utterances"]

        return actions_dict

    def add_state(self, state, metadata, step_index, reward: float | None = None):
        # VWA format:
        # state_info = {observation: {image: Any, raw_screenshot: Any, text: str, bboxes: dict}, info: {"page": Page, "fail_error": str, "observation_metadata": dict}}
        observation: dict[str, Any] = state.get("observation", {})
        state_info: dict[str, Any] = state.get("info", {})

        # Get images in observation
        # images = [observation.get("image")] if observation.get("image") is not None else []
        images = [observation.get("raw_screenshot")] if observation.get("raw_screenshot") is not None else []

        # Get text observation
        text = observation.get("text") if observation.get("text") is not None else ""

        # Get URL
        url = state_info.get("page", {}).url if state_info.get("page", None) is not None else ""

        # Get bounding boxes for images in observation, if any
        nodes_bbox_info = state_info.get("observation_metadata", {}).get("image", {}).get("obs_nodes_bbox_info", {})
        nodes_info_semantic_img = state_info.get("observation_metadata", {}).get("image", {}).get("obs_nodes_semantic_info", {})
        img_obs_data = {}
        for id, info in nodes_bbox_info.items():
            img_obs_data[id] = {
                "bbox": info.get("bbox", []),
                "semantic": nodes_info_semantic_img.get(id, {}),
            }

        # If no bbox data, collect info from 'image' observation; otherwise, skip as repeated info
        if not img_obs_data:
            img_obs_data = state_info.get("observation_metadata", {}).get("image", {})

        text_obs_data = state_info.get("observation_metadata", {}).get("text", {})

        state_dict = {
            "images": images,
            "text": text,
            "url": url,
            "env_error": metadata.get("fail_error", ""),
            "step_index": step_index,
            "reward": reward,
            "metadata": {
                "nodes_info_image_observations": [img_obs_data],
                "nodes_info_text_observations": [text_obs_data],
            },
        }
        if "scores_per_round" in state:
            state_dict["scores_per_round"] = state["scores_per_round"]
        if "retrieved_knowledge" in state:
            state_dict["retrieved_knowledge"] = state["retrieved_knowledge"]

        return state_dict

    def end_episode(
        self,
        reward: float,
        total_time_seconds: float,
        episode_completed: bool,
        terminated: bool | None = None,
        truncated: bool | None = None,
    ):
        self.cum_reward = reward
        self.total_time_seconds = total_time_seconds
        self.episode_completed = episode_completed
        self.terminated = terminated
        self.truncated = truncated

    def _save_images_to_files(
        self,
        images: list[Any],
        images_dir: str,
        final_output_dir: str,
        img_filenames: list[str] = [],
        img_filename_template: str = "img_{img_idx}.png",
    ) -> list[str]:
        """Save images to files and return relative paths."""
        if images is None or len(images) == 0:
            return []

        # Regularize arguments
        if img_filenames and len(img_filenames) != len(images):
            raise ValueError(f"Number of image filenames ({len(img_filenames)}) does not match number of images ({len(images)})")

        if not img_filenames:
            img_filenames = [img_filename_template.format(img_idx=img_idx) for img_idx in range(len(images))]

        # Convert images to files
        saved_paths: list[str] = []
        try:
            for img_idx, img in enumerate(images):
                img_name = img_filenames[img_idx]
                img_path = os.path.join(images_dir, img_name)
                saved_path = any_to_path(img, out_path=img_path, overwrite=True)
                saved_paths.append(os.path.relpath(saved_path, final_output_dir))
            return saved_paths
        except Exception as e:
            logger.error(f"Failed to save images to files: {e}", exc_info=True)
            raise e

    def _serialize_state(
        self,
        state: dict[str, Any],
        images_dir: str,
        final_output_dir: str,
        step_idx: int,
    ) -> dict[str, Any]:
        if not state:
            return {}

        # Serialize each state
        new_state: dict[str, Any] = {k: v for k, v in state.items() if k != "images"}
        # Convert images to local paths relative to output_dir
        state_img_paths = self._save_images_to_files(
            state.get("images", []),
            images_dir,
            final_output_dir,
            img_filename_template=f"step_{step_idx}_img_{{img_idx}}.png",
        )
        new_state["images"] = state_img_paths

        return new_state

    def _serialize_actions(self, actions: dict[str, Any], images_dir: str, final_output_dir: str, step_idx: int) -> dict[str, Any]:
        if not actions:
            return {}

        actions_per_agent = actions.get("agents", {})
        new_actions_per_agent: dict[str, Any] = {}
        for agent_id, atomic_actions in actions_per_agent.items():
            new_atomic_list: list[dict[str, Any]] = []
            for aa_idx, atomic in enumerate(atomic_actions or []):
                new_atomic = {k: v for k, v in atomic.items() if k != "generated_images"}
                new_atomic["generated_images"] = self._save_images_to_files(
                    atomic.get("generated_images", []),
                    images_dir,
                    final_output_dir,
                    img_filename_template=f"step_{step_idx}_action_{agent_id}_{aa_idx}_img_{{img_idx}}.png",
                )
                new_atomic_list.append(new_atomic)
            new_actions_per_agent[agent_id] = new_atomic_list

        new_actions = {
            "agents": new_actions_per_agent,
            "metadata": actions.get("metadata", {}),
        }

        return new_actions

    def dump_to_json(self, out_file_path: str | None = None):
        if out_file_path is None:
            out_file_path = self.file_path

        if not out_file_path.endswith(".json"):
            out_file_path = f"{out_file_path}.json"

        if not out_file_path:
            raise ValueError("Error dumping trajectory: missing output file path")

        # Determine output directory preference
        out_dir = os.path.dirname(out_file_path)

        os.makedirs(out_dir, exist_ok=True)
        images_dir = os.path.join(out_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Build a JSON-serializable copy with image paths
        serializable_objective = {k: v for k, v in self.objective.items() if k != "images"}
        serializable_objective["images"] = []

        for img_idx, img in enumerate(self.objective.get("images", [])):
            if isinstance(img, str) and os.path.exists(img):
                img_path = img
            else:
                img_path = self._save_images_to_files(
                    [img],
                    images_dir,
                    out_dir,
                    img_filename_template=f"objective_img_{img_idx}.png",
                )
            serializable_objective["images"].append(img_path)

        serializable_steps: list[dict[str, Any]] = []
        for step_idx, step in enumerate(self.steps):
            new_step: dict[str, Any] = {k: v for k, v in step.items() if k != "state" and k != "actions"}
            new_step["state"] = self._serialize_state(step.get("state", {}), images_dir, out_dir, step_idx)

            new_step["actions"] = self._serialize_actions(step.get("actions", {}), images_dir, out_dir, step_idx)
            serializable_steps.append(new_step)

        serializable = {k: self._make_json_serializable(v) for k, v in self.__dict__.items() if k not in ["objective", "steps"]}
        serializable["steps"] = serializable_steps
        serializable["metadata"] = self.metadata

        with open(out_file_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Trajectory dumped to {out_file_path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "objective": self.objective,
            "args": self._make_json_serializable(self.args),
            "config": self._make_json_serializable(self.config),
            "steps": self.steps,
            "cum_reward": self.cum_reward,
            "experiment_name": self.experiment_name,
            "datetime": self.datetime,
            "total_time_seconds": self.total_time_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VWA_Trajectory":
        """
        Create a VWA_Trajectory object from a dictionary.

        Args:
            data: Dictionary containing trajectory data

        Returns:
            VWA_Trajectory object
        """
        objective = data.get("objective", {})
        objective_text = objective.get("text", "")
        objective_imgs = objective.get("images", [])

        # Create the trajectory object
        trajectory = cls(
            task_id=data.get("task_id", ""),
            objective_text=objective_text,
            objective_imgs=objective_imgs,
            domain=data.get("domain", ""),
            args=data.get("args"),
            config=data.get("config"),
            experiment_name=data.get("experiment_name"),
            config_file=data.get("config_file", ""),
            metadata=data.get("metadata", {}),
        )

        # Set additional attributes that may have been set after initialization
        trajectory.cum_reward = data.get("cum_reward", float("nan"))
        trajectory.datetime = data.get("datetime")
        trajectory.total_time_seconds = data.get("total_time_seconds")
        trajectory.terminated = data.get("terminated")
        trajectory.truncated = data.get("truncated")
        trajectory.episode_completed = data.get("episode_completed", False)
        trajectory.info = data.get("info", {})

        # Load steps
        trajectory.steps = data.get("steps", [])

        return trajectory

    @classmethod
    def from_json(cls, json_path: str | Path) -> "VWA_Trajectory":
        """
        Create a VWA_Trajectory object from a JSON file.

        Args:
            json_path: Path to the JSON file containing trajectory data

        Returns:
            VWA_Trajectory object
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # Convert image paths to absolute paths
        for step in data["steps"]:
            for i, img in enumerate(step["state"]["images"]):
                # step["state"]["images"][i] = os.path.join(data["output_dir"], img)
                step["state"]["images"][i] = Path(json_path).parent.joinpath(img).resolve()

            for agent_id, atomic_actions in step["actions"]["agents"].items():
                for atomic_action in atomic_actions:
                    for i, img in enumerate(atomic_action["generated_images"]):
                        atomic_action["generated_images"][i] = os.path.join(data["output_dir"], img)

        if not data.get("config", {}):
            data["config"] = get_config(data.get("task_id", ""), data.get("domain", ""))

        if "objective" not in data:
            # Try to get from config file
            config_dict = data.get("config", {})
            if not config_dict:
                config_dict = get_config(data.get("task_id", ""), data.get("domain", ""))

            if config_dict:
                data["config"] = config_dict
                data["objective"] = {}
                data["objective"]["text"] = config_dict.get("intent", "")

                # Regularize image paths data type
                image_paths = config_dict.get("image", None)
                image_paths = [] if image_paths is None else image_paths
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                data["objective"]["images"] = image_paths

        else:
            for i, img in enumerate(data.get("objective", {}).get("images", [])):
                # if objective image is relative to output_dir, convert to absolute path
                try:
                    Path(img).relative_to(data["output_dir"])
                    data["objective"]["images"][i] = os.path.join(data["output_dir"], img)
                except Exception:
                    pass

        data["file_path"] = json_path
        return cls.from_dict(data)

    @classmethod
    def from_json_string(cls, json_string: str) -> "VWA_Trajectory":
        """
        Create a VWA_Trajectory object from a JSON string.

        Args:
            json_string: JSON string containing trajectory data

        Returns:
            VWA_Trajectory object
        """
        data = json.loads(json_string)
        return cls.from_dict(data)

    def annotate_images(self, state, actions, ann_types: list[str]):
        """
        Annotate the images based on the actions.
        """
        bboxes_list = []
        all_coordinates = []
        sup_ann_bbox = ["som", "bbox"]
        sup_ann_coord = ["coord", "coordinates", "dot"]

        if len(state["images"]) == 0:
            return

        if any(supported_ann in ann_types for supported_ann in sup_ann_bbox):
            bboxes_list = state["metadata"]["nodes_info_image_observations"]

        if any(supported_ann in ann_types for supported_ann in sup_ann_coord):
            agents = actions.get("agents", {})

            # Collect all coordinates from all agents/actions for this step
            for agent_id, atomic_actions in agents.items():
                for atomic_action in atomic_actions:
                    coordinates = atomic_action.get("coordinates", {})
                    if coordinates and "x" in coordinates and "y" in coordinates:
                        coord_entry = {
                            "x": coordinates["x"],
                            "y": coordinates["y"],
                            "relative": True,
                        }
                        all_coordinates.append(coord_entry)

        # Ensure we have matching lengths for images, bboxes, and coordinates
        num_images = len(state["images"])

        # Pad bboxes_list to match number of images
        while len(bboxes_list) < num_images:
            bboxes_list.append(None)
        bboxes_list = bboxes_list[:num_images]

        # Create coordinates_list: put all coordinates on last image, empty on others
        if any(supported_ann in ann_types for supported_ann in ["coord", "coordinates", "dot"]) and all_coordinates:
            coordinates_list = [None] * (num_images - 1) + [all_coordinates]
        else:
            coordinates_list = [None] * num_images
        coordinates_list = coordinates_list[:num_images]

        for i, img in enumerate(state["images"]):
            annotated = annotate_image(img, bboxes_list[i], coordinates_list[i], marker_style="dot")
            state["images"][i] = annotated

    def has_verifier_calls(self):
        for step in self.steps:
            if "scores_per_round" in step["state"]:
                return True
        return False

    def get_scores_for_round_step(self, round_idx: int, call_max: int | None = None, call_min: int = 0):
        verifier_call_int = 0
        for step in self.steps:
            if verifier_call_int < call_min:
                continue
            if call_max is not None and verifier_call_int > call_max:
                break
            if "scores_per_round" in step["state"]:
                verifier_call_int += 1
                for score_entry in step["state"]["scores_per_round"]:
                    if score_entry["round"] == round_idx:
                        return score_entry["score"], score_entry
                    break
        return None, None
