import json
import traceback
from pathlib import Path

from core_utils.file_utils import find_files, get_common_paths
from core_utils.image_utils import any_to_pil
from core_utils.logger_utils import logger
from llms.translate import batch_to_english

TASK_CONFIGS_PATH = Path("osw_traces/task_configs")


def load_evaluator_info(domain, task_id):
    """Load evaluator info from the task config JSON, or return None if not found."""
    config_path = TASK_CONFIGS_PATH / domain / f"{task_id}.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("evaluator")
    except Exception as e:
        print(f"Error loading evaluator for {domain}/{task_id}: {e}\n{traceback.format_exc()}")
        return None


def annotate_action_on_image(img, action, marker_style="dot", dot_radius=8, square_side=20):
    """
    Annotate the given image based on the click action.

    This function looks for one or more occurrences of 'start_box' in the action string.
    For each found box, the coordinates are extracted and processed:
      - If the box is specified as a 4-tuple, it's interpreted as [x1, y1, x2, y2],
        and the marker is placed at the rectangle's center.
      - If the box is specified as a 2-tuple, that point is considered the center.

    Depending on the marker_style:
      - "square" draws a square centered at the computed location.
      - "dot" draws a filled red dot at the location.

    Parameters:
        img: A PIL Image to annotate.
        action: A string containing one or more start_box definitions.
        marker_style: "square" or "dot" (default) for a filled red dot.
    """
    import ast
    import re

    from PIL import ImageDraw

    # Find all occurrences of start_box in the action string.
    box_matches = re.findall(r"start_box\s*=\s*[\"']([^\"']+)[\"']", action)
    if not box_matches:
        return img

    draw = ImageDraw.Draw(img)
    for box_str in box_matches:
        try:
            # Safely evaluate the start_box string.
            start_box = ast.literal_eval(box_str)
        except Exception as e:
            # Skip this box if evaluation fails.
            continue

        if isinstance(start_box, (list, tuple)):
            if len(start_box) == 4:
                x1, y1, x2, y2 = start_box  # e.g., a rectangle [x1, y1, x2, y2]
            elif len(start_box) == 2:
                x1, y1 = start_box
                x2, y2 = x1, y1
            else:
                continue  # Skip if it's not a 2-tuple or a 4-tuple.

            # Retrieve image dimensions.
            im_width, im_height = img.width, img.height

            # Determine if the coordinates are normalized (0..1 range) and if so, scale them.
            if max(x1, x2, y1, y2) <= 1:
                center_x = round(((x1 + x2) / 2) * im_width)
                center_y = round(((y1 + y2) / 2) * im_height)
            else:
                center_x = round((x1 + x2) / 2)
                center_y = round((y1 + y2) / 2)

            # Draw marker based on the specified style.
            if marker_style == "square":
                half_side = square_side // 2
                draw.rectangle(
                    [(center_x - half_side, center_y - half_side), (center_x + half_side, center_y + half_side)],
                    outline="red",
                    width=3,
                )
            elif marker_style == "dot":
                draw.ellipse(
                    [(center_x - dot_radius, center_y - dot_radius), (center_x + dot_radius, center_y + dot_radius)],
                    fill="red",
                    outline="red",
                )

    return img


def trace_to_english(trace_path: str | Path = "", trace_data: dict | None = None):
    if not trace_path and trace_data is None:
        raise ValueError("Either trace_path or trace_data must be provided.")

    if trace_data is None:
        trace_data = json.load(open(trace_path))

    if not trace_data and not trace_path:
        raise ValueError("Either trace_data or trace_path must be provided.")

    # Collect only steps that don't already have a translation.
    translatable_texts = []
    indices = []

    assert trace_data is not None
    for idx, step in enumerate(trace_data["steps"]):
        if not step.get("generated_text_en"):
            translatable_texts.append(step["generated_text"])
            indices.append(idx)

    if translatable_texts:
        translated_texts = batch_to_english(translatable_texts)
        if translated_texts is None:
            trace_path = trace_data.get("trajectory_path", "")
            logger.warning(f"{__file__}: Failed to translate texts while loading trace from {trace_path}.")
            return trace_data

        for idx, translated_text in zip(indices, translated_texts):
            trace_data["steps"][idx]["generated_text_en"] = translated_text

        # Write to file if a trace_path was provided.
        traj_path = trace_path or trace_data["trajectory_path"]
        if traj_path:
            with open(traj_path, "w") as f:
                json.dump(trace_data, f, indent=4)

    return trace_data


class TrajectoryView:
    def __init__(self, trace_data=None, trajectory_path: str | Path = "", to_english=False):
        self._states: list = []
        self._actions: list = []

        if trace_data:
            self.base_path = trace_data.get("base_path", "")
            self.trajectory = trace_data

        elif trajectory_path:
            self.base_path = Path(trajectory_path).parent
            with open(trajectory_path, "r") as f:
                self.trajectory = json.load(f)
        else:
            raise ValueError("Either trace_data or trajectory_path must be provided.")

        self.score = self.trajectory.get("score", None)
        self.domain = self.trajectory.get("domain", "")
        self.objective = self.trajectory.get("objective", None)
        self.task_id = self.trajectory.get("task_id", "")
        if to_english:
            try:
                trace_to_english(trace_path=trajectory_path, trace_data=trace_data)
            except Exception as e:
                logger.warning(f"{__file__}: Failed to translate texts while loading trace from {trajectory_path}: {str(e)}")
                # print(f"{__file__}: Failed to translate texts while loading trace from {trajectory_path}.")

        self.build_trajectory_view(trajectory_data=self.trajectory)

    def parse_thought_action(self, model_id="ui_tars"):
        if "ui_tars" in model_id:
            pass
        # TODO

    def get_images(self, imgs_rel_paths: list[str] | str | Path):
        imgs = []
        if not isinstance(imgs_rel_paths, list):
            imgs_rel_paths = [imgs_rel_paths]  # type:ignore

        for img_rel_path in imgs_rel_paths:  # type:ignore
            img_path = self.base_path / img_rel_path
            imgs.append(any_to_pil(img_path))
        return imgs

    def build_trajectory_view(self, trajectory_data: dict = {}):
        traj_data = self.trajectory if not trajectory_data else trajectory_data

        self._states = []
        self._actions = []

        images0 = self.get_images(traj_data["initial_screenshot"])
        self._states.append(
            {
                "observation": {"images": images0, "texts": ""},
            }
        )

        self._actions.append(
            {"texts": [traj_data["steps"][0].get("generated_text")], "images": []},
        )

        steps = traj_data["steps"]
        for i in range(1, len(steps)):
            state_dict = {}
            state_dict["observation"] = {"images": self.get_images(steps[i - 1]["screenshots"]), "texts": []}
            if steps[i].get("verifier_loop"):
                state_dict["verifier_loop"] = steps[i].get("verifier_loop", False)
            self._states.append(state_dict)

            if steps[i].get("generated_text_en"):
                self._actions.append({"texts": [steps[i]["generated_text_en"]], "images": []})
            else:
                self._actions.append({"texts": [steps[i]["generated_text"]], "images": []})

        return self._states, self._actions

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions


def load_trajectory_data(trajectory_path: str):
    with open(trajectory_path, "r") as f:
        data = json.load(f)
    return data


def get_experiments_dirs(base_dir, traj_file_pattern: str = "trajectory.json"):
    """Returns a list of experiments dirs and a dict of source_dir -> list of trajectory files"""
    traj_files = find_files(start_dir=base_dir, filename=traj_file_pattern, downwards=True)

    # Get shortest common paths = experiments dirs
    _, source_dirs, _ = get_common_paths(traj_files, relative_to=base_dir)

    source_to_files = {}
    for source_dir in source_dirs:
        source_to_files[source_dir] = find_files(start_dir=source_dir, filename=traj_file_pattern, downwards=True)

    return source_dirs, source_to_files
