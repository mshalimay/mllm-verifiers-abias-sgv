import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from agrb.constants import AGRB_TO_VWA, LOCAL_INPUT_IMAGES_DIR_TEMPLATE, VWA_TASK_CONFIGS
from core_utils.file_utils import find_files
from core_utils.image_utils import any_to_pil

try:
    from agrb.constants import ANNOTATIONS_CSV
except ImportError:
    ANNOTATIONS_CSV = ""

AGRB_TO_VWA_DF = None


def map_eval_to_score(eval_str: str):
    """
    Map the evaluation string to a score.
    """
    if eval_str == "Successful":
        return 1
    elif eval_str == "Unsuccessful":
        return 0
    else:
        return np.nan


def get_human_annotation(task_id: str, benchmark: str, exp_name: str) -> dict[str, Any] | None:
    """
    Get human annotation data from the CSV file.

    Args:
        task_id: The task ID to look up
        benchmark: The benchmark name (e.g., 'visualwebarena')
        exp_name: The experiment name

    Returns:
        Dictionary containing annotation data or None if not found
    """
    try:
        annotations_df = pd.read_csv(ANNOTATIONS_CSV)

        # Find matching annotation based on task_id, benchmark, and exp_name
        matching_annotation = annotations_df[(annotations_df["task_id"] == task_id) & (annotations_df["benchmark"] == benchmark) & (annotations_df["exp_name"] == exp_name)]

        if len(matching_annotation) > 0:
            mean_score = matching_annotation["trajectory_success"].map(map_eval_to_score).mean()
            return {
                "human_mean_score": mean_score,
            }
    except Exception as e:
        print(f"Error getting human annotation for task_id: {task_id}, benchmark: {benchmark}, exp_name: {exp_name}: {e}")
        return None


def annotate_action_on_image(img, bboxes, marker_style="dot", dot_radius=8, square_side=20, min_width=8, min_height=8) -> tuple[Image.Image, list[tuple[int, int]]]:
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
    draw = ImageDraw.Draw(img)
    if not isinstance(bboxes, list):
        bboxes = [bboxes]

    centers = []
    for bbox in bboxes:
        if not bbox.get("bbox"):
            continue

        if not bbox.get("set_of_marks"):
            continue

        # Expect [x, y, w, h]
        x, y, w, h = bbox["bbox"]
        if w < min_width or h < min_height:
            continue
        if bbox.get("visibility", 1.0) <= 0.0:
            continue

        # Color + geometry
        left, top = float(x), float(y)
        right, bottom = left + float(w), top + float(h)

        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

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

        centers.append((center_x, center_y))

    return img, centers


# ---- draw bboxes on screenshot ----
def write_bbox_to_screenshot(
    screenshot,
    bbox_entries: dict,
    out_path=None,
    min_width=8,
    min_height=8,
    padding=0,
    border=2,
    add_ids=True,
):
    """
    bbox_entries: dict of {id: {'bbox':[x,y,w,h], 'visibility':float, 'clickable':bool, ...}}
    """
    img = any_to_pil(screenshot)
    draw = ImageDraw.Draw(img)

    # Font: try your SourceCodePro, otherwise default
    try:
        font = ImageFont.truetype("media/SourceCodePro-SemiBold.ttf", 16)
        font_size = 16
    except Exception:
        font = ImageFont.load_default()
        font_size = 12

    # Matplotlib categorical color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Iterate and draw
    i = 0
    for bbox_id, spec in bbox_entries.items():
        if not spec.get("bbox"):
            continue

        if not spec.get("set_of_marks"):
            continue

        # Expect [x, y, w, h]
        x, y, w, h = spec["bbox"]
        if w < min_width or h < min_height:
            continue
        if spec.get("visibility", 1.0) <= 0.0:
            continue

        # Color + geometry
        color = color_cycle[i % len(color_cycle)]
        left, top = float(x), float(y)
        right, bottom = left + float(w), top + float(h)

        # outline rectangle
        draw.rectangle(
            [left - padding, top - padding, right + padding, bottom + padding],
            outline=color,
            width=border,
        )

        # label (1-based)
        if add_ids:
            # label = str(i + 1)
            label = bbox_id
            # try to place above box; if not enough room, place inside top-left
            tw, th = draw.textlength(label, font=font), font_size
            tx, ty = left, max(0, top - th - 2)
            # background rect for legibility
            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=color)
            draw.text((tx + 2, ty + 1), label, font=font, fill="white")

        i += 1

    return img
    # # save
    # if out_path is None:
    #     stem, ext = os.path.splitext(screenshot_img_path)
    #     out_path = f"{stem}_with_boxes.png"
    # img.save(out_path)
    # return out_path


def load_trajectory_data(trajectory_path: str) -> Dict:
    """
    Load trajectory data from a JSON file.

    Args:
        trajectory_path: Path to the trajectory JSON file

    Returns:
        Dictionary containing trajectory data
    """
    with open(trajectory_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def find_trajectory_files(base_dir: str, filename_pattern: str = "*.json") -> List[str]:
    """
    Find trajectory files using utils.file_utils.find_files if available, otherwise fallback.

    Args:
        base_dir: Base directory to search
        filename_pattern: Pattern to match filenames

    Returns:
        List of file paths
    """
    return find_files(start_dir=base_dir, filename=filename_pattern, downwards=True)


def update_task_intent(
    vwa_domain: str,
    vwa_task_id: str,
) -> str:
    try:
        with open(VWA_TASK_CONFIGS, "r", encoding="utf-8") as f:
            vwa_tasks = json.load(f)
        for task in vwa_tasks:
            if task["domain"] == vwa_domain and str(task["task_id"]) == str(vwa_task_id):
                return task["intent"]
        return ""
    except Exception as e:
        print(f"Error updating task intent for domain: {vwa_domain}, task_id: {vwa_task_id}: {e}")
        return ""


def get_vwa_domain_task_id(agrb_task_id: str, experiment_name: str) -> tuple[str, str]:
    global AGRB_TO_VWA_DF
    if AGRB_TO_VWA_DF is None:
        AGRB_TO_VWA_DF = pd.read_csv(AGRB_TO_VWA)
    df = AGRB_TO_VWA_DF
    # Filter by experiment_name
    # df = df[df["experiment_name"] == experiment_name]
    # Filter by trajectory_path
    df = df[df["task_id"] == agrb_task_id]
    df = df[df["exp_name"] == experiment_name]
    df_domain_task_id = df[["domain_task_id"]]
    domain, task_id = df_domain_task_id.iloc[0].values[0].split("_")
    return domain, task_id


def get_local_input_image_path(agrb_task_id: str, exp_name: str) -> list[Path]:
    domain, task_id = get_vwa_domain_task_id(agrb_task_id, exp_name)
    input_images_dir = LOCAL_INPUT_IMAGES_DIR_TEMPLATE.format(domain=domain, task_id=task_id)
    # Get all png files in the directory
    input_image_paths = []
    for file in os.listdir(input_images_dir):
        file_path = Path(input_images_dir) / file
        # is file
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif"}:
            input_image_paths.append(file_path)
    return input_image_paths
