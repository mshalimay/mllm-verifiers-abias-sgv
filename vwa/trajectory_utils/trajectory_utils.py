from typing import Any

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from core_utils.image_utils import any_to_pil


def _compute_local_luminance(img: Image.Image, x: float, y: float, sample_radius: int = 6) -> float:
    """Approximate local luminance around (x, y) using grayscale (L mode)."""
    lum_img = img.convert("L")
    cx, cy = int(x), int(y)
    left = max(cx - sample_radius, 0)
    top = max(cy - sample_radius, 0)
    right = min(cx + sample_radius, lum_img.width - 1)
    bottom = min(cy + sample_radius, lum_img.height - 1)

    if right <= left or bottom <= top:
        val = lum_img.getpixel((min(max(cx, 0), lum_img.width - 1), min(max(cy, 0), lum_img.height - 1)))
        if isinstance(val, tuple):
            if len(val) == 0:
                return 0.0
            return float(sum(val) / len(val))
        if val is None:
            return 0.0
        return float(val)

    region = lum_img.crop((left, top, right, bottom))
    pixels = list(region.getdata())
    if not pixels:
        return 0.0
    total = 0.0
    for v in pixels:
        if isinstance(v, tuple):
            if len(v) == 0:
                continue
            total += float(sum(v) / len(v))
        elif v is None:
            continue
        else:
            total += float(v)
    return total / len(pixels)


def _auto_contrast_bw(img: Image.Image, x: float, y: float) -> str:
    """Return 'white' for dark backgrounds, else 'black'."""
    return "white" if _compute_local_luminance(img, x, y) < 128 else "black"


def annotate_marker_on_image(
    img,
    coordinates,
    marker_style="dot",
    dot_radius=8,
    square_side=20,
    marker_color: str = "#FF00FF",  # magenta
    add_halo: bool = True,
    halo_width: int = 3,
    halo_color: str | None = None,
) -> tuple[Image.Image, Any]:
    """
    Annotate one or more point markers on the image.

    coordinates can be a single dict or a list of dicts, each with keys:
      - x, y: location of the point
      - relative (optional): if True, x and y are in [0,1] relative to image size; otherwise pixels

    marker_style: "square" draws a square centered at the point; "dot" draws a filled dot.
    marker_color: accepts any PIL color string (e.g., '#FF00FF', 'cyan'). Default is bright magenta for visibility.
    add_halo: draw a black/white halo around the marker to maintain visibility on any background.
    halo_width: thickness of the halo stroke.
    halo_color: override the halo color; if None, auto-selects black or white based on local luminance.
    """
    img = any_to_pil(img)
    draw = ImageDraw.Draw(img)

    if not coordinates:
        return img, coordinates

    coords_list: list[dict]
    if isinstance(coordinates, dict):
        coords_list = [coordinates]
    else:
        coords_list = list(coordinates)

    for coord in coords_list:
        if not coord:
            continue
        x = coord.get("x")
        y = coord.get("y")
        if x is None or y is None:
            continue

        if coord.get("relative"):
            x = float(x) * img.width
            y = float(y) * img.height
        else:
            x = float(x)
            y = float(y)

        # Check if within image bounds in Y axis; if not, offset a bit until it is
        while y < 0 or y > img.height:
            y = y + 0.01 * (1 if y < 0 else -1)

        while x < 0 or x > img.width:
            x = x + 0.01 * (1 if x < 0 else -1)

        # Halo color selection
        effective_halo_color = halo_color if halo_color is not None else _auto_contrast_bw(img, x, y)

        if marker_style == "square":
            half_side = square_side // 2
            left = x - half_side
            top = y - half_side
            right = x + half_side
            bottom = y + half_side

            if add_halo and halo_width > 0:
                draw.rectangle(
                    [(left, top), (right, bottom)],
                    outline=effective_halo_color,
                    width=max(1, halo_width * 2),
                )
            draw.rectangle(
                [(left, top), (right, bottom)],
                outline=marker_color,
                width=3,
            )
        elif marker_style == "dot":
            # dot
            if add_halo and halo_width > 0:
                draw.ellipse(
                    [
                        (x - dot_radius, y - dot_radius),
                        (x + dot_radius, y + dot_radius),
                    ],
                    outline=effective_halo_color,
                    width=max(1, halo_width * 2),
                )
            draw.ellipse(
                [
                    (x - dot_radius, y - dot_radius),
                    (x + dot_radius, y + dot_radius),
                ],
                fill=marker_color,
                outline=marker_color,
            )
        else:
            raise NotImplementedError(f"Marker style {marker_style} not implemented")

    return img, coordinates


# ---- draw bboxes on screenshot ----
def write_bbox_to_screenshot(
    screenshot,
    bbox_entries: dict,
    bbox_padding=0,
    bbox_border=2,
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

    # Prepare to avoid overlapping labels and draw after boxes
    existing_text_rectangles: list[list[float]] = []
    text_to_draw: list[tuple[list[float], tuple[float, float], str, str]] = []

    # Iterate and draw
    i = 0
    for bbox_id, spec in bbox_entries.items():
        if not isinstance(spec, dict):
            continue

        if not spec.get("bbox"):
            continue

        # Expect [x, y, w, h] in pixels unless 'relative' is True
        left, top, right, bottom = spec["bbox"]

        if spec.get("visibility", 1.0) <= 0.0:
            continue

        # Color + geometry
        color = color_cycle[i % len(color_cycle)]

        # outline rectangle
        draw.rectangle(
            [left - bbox_padding, top - bbox_padding, right + bbox_padding, bottom + bbox_padding],
            outline=color,
            width=bbox_border,
        )

        # label (1-based)
        if add_ids:
            label = bbox_id
            # font metrics
            padding = 2
            text_width = draw.textlength(label, font=font)
            text_height = font_size

            # Candidate positions around the box
            text_positions = [
                (left - font_size, top - font_size),  # top-left
                (left, top - font_size),  # near top-left
                (right, top - font_size),  # top-right
                (right - font_size - 2 * padding, top - font_size),  # near top-right
                (left - font_size, bottom),  # bottom-left
                (left, bottom),  # near bottom-left
                (right - font_size - 2 * padding, bottom),  # near bottom-right
            ]

            # Viewport is the image
            viewport_width, viewport_height = img.width, img.height

            chosen_position: tuple[float, float] | None = None
            chosen_rect: list[float] | None = None

            for tx, ty in text_positions:
                new_text_rectangle = [
                    tx - padding,
                    ty - padding,
                    tx + text_width + padding,
                    ty + text_height + padding,
                ]
                # within viewport
                if not (new_text_rectangle[0] >= 0 and new_text_rectangle[1] >= 0 and new_text_rectangle[2] <= viewport_width and new_text_rectangle[3] <= viewport_height):
                    continue

                # no overlap with existing labels
                overlaps = False
                for ex in existing_text_rectangles:
                    # rectangles overlap check with small padding
                    if not (new_text_rectangle[2] < ex[0] - padding or new_text_rectangle[0] > ex[2] + padding or new_text_rectangle[3] < ex[1] - padding or new_text_rectangle[1] > ex[3] + padding):
                        overlaps = True
                        break
                if not overlaps:
                    chosen_position = (tx, ty)
                    chosen_rect = new_text_rectangle
                    break

            # fallback if none fit
            if chosen_position is None or chosen_rect is None:
                tx, ty = text_positions[0][0] + padding, text_positions[0][1]
                chosen_position = (tx, ty)
                chosen_rect = [
                    tx - padding,
                    ty - padding,
                    tx + text_width + padding,
                    ty + text_height + padding,
                ]

            existing_text_rectangles.append(chosen_rect)
            text_to_draw.append((chosen_rect, chosen_position, str(label), color))

        i += 1

    # Draw labels after placing all boxes
    for rect, pos, label, color in text_to_draw:
        draw.rectangle(rect, fill=color)
        draw.text(pos, label, font=font, fill="white")

    return img


def annotate_image(
    image: Any,
    bbox: dict | None = None,
    coordinates: dict | list[dict] | None = None,
    marker_style: str = "dot",
    marker_color: str = "#FF00FF",
    add_halo: bool = True,
    halo_width: int = 3,
    halo_color: str | None = None,
) -> Image.Image:
    """
    Annotate the given image based on provided bounding boxes and/or coordinates.

    Args:
        image: Any image-like input (path, bytes, PIL Image, numpy array)
        bbox: dict mapping id -> {'bbox': [x,y,w,h], ...}. Coordinates are in pixels unless 'relative' is True
        coordinates: a dict or list of dicts with keys {'x','y', 'relative'?}
        marker_style: style of the marker for coordinates
        marker_color: high-visibility default is bright magenta ('#FF00FF'). Try '#FFFF00' (yellow) or '#00FFFF' (cyan) as alternatives.
        add_halo/halo_width/halo_color: options for a black/white halo to preserve visibility against any background.
    """
    image = any_to_pil(image)
    if bbox:
        image = write_bbox_to_screenshot(image, bbox)

    if coordinates:
        image, _ = annotate_marker_on_image(
            image,
            coordinates,
            marker_style=marker_style,
            marker_color=marker_color,
            add_halo=add_halo,
            halo_width=halo_width,
            halo_color=halo_color,
        )

    return image
