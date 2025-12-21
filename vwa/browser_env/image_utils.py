# NOTE[mandrade]: added

import base64
import os
import re
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import requests  # type: ignore
from numpy.typing import NDArray
from PIL import Image
from PIL.ImageFile import ImageFile

ImageInput = Union[Image.Image, str, np.ndarray[Any, Any], bytes, ImageFile, Path]


VALID_IMAGE_FORMATS = ["JPG", "JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "TIF"]


# ===============================================================================
# LINK Type checking
# ===============================================================================
def is_url(img: str) -> bool:
    return img.startswith("http")


def is_image(img: Any, return_image: bool = False) -> bool | Image.Image | None:
    try:
        pil_img = any_to_pil(img)
        return pil_img if return_image else True
    except Exception:
        return False if not return_image else None


def is_b64_image(image: Any, return_image: bool = False) -> bool | Image.Image | None:
    if not isinstance(image, str):
        return False if not return_image else None
    # Accept a broad set of MIME subtypes (letters, digits, '+', '.', '-')
    if re.match(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", image):
        if return_image:
            return b64_to_pil(image)
        else:
            return True
    try:
        decoded = base64.b64decode(image)
        pil_img = Image.open(BytesIO(decoded))
        pil_img.load()
        return pil_img if return_image else True
    except Exception:
        return False if not return_image else None


def is_string(obj: Any) -> bool:
    return isinstance(obj, str) and not is_image(obj)


def is_path_img(img: Any) -> bool:
    if isinstance(img, (str, Path)) and os.path.exists(str(img)):
        try:
            _ = Image.open(img)
            return True
        except Exception:
            return False
    return False


def fmt_to_mime(fmt: str | None) -> str:
    return {"JPG": "jpeg", "JPEG": "jpeg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tiff", "TIF": "tiff"}.get(
        (fmt or "").upper(), (fmt or "png").lower()
    )


def validate_image_format(fmt: str | None) -> bool:
    if not fmt:
        return False
    return fmt.upper() in VALID_IMAGE_FORMATS


def infer_format(img: ImageInput) -> str | None:
    try:
        img = any_to_pil(img)
        return img.format
    except Exception:
        return None


def get_mime_type(img: ImageInput, fallback: str = "PNG") -> str:
    fmt = infer_format(img)
    if not fmt:
        return fallback
    return fmt_to_mime(fmt)


# ===============================================================================
# LINK Image conversion
# ===============================================================================
def get_image_from_url(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> Image.Image:
    if not headers:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
    response = requests.get(url, stream=True, headers=headers, timeout=timeout)
    return Image.open(response.raw)  # type: ignore


def numpy_to_pil(img_np: NDArray[Any]) -> Image.Image:
    return Image.fromarray(img_np)


def b64_to_pil(img_b64: str) -> Image.Image:
    # If there's a comma, split and decode only the base64 part
    # i.e., "data:image/png;base64,<...>"
    if img_b64.startswith("data:image"):
        _, base64_data = img_b64.split(",", 1)
    else:
        base64_data = img_b64
    decoded = base64.b64decode(base64_data)
    return Image.open(BytesIO(decoded))


def any_to_pil(img: ImageInput) -> Image.Image:
    # If image is a numpy array, convert to PIL.
    if isinstance(img, Image.Image):
        return img

    elif isinstance(img, np.ndarray):
        return Image.fromarray(img)

    elif is_path_img(img):
        return Image.open(img)

    elif isinstance(img, str):
        # If image is a base64 string, convert to PIL.
        _tmp = is_b64_image(img, return_image=True)
        if isinstance(_tmp, Image.Image):
            return _tmp

        elif is_url(img):
            return get_image_from_url(img)

        else:
            raise ValueError(f"Invalid image string: {img}")

    # If image is a bytes object, convert to PIL.
    elif isinstance(img, (bytes, bytearray)):
        return Image.open(BytesIO(img))

    else:
        raise ValueError(f"Invalid image type: {type(img)}")


def any_to_b64(img: ImageInput, add_header: bool = True, format: str | None = None) -> str:
    """Convert any image input to a base64-encoded string.
    Args:
        img (ImageInput): The image input to convert.
        add_header (bool, optional): Whether to add the "data:image/..." header. Defaults to True.
        format (str | None, optional): Image format to use for conversion. If not provided, try to infer from the image or fall back to PNG. Defaults to None.
    Raises:
        ValueError: If the image cannot be converted to base64.
    Returns:
        str: The image as a base64-encoded string.
    """

    if img is None:
        raise ValueError("No image provided for conversion to base64")

    if not validate_image_format(format):
        format = None

    pil_img: Image.Image | None = None

    if isinstance(img, str):
        # If b64 image already, add/remove header as needed.
        if is_b64_image(img):
            has_header = bool(re.match(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", img))
            if add_header:
                # If it already has a header, preserve as-is
                if has_header:
                    return img
                # Else add a header to raw base64 payload. Try to infer format if none provided.
                if format:
                    return f"data:image/{fmt_to_mime(format)};base64," + img
                else:
                    fmt = fmt_to_mime(infer_format(img) or "PNG")
                    return f"data:image/{fmt};base64," + img
            else:
                # Strip header if present; else return raw payload
                if has_header:
                    _, b64_data = img.split(",", 1)
                    return b64_data
                return img

        # If image is a URL, download it and convert to PIL.
        elif is_url(img):
            pil_img = get_image_from_url(img)

        # If image is a path, open it as a PIL image.
        elif is_path_img(img):
            p = Path(img)
            raw = p.read_bytes()
            fmt = infer_format(img) or "PNG"
            payload = base64.b64encode(raw).decode("ascii")
            if add_header:
                return f"data:image/{fmt_to_mime(fmt)};base64," + payload
            else:
                return payload

        else:
            raise ValueError(f"Invalid image string: {img}")

    # If image is a numpy array, convert to PIL.
    elif isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)

    # If image is a bytes object, convert to PIL.
    elif isinstance(img, (bytes, bytearray)):
        raw = bytes(img)
        try:
            fmt = Image.open(BytesIO(raw)).format or "PNG"
        except Exception:
            fmt = "PNG"
        payload = base64.b64encode(raw).decode("ascii")
        if add_header:
            return f"data:image/{fmt_to_mime(fmt)};base64," + payload
        else:
            return payload

    # If image is a PIL image, return it as is.
    elif isinstance(img, Image.Image):
        pil_img = img

    else:
        # Fallback: try to convert any other type to PIL.
        pil_img = any_to_pil(img)

    if not pil_img:
        raise ValueError("Failed to convert image to base64")

    # Convert PIL image to base64.
    with BytesIO() as image_buffer:
        img_format = format or pil_img.format or "PNG"
        pil_img.save(image_buffer, format=img_format)
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        if add_header:
            img_b64 = f"data:image/{fmt_to_mime(img_format)};base64," + img_b64
    return img_b64


def any_to_bytes(img: ImageInput, format: str | None = None) -> bytes:
    """Convert any image input to bytes.

    Args:
        img (ImageInput): The image input to convert.
        format (str, optional): Image format to use for conversion. If not provided, try to infer from the image or fall back to PNG. Defaults to "".

    Raises:
        ValueError: If the image cannot be converted to bytes.

    Returns:
        bytes: The image data as bytes.
    """
    if img is None:
        raise ValueError("No image provided for conversion to bytes")

    if not validate_image_format(format):
        format = None

    format_change = False
    if format:
        orig_format = infer_format(img)
        format_change = orig_format != format

    if isinstance(img, (bytes, bytearray)) and not format_change:
        return bytes(img)

    # If input is a file path, return original file bytes
    if is_path_img(img) and not format_change:
        return Path(img).read_bytes()  # type: ignore

    # If input is a base64 string (with or without header), decode and return raw bytes
    if isinstance(img, str) and is_b64_image(img) and not format_change:
        has_header = bool(re.match(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", img))
        if has_header:
            _, payload = img.split(",", 1)
            return base64.b64decode(payload)
        else:
            return base64.b64decode(img)

    # Convert remaining cases to a PIL Image and then to bytes (re-encode).
    pil_img = any_to_pil(img)

    with BytesIO() as image_buffer:
        # Honor requested format if provided; else prefer the PIL image's format; else infer; else PNG
        img_format = format or (getattr(pil_img, "format", None) or infer_format(img) or "PNG")
        pil_img.save(image_buffer, format=img_format)
        return image_buffer.getvalue()


def any_to_path(img: ImageInput, out_path: str = "", overwrite: bool = True) -> str:
    pil_img: Image.Image | None = None
    # If is path and image exists
    if isinstance(img, Path) and img.exists():
        # If path and out_path, save the image in `img` to `out_path`
        if out_path and out_path != str(img):
            shutil.copy(img, out_path)
            return out_path
        else:
            return str(img)

    # If is path and image does not exist
    elif isinstance(img, Path) and not img.exists():
        raise FileNotFoundError(f"Image file not found: {img}")

    # If string image
    elif isinstance(img, str):
        tmp_img = is_b64_image(img, return_image=True)  # may be Image or None
        if isinstance(tmp_img, Image.Image):
            pil_img = tmp_img
            # If image is a base64 string, convert to PIL for later processing
            pass

        elif os.path.exists(img):
            # If string path and out_path, save the image in `img` to `out_path`
            if out_path and out_path != img:
                shutil.copy(img, out_path)
                return out_path
            else:
                return img

        elif is_url(img):
            # If string url, convert to PIL for later processing
            pil_img = get_image_from_url(img)

        else:
            raise ValueError(f"Invalid image string: {img}")

    # If image is a PIL image
    elif isinstance(img, Image.Image):
        pil_img = img

    # If image is a bytes object, convert to PIL.
    elif isinstance(img, bytes):
        pil_img = Image.open(BytesIO(img))

    else:
        # If image is a numpy array or other type, convert to PIL for later processing
        try:
            pil_img = any_to_pil(img)
        except Exception:
            raise ValueError(f"Invalid image type: {type(img)}")

    if pil_img is None:
        raise ValueError(f"Failed to convert image to path")

    else:
        if not out_path:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file_path = temp_file.name
            pil_img.save(temp_file_path)
            return temp_file_path
        elif out_path and overwrite:
            pil_img.save(out_path)
            return out_path
        else:
            if Path(out_path).exists():
                raise FileExistsError(f"Image file already exists: {out_path}. Use `overwrite=True` to overwrite.")
            else:
                pil_img.save(out_path)
                return out_path
