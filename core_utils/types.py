from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile

ImageInput = Union[Image.Image, str, np.ndarray[Any, Any], bytes, ImageFile, Path]
