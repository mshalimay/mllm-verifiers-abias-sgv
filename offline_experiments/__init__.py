# Import path bootstrap
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VWA_DIR = _REPO_ROOT / "vwa"
_OSW_DIR = _REPO_ROOT / "osw"
_AGRB_DIR = _REPO_ROOT / "agrb"
if _AGRB_DIR.is_dir():
    sys.path.insert(0, str(_AGRB_DIR))
if _VWA_DIR.is_dir():
    sys.path.insert(0, str(_VWA_DIR))
if _OSW_DIR.is_dir():
    sys.path.insert(0, str(_OSW_DIR))
