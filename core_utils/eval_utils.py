import random

import numpy as np
import torch

# ===============================================================================
# Other helpers
# ===============================================================================


def set_seed(seed: int | None = None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
