from typing import Union

from .actions import Action
from .env_utils import StateInfo

Trajectory = list[Union[StateInfo, Action]]
