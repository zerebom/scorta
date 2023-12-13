import os
import random

import numpy as np


def seed_everything(seed: int = 113) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
