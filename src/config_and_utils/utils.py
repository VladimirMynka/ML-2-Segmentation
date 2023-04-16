import os
import re

import torch


def get_last_dataset(path: os.PathLike):
    folders = [
        str(folder) for folder in os.listdir(path)
        if re.match(r"(\d+_)+(\d+)", str(folder))
    ]
    max_folder = max(folders)
    return os.path.join(path, max_folder)


def check_model(model, image_size, device):
    _ = model(torch.randn(
        (2, 3, image_size, image_size),
        device=device
    ))

