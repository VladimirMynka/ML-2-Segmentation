import logging
import os
import re

import torch

from src.config_and_utils.config import LoggerConfig


def init_logging(config: LoggerConfig = None):
    if config is None:
        config = LoggerConfig()
    logging.basicConfig(
        filename=config.log_file,
        encoding=config.encoding,
        level=config.level,
        format=config.format,
        datefmt=config.date_format
    )


def get_last_dataset(path: os.PathLike):
    folders = [
        str(folder) for folder in os.listdir(path)
        if re.match(r"(\d+_)+(\d+)", str(folder))
    ]
    max_folder = max(folders)
    return os.path.join(path, max_folder)


def check_model(model, image_size, device):
    _ = model(torch.randn(
        (2, 3, image_size[0], image_size[1]),
        device=device
    ))
