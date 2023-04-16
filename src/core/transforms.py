import torchvision.transforms as torchvision_t

from src.config_and_utils.config import NormalizeImageConfig


def get_train_transforms(config: NormalizeImageConfig):
    transforms = torchvision_t.Compose([
        torchvision_t.ToTensor(),
        torchvision_t.Normalize(config.mean, config.std),
    ])

    return transforms


def get_common_transforms(config: NormalizeImageConfig):
    transforms = torchvision_t.Compose([
        torchvision_t.ToTensor(),
        torchvision_t.Normalize(config.mean, config.std),
    ])
    return transforms
