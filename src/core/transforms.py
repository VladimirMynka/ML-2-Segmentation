import torchvision.transforms as torchvision_t

from src.config_and_utils.config import TransformsConfig


def get_train_transforms(config: TransformsConfig):
    transforms = torchvision_t.Compose([
        torchvision_t.ToTensor(),
        torchvision_t.Normalize(config.mean, config.std),
    ])

    return transforms


def get_common_transforms(config: TransformsConfig):
    transforms = torchvision_t.Compose([
        torchvision_t.ToTensor(),
        torchvision_t.Normalize(config.mean, config.std),
    ])
    return transforms
