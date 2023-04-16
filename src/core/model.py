import typing as t

from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, deeplabv3_resnet101

from src.config_and_utils.config import ModelConfig


class Model(nn.Module):
    mapper: t.Dict[str, t.Callable[[str], nn.Module]] = {
        "mbv3": deeplabv3_mobilenet_v3_large,
        "r50": deeplabv3_resnet50,
        "r101": deeplabv3_resnet101
    }

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        try:
            self.model = self.mapper[config.backbone_model](config.weights)
        except KeyError:
            raise KeyError("Wrong backbone model passed. Must be one of 'mbv3', 'r50' and 'r101'")

        self.model.classifier[4] = nn.LazyConv2d(config.n_classes, 1)
        self.model.aux_classifier[4] = nn.LazyConv2d(config.n_classes, 1)

    def forward(self, x):
        return self.model(x)
