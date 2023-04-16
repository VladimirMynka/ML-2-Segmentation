import os
import typing as t
from logging import Logger

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.config_and_utils.config import TrainPipelineConfig
from src.config_and_utils.utils import get_last_dataset, check_model
from src.core.device_data_loader import DeviceDataLoader
from src.core.image_dataset import ImageDataset
from src.core.losses.bce_iou_loss import BCEAndIouLoss
from src.core.losses.iou_metric import IoUMetric
from src.core.model import Model
from src.core.transforms import get_train_transforms, get_common_transforms


class TrainPipeline:
    def __init__(self, logger: Logger):
        self.config = TrainPipelineConfig()
        self.data_config = self.config.data_preparation_config
        self.model_config = self.config.model_config
        self.transforms_config = self.config.transforms_config

        self.device = self.config.device

        self.logger = logger

    def run(self):
        model = Model(self.model_config).to(self.device)

        # Check all is OK
        check_model(model, self.transforms_config.image_size, self.device)

        train_loader, valid_loader = self._init_data()
        use_dice = True if self.config.metric_name == "dice" else False

        metric_fn = IoUMetric(num_classes=self.model_config.n_classes, use_dice=use_dice).to(self.device)
        loss_fn = BCEAndIouLoss(use_dice=use_dice).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        self.train(model, train_loader, valid_loader, optimizer, loss_fn, metric_fn)

    def _init_data(self) -> t.Tuple[DeviceDataLoader, DeviceDataLoader]:
        train_path_x, train_path_y, valid_path_x, valid_path_y = self._get_folders()

        train_transforms = get_train_transforms(self.transforms_config)
        valid_transforms = get_common_transforms(self.transforms_config)

        train_ds = ImageDataset(
            train_path_x, train_path_y, train_transforms,
            affine=self._parse_affine(), image_size=self.transforms_config.image_size
        )
        valid_ds = ImageDataset(
            valid_path_x, valid_path_y, valid_transforms,
            affine=None, image_size=self.transforms_config.image_size
        )

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.config.batch_size, shuffle=False)

        train_loader = DeviceDataLoader(train_loader, self.device)
        valid_loader = DeviceDataLoader(valid_loader, self.device)

        return train_loader, valid_loader

    def _get_folders(self):
        dataset_path = get_last_dataset(self.config.data_preparation_config.dataset_path)

        valid_path_x = os.path.join(dataset_path, "val", "x")
        valid_path_y = os.path.join(dataset_path, "val", "y")

        train_path_x = os.path.join(dataset_path, "train", "x")
        train_path_y = os.path.join(dataset_path, "train", "y")

        return train_path_x, train_path_y, valid_path_x, valid_path_y

    def _parse_affine(self):
        affine = (
            self.transforms_config.max_rotate,
            self.transforms_config.max_translate,
            self.transforms_config.do_vertical_flip,
            self.transforms_config.do_horizontal_flip
        )
        return affine

    def train(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        loss_fn,
        metric_fn
    ):
        best_metric = 0.0

        for epoch in range(1, self.config.n_epochs + 1):

            logs = {
                'loss': [],
                self.config.metric_name: [],
                'val_loss': [],
                f'val_{self.config.metric_name}': []
            }

            model.train()
            train_loss, train_metric = self._step(
                model,
                epoch_num=epoch,
                loader=train_loader,
                optimizer_fn=optimizer,
                loss_fn=loss_fn,
                metric_fn=metric_fn,
                is_train=True,
                metric_name=self.config.metric_name,
            )

            model.eval()
            valid_loss, valid_metric = self._step(
                model,
                epoch_num=epoch,
                loader=valid_loader,
                loss_fn=loss_fn,
                metric_fn=metric_fn,
                is_train=False,
                metric_name=self.config.metric_name,
            )

            logs['loss'].append(train_loss)
            logs[self.config.metric_name].append(train_metric)
            logs['val_loss'].append(valid_loss)
            logs[f'val_{self.config.metric_name}'].append(valid_metric)

            if valid_metric >= best_metric:
                self.logger.info("\nSaving model.....")
                torch.save(model.state_dict(), self.model_config.save_path)
                best_metric = valid_metric

        return logs

    def _step(
        self,
        model: torch.nn.Module,
        epoch_num: int = 0,
        loader: DataLoader = None,
        optimizer_fn: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        metric_fn: torch.nn.Module = None,
        is_train: bool = False,
        metric_name: str = "iou"
    ):
        loss_record = MeanMetric()
        metric_record = MeanMetric()

        loader_len = len(loader)

        text = "TRAIN" if is_train else "VALID"

        for data in tqdm(
            iterable=loader,
            total=loader_len,
            dynamic_ncols=True,
            desc=f"{text} :: Epoch: {epoch_num}"
        ):
            self._step_one_batch(model, data, optimizer_fn, loss_fn, metric_fn, is_train, loss_record, metric_record)

        current_loss = loss_record.compute()
        current_metric = metric_record.compute()

        self.logger.info(
            f"\rEpoch {epoch_num:>03} :: "
            f"{text} :: "
            f"LOSS: {loss_record.compute()}, "
            f"{metric_name.upper()}: {metric_record.compute()}\t\t\t\t",
        )

        return current_loss, current_metric

    @staticmethod
    def _step_one_batch(
        model: torch.nn.Module,
        data: t.Any,
        optimizer_fn: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        metric_fn: torch.nn.Module = None,
        is_train: bool = False,
        loss_record: MeanMetric = None,
        metric_record: MeanMetric = None
    ):
        if is_train:
            predicts = model(data[0])["out"]
        else:
            with torch.no_grad():
                predicts = model(data[0])["out"].detach()

        loss = loss_fn(predicts, data[1])

        if is_train:
            optimizer_fn.zero_grad()
            loss.backward()
            optimizer_fn.step()

        metric = metric_fn(predicts.detach(), data[1])

        loss_value = loss.detach().item()
        metric_value = metric.detach().item()

        loss_record.update(loss_value)
        metric_record.update(metric_value)
