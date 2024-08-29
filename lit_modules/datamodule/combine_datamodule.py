import os
from typing import Optional, Dict
from argparse import Namespace
import pytorch_lightning as pl
import torchvision
from lightning.pytorch.utilities.combined_loader import CombinedLoader


class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Namespace, datasets: Dict[str, pl.LightningDataModule]):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets = datasets
        self.mode = (
            self.hparams.mode
        )  # 'max_size_cycle' or other modes supported by CombinedLoader

        self.task_type = "combined"

    def prepare_data(self):
        for dataset in self.datasets.values():
            dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        for dataset in self.datasets.values():
            dataset.setup(stage)

    def _get_combined_loader(self, loader_method: str) -> CombinedLoader:
        loaders = {}
        for name, dataset in self.datasets.items():
            loader = getattr(dataset, loader_method)()
            if loader is not None:  # Some datasets might not have all loader types
                loaders[name] = loader
        return CombinedLoader(loaders, mode=self.mode)

    def train_dataloader(self) -> CombinedLoader:
        return self._get_combined_loader("train_dataloader")

    def val_dataloader(self) -> CombinedLoader:
        return self._get_combined_loader("val_dataloader")

    def test_dataloader(self) -> CombinedLoader:
        return self._get_combined_loader("test_dataloader")

    def predict_dataloader(self) -> CombinedLoader:
        return self._get_combined_loader("predict_dataloader")

    def get_dataset_info(self) -> Dict[str, Dict[str, int]]:
        info = {}
        for name, dataset in self.datasets.items():
            info[name] = {
                "train": (
                    len(dataset.train_dataloader())
                    if hasattr(dataset, "train_dataloader")
                    else 0
                ),
                "val": (
                    len(dataset.val_dataloader())
                    if hasattr(dataset, "val_dataloader")
                    else 0
                ),
                "test": (
                    len(dataset.test_dataloader())
                    if hasattr(dataset, "test_dataloader")
                    else 0
                ),
            }
        return info

    def log_samples_to_tensorboard(self, logger):
        if self.task_type == "classification" or self.task_type == "combined":
            # Get a batch of data
            batch = next(iter(self.train_dataloader()))
            images, labels = batch
            if self.task_type == "combined":
                images, labels = images["classification"], labels["classification"]

            # Create a grid of images
            grid = torchvision.utils.make_grid(images)
            logger.experiment.add_image("sample_images", grid, 0)

            # Log labels
            if self.task_type == "classification":
                class_names = [f"Class_{i}" for i in range(self.num_classes)]
                # label_names = [class_names[label] for label in labels]
                logger.experiment.add_text("sample_labels", ", ".join(class_names), 0)
            elif self.task_type == "combined":
                logger.experiment.add_text(
                    "sample_classification_labels", str(labels.tolist()), 0
                )

        if self.task_type == "regression" or self.task_type == "combined":
            batch = next(iter(self.train_dataloader()))
            images, labels = batch
            if self.task_type == "combined":
                images, labels = images["regression"], labels["regression"]

            # Create a grid of images
            grid = torchvision.utils.make_grid(images)
            logger.experiment.add_image("sample_regression_images", grid, 0)

            # Log labels
            logger.experiment.add_text(
                "sample_regression_labels", str(labels.tolist()), 0
            )
