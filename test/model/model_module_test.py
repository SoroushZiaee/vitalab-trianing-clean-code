import sys
import os
from argparse import Namespace
from typing import Dict, Optional
from tqdm import tqdm
from pprint import pprint

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
parent_dir = os.path.dirname(parent_dir)

print(f"{script_dir = }")
print(f"{parent_dir = }")
sys.path.append(parent_dir)

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import lightning as L
from lightning import LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import CombinedLoader
from tqdm import tqdm
import numpy as np

# Assuming ModelLightning is in the same directory
from lit_modules.modules import ModelLightning


class SyntheticDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        num_classes=5,
        image_size=224,
        task_type="classification",
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.task_type = task_type

        self.data = torch.randn(num_samples, 3, image_size, image_size)

        if task_type == "classification":
            self.labels = torch.randint(0, num_classes, (num_samples,))
        elif task_type == "regression":
            self.labels = torch.randn(num_samples, 1)
        else:
            raise ValueError(
                "Invalid task_type. Choose 'classification' or 'regression'."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SyntheticDataModule(LightningDataModule):
    def __init__(
        self,
        num_samples=1000,
        num_classes=5,
        image_size=224,
        batch_size=32,
        task_type="classification",
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.task_type = task_type

        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def setup(self, stage: Optional[str] = None):
        dataset = SyntheticDataset(
            self.num_samples, self.num_classes, self.image_size, self.task_type
        )

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

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


class CombinedSyntheticDataModule(LightningDataModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets = {
            "classification": SyntheticDataModule(
                num_samples=hparams.num_samples,
                num_classes=hparams.num_classes,
                image_size=hparams.image_size,
                batch_size=hparams.batch_size,
                task_type="classification",
            ),
            "regression": SyntheticDataModule(
                num_samples=hparams.num_samples,
                num_classes=1,
                image_size=hparams.image_size,
                batch_size=hparams.batch_size,
                task_type="regression",
            ),
        }
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
        return self._get_combined_loader(
            "test_dataloader"
        )  # Using test_dataloader for predict

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
        batch, batch_idx, _ = next(iter(self.train_dataloader()))

        print(f"{type(batch) = }")
        print(f"{batch.keys() = }")

        clf_batch, reg_batch = batch["classification"], batch["regression"]

        print(f"{type(clf_batch) = }")
        print(f"{len(clf_batch) = }")

        print(f"{type(reg_batch) = }")
        print(f"{len(reg_batch) = }")

        images, labels = clf_batch
        # Create a grid of images
        grid = torchvision.utils.make_grid(images)
        logger.experiment.add_image("sample_images", grid, 0)
        logger.experiment.add_text(
            "sample_classification_labels", str(labels.tolist()), 0
        )

        images, labels = reg_batch
        grid = torchvision.utils.make_grid(images)
        logger.experiment.add_image("sample_regression_images", grid, 0)
        logger.experiment.add_text("sample_regression_labels", str(labels.tolist()), 0)


def main():
    # Choose the task type
    task_type = (
        "classification"  # Options: "classification", "regression", or "combined"
    )

    num_classes = 2
    num_samples = 1000
    image_size = 224  # inception_v3 requires 299
    batch_size = 64

    # Create TensorBoard logger
    logger = TensorBoardLogger(
        "tb_logs", name=f"synthetic_model_{task_type}", log_graph=True
    )

    if task_type == "combined":
        print("Creating combined synthetic data module.")

        # Create the combined synthetic datamodule
        hparams = Namespace(
            num_samples=num_samples,
            num_classes=num_classes,
            image_size=image_size,
            batch_size=batch_size,
            mode="max_size_cycle",
        )
        datamodule = CombinedSyntheticDataModule(hparams)
    else:
        # Create the single task synthetic datamodule
        if task_type == "regression":
            num_classes = 1
        datamodule = SyntheticDataModule(
            num_samples=num_samples,
            num_classes=num_classes,
            image_size=image_size,
            batch_size=batch_size,
            task_type=task_type,
        )

    # Prepare data and setup
    datamodule.prepare_data()
    datamodule.setup()

    if task_type in ["regression", "classification", "combined"]:
        datamodule.log_samples_to_tensorboard(logger)

    # Define hyperparameters for the model
    model_hparams = {
        "arch": "alexnet",
        "use_blurpool": True,
        "pretrained": False,
        "num_classes": num_classes,
        "lr": 0.0001,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "nesterov": True,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "task_type": task_type,
        "experiment": "one",
        "optimizer": "sgd",
        "scheduler": "plateau",
        "step_size": 30,
        "max_epochs": 100,
    }

    pprint(model_hparams)

    # Create the model
    model = ModelLightning(model_hparams)

    # Create a trainer with TensorBoard logger
    trainer = L.Trainer(
        max_epochs=90,
        limit_train_batches=20,
        limit_val_batches=10,
        logger=logger,
        log_every_n_steps=1,
        accelerator="auto",
        strategy="auto",
        # strategy="ddp_find_unused_parameters_true",  # Inception_v3 requires this
        devices="auto",
        fast_dev_run=False,
    )

    # Fit the model
    trainer.fit(model, datamodule=datamodule)

    print("\nTraining complete. TensorBoard logs saved in 'tb_logs' directory.")
    print("To view the logs, run: tensorboard --logdir=tb_logs")


if __name__ == "__main__":
    main()
