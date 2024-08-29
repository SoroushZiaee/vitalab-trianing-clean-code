import sys
import os
import argparse
import yaml
from argparse import Namespace
from typing import Dict, Optional
from tqdm import tqdm
from pprint import pprint

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

import torch

torch.set_float32_matmul_precision("medium")  # or 'medium' for more speed
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import lightning as L
from lightning import LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import CombinedLoader

from lit_modules.modules import ModelLightning
from lit_modules.datamodule import (
    ImageNetDataModule,
    LaMemDataModule,
    CombinedDataModule,
)


def test_datamodule_batches(datamodule):
    # Ensure the datamodule is set up
    datamodule.setup()

    # Get the train dataloader
    train_dataloader = datamodule.train_dataloader()

    # Check if it's a DataLoader instance
    assert isinstance(
        train_dataloader, DataLoader
    ), "train_dataloader() should return a DataLoader"

    # Fetch 5 batches
    batches = []
    for i, batch in enumerate(train_dataloader):
        if i >= 5:
            break
        batches.append(batch)

    # Check if we got 5 batches
    assert len(batches) == 5, f"Expected 5 batches, but got {len(batches)}"

    # Check the structure of each batch
    for i, batch in enumerate(batches):
        # Assuming the batch contains images and labels
        assert (
            len(batch) == 2
        ), f"Batch {i} should contain 2 elements (images and labels)"

        images, labels = batch

        # Check if images are tensors with the correct shape
        assert isinstance(
            images, torch.Tensor
        ), f"Batch {i}: Images should be a torch.Tensor"
        assert (
            images.dim() == 4
        ), f"Batch {i}: Images should have 4 dimensions (batch_size, channels, height, width)"
        assert images.shape[1] == 3, f"Batch {i}: Images should have 3 channels"

        # Check if labels are tensors with the correct shape
        assert isinstance(
            labels, torch.Tensor
        ), f"Batch {i}: Labels should be a torch.Tensor"
        assert labels.dim() == 1, f"Batch {i}: Labels should have 1 dimension"

        # Check if the batch size matches the configured batch size
        assert (
            images.shape[0] == datamodule.hparams.batch_size
        ), f"Batch {i}: Incorrect batch size"
        assert (
            labels.shape[0] == datamodule.hparams.batch_size
        ), f"Batch {i}: Incorrect batch size for labels"

    print("All 5 batches successfully fetched and validated.")


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


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config_path):
    # Load configuration from YAML file
    config = load_config(config_path)

    # Extract parameters from config
    task_type = config.get("task_type", "classification")
    max_epochs = config.get("max_epochs", 100)
    arch = config.get("arch", "alexnet")
    optimizer = config.get("optimizer", "sgd")
    num_classes = config.get("num_classes", 1000)
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 4)

    # Define hyperparameters
    imagenet_hparams = Namespace(
        data_dir="/home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/imagenet",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        temp_extract=config.get("temp_extract", True),
        pin_memories=config.get("pin_memories", [False, False, False]),
    )

    # Create the DataModule
    imagenet_dm = ImageNetDataModule(imagenet_hparams)

    # Define hyperparameters
    lamem_hparams = Namespace(
        data_dir="/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/lamem/lamem_images/lamem",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        change_labels=config.get("change_labels", False),
        pin_memories=config.get("pin_memories", [False, False, False]),
    )

    # Create the DataModule
    lamem_dm = LaMemDataModule(lamem_hparams)

    # Prepare data and setup

    # Create a dictionary of datasets
    datasets = {"regression": lamem_dm, "classification": imagenet_dm}

    # Define hyperparameters for the combined datamodule
    combined_hparams = Namespace(mode="max_size_cycle")

    # Create the combined datamodule
    datamodule = CombinedDataModule(combined_hparams, datasets)
    datamodule.prepare_data()
    datamodule.setup()

    # test_datamodule_batches(datamodule)

    # Create TensorBoard logger
    # Create TensorBoard logger
    logger = TensorBoardLogger(
        config.get("log_dir", "experiments"), name=f"{arch}/{task_type}", log_graph=True
    )

    # if task_type in ["regression", "classification", "combined"]:
    #     datamodule.log_samples_to_tensorboard(logger)

    # Define hyperparameters for the model
    model_hparams = {
        "arch": arch,
        "pretrained": config.get("pretrained", False),
        "use_blurpool": config.get("use_blurpool", False),
        "num_classes": num_classes,
        "lr": config.get("lr", 0.01),
        "weight_decay": config.get("weight_decay", 0.0001),
        "momentum": config.get("momentum", 0.9),
        "nesterov": config.get("nesterov", True),
        "norm_mean": config.get("norm_mean", [0.485, 0.456, 0.406]),
        "norm_std": config.get("norm_std", [0.229, 0.224, 0.225]),
        "task_type": task_type,
        "experiment": config.get("experiment", "one"),
        "optimizer": optimizer,
        "scheduler": config.get("scheduler", "step"),
        "step_size": config.get("step_size", 30),
        "max_epochs": max_epochs,
        "lr_gamma": config.get("lr_gamma", 0.1),
    }

    pprint(model_hparams)

    # Create the model
    model = ModelLightning(model_hparams)

    # Create a trainer with TensorBoard logger
    trainer = L.Trainer(
        max_epochs=max_epochs,
        # limit_train_batches=config.get("limit_train_batches", 20),
        # limit_val_batches=config.get("limit_val_batches", 10),
        logger=logger,
        log_every_n_steps=config.get("log_every_n_steps", 1),
        accelerator=config.get("accelerator", "auto"),
        strategy="ddp",
        devices=config.get("devices", "auto"),
        fast_dev_run=config.get("fast_dev_run", False),
        sync_batchnorm=config.get("sync_batchnorm", True),
        num_nodes=config.get("num_nodes", 1),
    )

    # Fit the model
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )

    print(
        f"\nTraining complete. TensorBoard logs saved in '{config.get('log_dir', 'experiments')}' directory."
    )
    print("To view the logs, run: tensorboard --logdir=experiments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with YAML config")
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    main(args.config)
