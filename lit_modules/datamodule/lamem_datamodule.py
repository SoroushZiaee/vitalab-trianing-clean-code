import os
from typing import List, Optional
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from argparse import Namespace
import torch

from datasets.LaMem import LaMem


class LaMemDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.data_dir = self.hparams.data_dir
        self.image_size = self.hparams.image_size
        self.batch_size = self.hparams.batch_size
        self.num_workers = self.hparams.num_workers
        self.change_labels = self.hparams.change_labels
        self.pin_memory_train, self.pin_memory_val, self.pin_memory_test = (
            self.hparams.pin_memories
        )

        splits_list = os.listdir(os.path.join(self.data_dir, "splits"))
        self.train_splits = list(sorted(filter(lambda x: "train" in x, splits_list)))
        self.val_splits = list(sorted(filter(lambda x: "val" in x, splits_list)))
        self.test_splits = list(sorted(filter(lambda x: "test" in x, splits_list)))

        # Use one Split train_1.csv
        self.train_splits = [self.train_splits[0]]
        self.val_splits = [self.val_splits[0]]
        self.test_splits = [self.test_splits[0]]

        self.dims = (3, self.image_size, self.image_size)

        self.task_type = "regression"  # regression task

    def prepare_data(self):
        # Check if the data is already downloaded
        if not os.path.exists(os.path.join(self.data_dir, "images")):
            print("LaMem data not found. Please download the dataset manually.")

    def setup(self, stage: Optional[str] = None):
        stats_tensor = torch.load(
            "/home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/datasets/LaMem/support_files/lamem_mean_std_tensor.pt"
        ).numpy()
        LAMEM_MEAN, LAMEM_STD = stats_tensor[:3], stats_tensor[3:]

        if stage == "fit" or stage is None:
            self.lamem_train = self.get_dataset(
                self.train_splits,
                self.train_transform(mean=LAMEM_MEAN, std=LAMEM_STD),
            )
            self.lamem_val = self.get_dataset(
                self.val_splits, self.val_transform(mean=LAMEM_MEAN, std=LAMEM_STD)
            )

        if stage == "test" or stage is None:
            self.lamem_test = self.get_dataset(
                self.test_splits, self.val_transform(mean=LAMEM_MEAN, std=LAMEM_STD)
            )

        if stage == "fit" or stage is None:
            print(f"Train dataset size: {len(self.lamem_train)}")
            print(f"Validation dataset size: {len(self.lamem_val)}")

    def get_dataset(self, splits: List[str], transform):
        return LaMem(
            root=self.data_dir,
            splits=splits,
            transforms=transform,
            change_labels=self.change_labels,
        )

    def train_transform(self, mean=[1, 1, 1], std=[0, 0, 0]):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def val_transform(self, mean=[1, 1, 1], std=[0, 0, 0]):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            self.lamem_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.lamem_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_val,
        )

    def test_dataloader(self):
        return DataLoader(
            self.lamem_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_test,
        )

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
