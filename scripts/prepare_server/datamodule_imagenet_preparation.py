import sys
import os
from argparse import Namespace

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
parent_dir = os.path.dirname(parent_dir)

print(f"{script_dir = }")
print(f"{parent_dir = }")
sys.path.append(parent_dir)

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lit_modules.datamodule import ImageNetDataModule


def main():
    # Define hyperparameters
    hparams = Namespace(
        data_dir="/home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/imagenet",
        image_size=224,
        batch_size=32,
        num_workers=4,
        temp_extract=False,
        pin_memories=[False, False, False],  # [train, val, test]
    )

    # Create the DataModule
    data_module = ImageNetDataModule(hparams)

    # Prepare data and setup
    data_module.prepare_data()
    data_module.setup()

    # Print dataset sizes
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")

    # Test dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Test a single batch
    for batch in train_loader:
        images, labels = batch
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break

    # Test transforms
    print("Train transform:")
    print(data_module.train_transform())
    print("\nValidation transform:")
    print(data_module.val_transform())


if __name__ == "__main__":
    main()
