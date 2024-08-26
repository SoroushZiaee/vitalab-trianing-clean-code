import sys
import os
from argparse import Namespace
from tqdm import tqdm

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
parent_dir = os.path.dirname(parent_dir)

print(f"{script_dir = }")
print(f"{parent_dir = }")
sys.path.append(parent_dir)

from lit_modules.datamodule import ImageNetDataModule
from lit_modules.datamodule import LaMemDataModule
from lit_modules.datamodule import CombinedDataModule


def main():
    # Define hyperparameters for each dataset
    lamem_hparams = Namespace(
        data_dir="/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/lamem/lamem_images/lamem",
        image_size=224,
        batch_size=32,
        num_workers=4,
        change_labels=False,
    )

    imagenet_hparams = Namespace(
        data_dir="/home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/imagenet",
        image_size=224,
        batch_size=32,
        num_workers=4,
        temp_extract=True,
    )

    # Create dataset instances
    lamem_dataset = LaMemDataModule(lamem_hparams)
    imagenet_dataset = ImageNetDataModule(imagenet_hparams)

    # Create a dictionary of datasets
    datasets = {"lamem": lamem_dataset, "imagenet": imagenet_dataset}
    # datasets = {"lamem": lamem_dataset}

    # Define hyperparameters for the combined datamodule
    combined_hparams = Namespace(mode="max_size_cycle")

    # Create the combined datamodule
    combined_datamodule = CombinedDataModule(combined_hparams, datasets)

    # Prepare data and setup
    combined_datamodule.prepare_data()
    combined_datamodule.setup()

    # Get dataset info
    dataset_info = combined_datamodule.get_dataset_info()
    print("Dataset Info:")
    for name, info in dataset_info.items():
        print(f"  {name}:")
        for loader_type, size in info.items():
            print(f"    {loader_type}: {size} batches")

    # Test dataloaders
    train_loader = combined_datamodule.train_dataloader()
    val_loader = combined_datamodule.val_dataloader()
    test_loader = combined_datamodule.test_dataloader()

    # Process a few batches with progress bar
    num_batches_to_process = 10
    print(f"\nProcessing {num_batches_to_process} batches:")
    for batch_idx, batch in enumerate(tqdm(train_loader, total=num_batches_to_process)):
        if batch_idx >= num_batches_to_process:
            break

        data, batch_i, dataloader_idx = batch

        print(f"\nBatch {batch_i + 1} from dataloader {dataloader_idx + 1}:")

        for dataset_name, (images, labels) in data.items():
            print(f"\n{dataset_name.capitalize()} batch {batch_idx + 1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")

            # Print the first few labels
            print(f"  First few labels: {labels[:5]}")

            # If you want to check the image tensor values
            print(
                f"  Image tensor min: {images.min()}, max: {images.max()}, mean: {images.mean()}"
            )

        print("-" * 50)  # Separator between batches


if __name__ == "__main__":
    main()
