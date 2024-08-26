import sys
import os
import numpy as np
from torchvision import transforms
import PIL.Image

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
parent_dir = os.path.dirname(parent_dir)

print(f"{script_dir = }")
print(f"{parent_dir = }")
sys.path.append(parent_dir)

from datasets.LaMem import LaMem


def get_train_transform(
    resize: int = 256, desired_image_size: int = 224, mean=[104, 117, 128]
):
    return transforms.Compose(
        [
            transforms.Resize((resize, resize), PIL.Image.BILINEAR),
            lambda x: np.array(x),
            lambda x: np.subtract(
                x[:, :, [2, 1, 0]], mean
            ),  # Subtract average mean from image (opposite order channels)
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=13),
            transforms.CenterCrop(desired_image_size),  # Center crop to 224x224
        ]
    )


def get_val_transform(
    resize: int = 256, desired_image_size: int = 224, mean=[104, 117, 128]
):
    return transforms.Compose(
        [
            transforms.Resize((resize, resize), PIL.Image.BILINEAR),
            lambda x: np.array(x),
            lambda x: np.subtract(
                x[:, :, [2, 1, 0]], mean
            ),  # Subtract average mean from image (opposite order channels)
            transforms.ToTensor(),
            transforms.CenterCrop(desired_image_size),  # Center crop to 224x224
        ]
    )


def main():
    root = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/lamem/lamem_images/lamem"
    splits_list = os.listdir(os.path.join(root, "splits"))
    train_splits = list(sorted(filter(lambda x: "train" in x, splits_list)))
    val_splits = list(sorted(filter(lambda x: "val" in x, splits_list)))
    test_splits = list(sorted(filter(lambda x: "test" in x, splits_list)))
    train_splits = [train_splits[0]]
    val_splits = [val_splits[0]]
    test_splits = [test_splits[0]]
    change_labels = False

    train_dataset = LaMem(
        root=root,
        splits=train_splits,
        transforms=get_train_transform(),
        change_labels=change_labels,
    )
    print(f"Len of train dataset: {len(train_dataset)}")

    val_dataset = LaMem(
        root=root,
        splits=val_splits,
        transforms=get_val_transform(),
        change_labels=change_labels,
    )
    print(f"Len of val dataset: {len(val_dataset)}")

    test_dataset = LaMem(
        root=root,
        splits=test_splits,
        transforms=get_val_transform(),
        change_labels=change_labels,
    )
    print(f"Len of test dataset: {len(test_dataset)}")

    total_len = len(test_dataset) + len(val_dataset) + len(train_dataset)
    print(f"Total number of samples: {total_len}")


if __name__ == "__main__":
    main()
