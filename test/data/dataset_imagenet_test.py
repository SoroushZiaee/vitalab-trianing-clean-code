import sys
import os

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
parent_dir = os.path.dirname(parent_dir)

print(f"{script_dir = }")
print(f"{parent_dir = }")
sys.path.append(parent_dir)

from torchvision import transforms as transform_lib
from datasets.ImageNet import ImageNet


def train_transform(image_size: int = 224):
    """
    The standard imagenet transforms
    """
    preprocessing = transform_lib.Compose(
        [
            transform_lib.RandomResizedCrop(image_size),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
            # imagenet_normalization(), # model does it's own normalization!
        ]
    )
    return preprocessing


def main():
    # Train dataset
    root = "/home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/imagenet"
    type_ = "train"
    train_dataset = ImageNet(
        root=root,
        split=type_,
        temp_extract=True,  # hparams.temp_extract, # True
        # dst_meta_path=self.meta_dir,
        transform=train_transform(),
    )

    # Validation dataset
    root = "/home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/imagenet"
    type_ = "val"
    val_dataset = ImageNet(
        root=root,
        split=type_,
        temp_extract=True,  # hparams.temp_extract, # True
        # dst_meta_path=self.meta_dir,
        transform=train_transform(),
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")


if __name__ == "__main__":
    main()
