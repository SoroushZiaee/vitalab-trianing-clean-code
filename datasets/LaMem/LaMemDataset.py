import os
import logging
from typing import List
import torch
import pandas as pd
from torch.utils.data import Dataset
import PIL.Image
from torchvision import transforms
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LaMem(Dataset):

    def __init__(
        self, root: str, splits: List[str], transforms=None, change_labels: bool = False
    ):
        logger.info(f"Initializing LaMem dataset with root: {root}")
        logger.info(f"Splits: {splits}")
        logger.info(f"Transforms: {transforms}")
        logger.info(f"Change labels: {change_labels}")

        self.mem_frame = pd.concat(
            [
                pd.read_csv(
                    os.path.join(root, "splits", split),
                    delimiter=",",
                )
                for split in splits
            ],
            axis=0,
        )
        logger.info(f"Loaded {len(self.mem_frame)} samples from splits")

        if change_labels:
            logger.warning("Changing labels - this will randomize memo scores!")
            logger.debug(
                f"Before changing: First 10 memo scores: {self.mem_frame['memo_score'][:10].tolist()}"
            )
            self.mem_frame["memo_score"] = (
                self.mem_frame["memo_score"].sample(frac=1).reset_index(drop=True)
            )
            logger.debug(
                f"After changing: First 10 memo scores: {self.mem_frame['memo_score'][:10].tolist()}"
            )

        self.transforms = transforms
        self.images_path = os.path.join(root, "images")
        logger.info(f"Images path set to: {self.images_path}")

    def __len__(self):
        return len(self.mem_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_path, self.mem_frame["image_name"][idx])
        logger.debug(f"Loading image: {img_name}")

        try:
            image = PIL.Image.open(img_name).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_name}: {str(e)}")
            raise

        mem_score = self.mem_frame["memo_score"][idx]
        target = float(mem_score)
        target = torch.tensor(target)

        if self.transforms:
            try:
                image = self.transforms(image)
            except Exception as e:
                logger.error(f"Error applying transforms to image {img_name}: {str(e)}")
                raise

        logger.debug(f"Returning image {img_name} with target {target}")
        return image, target


# Note: 00050591.jpg is removed because it can't be loaded in ffcv.
logger.warning("Note: 00050591.jpg is removed because it can't be loaded in ffcv.")
