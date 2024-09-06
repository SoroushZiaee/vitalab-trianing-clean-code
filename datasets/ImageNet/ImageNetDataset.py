import os
import sys
import hashlib
import logging
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm
from joblib import Parallel, delayed

import torch
from torchvision.datasets.folder import ImageFolder

# from torchvision.datasets.utils import extract_archive

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

ARCHIVE_META = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "9aa5a8351aa504e3bd61f1fe87df7b76"),
}

META_FILE = "meta.bin"


class ImageNet(ImageFolder):
    def __init__(
        self, root: str, split: str = "train", temp_extract: bool = False, **kwargs: Any
    ) -> None:
        logging.info(
            f"Initializing ImageNet dataset with root: {root}, split: {split}, temp_extract: {temp_extract}"
        )
        self.root = os.path.expanduser(root)

        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.temp_extract = temp_extract

        print(f"self.temp_extract: {self.temp_extract}")

        if self.temp_extract:
            self.temp_dir = tempfile.mkdtemp(dir=os.environ.get("SLURM_TMPDIR", "/tmp"))
            logging.info(f"Using temporary directory for extraction: {self.temp_dir}")
        else:
            self.temp_dir = os.path.join(os.environ.get("SLURM_TMPDIR", "/tmp"), "data")

        logging.info("Parsing archives...")
        self.parse_archives()
        logging.info("Loading metadata...")
        wnid_to_classes = load_meta_file(self.root)[0]

        logging.info("Initializing ImageFolder...")
        super().__init__(self.split_folder, **kwargs)

        logging.info("Setting up class information...")
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss
        }
        logging.info("ImageNet initialization complete.")

    def parse_archives(self) -> None:
        logging.info("Checking for metadata file...")
        # if not check_integrity(os.path.join(self.root, META_FILE)):
        # logging.info("Metadata file not found. Parsing devkit archive...")
        parse_devkit_archive(self.root)

        logging.info(f"Checking for {self.split} folder...")
        if not os.path.isdir(self.split_folder):
            if self.split == "train":
                logging.info("Parsing train archive...")
                parse_train_archive(self.root, temp_dir=self.temp_dir)
            elif self.split == "val":
                logging.info("Parsing validation archive...")
                parse_val_archive(self.root, temp_dir=self.temp_dir)

    @property
    def split_folder(self) -> str:
        if self.temp_extract:
            return os.path.join(self.temp_dir, self.split)
        else:
            return os.path.join(self.temp_dir, self.split)

    def extra_repr(self) -> str:
        return f"Split: {self.split}"

    def __del__(self):
        if self.temp_extract:
            logging.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)


def verify_str_arg(value: str, arg_name: str, valid_values: tuple) -> str:
    """
    Verify that the given string argument is one of the valid values.

    Args:
        value (str): The value to verify.
        arg_name (str): The name of the argument (used in error messages).
        valid_values (tuple): A tuple of valid string values.

    Returns:
        str: The input value if it's valid.

    Raises:
        ValueError: If the input value is not one of the valid values.
    """
    if value not in valid_values:
        valid_values_str = ", ".join(repr(v) for v in valid_values)
        raise ValueError(
            f"{arg_name} should be one of {valid_values_str}, but got {value!r}"
        )
    return value


def load_meta_file(
    root: str, file: Optional[str] = None
) -> Tuple[Dict[str, str], List[str]]:
    logging.info("Loading metadata file...")
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        logging.info("Metadata file found and integrity verified.")
        return torch.load(file)
    else:
        logging.error(f"Metadata file not found or corrupted: {file}")
        raise RuntimeError(
            f"The meta file {file} is not present in the root directory or is corrupted."
        )


def _verify_archive(root: str, file: str, md5: str) -> None:
    logging.info(f"Verifying archive: {file}")
    if not check_integrity(os.path.join(root, file), md5):
        logging.error(f"Archive not found or corrupted: {file}")
        raise RuntimeError(
            f"The archive {file} is not present in the root directory or is corrupted."
        )
    logging.info("Archive verification successful.")


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    logging.info(f"Checking integrity of file: {fpath}")
    if not os.path.isfile(fpath):
        logging.warning(f"File not found: {fpath}")
        return False
    if md5 is None:
        logging.info("No MD5 provided, skipping checksum verification.")
        return True
    return check_md5(fpath, md5)


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    logging.info(f"Checking MD5 for file: {fpath}")
    return md5 == calculate_md5(fpath, **kwargs)


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    logging.info(f"Calculating MD5 for file: {fpath}")
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in tqdm(
            iter(lambda: f.read(chunk_size), b""), desc="Calculating MD5"
        ):
            md5.update(chunk)
    return md5.hexdigest()


def parse_devkit_archive(root: str, file: Optional[str] = None) -> None:
    logging.info("Parsing devkit archive...")
    import scipy.io as sio

    def parse_meta_mat(
        devkit_root: str,
    ) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
        logging.info("Parsing meta.mat file...")
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [
            meta[idx]
            for idx, num_children in enumerate(nums_children)
            if num_children == 0
        ]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(", ")) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
        logging.info("Parsing validation ground truth...")
        file = os.path.join(
            devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt"
        )
        with open(file) as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir() -> Iterator[str]:
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ARCHIVE_META["devkit"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    # _verify_archive(root, file, md5)

    with get_tmp_dir() as tmp_dir:
        logging.info(f"Extracting devkit archive to temporary directory: {tmp_dir}")
        extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        logging.info("Saving parsed metadata...")
        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def parse_train_archive(
    root: str,
    file: Optional[str] = None,
    folder: str = "train",
    temp_dir: Optional[str] = None,
) -> None:
    logging.info("Parsing train archive...")
    archive_meta = ARCHIVE_META["train"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    extract_dir = temp_dir if temp_dir else root
    train_root = os.path.join(extract_dir, folder)
    logging.info(f"Extracting train archive to: {train_root}")
    extract_archive(os.path.join(root, file), train_root)

    archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
    logging.info("Extracting individual synset archives...")
    for i, archive in tqdm(enumerate(archives), desc="Extracting synset archives"):
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)

        # if i == 10:
        #     logging.info("Reach to 10 synset archives.")
        #     break


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with tarfile.open(from_path, "r") as tar:
        tar.extractall(path=to_path)

    if remove_finished:
        os.remove(from_path)


def parse_val_archive(
    root: str,
    file: Optional[str] = None,
    wnids: Optional[List[str]] = None,
    folder: str = "val",
    temp_dir: Optional[str] = None,
) -> None:
    logging.info("Parsing validation archive...")
    archive_meta = ARCHIVE_META["val"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    if wnids is None:
        wnids = load_meta_file(root)[1]

    _verify_archive(root, file, md5)

    extract_dir = temp_dir if temp_dir else root
    val_root = os.path.join(extract_dir, folder)
    logging.info(f"Extracting validation archive to: {val_root}")
    extract_archive(os.path.join(root, file), val_root)

    images = sorted(os.path.join(val_root, image) for image in os.listdir(val_root))

    logging.info("Creating synset folders for validation images...")
    for wnid in tqdm(set(wnids), desc="Creating synset folders"):
        os.makedirs(os.path.join(val_root, wnid), exist_ok=True)

    logging.info("Moving validation images to their respective synset folders...")
    for wnid, img_file in tqdm(zip(wnids, images), desc="Moving validation images"):
        shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))


@contextmanager
def get_tmp_dir(base_dir: Optional[str] = None) -> Iterator[str]:
    tmp_dir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


logging.info("ImageNet module loaded.")
