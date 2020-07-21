from pathlib import Path

import os


def get_project_root() -> Path:
    return Path(
        os.path.dirname(os.path.realpath("__file__")).rsplit(os.sep, 2)[0])


SOURCE = get_project_root()
DATASET_PATH = SOURCE / "storage/inpainting-dataset"

IMAGES_PATH = DATASET_PATH / "images"
MASKS_PATH = DATASET_PATH / "masks"
TRAIN_DATA_PATH = IMAGES_PATH / "train"
VAL_DATA_PATH = IMAGES_PATH / "validation"
TRAIN_MASK_PATH = MASKS_PATH / "train"
VAL_MASK_PATH = MASKS_PATH / "validation"
