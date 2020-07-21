from painter.imports import *
from .env_path import *
SOURCE = get_project_root()
DATASET_PATH = SOURCE / "storage/inpainting-dataset"
IMAGES_PATH = DATASET_PATH / "images"
MASK_PATH = DATASET_PATH / "masks"