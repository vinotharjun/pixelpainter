from PIL import Image
import numpy as np
import h5py
import copy
from painter.imports.torchpacks import *
from .env_path import *
from pathlib import Path
SOURCE = get_project_root()
DATASET_PATH = SOURCE / "storage/inpainting-dataset"
IMAGES_PATH = DATASET_PATH / "images"
MASK_PATH = DATASET_PATH / "masks"