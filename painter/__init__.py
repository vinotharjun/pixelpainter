# import torch, torch.nn.functional as F
# from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
# from torch import nn, optim, as_tensor
# from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
# from torch.nn.utils import weight_norm, spectral_norm
from .env_path import *
from pathlib import Path
SOURCE = get_project_root()
DATASET_PATH = SOURCE / "storage/inpainting-dataset"
IMAGES_PATH = DATASET_PATH / "images"
MASK_PATH = DATASET_PATH / "masks"