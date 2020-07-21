import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .genmask import MaskGenerator
import h5py
import copy


class CustomDataset(Dataset):
    def __init__(self,
                 hr_shape,
                 datatype="train",
                 dataset_path="../dataset/train_data.h5",
                 mask_path="../dataset/masks"):
        super(CustomDataset, self).__init__()
        hr_height, hr_width = hr_shape
        self.datatype = datatype
        self.mask_path = mask_path
        self.dataset_path = dataset_path
        self.mask_generator = MaskGenerator(height=hr_height,
                                            width=hr_width,
                                            channels=3,
                                            filepath=self.mask_path)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.image_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_height), Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize(self.mean, self.std)
        ])
        self.file = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.dataset_path, 'r')[self.datatype]
        image_array = self.file[index, ...]
        switch = bool(random.getrandbits(1))
        mask_array = self.mask_generator.sample(switcher=switch)

        input_array = image_array * mask_array

        img = Image.fromarray(image_array).convert("RGB")
        mask = Image.fromarray(mask_array).convert("RGB")
        input_image = Image.fromarray(input_array).convert("RGB")

        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        input_image = self.image_transform(input_image)

        return {"ground_truth": img, "mask": mask, "input": input_image}

    def __len__(self):
        if self.datatype == "train":
            return 83308
        else:
            return 9255
