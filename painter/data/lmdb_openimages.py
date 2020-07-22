from painter import *
import lz4framed
import numpy as np
from typing import Any
from painter.data.genmask import MaskGenerator

import lz4framed
import numpy as np
from typing import Any
# import nonechucks as nc
from torch.utils.data import Dataset, DataLoader
from painter.data.genmask import MaskGenerator


class InvalidFileException(Exception):
    pass


class LMDBDataset(Dataset):
    def __init__(self,
                 datatype,
                 lmdb_store_path,
                 mask_path,
                 hr_shape=(192, 192),
                 gen_only=True):
        super().__init__()
        assert os.path.isfile(
            lmdb_store_path), f"LMDB store '{lmdb_store_path} does not exist"
        assert not os.path.isdir(
            lmdb_store_path
        ), f"LMDB store name should a file, found directory: {lmdb_store_path}"

        self.lmdb_store_path = lmdb_store_path
        self.lmdb_connection = lmdb.open(lmdb_store_path,
                                         subdir=False,
                                         readonly=True,
                                         lock=False,
                                         readahead=False,
                                         meminit=False)

        with self.lmdb_connection.begin(write=False) as lmdb_txn:
            self.length = lmdb_txn.stat()['entries'] - 1
            self.keys = pyarrow.deserialize(
                lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            print(f"Total records: {len(self.keys), self.length}")
        self.transform = transform
        self.datatype = datatype
        self.mask_path = mask_path / datatype

        self.hr_height, self.hr_width = hr_shape
        self.mask_generator = MaskGenerator(height=self.hr_height,
                                            width=self.hr_width,
                                            channels=3,
                                            filepath=self.mask_path)

        self.gen_only = gen_only
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.image_transform = torchvision.transforms.Compose([
            transforms.Resize((self.hr_height, self.hr_height), Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, index):

        lmdb_value = None
        with self.lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[index])
        assert lmdb_value is not None, f"Read empty record for key: {self.keys[index]}"

        img_name, img_arr, img_shape = LMDBDataset.decompress_and_deserialize(
            lmdb_value=lmdb_value)
        image_array = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        if self.gen_only == True:
            mask_array = self.mask_generator.sample(switcher=True)
        else:
            switch = bool(random.getrandbits(1))
            mask_array = self.mask_generator.sample(switcher=switch)
        input_array = image_array * mask_array

        input_image = Image.fromarray(input_array).convert("RGB")
        mask = Image.fromarray(mask_array).convert("RGB")
        img = Image.fromarray(image_array).convert("RGB")

        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        input_image = self.image_transform(input_image)

        return {"ground_truth": img, "mask": mask, "input": input_image}

    @staticmethod
    def decompress_and_deserialize(lmdb_value: Any):
        return pyarrow.deserialize(lz4framed.decompress(lmdb_value))

    def __len__(self):
        return self.length
