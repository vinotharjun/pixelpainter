from painter import *
from .genmask import MaskGenerator


class OpenImages(Dataset):
    def __init__(self,
                 hr_shape,
                 datatype="train",
                 dataset_path=IMAGES_PATH,
                 mask_path=MASKS_PATH):
        super(OpenImages, self).__init__()
        self.hr_height, self.hr_width = hr_shape
        self.mask_path = mask_path / datatype
        self.dataset_path = dataset_path / datatype
        self.files = [
            self.dataset_path / i for i in os.listdir(self.dataset_path)
        ]

        self.mask_generator = MaskGenerator(height=self.hr_height,
                                            width=self.hr_width,
                                            channels=3,
                                            filepath=self.mask_path)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.image_transform = transforms.Compose([
            transforms.Resize((self.hr_height, self.hr_height), Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, index):
        image_path = self.files[index]

        img = Image.open(image_path).convert("RGB").resize(
            (self.hr_height, self.hr_width))

        image_array = np.array(img).astype(np.uint8)
        switch = bool(random.getrandbits(1))
        mask_array = self.mask_generator.sample(switcher=switch)

        input_array = image_array * mask_array

        input_image = Image.fromarray(input_array).convert("RGB")
        mask = Image.fromarray(mask_array).convert("RGB")

        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        input_image = self.image_transform(input_image)

        return {"ground_truth": img, "mask": mask, "input": input_image}

    def __len__(self):
        return len(self.files)
