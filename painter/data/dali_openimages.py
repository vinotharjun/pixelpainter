from painter import *
from painter.imports.dalipacks import *
from .genmask import MaskGenerator


class DALIDataloader(DALIGenericIterator):
    def __init__(self,
                 pipeline,
                 size,
                 batch_size,
                 auto_reset=True,
                 onehot_label=False):
        self.size = size
        self.batch_size = batch_size
        super().__init__(pipelines=pipeline,
                         size=size,
                         auto_reset=auto_reset,
                         output_map=output_map)

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        data = super().__next__()[0]
        return {
            "input": data["input"],
            "mask": data["mask"],
            "ground_truth": data["ground_truth"]
        }

    def __len__(self):
        if self.size % self.batch_size == 0:
            return self.size // self.batch_size
        else:
            return self.size // self.batch_size + 1


class HybridTrainPipe(Pipeline):
    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 data_dir,
                 crop,
                 dali_cpu=False,
                 local_rank=0,
                 world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads,
                                              device_id,
                                              seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir,
                                    shard_id=local_rank,
                                    file_list=data_dir / "file_paths.txt",
                                    num_shards=world_size,
                                    random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu",
                                         size=crop,
                                         random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.mask_generator = MaskGenerator(height=crop[0],
                                            width=crop[1],
                                            channels=3,
                                            filepath=MASKS_PATH / "train")
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.jpegs, _ = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        input_image = copy.deepcopy(output)
        switch = bool(random.getrandbits(1))
        mask_array = self.mask_generator.sample(switcher=switch)
        mask_array = torch.from_array(mask_array).permute(2, 0, 1)
        input_image[mask_array == 0] = 0
        return {
            "input": input_image,
            "mask": mask_array,
            "ground_truth": output
        }
