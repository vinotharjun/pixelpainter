from painter import *
import wandb


def im_convert(tensor,
               denormalize=True,
               denormalize_mean=(0.485, 0.456, 0.406),
               denormalize_std=(0.229, 0.224, 0.225)):
    if tensor.ndimension() == 4:
        tensor = tensor.squeeze(0)

    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    if denormalize:
        image = image * np.array(denormalize_std) + np.array(denormalize_mean)
        image = image.clip(0, 1)
    return image


def minmax(tensor):
    return torch.max(tensor), torch.min(tensor)


def getsample(loader):
    a = iter(loader)
    a = next(a)
    return a


def psnr(A, B):
    mse = np.square(np.subtract(A, B)).mean()
    if mse == 0:
        return 100.
    return 10 * np.log10(1. / mse)


def validate_single(image1, image2):
    def luma(im):
        r = im[0, :, :]
        g = im[1, :, :]
        b = im[2, :, :]
        return .299 * r + .587 * g + .114 * b

    psnr_val = psnr(luma(image1), luma(image2))
    return psnr_val


def convert_batch(tensor):
    return tensor.cpu().detach().numpy()


def ifnone(a: Any, b: Any) -> Any:
    return b if a is None else a


def validate(arr1, arr2):
    arr1 = convert_batch(arr1)
    arr2 = convert_batch(arr2)
    vals = []
    for i in range(arr1.shape[0]):
        vals.append(validate_single(arr1[i], arr2[i]))
    return sum(vals) / len(vals)


def save_result(arr, denormalize=False):
    ret = []
    for i in range(arr.shape[0]):
        a = im_convert(arr[i], denormalize=denormalize)
        im = Image.fromarray(np.uint8(a * 255))
        ret.append(wandb.Image(im))
    return ret

def to_snake_case(string):
    """Converts CamelCase string into snake_case."""
    
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def classname(obj):
    return obj.__class__.__name__

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
