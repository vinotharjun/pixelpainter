from painter import *


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
