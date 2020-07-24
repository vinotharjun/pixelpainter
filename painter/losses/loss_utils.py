from painter.imports import *


def gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    x = input_tensor.view(b, c, -1)
    return torch.bmm(x, x.transpose(1, 2)) / (c * h * w)


def children(m):
    return list(m.children())


class SaveFeatures():
    stored = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.stored = output

    def remove(self):
        self.hook.remove()


def gram_loss(input_tensor, target):
    return F.l1_loss(gram_matrix(input_tensor), gram_matrix(target))
