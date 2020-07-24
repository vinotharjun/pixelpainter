from painter import *
from typing import *
intTuple = Tuple[int, int]


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0
                or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


def ifnone(a: Any, b: Any) -> Any:
    return b if a is None else a


def init_default(m: nn.Conv2d,
                 func: Callable = nn.init.kaiming_normal_) -> nn.Conv2d:
    if func:
        if hasattr(m, 'weight'):
            func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.)
    return m


def relu(inplace: bool = False, leaky: float = None) -> torch.nn.Module:
    return nn.LeakyReLU(
        inplace=inplace,
        negative_slope=leaky) if leaky is not None else nn.ReLU(
            inplace=inplace)


def conv2d(ni: int,
           nf: int,
           ks: Union[int, intTuple] = 3,
           stride: Union[int, intTuple] = 1,
           padding: Union[int, intTuple] = None,
           dilation: Union[int, intTuple] = 1,
           groups: int = 1,
           bias: bool = False,
           init: Callable = nn.init.kaiming_normal_) -> nn.Conv2d:
    if padding is None:
        padding = ks // 2
    layer = nn.Conv2d(ni,
                      nf,
                      kernel_size=ks,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias)
    if init is None:
        return layer
    else:
        return init_default(layer, init)


def conv_layer(ni: int,
               nf: int,
               kernel_size: Union[int, intTuple] = 3,
               stride: Union[int, intTuple] = 1,
               padding: Union[int, intTuple] = None,
               bias: bool = None,
               use_batch_norm: bool = True,
               use_activation: bool = True,
               leaky: float = None,
               init: Callable = nn.init.kaiming_normal_,
               **kwargs):

    if padding is None:
        padding = (kernel_size - 1) // 2
    # if we use batch norm ,bias term will be cancelled out even if we included
    if bias is None:
        bias = not use_batch_norm
    conv_func = nn.Conv2d
    conv = init_default(
        conv_func(ni,
                  nf,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding), init)
    if not use_batch_norm:
        conv = nn.utils.weight_norm(conv)
    layers = [conv]
    if use_activation:
        layers.append(relu(True, leaky))
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)
