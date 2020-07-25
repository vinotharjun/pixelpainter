from .layers import *
from typing import *


class PixelShuffle_ICNR(nn.Module):
    def __init__(self,
                 ni: int,
                 nf: int = None,
                 scale: int = 2,
                 blur: bool = False,
                 leaky: float = 0.01) -> None:
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_layer(ni,
                               nf * (scale**2),
                               kernel_size=1,
                               use_activation=False,
                               use_batch_norm=False)
        self.icnr(self.conv[0].weight, scale)
        self.shuffle = nn.PixelShuffle(scale)
        self.pad = nn.ReflectionPad2d(1)
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur
        self.relu = relu(True, leaky=leaky)

    def forward(self, x: torch.tensor):
        x = self.conv(x)
        x = self.relu(x)
        x = self.shuffle(x)

        if self.do_blur:
            x = self.pad(x)
            return self.blur(x)
        else:
            return x

    def icnr(self,
             x: torch.tensor,
             scale: int = 2,
             init: Callable = nn.init.kaiming_normal_):
        ni, nf, h, w = x.shape
        ni2 = int(ni / (scale**2))
        k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale**2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        x.data.copy_(k)


#layer courtesy
class PartialConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: Union[int, intTuple] = 1,
                 padding: Union[int, intTuple] = 0,
                 dilation: Union[int, intTuple] = 1,
                 groups: Union[int, intTuple] = 1,
                 bias: bool = True):
        super().__init__()
        self.input_conv = conv2d(in_channels, out_channels, kernel_size,
                                 stride, padding, dilation, groups, bias)
        self.mask_conv = conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                dilation,
                                groups,
                                False,
                                init=None)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input: torch.tensor, mask: torch.tensor) -> torch.tensor:

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1,
                                                    1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PartialConvEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, intTuple] = 3,
                 stride: Union[int, intTuple] = 1,
                 padding: Union[int, intTuple] = 1,
                 activation: str = "relu",
                 batch_norm_enable: bool = True,
                 bias: bool = False):
        super().__init__()
        self.batch_norm_enable = batch_norm_enable
        self.conv = PartialConvLayer(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)
        if batch_norm_enable:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, input_tensor, input_mask):
        out, out_mask = self.conv(input_tensor, input_mask)
        if self.batch_norm_enable:
            out = self.batch_norm(out)
        if self.activation:
            out = self.activation(out)
        return out, out_mask


class PartialConvDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, intTuple] = 3,
        stride: Union[int, intTuple] = 1,
        padding: Union[int, intTuple] = 1,
        activation: str = "relu",
        batch_norm_enable: bool = True,
        blur: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.batch_norm_enable = batch_norm_enable
        self.blur = blur
        self.activation = activation
        self.conv = PartialConvLayer(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)
        if batch_norm_enable:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "relu":
            self.activation = nn.ReLU()
        self.upsample_layer = PixelShuffle_ICNR(out_channels,
                                                out_channels,
                                                scale=2,
                                                blur=self.blur)

    def forward(self, input_tensor, input_mask):
        out, out_mask = self.conv(input_tensor, input_mask)
        if self.batch_norm_enable:
            out = self.batch_norm(out)
        if self.activation:
            out = self.activation(out)
        out = self.upsample_layer(out)
        ou_maskt = F.interpolate(out_mask, size=out.shape[-2:])
        return out, out_mask


class SequentialEx(nn.Module):
    def __init__(self, *layers):
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i):
        return self.layers[i]

    def append(self, l):
        return self.layers.append(l)

    def extend(self, l):
        return self.layers.extend(l)

    def insert(self, i, l):
        return self.layers.insert(i, l)
