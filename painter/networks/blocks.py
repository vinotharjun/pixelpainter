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
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
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

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input: torch.tensor, mask: torch.tensor) -> torch.tensor:
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

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


class PartialConvBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 bn=True,
                 sample='none-3',
                 activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConvLayer(in_ch,
                                         out_ch,
                                         kernel_size=5,
                                         stride=2,
                                         padding=2,
                                         bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConvLayer(in_ch,
                                         out_ch,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConvLayer(in_ch,
                                         out_ch,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         bias=conv_bias)
        else:
            self.conv = PartialConvLayer(in_ch,
                                         out_ch,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=conv_bias)
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask
