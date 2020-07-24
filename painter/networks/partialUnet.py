from .layers import *
from .blocks import *


class PartialConvEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
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
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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
        out_mask = F.interpolate(out_mask, scale_factor=2, mode='nearest')
        return out, out_mask


class PartialConvUNet(nn.Module):
    def __init__(self, depth=8):
        super().__init__()
        self.depth = 8
        self.encoder1 = PartialConvEncoderBlock(in_channels=3,
                                                out_channels=64,
                                                kernel_size=7,
                                                stride=2,
                                                padding=3,
                                                batch_norm_enable=False,
                                                activation="relu")
        self.encoder2 = PartialConvEncoderBlock(in_channels=64,
                                                out_channels=128,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2,
                                                batch_norm_enable=True,
                                                activation="relu")
        self.encoder3 = PartialConvEncoderBlock(in_channels=128,
                                                out_channels=256,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                batch_norm_enable=True,
                                                activation="relu")
        self.encoder4 = PartialConvEncoderBlock(in_channels=256,
                                                out_channels=512,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                batch_norm_enable=True,
                                                activation="relu")
        for i in range(5, depth + 1):
            name = "encoder{:d}".format(i)
            setattr(
                self, name,
                PartialConvEncoderBlock(in_channels=512,
                                        out_channels=512,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        batch_norm_enable=True,
                                        activation="relu"))
        self.bottleneck = PartialConvDecoderBlock(512, 512, 3, 1, 1)

        for i in reversed(range(5, depth + 1)):
            name = 'decoder{:d}'.format(i)
            setattr(
                self, name,
                PartialConvDecoderBlock(
                    in_channels=512 + 512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    batch_norm_enable=True,
                    activation="leaky",
                ))
        self.decoder4 = PartialConvDecoderBlock(
            in_channels=512 + 512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            batch_norm_enable=True,
            activation="leaky",
        )
        self.decoder3 = PartialConvDecoderBlock(
            in_channels=256 + 256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            batch_norm_enable=True,
            activation="leaky",
        )
        self.decoder2 = PartialConvDecoderBlock(
            in_channels=128 + 128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            batch_norm_enable=True,
            activation="leaky",
        )
        self.decoder1 = PartialConvDecoderBlock(in_channels=64 + 64,
                                                out_channels=32,
                                                kernel_size=32,
                                                stride=1,
                                                padding=1,
                                                batch_norm_enable=False,
                                                activation=None)
        self.final_layer = PartialConvLayer(in_channels=32 + 3,
                                            out_channels=3,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            upsampled = F.interpolate(upsampled, size=bypass.shape[-2:])

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, mask):
        encoder_dict = {}
        mask_dict = {}

        key_prev = "0"
        encoder_dict[key_prev], mask_dict[key_prev] = x, mask

        for i in range(1, self.depth + 1):

            encoder_key = "encoder" + str(i)
            key = str(i)
            encoder_dict[key], mask_dict[key] = getattr(self, encoder_key)(
                encoder_dict[key_prev], mask_dict[key_prev])
            key_prev = key

        outdata, outmask = encoder_dict[key_prev], mask_dict[key_prev]
        #bottle neck layer
        outdata, outmask = self.bottleneck(outdata, outmask)

        for i in reversed(range(1, self.depth + 1)):
            enc_h_key = str(i)
            dec_l_key = 'decoder' + str(i)

            outdata = self.crop_and_concat(outdata,
                                           encoder_dict[enc_h_key],
                                           crop=True)
            outmask = self.crop_and_concat(outmask,
                                           mask_dict[enc_h_key],
                                           crop=True)
            print(outdata.shape, outmask.shape, enc_h_key, dec_l_key)
            outdata, outmask = getattr(self, dec_l_key)(outdata, outmask)
        outdata, outmask = self.crop_and_concat(
            outdata, encoder_dict["0"],
            crop=True), self.crop_and_concat(outmask,
                                             mask_dict["0"],
                                             crop=True)
        outdata, outmask = self.final_layer(outdata, outmask)

        return outdata, outmask
