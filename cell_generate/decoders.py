import torch
import torch.nn as nn
import copy


class Decoder(nn.Module):
    def __init__(self, n_input_channels=2, input_shape=128):
        super().__init__()
        self.filters = [16, 32, 64, 128, 256, 512]
        self.input_shape = input_shape
        bias = False

        out_pad = 1 if input_shape // 2 // 2 // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv6 = nn.ConvTranspose3d(
            self.filters[5],
            self.filters[4],
            3,
            stride=2,
            padding=0,
            output_padding=out_pad,
            bias=bias,
        )
        self.bn6 = nn.BatchNorm3d(self.filters[4])

        out_pad = 1 if input_shape // 2 // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv5 = nn.ConvTranspose3d(
            self.filters[4],
            self.filters[3],
            5,
            stride=2,
            padding=2,
            output_padding=out_pad,
            bias=bias,
        )
        self.bn5 = nn.BatchNorm3d(self.filters[3])
        out_pad = 1 if input_shape // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose3d(
            self.filters[3],
            self.filters[2],
            5,
            stride=2,
            padding=2,
            output_padding=out_pad,
            bias=bias,
        )
        self.bn4 = nn.BatchNorm3d(self.filters[2])
        out_pad = 1 if input_shape // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose3d(
            self.filters[2],
            self.filters[1],
            5,
            stride=2,
            padding=2,
            output_padding=out_pad,
            bias=bias,
        )
        self.bn3 = nn.BatchNorm3d(self.filters[1])
        out_pad = 1 if input_shape // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose3d(
            self.filters[1],
            self.filters[0],
            5,
            stride=2,
            padding=2,
            output_padding=out_pad,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm3d(self.filters[0])
        out_pad = 1 if input_shape % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose3d(
            self.filters[0],
            n_input_channels,
            5,
            stride=2,
            padding=2,
            output_padding=out_pad,
            bias=bias,
        )
        self.leaky6 = nn.LeakyReLU(negative_slope=0.2)
        self.leaky5 = copy.deepcopy(self.leaky6)
        self.leaky4 = copy.deepcopy(self.leaky6)
        self.leaky3 = copy.deepcopy(self.leaky6)
        self.leaky2 = copy.deepcopy(self.leaky6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(
            x.size(0),
            self.filters[5],
            ((self.input_shape // 2 // 2 // 2 // 2 // 2 - 1) // 2),
            ((self.input_shape // 2 // 2 // 2 // 2 // 2 - 1) // 2),
            ((self.input_shape // 2 // 2 // 2 // 2 // 2 - 1) // 2),
        )
        x = self.leaky6(self.bn6(self.deconv6(x)))
        x = self.leaky5(self.bn5(self.deconv5(x)))
        x = self.leaky4(self.bn4(self.deconv4(x)))
        x = self.leaky3(self.bn3(self.deconv3(x)))
        x = self.leaky2(self.bn2(self.deconv2(x)))
        x = self.deconv1(x)
        out = self.sigmoid(x)

        return out


if __name__ == "__main__":
    inp = torch.rand((1, 512))
    model = Decoder()
    out = model(inp)
    print(out.shape)
