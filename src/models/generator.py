# Code based on
# https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/transformer_net.py
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9f8f61e5a375c2e01c5187d093ce9c2409f409b0/models/networks.py#L316C7-L316C22
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
# https://www.youtube.com/watch?v=4LktBHGCNfw&ab_channel=AladdinPersson
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9f8f61e5a375c2e01c5187d093ce9c2409f409b0/models/networks.py#L377

import torch.nn as nn
from config import num_res_blocks

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()

        layers = []

        ### apply RELU
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim, affine=True), nn.ReLU(True)]

        ### No Relu
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim, affine=True)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x) + x
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ngf = 64

        model = []
        ### Initial convolution
        model += [nn.Conv2d(3, ngf, kernel_size=7, bias=True, padding=3, padding_mode="reflect"),
                 nn.ReLU(True)]

        ### Downsampling
        model += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True, padding_mode="reflect"),
                      nn.InstanceNorm2d(ngf * 2, affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf * 2, ngf * (2**2), kernel_size=3, stride=2, padding=1, bias=True, padding_mode="reflect"),
                      nn.InstanceNorm2d(ngf * (2**2), affine=True),
                      nn.ReLU(True)]

        ### Residual Blocks
        for i in range(num_res_blocks):
          model += [ResidualBlock(ngf * (2**2))]

        ### Upsampling
        model += [nn.ConvTranspose2d(ngf * (2**2), ngf * 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(ngf * 2, affine=True),
                      nn.ReLU(True)]

        model += [nn.ConvTranspose2d(ngf * 2, ngf,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(ngf, affine=True),
                      nn.ReLU(True)]

        ### Final convolution
        model += [nn.Conv2d(ngf, 3, kernel_size=7, padding=3, padding_mode="reflect"), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()
    self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
        nn.InstanceNorm2d(out_channels), nn.LeakyReLU(0.2))

  def forward(self, x):
    return self.conv(x)