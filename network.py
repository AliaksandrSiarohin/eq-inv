import torch
from torch import nn
import torch.nn.functional as F

from layers import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, DiscriminatorBlock


class Generator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, max_features=256, num_down_blocks=2, num_bottleneck_blocks=6):
        super(Generator, self).__init__()

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def forward(self, source_image):
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.tanh(out)

        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512, sn=True):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            block = DiscriminatorBlock(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                        min(max_features, block_expansion * (2 ** (i + 1))),
                                        norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn)
            down_blocks.append(block)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
           self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        out = x
        for down_block in self.down_blocks:
            out = down_block(out)
        out = self.conv(out)
        return out
