import torch
import torch.nn.functional as F
from torch import nn

from layers import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, DiscriminatorBlock


class Generator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, max_features=256, num_down_blocks=2, num_bottleneck_blocks=6,
                 equivariance=None, pool='avg', upsample_mode='nearest'):
        super(Generator, self).__init__()
        assert equivariance in [None, 'p4m']
        assert pool in ['avg', 'blur']
        assert upsample_mode in ['nearest', 'bilinear']

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=7, padding=3,
                                 lift=True, equivariance=equivariance)

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=3, padding=1,
                                           equivariance=equivariance, pool=pool))

        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=3, padding=1,
                                       equivariance=equivariance, mode=upsample_mode))
        self.up_blocks = nn.ModuleList(up_blocks)

        bottleneck = []
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            bottleneck.append(ResBlock2d(in_features, kernel_size=3, padding=1,
                                         equivariance=equivariance))
        self.bottleneck = nn.ModuleList(bottleneck)

        self.final = SameBlock2d(block_expansion, num_channels, kernel_size=7, padding=3,
                                 equivariance=equivariance, last=True)
        self.num_channels = num_channels

    def forward(self, source_image, source):
        out = self.first(source_image, source)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out, source)

        for i in range(len(self.bottleneck)):
            out = self.bottleneck[i](out, source)

        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out, source)
        out = self.final(out, source)
        out = F.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=True, equivariance=None, pool='avg'):
        super(Discriminator, self).__init__()
        assert equivariance in [None, 'p4m']
        self.equivariance = equivariance

        blocks = []
        for i in range(num_blocks):
            in_channels = num_channels if i == 0 else min(max_features, block_expansion * (2 ** (i - 1)))
            out_channels = min(max_features, block_expansion * (2 ** i))
            block = DiscriminatorBlock(in_channels, out_channels, norm=(i != 0), lift=(i == 0), equivariance=equivariance,
                                       kernel_size=5, padding=0, pool=None if (i == num_blocks - 1) else pool, sn=sn)
            blocks.append(block)

        blocks.append(DiscriminatorBlock(blocks[-1].conv.out_channels, 1, kernel_size=1, norm=False, pool=None,
                                         last=True, sn=sn, equivariance=equivariance))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = x
        for down_block in self.blocks:
            out = down_block(out)
        return out
