import torch.nn.functional as F
from torch import nn

from layers import ResBlock2d, UpBlock2d, Block2d


class Generator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, num_down_blocks=2, num_bottleneck_blocks=6,
                 equivariance=None, scales=1, pool='avg', upsample='nearest', sn=False, bn='instance',
                 group_pool='avg'):
        assert group_pool in ['max', 'avg']

        super(Generator, self).__init__()

        self.first = Block2d(num_channels, block_expansion, kernel_size=7, padding=3, pool_type=None,
                             lift=True, equivariance=equivariance, scales=scales, sn=sn, bn_type=bn)

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = block_expansion * (2 ** i)
            out_features = block_expansion * (2 ** (i + 1))
            down_blocks.append(Block2d(in_features, out_features, kernel_size=3, padding=1, bn_type=bn,
                                       equivariance=equivariance, pool_type=pool, scales=scales, sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = block_expansion * (2 ** (num_down_blocks - i))
            out_features = block_expansion * (2 ** (num_down_blocks - i - 1))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=3, padding=1, scales=scales,
                                       equivariance=equivariance, upsample_type=upsample, bn_type=bn))
        self.up_blocks = nn.ModuleList(up_blocks)

        bottleneck = []
        in_features = block_expansion * (2 ** num_down_blocks)
        for i in range(num_bottleneck_blocks):
            bottleneck.append(ResBlock2d(in_features, kernel_size=3, padding=1, scales=scales,
                                         equivariance=equivariance, bn_type=bn, sn=sn))
        self.bottleneck = nn.ModuleList(bottleneck)

        self.final = Block2d(block_expansion, num_channels, kernel_size=7, padding=3,
                             equivariance=equivariance, scales=scales, bn_type=None, sn=sn, group_pool=group_pool)
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
    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4,
                 equivariance=None, scales=1, pool='avg', sn=False, bn='instance', group_pool='avg'):
        super(Discriminator, self).__init__()
        assert group_pool in ['max', 'avg']

        blocks = []
        for i in range(num_blocks):
            in_channels = num_channels if i == 0 else block_expansion * (2 ** (i - 1))
            out_channels = block_expansion * (2 ** i)
            block = Block2d(in_channels, out_channels, bn_type=None if (i == 0) else bn,
                            lift=(i == 0), equivariance=equivariance, scales=scales, sn=sn,
                            kernel_size=5, padding=2, pool_type=None if (i == num_blocks - 1) else pool,
                            activation='leaky_relu')
            blocks.append(block)

        blocks.append(Block2d(blocks[-1].conv.out_channels, 1, kernel_size=1, padding=1,
                              bn_type=None, pool_type=None, group_pool=group_pool,
                              sn=sn, equivariance=equivariance, scales=scales))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = x
        for down_block in self.blocks:
            out = down_block(out)
        return out
