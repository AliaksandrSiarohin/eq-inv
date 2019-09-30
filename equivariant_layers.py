from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
from torch import nn
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm3d
import torch.nn.functional as F


class EqResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(EqResBlock2d, self).__init__()
        self.conv1 = P4MConvP4M(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                padding=padding)
        self.conv2 = P4MConvP4M(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class EqUpBlock2d(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(EqUpBlock2d, self).__init__()

        self.conv = P4MConvP4M(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class EqDownBlock2d(nn.Module):
    """
    Simple block for processinGg video (encoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(EqDownBlock2d, self).__init__()
        self.conv = P4MConvP4M(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class EqFirstBlock2d(nn.Module):
    """
    Simple block with group convolution.
    """

    def __init__(self, in_features, out_features, kernel_size=7, padding=3):
        super(EqFirstBlock2d, self).__init__()
        self.conv = P4MConvZ2(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class EqLastBlock2d(nn.Module):
    """
    Simple block with group convolution.
    """

    def __init__(self, in_features, out_features, kernel_size=7, padding=3):
        super(EqLastBlock2d, self).__init__()
        self.conv = P4MConvP4M(in_channels=in_features, out_channels=out_features,
                               kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out = out.max(dim=2)[0]
        return out


class EqDiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False, first=False):
        super(EqDiscriminatorBlock, self).__init__()
        if first:
            self.conv = P4MConvZ2(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)
        else:
            self.conv = P4MConvP4M(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm3d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out
