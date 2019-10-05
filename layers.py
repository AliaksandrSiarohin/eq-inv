import torch
import torch.nn.functional as F
from torch import nn

from equivariant_layers import P4MConvP4M, P4MConvZ2
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding, equivariance=None):
        super(ResBlock2d, self).__init__()
        assert equivariance in [None, 'p4m']

        if equivariance is None:
            self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                   padding=padding)
            self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                   padding=padding)
        else:
            self.conv1 = P4MConvP4M(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                    padding=padding)
            self.conv2 = P4MConvP4M(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                    padding=padding)

        self.norm1 = BatchNorm(in_features, affine=True)
        self.norm2 = BatchNorm(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1,
                 mode='nearest', equivariance=None):
        super(UpBlock2d, self).__init__()
        assert equivariance in [None, 'p4m']

        if equivariance is None:
            self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                  padding=padding, groups=groups)
        else:
            self.conv = P4MConvP4M(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                   padding=padding)

        if equivariance is None:
            self.interpolation_mode = mode
            self.scale_factor = 2
        else:
            self.interpolation_mode = mode.replace('bi', 'tri')
            self.scale_factor = (1, 2, 2)
        self.norm = BatchNorm(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale_factor, mode=self.interpolation_mode)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1,
                 pool='avg', equivariance=None):
        super(DownBlock2d, self).__init__()
        assert pool in ['avg', 'blur']

        if equivariance is None:
            self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                  padding=padding, groups=groups)
        else:
            self.conv = P4MConvP4M(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                   padding=padding)
        self.norm = BatchNorm(out_features, affine=True)

        if equivariance is None and pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=(2, 2))

        if equivariance is not None and pool == 'avg':
            self.pool = nn.AdaptiveAvgPool3d(kernel_size=(1, 2, 2))

        if pool == 'blur':
            self.pool = AntiAliasConv(stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block preserve resolution
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, equivariance=None,
                 lift=False, last=False):
        super(SameBlock2d, self).__init__()
        assert equivariance in [None, 'p4m']

        if equivariance is None:
            self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                                  kernel_size=kernel_size, padding=padding, groups=groups)
        else:
            if lift:
                self.conv = P4MConvZ2(in_channels=in_features, out_channels=out_features,
                                      kernel_size=kernel_size, padding=padding)
            else:
                self.conv = P4MConvP4M(in_channels=in_features, out_channels=out_features,
                                       kernel_size=kernel_size, padding=padding)

        if not last:
            self.norm = BatchNorm(out_features, affine=True)

        self.group_pool = last and (self.equivariance is not None)

    def forward(self, x):
        out = self.conv(x)
        if self.group_pool:
            out = out.max(dim=2)[0]
            return out
        out = self.norm(out)
        out = F.relu(out)
        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=False, kernel_size=5, pool=False, sn=False,
                 equivariance=None, padding=2, lift=False, last=False):
        super(DiscriminatorBlock, self).__init__()
        assert equivariance in [None, 'p4m']
        assert pool in [None, 'avg', 'blur']

        if equivariance is None:
            self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                                  kernel_size=kernel_size, padding=padding)
        else:
            if lift:
                self.conv = P4MConvZ2(in_channels=in_features, out_channels=out_features,
                                      kernel_size=kernel_size, padding=padding)
            else:
                self.conv = P4MConvP4M(in_channels=in_features, out_channels=out_features,
                                       kernel_size=kernel_size, padding=padding)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = torch.nn.modules.instancenorm._InstanceNorm(out_features, affine=True)
        else:
            self.norm = None

        if equivariance is None and pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=(2, 2))

        if equivariance is not None and pool == 'avg':
            self.pool = nn.AdaptiveAvgPool3d(kernel_size=(1, 2, 2))

        if pool == 'blur':
            self.pool = AntiAliasConv(stride=2)

        self.group_pool = last and (self.equivariance is not None)

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.group_pool:
            out = out.max(dim=2)[0]
            return out
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = self.pool(out)
        return out


class AntiAliasConv(nn.Module):
    def __init__(self, stride, sigma=None, kernel_size=None):
        super(AntiAliasConv, self).__init__()

        if sigma is None:
            sigma = (stride - 1) / 2

        if kernel_size is None:
            kernel_size = 2 * round(sigma * 4) + 1

        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        mean = (kernel_size - 1) / 2
        kernel = torch.exp(-(torch.arange(kernel_size, dtype=torch.float32) - mean) ** 2 / (2 * sigma ** 2))

        kernel = kernel / torch.sum(kernel)

        self.register_buffer('weight', kernel)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, input):
        # Put equivariance dimensions into batch dimension
        shape = input.shape
        if len(shape) != 4:
            input = input.permute([0] + list(range(2, len(shape) - 2)) + [1, -2, -1])
            before_view_shape = list(input.shape)
            input = input.view(-1, shape[1], shape[-2], shape[-1])

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))

        # Apply the convolution separately for height and width
        kernel_w = self.weight.view(1, 1, -1, 1).repeat(shape[1], 1, 1, 1)
        kernel_h = self.weight.view(1, 1, 1, -1).repeat(shape[1], 1, 1, 1)

        out = F.conv2d(out, weight=kernel_w, groups=shape[1], stride=(self.stride, 1))
        out = F.conv2d(out, weight=kernel_h, groups=shape[1], stride=(1, self.stride))

        # Restore equivariance dimensions
        if len(shape) != 4:
            before_view_shape[-1] = out.shape[-1]
            before_view_shape[-2] = out.shape[-2]
            out = out.view(*before_view_shape)
            out = out.permute([0, -3] + list(range(1, len(shape) - 3)) + [-2, -1])

        return out
