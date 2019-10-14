import torch
import torch.nn.functional as F
from torch import nn

from equivariant_layers import P4MConvP4M, P4MConvZ2
from sync_batchnorm import _SynchronizedBatchNorm as BatchNorm


class _InstanceNorm(torch.nn.modules.instancenorm._InstanceNorm):
    def _check_input_dim(self, input):
        None


class Identity(nn.Module):
    def forward(self, input, *args):
        return input


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
        shape = input.shape

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))

        # Apply the convolution separately for height and width
        kernel_w = self.weight.view(1, 1, -1, 1).repeat(shape[1], 1, 1, 1)
        kernel_h = self.weight.view(1, 1, 1, -1).repeat(shape[1], 1, 1, 1)

        out = F.conv2d(out, weight=kernel_w, groups=shape[1], stride=(self.stride, 1))
        out = F.conv2d(out, weight=kernel_h, groups=shape[1], stride=(1, self.stride))

        return out


class BN(nn.Module):
    def __init__(self, in_features, type):
        super(BN, self).__init__()

        assert type in [None, 'bn', 'adabn', 'instance']
        
        self.type = type
        if self.type is None:
            return

        if type.endswith('bn'):
            self.norm_source = BatchNorm(in_features, affine=True)
        elif type == 'instance':
            self.norm_source = _InstanceNorm(in_features, affine=True)
           
        if type == 'adabn':
            self.norm_target = BatchNorm(in_features, affine=True)

            self.norm_target.weight = self.norm_source.weight
            self.norm_target.bias = self.norm_source.bias
        else:
            self.norm_target = None

        self.type = type

    def forward(self, x, source):
        if self.type is None:
            return x
        source = source or (self.norm_target is None)
        shape = x.shape
        if len(shape) >= 6:
            x = x.view(shape[0], shape[1], -1)

        if source:
            out = self.norm_source(x)
        else:
            out = self.norm_target(x)

        if len(shape) >= 6:
            out = out.view(shape)

        return out

    def extra_repr(self):
        return 'type={}'.format(self.type)


class Pool(nn.Module):
    def __init__(self, type):
        super(Pool, self).__init__()
        assert type in [None, 'avg', 'blur']

        if type == 'blur':
            self.pool = AntiAliasConv(stride=2)
        elif type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2)

        self.type = type

    def forward(self, input):
        if self.type is None:
            return input

        shape = input.shape
        # Put equivariance dimensions in channels
        if len(shape) != 4:
            input = input.view(shape[0], -1, shape[-2], shape[-1])

        out = self.pool(input)

        # Restore equivariance dimensions
        if len(shape) != 4:
            shape = list(shape)
            shape[-2] = out.shape[-2]
            shape[-1] = out.shape[-1]
            out = out.view(*shape)

        return out

    def extra_repr(self):
        return 'type={}'.format(self.type)


class Upsample(nn.Module):
    def __init__(self, type):
        assert type in ['nearest', 'bilinear']
        super(Upsample, self).__init__()
        self.type = type

    def forward(self, input):
        shape = input.shape
        # Put equivariance dimensions in channels
        if len(shape) != 4:
            input = input.view(shape[0], -1, shape[-2], shape[-1])

        out = F.interpolate(input, scale_factor=2, mode=self.type)

        # Restore equivariance dimensions
        if len(shape) != 4:
            shape = list(shape)
            shape[-2] = out.shape[-2]
            shape[-1] = out.shape[-1]
            out = out.view(*shape)

        return out

    def extra_repr(self):
        return 'type={}'.format(self.type)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, equivariance, kernel_size=(3, 3), padding=(1, 1), lift=False,
                 scales=4, sn=False, stride=1):
        super(Conv, self).__init__()
        assert equivariance in [None, 'p4m']

        rot_layer = P4MConvZ2 if lift else P4MConvP4M

        if equivariance is None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        elif equivariance == 'p4m':
            self.conv = rot_layer(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                  kernel_size=kernel_size, padding=padding, scale_equivariance=(scales != 1))

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.equivariance = equivariance
        self.lift = lift
        self.scales = scales
        self.sn = sn

        if self.lift and self.scales != 1:
            self.pyramide = []
            for i in range(scales):
                if i == 0:
                    self.pyramide.append(Identity())
                else:
                    self.pyramide.append(AntiAliasConv(stride=1, sigma=2 ** i, kernel_size=1 + 4 * 2 ** i))

            self.pyramide = nn.ModuleList(self.pyramide)

    def forward(self, input):
        if self.lift and self.scales != 1:
            maps = []
            for module in self.pyramide:
                maps.append(module(input).unsqueeze(2))
            input = torch.cat(maps, dim=2)
 
        if self.scales != 1 and self.equivariance != 'p4m':
            maps = []
            for i in range(input.shape[2]):
                maps.append(F.conv2d(input[:, :, i], weight=self.conv.weight, bias=self.conv.bias, stride=self.stride,
                                     padding=self.padding * 2 ** i, dilation=2 ** i).unsqueeze(2))

            out = torch.cat(maps, dim=2)
            return out

        return self.conv(input)

    def extra_repr(self):
        return 'equivariance={}, lift={}, scales={}, sn={}'.format(self.equivariance, self.lift, self.scales, self.sn)


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding, equivariance=None, scales=1, bn_type='instance', sn=False):
        super(ResBlock2d, self).__init__()

        self.conv1 = Conv(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                          padding=padding, equivariance=equivariance, scales=scales, sn=sn)
        self.conv2 = Conv(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                          padding=padding, equivariance=equivariance, scales=scales, sn=sn)

        self.norm1 = BN(in_features, type=bn_type)
        self.norm2 = BN(in_features, type=bn_type)

    def forward(self, x, source=True):
        out = self.norm1(x, source)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out, source)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, equivariance=None, scales=1,
                 bn_type='instance', upsample_type='nearest', sn=True):
        super(UpBlock2d, self).__init__()

        self.conv = Conv(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                         padding=padding, equivariance=equivariance, scales=scales, sn=sn)
        self.upsample = Upsample(type=upsample_type)
        self.norm = BN(out_features, type=bn_type)

    def forward(self, x, source=True):
        out = self.upsample(x)
        out = self.conv(out)
        out = self.norm(out, source)
        out = F.relu(out)
        return out


class Block2d(nn.Module):
    """
    Conv-bn-pool block
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, equivariance=None, scales=1,
                 bn_type='instance', pool_type='avg', activation='relu', lift=False, group_pool=None, sn=False):
        super(Block2d, self).__init__()
        assert activation in ['relu', 'leaky_relu']
        assert group_pool in [None, 'avg', 'max', 'identity', 'linear']
        self.conv = Conv(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                         padding=padding, equivariance=equivariance, lift=lift, sn=sn, scales=scales,
                         stride=2 if (pool_type == 'stride') else 1)

        self.norm = BN(out_features, type=bn_type)
        self.pool = Pool(type=pool_type if pool_type != 'stride' else None)

        if activation == 'relu':
            self.activation = nn.ReLU
        else:
            self.activation = nn.LeakyReLU(0.2)

        self.group_pool = group_pool

        if self.group_pool == 'linear':
            self.linear_project = nn.Conv3d(out_features, out_features, groups=out_features,
                                            kernel_size=(scales * 8, 1, 1))

    def forward(self, x, source=True):
        out = self.conv(x)
        if self.group_pool is not None:
            if len(out.shape) == 4 or self.group_pool == 'identity':
                return out
            shape = out.shape
            out = out.view(shape[0], shape[1], -1, shape[-2], shape[-1])
            if self.group_pool == 'max':
                out = out.max(dim=2)[0]
            elif self.group_pool == 'avg':
                out = out.mean(dim=2)
            elif self.group_pool == 'linear':
                out = self.linear_project(out).squeeze(2)
            return out
        out = self.norm(out, source)
        out = F.relu(out)
        out = self.pool(out)
        return out
