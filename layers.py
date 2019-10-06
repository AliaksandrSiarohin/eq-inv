import torch
import torch.nn.functional as F
from torch import nn

from equivariant_layers import P4MConvP4M, P4MConvZ2
from sync_batchnorm import _SynchronizedBatchNorm as BatchNorm


class AdaBN(nn.Module):
    def __init__(self, in_features):
        super(AdaBN, self).__init__()

        self.norm_source = BatchNorm(in_features, affine=True)
        self.norm_target = BatchNorm(in_features, affine=True)

        self.norm_target.weight = self.norm_source.weight
        self.norm_target.bias = self.norm_source.bias

    def forward(self, x, source):
        if source:
            out = self.norm_source(x)
        else:
            out = self.norm_target(x)

        return out

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

        self.norm1 = AdaBN(in_features)
        self.norm2 = AdaBN(in_features)

    def forward(self, x, source = True):
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
        self.norm = AdaBN(out_features)

    def forward(self, x, source=True):
        out = F.interpolate(x, scale_factor=self.scale_factor, mode=self.interpolation_mode)
        out = self.conv(out)
        out = self.norm(out, source)
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
        self.norm = AdaBN(out_features) 

        if equivariance is None and pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=(2, 2))

        if equivariance is not None and pool == 'avg':
            self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

        if pool == 'blur':
            self.pool = AntiAliasConv(stride=2)

    def forward(self, x, source=True):
        out = self.conv(x)
        out = self.norm(out, source)
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
            self.norm = AdaBN(out_features)

        self.group_pool = last and (equivariance is not None)

    def forward(self, x, source=True):
        out = self.conv(x)
        if self.group_pool:
            out = out.max(dim=2)[0]
            return out
        out = self.norm(out, source)
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
            self.norm = _InstanceNorm(out_features, affine=True)
        else:
            self.norm = None

        if equivariance is None and pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=(2, 2))

        if equivariance is not None and pool == 'avg':
            self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

        if pool == 'blur':
            self.pool = AntiAliasConv(stride=2)
         
        if pool is None:
            self.pool = None

        self.group_pool = last and (equivariance is not None)

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


class _InstanceNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(_InstanceNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    'Unexpected running stats buffer(s) {names} for {klass} '
                    'with track_running_stats=False. If state_dict is a '
                    'checkpoint saved before 0.4.0, this may be expected '
                    'because {klass} does not track running stats by default '
                    'since 0.4.0. Please remove these keys from state_dict. If '
                    'the running stats are actually needed, instead set '
                    'track_running_stats=True in {klass} to enable them. See '
                    'the documentation of {klass} for details.'
                        .format(names=" and ".join('"{}"'.format(k) for k in running_stats_keys),
                                klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)

        super(_InstanceNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, input):
        self._check_input_dim(input)

        return F.instance_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)


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
            input = input.contiguous().view(-1, shape[1], shape[-2], shape[-1])

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
            out = out.permute([0, -3] + list(range(1, len(shape) - 3)) + [-2, -1]).contiguous()

        return out
