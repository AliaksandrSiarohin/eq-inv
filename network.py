import torch.nn.functional as F
from torch import nn

from layers import ResBlock2d, UpBlock2d, Block2d


class Generator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, num_down_blocks=2, num_bottleneck_blocks=6,
                 equivariance=None, scales=1, pool='avg', upsample='nearest', sn=False, bn='instance',
                 group_pool='avg'):
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
    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, kernel_size=4,
                 equivariance=None, scales=1, pool='avg', sn=False, bn='instance', group_pool='avg'):
        super(Discriminator, self).__init__()

        blocks = []
        for i in range(num_blocks):
            in_channels = num_channels if i == 0 else block_expansion * (2 ** (i - 1))
            out_channels = block_expansion * (2 ** i)
            block = Block2d(in_channels, out_channels, bn_type=None if (i == 0) else bn,
                            lift=(i == 0), equivariance=equivariance, scales=scales, sn=sn,
                            kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                            pool_type=None if (i == num_blocks - 1) else pool,
                            activation='leaky_relu')
            blocks.append(block)

        blocks.append(Block2d(out_channels, 1, kernel_size=1, padding=0,
                              bn_type=None, pool_type=None, group_pool=group_pool,
                              sn=sn, equivariance=equivariance, scales=scales))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = x
        for down_block in self.blocks:
            out = down_block(out)
        return out




class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, **kwargs):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)




class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d,
                  use_dropout=False, n_blocks=9, padding_type='reflect', **kwargs):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, source):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
