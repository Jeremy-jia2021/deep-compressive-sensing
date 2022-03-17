import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class encoder_decoder(nn.Module):
    def __init__(self, opt, norm_layer=nn.InstanceNorm2d, use_dropout=True, padding_type='reflect'):
        super(encoder_decoder, self).__init__()

        opt_net = opt['network_G']
        input_nc = opt_net['in_nc']
        output_nc = opt_net['out_nc']
        ngf = opt_net['nf']
        n_blocks = opt_net['nb']
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)

        # layers = []
        # layers.append(nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.InstanceNorm2d(ngf, affine=True, track_running_stats=True))
        # layers.append(nn.ReLU(inplace=True))
        #
        # # Down-sampling layers.
        # curr_dim = ngf
        # for i in range(2):
        #     layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim * 2
        #
        # self.encoder = nn.Sequential(*layers)
        #
        # layers = []
        # # Bottleneck layers.
        # for i in range(repeat_num):
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        #
        # # Up-sampling layers.
        # for i in range(2):
        #     layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim // 2
        #
        # layers.append(nn.Conv2d(curr_dim, output_nc, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Tanh())
        #
        # self.decoder = nn.Sequential(*layers)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
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

        self.encoder = nn.Sequential(*model)


        model = []

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

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

        self.decoder = nn.Sequential(*model)

        # initialization
        # mutil.initialize_weights([self.encoder, self.decoder], 0.1)

    def forward(self, input, is_bottleneck_fea=False, is_output_z=False, is_output_range=False):
        """Standard forward"""
        if is_output_z:
            return self.encoder(input)
        else:
            if is_bottleneck_fea:
                return self.decoder(input)
            else:
                bottleneck_fea = self.encoder(input)
                range = [torch.max(bottleneck_fea), torch.min(bottleneck_fea)]
                output = self.decoder(bottleneck_fea)
                if is_output_range:
                    return output, bottleneck_fea, range
                else:
                    return output, bottleneck_fea

        # if is_output_z:
        #     bottleneck_fea = self.encoder(input)
        #     return quantizer(bottleneck_fea)
        # else:
        #     if is_bottleneck_fea:
        #         return self.decoder(input)
        #     else:
        #         bottleneck_fea = self.encoder(input)
        #         z = quantizer(bottleneck_fea)
        #         output = self.decoder(z)
        #         return output, z

def quantizer(w, temperature=1, L=5):
    """
    Quantize feature map over L centers to obtain discrete $\hat{w}$
     + Centers: {-2,-1,0,1,2}
     + TODO:    Toggle learnable centers?
    """
    # with tf.variable_scope('quantizer_{}'.format(scope, reuse=reuse)):
    #     centers = tf.cast(tf.range(-2, 3), tf.float32)
    #     # Partition W into the Voronoi tesellation over the centers
    #     w_stack = tf.stack([w for _ in range(L)], axis=-1)
    #     w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)
    #
    #     smx = tf.nn.softmax(-1.0 / temperature * tf.abs(w_stack - centers), dim=-1)
    #     # Contract last dimension
    #     w_soft = tf.einsum('ijklm,m->ijkl', smx,
    #                        centers)  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))
    #
    #     # Treat quantization as differentiable for optimization
    #     w_bar = tf.round(tf.stop_gradient(w_hard - w_soft) + w_soft)
    #
    #     return w_bar

    centers = torch.arange(0, 5).float().cuda()
    # Partition W into the Voronoi tesellation over the centers
    w_stack = torch.stack([w for _ in range(L)], axis=-1)
    w_hard = torch.argmin(torch.abs(w_stack - centers), axis=-1).float() + torch.argmin(centers)

    smx = F.softmax(-1.0 / temperature * torch.abs(w_stack - centers), dim=-1)
    # Contract last dimension
    w_soft = torch.einsum('ijklm,m->ijkl', smx,
                       centers)  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))

    # Treat quantization as differentiable for optimization
    w_bar = torch.round((w_hard - w_soft).detach() + w_soft)

    return w_bar

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

