import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, norm_layer=nn.InstanceNorm2d):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True),
            #norm_layer(dim_out),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True))
            #norm_layer(dim_out))

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
        self.is_train = opt['is_train']
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

        layers = []
        layers.append(nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3, bias=True))
        #layers.append(norm_layer(ngf))
        layers.append(nn.ReLU(inplace=True))
        n_downsampling = 4

        # Down-sampling layers.
        curr_dim = ngf
        for i in range(n_downsampling):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=True))
            #layers.append(norm_layer(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.encoder = nn.Sequential(*layers)

        layers = []
        # Bottleneck layers.
        for i in range(n_blocks):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm_layer=norm_layer))
        # Up-sampling layers.
        for i in range(n_downsampling):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=True))
            #layers.append(norm_layer(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers.append(nn.Conv2d(curr_dim, output_nc, kernel_size=7, stride=1, padding=3, bias=True))
        layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*layers)

        self.Quantizer = Quantizer()


    def forward(self, input, current_step=0,
                is_bottleneck_fea=False,
                is_output_z=False,
                is_output_range=False,
                is_quantize=False):

        if self.is_train:
            is_quantize = True

        if is_output_z:
            bottleneck_fea = self.encoder(input)
            if is_quantize:
                return self.Quantizer(bottleneck_fea)
            else:
                return bottleneck_fea

        else:
            if is_bottleneck_fea:
                if is_quantize:
                    z = self.Quantizer(input)
                else:
                    z = input
                return self.decoder(z)
            else:
                bottleneck_fea = self.encoder(input)
                z = self.Quantizer(bottleneck_fea) #if current_step>100 or current_step==-1 else bottleneck_fea
                #z = bottleneck_fea
                range = [torch.min(z), torch.max(z)]
                output = self.decoder(z)
                if is_output_range:
                    return output, z, range
                else:
                    return output, z


class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        levels_first, levels_last, L = 0, 1, 20 #0, 0.5, 10
        self.centers = nn.Parameter(torch.linspace(levels_first, levels_last, L), requires_grad=False)
        self.sigma = 1.0 # 1.0e10 fof L=512, otherwise 1.0e20
        self.L = L #self.centers.size()[0]

    def __repr__(self):
        return '{}(sigma={})'.format(self._get_name(), self.sigma)

    def forward(self, x):
        w_stack = torch.stack([x for _ in range(self.L)], axis=-1)
        d = torch.pow(w_stack - self.centers, 2)
        #d = torch.abs(w_stack - self.centers)
        smx = F.softmax(-self.sigma * d, dim=-1)
        # Contract last dimension
        w_soft = torch.einsum('ijklm,m->ijkl', smx, self.centers)  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))
        w_hard = self.centers[torch.argmin(d.detach(), axis=-1)]
        # Treat quantization as differentiable for optimization
        w_soft.data = w_hard  # assign data, keep gradient
        """
            Refer to https://blog.csdn.net/dss_dssssd/article/details/89526623
            The sentence below is an safe alternative
            w_soft = torch.round((w_hard - w_soft).detach() + w_soft)
        """
        return w_soft

    def Eta(self, x, sigma=0.1):
        w_stack = torch.stack([x for _ in range(self.L)], axis=-1)
        P = torch.zeros_like(self.centers)

        d = torch.pow(w_stack - self.centers, 2)
        smx = F.softmax(-sigma * d, dim=-1)
        index = torch.argmin(d.detach(), axis=-1)
        for i in range(self.L):
            tmp = (index == i).sum()/(self.L*x.numel())
            P[i] = 1e-40 if tmp==0 else tmp
        # Contract last dimension
        w_soft = -torch.einsum('ijklm,m->ijkl', smx, torch.log(P))/x.numel()  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))
        return w_soft.sum()

    def Etaa(self, x, sigma=0.1):
        #   -0.0588, -0.0510, -0.0431, -0.0353, -0.0275, -0.0196, -0.0118, -0.0039,
        levels_first, levels_last, L = 0, 1, 40#.0050,1,100#.0067, 1, 75 #0.0039, 1, 128#0.0526, 1, 10 #0.0256, 1, 20
        centers = nn.Parameter(torch.linspace(levels_first, levels_last, L), requires_grad=False).to(x.device)

        w_stack = torch.stack([x for _ in range(L)], axis=-1)
        P = torch.zeros_like(centers)

        d = torch.pow(w_stack - centers, 2)
        smx = F.softmax(-sigma * d, dim=-1).sum([0,1,2,3])/x.numel()
        index = torch.argmin(d, axis=-1)
        for i in range(L):
            tmp = (index == i).sum()/(L*x.numel())
            P[i] = 1e-40 if tmp==0 else tmp
        # Contract last dimension
        w_soft = -smx*torch.log(P)

        #-torch.einsum('ijklm,m->ijkl', smx, torch.log(P))/x.numel()  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))
        return w_soft.sum()


    # def Eta(self, x):
    #     w_stack = torch.stack([x for _ in range(self.L)], axis=-1)
    #     P = torch.zeros_like(self.centers)
    #
    #     d = torch.pow(w_stack - self.centers, 2)
    #     smx = F.softmax(-self.sigma * d, dim=-1)
    #     for i in range(self.L):
    #         P[i] = (x == self.centers[i]).sum()/(self.L*x.numel()) + 1e-40
    #
    #     # Contract last dimension
    #     w_soft = -torch.einsum('ijklm,m->ijkl', smx, torch.log(P))/(self.L*x.numel())  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))
    #
    #     return w_soft.sum()
