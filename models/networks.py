import torch
import logging
import models.modules.encoder_decoder_arch as encoder_decoder_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.module_util import init_weights
from models.modules.loss import VGGFeatureExtractor

logger = logging.getLogger('base')

####################
# define network
####################
#### Generator
def define_G(opt):
    net = encoder_decoder_arch.encoder_decoder(opt)
    #init_weights(net,init_type='kaiming')
    return net

#### Discriminator
def define_D(opt):
    if opt['network_D']['d_type'] == 'vgg':
        D = eval('SRGAN_arch.Discriminator_VGG')
    elif opt['network_D']['d_type'] == 'wgan':
        D = eval('SRGAN_arch.Discriminator_wgan')
    else:
        D = eval('SRGAN_arch.Discriminator_nlayer')
    print(D.__name__,'is used!')

    opt_net = opt['network_D']
    net = D(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    #init_weights(net,init_type='kaiming')
    return net


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 14 #14
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
