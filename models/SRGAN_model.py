import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss,GradientPenaltyLoss

import os

logger = logging.getLogger('base')


class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            self.netD = DataParallel(self.netD)

            self.netG.train()
            self.netD.train()
        else:
            self.netG.eval()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].cuda(non_blocking=True)#.to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.cuda(non_blocking=True)#.to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.fake_H, quantized_fea, fea_range = self.netG(self.var_L, current_step=step, is_output_range=True)
        # non_zero_percentage = torch.nonzero(torch.abs(quantized_fea)>0.3).size(0)/(320*320*3)*100
        # print('Nonzero (>0.1) percentage: '+str(non_zero_percentage))
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

            if self.l_gan_w > 0:
                pred_g_fake = self.netD(self.fake_H)
                if self.opt['train']['gan_type'] == 'gan':
                    l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                elif self.opt['train']['gan_type'] == 'ragan':
                    pred_d_real = self.netD(self.var_ref).detach()
                    l_g_gan = self.l_gan_w * (
                        self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                elif self.opt['train']['gan_type'] == 'wgan':
                    l_g_gan = self.cri_gan(pred_g_fake, True)
                else:
                    raise NotImplementedError('GAN type [{:s}] is not found'.format(self.opt['train']['gan_type']))
                l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        if self.l_gan_w > 0:
            for p in self.netD.parameters():
                p.requires_grad = True

            self.optimizer_D.zero_grad()
            l_d_total = 0
            pred_d_real = self.netD(self.var_ref)
            pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
            if self.opt['train']['gan_type'] == 'gan':
                l_d_real = self.cri_gan(pred_d_real, True)
                l_d_fake = self.cri_gan(pred_d_fake, False)
                l_d_total = l_d_real + l_d_fake
            elif self.opt['train']['gan_type'] == 'ragan':
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                l_d_total = (l_d_real + l_d_fake) / 2
            elif self.opt['train']['gan_type'] == 'wgan':
                l_d_gp = GradientPenaltyLoss(self.netD, self.var_ref, self.fake_H.detach())
                l_d_total = self.cri_gan(pred_d_fake, False) + self.cri_gan(pred_d_real, True) + 10*l_d_gp
            else:
                raise NotImplementedError('GAN type [{:s}] is not found'.format(self.opt['train']['gan_type']))
            l_d_total.backward()
            self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['G/l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['G/l_g_fea'] = l_g_fea.item()
            if self.l_gan_w > 0:
                self.log_dict['G/l_g_gan'] = l_g_gan.item()

            # self.log_dict['fea_range/low'] = torch.min(fea_range[0]).item()
            # self.log_dict['fea_range/upper'] = torch.max(fea_range[1]).item()

        if self.l_gan_w > 0:
            self.log_dict['D/D_real'] = torch.mean(pred_d_real.detach())
            self.log_dict['D/D_fake'] = torch.mean(pred_d_fake.detach())
            if self.opt['train']['gan_type'] == 'wgan':
                self.log_dict['D/l_d_gp'] = l_d_gp.item()
                self.log_dict['D/l_d_total'] = l_d_total.item()
            else:
                self.log_dict['D/l_d_real'] = l_d_real.item()
                self.log_dict['D/l_d_fake'] = l_d_fake.item()

    def test(self, current_step):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, self.bottleneck_fea = self.netG(input=self.var_L, current_step=current_step)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L
        out_dict['SR'] = self.fake_H
        out_dict['bottleneck_fea'] = self.bottleneck_fea
        if need_GT:
            out_dict['GT'] = self.var_H
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)

        # save entire G and D
        save_filename = 'entire_G.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        torch.save(self.netG, save_path)
