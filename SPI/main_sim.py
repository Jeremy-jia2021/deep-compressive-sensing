from torch.autograd import Variable
import scripts.operation as operation
import torch
import torch.nn as nn
from utils import util
from data import util as data_util
import numpy as np
import cv2
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.models as models
import spectral
from tensorboardX import SummaryWriter
from models.modules.encoder_decoder_arch import Quantizer
import cv2 as cv
import os
from PIL import Image
import math
import shutil
import copy
import scipy.io as io
import util as SPI_util
from SPI.data.data_loader import create_dataloader, create_dataset
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import argparse
import options.options as option
import utils.util as util
from models import create_model
import utils.lr_scheduler as lr_scheduler
import utils.admm as optimizer
import time

#%% options
TensorBoard_path = '../tb_logger/' + 'optimization'
pattern_path = '../../SPI_patterns'
test_img_path = './data/classic/lena.bmp'
illum_pattern = 'DCT'
SNR = 30

GT_size = 128
use_tensorboard = True
#%%

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

class Solver(object):
    """Solver for training and testing StarGAN."""
    def __init__(self):
        """Initialize configurations."""
        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #### create model
        model = create_model(opt)
        self.model = model.netG
        for param in self.model.parameters():
            param.requires_grad = False

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        # Training configurations.
        self.batch_size = 1
        self.num_iters = 100
        self.lr = .5e-1
        self.beta1 = 0.9
        self.beta2 = 0.999
        # generate SPI masks
        self.Mask_generation()

        # Build the model and tensorboard.
        self.cri_l2 = nn.MSELoss().to(self.device)
        self.cri_tv = SPI_util.TVLoss().to(self.device)
        self.cri_l1 = nn.L1Loss().to(self.device)
        self.cri_sl1 = nn.SmoothL1Loss().to(self.device)
        self.Net_VGG = SPI_util.Net_VGG().to(self.device)
        if use_tensorboard:
            self.build_tensorboard()

        # load Quantizer defined in the original encoder_decorder network
        self.Quantizer = Quantizer().to(self.device)

    def cri_sparsity(self, fea):
        fea_space = fea.view(fea.size(0), -1)
        dim = fea_space.size(1)
        sparseness = (dim**0.5 - torch.norm(fea_space, p=1, dim=1) / torch.norm(fea_space, p=2, dim=1))/(dim**0.5 - 1)
        return 1.0/sparseness.mean()

    def Mask_generation(self, size=GT_size, is_update=True):
        if is_update:
            self.SPI_mask, self.SPI_index, _ = SPI_util.get_pattern(illum_pattern, size)
            total = self.SPI_mask.shape[0]
            SPI_mask = self.SPI_mask[0:total//2,] - self.SPI_mask[total//2:,]
            #io.savemat(pattern_path + '/SPImask_'+illum_pattern+'80.mat', {'Pattern': SPI_mask})
            torch.save(SPI_mask, pattern_path + '/SPImask_'+illum_pattern+'05.pt')
        else:
            _, self.SPI_index, _ = SPI_util.get_pattern(illum_pattern, size)
            tmp = io.loadmat(pattern_path + '/SPImask.mat')
            self.SPI_mask = tmp['Pattern']
        self.SPI_mask = torch.from_numpy(self.SPI_mask[np.newaxis,:, :, :]).float().to(self.device)

    def setup_optimizer(self, initial_rec):
        z = self.encoder_output(initial_rec.to(self.device))
        self.z = Variable(z, requires_grad=True)  # randn
        self.optimizer = torch.optim.RMSprop([self.z], self.lr, alpha=.9)#torch.optim.Adam([self.z], self.lr, [self.beta1, self.beta2])
        milestones = [500, 700, 800, 900, 1000, 1200] # if SNR > 25 else [500, 1000, 1500]
        self.scheduler = lr_scheduler.MultiStepLR_Restart(self.optimizer, milestones,
                                                     restarts=[],
                                                     weights=[],
                                                     gamma=0.5,
                                                     clear_state=[])  # 200, 400, 600, 1000, 1500


    def encoder_output(self, input_img):
        # input_img = SPI_util.norm(input_img)
        return self.model(input=input_img, is_output_z=True)

    # def encoder_decoder_output(self, input_fea):
    #     # input_fea = SPI_util.norm(input_fea)
    #     output = self.model(input=input_fea, is_bottleneck_fea=True)
    #     rec_z = self.model(input=output, is_output_z=True)
    #     return output, rec_z

    def decoder(self, input):
        #input = SPI_util.norm(input)
        #input.clamp(0,1)
        return self.model(input=input, is_bottleneck_fea=True, is_quantize=True)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.tb_logger = SummaryWriter(log_dir=TensorBoard_path)

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def train(self):
        f_conv_psnr = []
        f_deep_psnr = []
        f_conv_ssim = []
        f_deep_ssim = []
        f_conv_mse  = []
        f_deep_mse  = []

        input_path = []
        if os.path.isfile(test_img_path):
            input_path = test_img_path
            total = 1
        else:
            try:
                count = 0
                for file in os.listdir(test_img_path):
                    input_path.append(os.path.join(test_img_path, file))
                    count += 1
                    if count > 20:
                        break
                total = len(input_path)
            except:
                raise ValueError(test_img_path+' is an invalid path')

        for loops in range(total): # [5, 17, 9]
            if loops == 9:
                continue
            try:
                test_data = SPI_util.load_images(input_path[loops], GT_size).to(self.device)
            except:
                test_data = SPI_util.load_images(input_path, GT_size).to(self.device)
            idea_meas = SPI_util.get_sim_meas(test_data, self.SPI_mask)
            # io.savemat(pattern_path + '/idea_meas.mat', {'idea_meas': idea_meas.cpu().numpy()})
            # plt.plot(idea_meas.squeeze().cpu())
            # plt.show()

            white_image = torch.ones_like(test_data) * 0.02 # Fourier:0.1, others:0.02
            white_meas = SPI_util.get_sim_meas(white_image, self.SPI_mask)
            noise = util.add_Gaussian_noise(white_meas, SNR=SNR)
            meas = idea_meas + noise.to(self.device).unsqueeze(2).unsqueeze(3)

            # noise = 1e1*torch.normal(mean=0, std=torch.tensor(.5).expand_as(idea_meas)).cuda()
            # meas = idea_meas + noise
            self.test_measurement = SPI_util.diff_meas(meas) if illum_pattern is not "Fourier" else meas

            conv_rec = SPI_util.get_conv_rec(self.test_measurement, self.SPI_mask, self.SPI_index).to(self.device)
            # conv_rec = torch.from_numpy(cv.GaussianBlur(conv_rec.squeeze().cpu().numpy(), (3,3), 0)).to(self.device).unsqueeze(0).unsqueeze(1)
            self.setup_optimizer(conv_rec)

            print('Start iterating... {: d}'.format(loops))
            start = time.time()
            for iteration in range(50000): #2500
                # =================================================================================== #
                #                                      2. Training                                    #
                # =================================================================================== #
                self.reset_grad()
                # Decay learning rates.
                self.scheduler.step()

                img = self.decoder(self.z)
                rec_measurement = SPI_util.get_sim_meas(img*1, self.SPI_mask)* 1# self.k 1.3
                rec_measurement = SPI_util.diff_meas(rec_measurement) if illum_pattern is not "Fourier" else rec_measurement

                # loss calculation
                loss  = 1 * self.cri_l2(rec_measurement, self.test_measurement*.8) # 2
                loss += .01 * self.cri_sparsity(self.z)
                loss += .05 * torch.norm(self.z, p=1)
                #loss += 3 * self.cri_l2(self.z, self.Quantizer(self.z))
                loss += .01 * self.Quantizer.Eta(self.z, sigma=1.0)
                #loss += .1 * self.cri_tv(img)


                # loss  = 2 * self.cri_l1(rec_measurement, self.test_measurement)  # 2
                # loss += 3 * self.cri_l1(self.z, rec_z)  # 3  big value (10) will lead to stripe artifacts
                # loss += 3 * self.cri_l2(self.z, self.Quantizer(self.z))  # 3
                # loss += 30 * self.cri_tv(output)  # 1

                loss.backward()
                self.optimizer.step()

                # =================================================================================== #
                #                                     3. Validation                                   #
                # =================================================================================== #
                if iteration !=0 and iteration % 20 == 0:
                    end = time.time()
                    # normalize images
                    output = SPI_util.normalize_0_to_1(img.detach())
                    test_data = SPI_util.normalize_0_to_1(test_data)
                    conv_rec = SPI_util.normalize_0_to_1(conv_rec)
                    SR_cube = output.detach().cpu().numpy()
                    GT_data = test_data.detach().cpu().numpy()
                    CONV_REC = conv_rec.cpu().numpy()

                    # Calculate PSNR
                    # cv.PSNR(CONV_REC * 256, GT_data * 256)
                    conv_psnr = util.calculate_psnr(CONV_REC * 255, GT_data * 255)
                    deep_psnr = util.calculate_psnr(SR_cube * 255, GT_data * 255)
                    conv_ssim = util.calculate_ssim(CONV_REC * 255, GT_data * 255)
                    deep_ssim = util.calculate_ssim(SR_cube * 255, GT_data * 255)
                    conv_mse = util.calculate_mse(CONV_REC * 255, GT_data * 255)
                    deep_mse = util.calculate_mse(SR_cube * 255, GT_data * 255)

                    self.tb_logger.add_scalar('Conv/PSNR', conv_psnr, iteration)
                    self.tb_logger.add_scalar('Conv/SSIM', conv_ssim, iteration)
                    self.tb_logger.add_scalar('Conv/MSE', conv_mse, iteration)
                    self.tb_logger.add_scalar('Deep/PSNR', deep_psnr, iteration)
                    self.tb_logger.add_scalar('Deep/SSIM', deep_ssim, iteration)
                    self.tb_logger.add_scalar('Deep/MSE', deep_mse, iteration)
                    self.tb_logger.add_scalar('loss/loss', loss, iteration)

                    output_grid = make_grid(output[0:6, ...], normalize=True)
                    test_data_grid = make_grid(test_data[0:6, ...], normalize=True)
                    conv_rec_grid = make_grid(conv_rec[0:6, ...], normalize=True)

                    self.tb_logger.add_image('Res/Learned', output_grid, global_step=iteration, dataformats='CHW')
                    self.tb_logger.add_image('Res/GT', test_data_grid, global_step=iteration, dataformats='CHW')
                    self.tb_logger.add_image('Res/Conv.', conv_rec_grid, global_step=iteration, dataformats='CHW')
                    #self.tb_logger.add_image('Res/bottleneck_fea_img', self.z.squeeze(), global_step=iteration, dataformats='CHW')
                    print('iteration = {:d}, duration = {:.3f}'.format(iteration, end-start))
                    print('Loss on Quantizer = {:.5f}'.format(self.cri_l1(self.z, self.Quantizer(self.z)) ))
                    start = time.time()
                  #  print('output_max = {:.3f}, k1 = {:.6f}'.format(torch.max(output[0,0,...]), self.k))

            f_conv_psnr.append(conv_psnr)
            f_deep_psnr.append(deep_psnr)
            f_conv_ssim.append(conv_ssim)
            f_deep_ssim.append(deep_ssim)
            f_conv_mse.append(conv_mse)
            f_deep_mse.append(deep_mse)

            rmse_percent = (-deep_mse + conv_mse) / conv_mse * 100
            psnr_percent = (deep_psnr - conv_psnr) / conv_psnr * 100
            ssim_percent = (deep_ssim - conv_ssim) / conv_ssim * 100
            print('rmse = {:.2f}%, '.format(rmse_percent))
            print('psnr = {:.2f}%, '.format(psnr_percent))
            print('ssim = {:.2f}%, '.format(ssim_percent))

        f_conv_psnr = np.array(f_conv_psnr)
        f_deep_psnr = np.array(f_deep_psnr)
        f_conv_ssim = np.array(f_conv_ssim)
        f_deep_ssim = np.array(f_deep_ssim)
        f_conv_mse = np.array(f_conv_mse)
        f_deep_mse = np.array(f_deep_mse)
        print('deep_mse,  conv_mse  = {:.1f}, {:.1f}'.format(np.mean(f_deep_mse), np.mean(f_conv_mse)))
        print('deep_psnr, conv_psnr = {:.2f}, {:.2f}'.format(np.mean(f_deep_psnr), np.mean(f_conv_psnr)))
        print('deep_ssim, conv_ssim = {:.4f}, {:.4f}'.format(np.mean(f_deep_ssim), np.mean(f_conv_ssim)))

        rmse_percent = (-f_deep_mse + f_conv_mse) / f_conv_mse * 100
        psnr_percent = (f_deep_psnr - f_conv_psnr) / f_conv_psnr * 100
        ssim_percent = (f_deep_ssim - f_conv_ssim) / f_conv_ssim * 100
        # rmse_percent, psnr_percent, ssim_percent = [rmse_percent, psnr_percent, ssim_percent for x in ssim_percent if x>0]
        print('rmse = {:.2f} +/- {:.2f}%, '.format(np.mean(rmse_percent), np.std(rmse_percent)))
        print('psnr = {:.2f} +/- {:.2f}%, '.format(np.mean(psnr_percent), np.std(psnr_percent)))
        print('ssim = {:.2f} +/- {:.2f}%, '.format(np.mean(ssim_percent), np.std(ssim_percent)))

def main():
    filelist = [f for f in os.listdir(pattern_path) if f.endswith(".jpg")]
    for f in filelist:
        os.remove(os.path.join(pattern_path, f))
    os.system('rm -r /mnt/home/jeremy/networks/CS-GAN-test1-tju/tb_logger/optimization')

    my_whole_seed = 222
    random.seed(my_whole_seed)
    np.random.seed(my_whole_seed)
    torch.manual_seed(my_whole_seed)
    torch.cuda.manual_seed_all(my_whole_seed)
    torch.cuda.manual_seed(my_whole_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(my_whole_seed)

    solver = Solver()
    solver.train()

if __name__ == '__main__':
    main()