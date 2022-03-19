from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import cv2
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.models as models
import spectral
from tensorboardX import SummaryWriter
from models.modules.encoder_decoder_arch import Quantizer
#from BlueNoise import GetVoidAndClusterBlueNoise as BN_gen
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

TensorBoard_path = '../tb_logger/' + 'optimization'
pattern_path = './patterns'
test_img_path = './data/classic/meas_cameraman.mat'
measurement_path = './data/experiments/1-21-2022/DCT/7min/data.mat'
illum_pattern = 'DCT'

#### options
GT_size = 128
use_tensorboard = True

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
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Training configurations.
        self.batch_size = 1
        self.num_iters = 100
        self.num_iters_decay = 6000
        self.lr_update_step = 0.1
        '''
        SR=2%, lr=0.005
        SR=4%, lr=0.003
        SR=6%, lr=0.003
        SR=4%, lr=0.003  
        SNR=15dB, lr=0.0008
        '''
        self.lr =  5e-1 # Hadamard 0.0005  Fourier 0.002 DCT 0.0005
        self.beta1 = 0.5
        self.beta2 = 0.999
        # generate SPI masks
        self.Mask_generation()
        # get test images
        # dataset = create_dataset(test_img_path)
        # self.test_loader = create_dataloader(dataset)

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
            self.SPI_mask, self.SPI_index, self.truncated_index = SPI_util.get_pattern(illum_pattern, size)
            io.savemat(pattern_path + 'SPImask.mat', {'name': self.SPI_mask})
        else:
            _, self.SPI_index, _ = SPI_util.get_pattern(illum_pattern, size)
            tmp = io.loadmat(pattern_path + '/SPImask.mat')
            self.SPI_mask = tmp['Pattern']
        self.SPI_mask = torch.from_numpy(self.SPI_mask[np.newaxis,:, :, :]).float().to(self.device)

    def setup_optimizer(self, initial_rec):
        # initial_rec is a 2d image
        z = self.encoder(initial_rec.to(self.device))
        self.z = nn.Parameter(z.clone(), requires_grad=True) # randn

        #self.z = Variable(z, requires_grad=True)  # randn
        # k = torch.tensor(data=1e-3, dtype=float, device=self.device)
        # self.k = Variable(k, requires_grad=True)  # randn
        self.optimizer = torch.optim.Adam([self.z], self.lr, [self.beta1, self.beta2])#torch.optim.RMSprop([self.z], self.lr, alpha=.9)
        #torch.optim.Adam([self.z], self.lr, [self.beta1, self.beta2])
        self.scheduler = lr_scheduler.MultiStepLR_Restart(self.optimizer, [500,1000, 1500, 2000],
                                                          restarts=[],
                                                          weights=[],
                                                          gamma=0.5,
                                                          clear_state=[])  # 200, 400, 600, 1000, 1500

    def encoder(self, input):
        #input = SPI_util.norm(input)
        return self.model(input=input, is_output_z=True)

    # def encoder_decoder_output(self, input):
    #     #input = SPI_util.norm(input)
    #     output = self.model(input=input, is_bottleneck_fea=True, is_quantize=True)
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
        test_data = SPI_util.load_images(test_img_path, GT_size).to(self.device)
        self.test_measurement = SPI_util.load_measurement(measurement_path, illum_pattern, self.truncated_index).to(self.device)
        if illum_pattern == 'Fourier':
            self.test_measurement = self.test_measurement - torch.mean(self.test_measurement)
        io.savemat(pattern_path + '/measurement.mat', {'measurement': self.test_measurement.squeeze().cpu().numpy()})
        # plt.plot(self.test_measurement.squeeze().cpu())
        # plt.show()
        conv_rec = SPI_util.get_conv_rec(self.test_measurement, self.SPI_mask, self.SPI_index).to(self.device)
        # if illum_pattern == 'Fourier':
        #     tmp = copy.deepcopy(conv_rec)
        #     tmp[:,:,0:5,:]=0
        #     self.setup_optimizer(tmp)
        # else:
        # conv_rec = torch.from_numpy(cv.GaussianBlur(conv_rec.squeeze().cpu().numpy(), (5,5), 0)).to(self.device).unsqueeze(0).unsqueeze(1)
        self.setup_optimizer(conv_rec)

        #noise = 1e8*torch.normal(mean=0, std=torch.tensor(.5).expand_as(self.test_measurement)).cuda(self.test_measurement.device)
        print('Start training...')
        for iteration in range(5000):
            # =================================================================================== #
            #                                      2. Training                                    #
            # =================================================================================== #
            self.reset_grad()
            self.scheduler.step()

            img = self.decoder(self.z)#torch.clamp(self.decoder(self.z),0,0.4)
            rec_measurement = SPI_util.get_sim_meas(img*11.0, self.SPI_mask) # TODO: Hadamard:4,  DCT:9
            rec_measurement = SPI_util.diff_meas(rec_measurement) if illum_pattern is not "Fourier" else rec_measurement

            # loss calculation
            # loss  = 2 * self.cri_l1(rec_measurement, self.test_measurement)  # 2
            # #loss += 3 * self.cri_l1(self.z, rec_z)  # 3  big value (10) will lead to stripe artifacts
            # loss += 3 * self.cri_l2(self.z, self.Quantizer(self.z))  # 3
            # loss += 300 * self.cri_tv(img)  # 1

            loss  = 1 * self.cri_l2(rec_measurement, self.test_measurement) # 2
            loss += .1 * self.cri_sparsity(self.z)
            loss += .3 * torch.norm(self.z, p=1)
            loss += .1 * self.Quantizer.Eta(self.z, sigma=1.0)
            #loss += .7 * self.cri_tv(img)

            loss.backward()
            self.optimizer.step()

            # =================================================================================== #
            #                                     3. Validation                                   #
            # =================================================================================== #
            if iteration % 20 == 0:
                print("max = %.3f, min = %.3f"%(img.max(),img.min()))
                output = SPI_util.normalize_0_to_1(img.detach()).permute(0,1,3,2)
                test_data = SPI_util.normalize_0_to_1(test_data)
                conv_recc = SPI_util.normalize_0_to_1(conv_rec).permute(0,1,3,2)
                SR_cube = output.cpu().numpy()
                GT_data = test_data.detach().cpu().numpy()
                CONV_REC = conv_recc.cpu().numpy()

                # Calculate PSNR
                conv_psnr = util.calculate_psnr(CONV_REC * 256, GT_data * 256)
                deep_psnr = util.calculate_psnr(SR_cube * 256, GT_data * 256)
                conv_ssim = util.calculate_ssim(CONV_REC * 256, GT_data * 256)
                deep_ssim = util.calculate_ssim(SR_cube * 256, GT_data * 256)
                conv_mse = util.calculate_mse(CONV_REC * 256, GT_data * 256)
                deep_mse = util.calculate_mse(SR_cube * 256, GT_data * 256)

                self.tb_logger.add_scalar('Conv/PSNR', conv_psnr, iteration)
                self.tb_logger.add_scalar('Conv/SSIM', conv_ssim, iteration)
                self.tb_logger.add_scalar('Conv/MSE', conv_mse, iteration)
                self.tb_logger.add_scalar('Deep/PSNR', deep_psnr, iteration)
                self.tb_logger.add_scalar('Deep/SSIM', deep_ssim, iteration)
                self.tb_logger.add_scalar('Deep/MSE', deep_mse, iteration)
                self.tb_logger.add_scalar('loss/loss', loss, iteration)

                output_grid = make_grid(output[0:4, ...], normalize=True)
                test_data_grid = make_grid(test_data[0:4, ...], normalize=True)
                conv_rec_grid = make_grid(conv_recc[0:4, ...], normalize=True)

                self.tb_logger.add_image('Res/Learned', output_grid, global_step=iteration, dataformats='CHW')
                self.tb_logger.add_image('Res/GT', test_data_grid, global_step=iteration, dataformats='CHW')
                self.tb_logger.add_image('Res/Conv.', conv_rec_grid, global_step=iteration, dataformats='CHW')
                #self.tb_logger.add_image('Res/bottleneck_fea_img', self.z.squeeze(), global_step=iteration, dataformats='CHW')
                print('iteration = ', iteration)

        print('f_conv_mse = {:.1f}, '.format(conv_mse))
        print('f_deep_mse = {:.1f}, '.format(deep_mse))
        print('f_conv_psnr = {:.2f}, '.format(conv_psnr))
        print('f_deep_psnr = {:.2f}, '.format(deep_psnr))
        print('f_conv_ssim = {:.4f}, '.format(conv_ssim))
        print('f_deep_ssim = {:.4f}, '.format(deep_ssim))
        #util.save_img(util.tensor2img(output_img), 'res.jpg')

def main():
    filelist = [f for f in os.listdir(pattern_path) if f.endswith(".jpg")]
    for f in filelist:
        os.remove(os.path.join(pattern_path, f))
    os.system('rm -r /data/jia/Networks/single-pixel/CS-GAN-test1/tb_logger/optimization')

    my_whole_seed = 222
    random.seed(my_whole_seed)
    np.random.seed(my_whole_seed)
    torch.manual_seed(my_whole_seed)
    torch.cuda.manual_seed_all(my_whole_seed)
    torch.cuda.manual_seed(my_whole_seed)
    np.random.seed(my_whole_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(my_whole_seed)

    solver = Solver()
    solver.train()

if __name__ == '__main__':
    main()
