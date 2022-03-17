import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import warnings
from data import util as data_util
import scipy.io as io
import scripts.operation as operation
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from utils import util


INVS_IMAGE = False
FLIP_IMAGE = False
ROT_90 = False
####################
# %% miscellaneous
####################
def index_element(src, trg):
    '''
    index the elements of truncated spectrum in the original spectrum
    :param src:
    :param trg:
    :return:
    '''
    truncated_index = []
    index_arr = np.array(src, dtype=float)
    for i in range(index_arr[0, :].size):
        truncated_index.append(list((index_arr[:, i] == trg).all(axis=1)).index(True))

    return truncated_index

def get_pattern(basis, size):
    '''
    supporting basis: DCT Fourier Hadamard
    '''
    #%%
    if basis == 'DCT':
        '''
        .5% 10 (82)    86
        1%  15 (189)   163
        2%  20 (331)   327
        4%  29 (683)   655
        6%  35 (995)   983
        8%  41 (1353)  1310
        10% 45 (1632)  1638
        16% 57 (2600)  2621
        32% 82 (5354)   5242
        50% 102 (8269)  8192
        '''
        u = np.arange(0, 128, 1)
        v = np.arange(0, 128, 1)
        uu, vv = np.meshgrid(u, v, sparse=False, indexing='ij')
        ### circle
        index = np.where((np.power(uu, 2) + np.power(vv, 2)) < 10 ** 2)
        ### the spectrum of original measurement
        index_full = np.array(np.where((np.power(uu, 2) + np.power(vv, 2)) < 8269 ** 2), dtype=float).T
        truncated_index = index_element(index, index_full)

        uu = uu[index]
        vv = vv[index]
        # uu[index]=1000
        # plt.imshow(uu, cmap='gray')
        # plt.show()
        xv, yv = np.meshgrid(range(size), range(size), sparse=False, indexing='ij')
        SPI_mask = []
        for i in range(np.size(uu)):  # np.size(uu)
            u, v = uu[i], vv[i]
            c_u = 2**-0.5 if u==0 else 1
            c_v = 2**-0.5 if v==0 else 1
            mask = c_u*c_v*2 / np.sqrt(size * size) * np.cos((2 * xv + 1) * u * np.pi / (2 * size)) * np.cos((2 * yv + 1) * v * np.pi / (2 * size))
            mask[mask < 0] = 0
            SPI_mask.append(mask)
        for i in range(np.size(uu)):  # np.size(uu)
            u, v = uu[i], vv[i]
            c_u = 2**-0.5 if u==0 else 1
            c_v = 2**-0.5 if v==0 else 1
            mask = - c_u*c_v*2 / np.sqrt(size * size) * np.cos((2 * xv + 1) * u * np.pi / (2 * size)) * np.cos((2 * yv + 1) * v * np.pi / (2 * size))
            mask[mask < 0] = 0
            SPI_mask.append(mask)

    # %%
    elif basis == 'Fourier':
        '''
        1%  .0570 (166)   163
        2%  .0830 (341)   327
        4%  .1120 (646)   655
        5%  .1200 
        6%  .1380 (983)   983
        8%  .1600 (1319)  1310 
        10% .1780 (1644)  1638
        '''
        u = np.arange(-0.448, 0.448, 0.0078) #(-0.448, 0.6, 0.0078)
        v = np.arange(-0.448, 0.448, 0.0078)
        uu, vv = np.meshgrid(u, v, sparse=False, indexing='ij')
        ### circle
        index = np.where((np.power(uu, 2) + np.power(vv, 2)) < 0.1120 ** 2)
        ### the spectrum of original measurement
        index_full = np.array(np.where((np.power(uu, 2) + np.power(vv, 2)) < 0.1600 ** 2), dtype=float).T
        truncated_index = index_element(index, index_full)

        uu = uu[index]
        vv = vv[index]
        xv, yv = np.meshgrid(range(size), range(size), sparse=False, indexing='ij')

        SPI_mask = []
        phi = [0, 2.0*np.pi/3.0, 4.0*np.pi/3.0]
        for i in range(len(phi)):
            for j in range(np.size(uu)):
                mask = 0+np.cos(2*np.pi*uu[j]*xv + 2*np.pi*vv[j]*yv + phi[i])
                SPI_mask.append(mask)

    # %%
    elif basis == 'Hadamard':
        SPI_mask = []
        H = hadamard(size ** 2)
        sub_mats = H.reshape(size, size, size ** 2)
        TVs = util.Total_var(sub_mats)
        index = np.argsort(TVs)

        samplings = int(size ** 2 * 0.32)
        index = index[0:samplings]
        tmask1 = np.array(sub_mats[..., index])
        tmask1[tmask1 < 0] = 0
        tmask2 = -np.array(sub_mats[..., index])
        tmask2[tmask2 < 0] = 0
        tmask = np.concatenate((tmask1, tmask2),axis=2)
        SPI_mask = np.transpose(tmask, (2, 0, 1)).tolist()

        truncated_index = range(samplings)
    else:
        raise ValueError('Invalid illumination pattern!')

    SPI_mask = SPI_mask / np.max(SPI_mask)
   # SPI_mask = SPI_mask[0:360]
    return SPI_mask, index, truncated_index

def load_measurement(measurement_path, illum_pattern, truncated_index):
    if illum_pattern == 'DCT':
        cof = 4.e3 # TODO: DCS:5e3, Wavelet:2e3
    elif illum_pattern == 'Hadamard':
        cof = 3.e3 # TODO: DCS:5e3, Wavelet:.9e3
    else:
        cof = 10e-3 #9e-3

    raw = []
    if os.path.isfile(measurement_path):
        tmp = io.loadmat(measurement_path)['data'].astype(float)
        tmp = tmp/tmp.max()*cof
        if INVS_IMAGE:
            tmp = -tmp

        raw = tmp[truncated_index].squeeze()
    else:
        meas_dir = os.path.join(measurement_path)
        try:
            for file in os.listdir(meas_dir):
                filename = os.path.join(meas_dir, file)
                if ".mat" in filename:
                    tmp = io.loadmat(filename)['data'].astype(float).squeeze() * cof
                    if INVS_IMAGE:
                        tmp = -tmp
                    tmp = tmp[truncated_index]
                    raw.append(tmp)
            print('Fourier sampling found!')
        except:
            raise ValueError(measurement_path+'is an invalid path')
        raw = np.hstack(raw).squeeze()
    # coff = 5.0e3/np.max(raw)
    # raw = raw*coff
    return torch.from_numpy(raw[np.newaxis, :, np.newaxis, np.newaxis]).float()
####################
# %% miscellaneous
####################

def normalize_0_to_1(mat):
    for i in range(mat.shape[0]):
        mat[i, 0, :, :] = (mat[i, 0, :, :] - mat[i, 0, :, :].min()) / (mat[i, 0, :, :].max() - mat[i, 0, :, :].min())
    return mat

def diff_meas(meas):
    lens_per_phase = int(meas.shape[1]/2)
    return meas[:, 0*lens_per_phase:1*lens_per_phase, :, :] - meas[:, 1*lens_per_phase:2*lens_per_phase, :, :]

def get_conv_rec(meas, mask, index):
    '''
    :param meas: size of [5, 933, 1, 1]
    :param mask: size of [1, 933, 128, 128]
    :param index: length of 2
    :return:
    '''
    if type(index) is not tuple or meas.shape[1] == len(index[0]):
        tmp = torch.mul(meas, diff_meas(mask))
        conv_rec = normalize_0_to_1(torch.sum(tmp, dim=1, keepdim=True))
        #conv_rec[conv_rec<0.1]=0
    else:
        lens_per_phase = int(meas.shape[1]/3)

        D1 = np.zeros([meas.shape[0], mask.shape[2], mask.shape[3]])
        D1[..., index[0], index[1]] = meas[..., 0*lens_per_phase:1*lens_per_phase, 0, 0].cpu()
        D2 = np.zeros([meas.shape[0], mask.shape[2], mask.shape[3]])
        D2[..., index[0], index[1]] = meas[..., 1*lens_per_phase:2*lens_per_phase, 0, 0].cpu()
        D3 = np.zeros([meas.shape[0], mask.shape[2], mask.shape[3]])
        D3[..., index[0], index[1]] = meas[..., 2*lens_per_phase:3*lens_per_phase, 0, 0].cpu()

        meass = (2*D1 - D2 - D3) + 3**0.5*1j*(D2-D3)

        conv_rec = torch.from_numpy(abs(np.fft.ifft2(np.fft.ifftshift(meass, axes=[1, 2]))))
        conv_rec = conv_rec[:, np.newaxis, :, :]

    return normalize_0_to_1(conv_rec.float())*0.9

def get_sim_meas(img, mask):
    res=[]
    input = img.reshape(img.shape[0], img.shape[1], img.shape[2]*img.shape[3], img.shape[1])
    SPI_mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[0], mask.shape[2]*mask.shape[3])
    for i in range(input.shape[0]):
        res.append(torch.matmul(SPI_mask[0,...], input[i,...]))
    return torch.stack(res, dim=0)


def get_sim_meass(meas, mask):
    res=[]
    SPI_mask = mask.reshape(mask.shape[0], mask.shape[2]*mask.shape[3], mask.shape[0], mask.shape[1])
    for i in range(meas.shape[0]):
        res.append(torch.matmul(SPI_mask[0,...], meas[i,...].permute([1,0,2])))
    res = torch.stack(res, dim=0)
    res = res.reshape(res.shape[0], 1, int(res.shape[1]**0.5), int(res.shape[1]**0.5))
    return res


####################
# %% miscellaneous
####################

def load_images(test_img_path, size):
    if os.path.isfile(test_img_path):
        if ".mat" in test_img_path:
            test_data = io.loadmat(test_img_path)['data'].astype(np.float32)
        else:
            test_data = data_util.read_img_grayscale(None, test_img_path)
    else:
        raise ValueError(test_img_path + 'is an invalid path')

    # rows, cols = test_data.shape
    # M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -90, 1)
    # test_data = cv2.warpAffine(test_data, M, (cols, rows))
    if FLIP_IMAGE:
        test_data = cv2.flip(test_data, 0)
    if ROT_90:
        test_data = cv2.transpose(test_data)
        test_data = cv2.flip(test_data, 0)
    '''
    test_dataa = test_data[0:158, 0:161]  #DHT/cameraman_100ms
    test_dataa = test_data[0:158, 0:161]  #DHT/cameraman_50ms
    test_dataa = test_data[0:158, 0:161]  #DHT/cameraman_25ms
    test_dataa = test_data[3:161,3:161]   #DHT/ghost_100ms
    test_dataa = test_data[2:161,3:161]   #DHT/ghost_50ms
    test_dataa = test_data[2:161,3:161]   #DHT/ghost_25ms
    test_dataa = test_data[0:-4, :]       #DFT/cameraman
    test_dataa = test_data[4:161, 3:161]  #DFT/ghost
    test_dataa = test_data[3:, :]         #DCT/cameraman
    test_dataa = test_data[3:161,3:161]   #DCT/ghost
    test_dataa = test_data[2:-3, 3:]      #DCT/cameraman_light
    test_dataa = test_data                #simulation
    '''
    test_dataa = test_data[3:, :]         #DCT/cameraman

    test_dataa = cv2.resize(np.copy(test_dataa), (size, size), interpolation=cv2.INTER_CUBIC)  # fx=0.5, fy=0.5,
    test_dataa = test_dataa[..., np.newaxis]  # keep only one channel
    test_dataa = operation.normalize_0_to_1(test_dataa)
    test_dataa = np.transpose(test_dataa, (2, 0, 1))[np.newaxis, :, :, :]
    return torch.from_numpy(test_dataa)


# TV loss(total variation regularizer)
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class Net_VGG(nn.Module):
    def __init__(self):
        super(Net_VGG, self).__init__()
        self.netF = models.vgg16(pretrained=True).features[:14]  #
        for param in self.netF.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = []
        for channel in range(x.size(1)):
            input = x[:, channel, ].unsqueeze(1)
            input = torch.cat([input, input, input], 1)
            output.append(self.netF(input))
        output = torch.cat(output, 1)
        return output

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def norm(x):
    """Convert the range from [0, 1]. to [-1, 1]"""
    # out = x*2-1
    # return out.clamp_(0, 1)

    return torch.clamp(x, 0., 1.)