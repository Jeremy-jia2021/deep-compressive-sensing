import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import scripts.operation as operation
import scipy.io as io

import matplotlib.pyplot as plt

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.float)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    #image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    #out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        # mean, var = 0, 0.001
        # self.noise = np.random.normal(mean, var ** 0.5, [opt['GT_size'], opt['GT_size'], 1])
        #self.register_buffer('noise', mean)

        self.mode = opt['mode']
        self.paths_GT = util.get_paths_from_images(opt['dataroot_GT'])
        self.paths_LQ = util.get_paths_from_images(opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        #print(GT_path)
        if self.mode == 'IMAGE':
            img_GT = util.read_img(None, GT_path)
        else:
            img_GT = operation.load_hscnn(GT_path)
            img_GT = cv2.resize(np.copy(img_GT), (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            img_GT = img_GT[..., 0, np.newaxis] # keep only one channel
            img_GT = operation.normalize_0_to_1(img_GT)

        # range = [np.max(img_GT), np.min(img_GT)]

        # get LQ image
        LQ_path = self.paths_LQ[index]
        if self.mode == 'IMAGE':
            img_LQ = util.read_img(None, LQ_path)
        else:
            img_LQ = operation.load_hscnn(LQ_path)
            img_LQ = cv2.resize(np.copy(img_LQ), (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            img_LQ = img_LQ[..., 0, np.newaxis]  # keep only one channel
            img_LQ = operation.normalize_0_to_1(img_LQ)  # *2 - 1

        # if self.opt['phase'] == 'train':
        #     img_LQ = img_LQ + self.noise

        # plt.imshow(img_LQ[...,0], cmap='gray')
        # plt.colorbar()
        # plt.show()
        # io.savemat('input.mat', {'input': img_LQ[...,0]})

        #if self.opt['phase'] == 'train':
        if img_LQ.ndim == 2:
            img_LQ = np.expand_dims(img_LQ, axis=2)
            img_GT = np.expand_dims(img_GT, axis=2)

        H, W, C = img_LQ.shape
        LQ_size = GT_size
        if self.opt['phase'] == 'train':
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
        else:
            rnd_h, rnd_w = 180, 180
        img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        rnd_h_GT, rnd_w_GT = int(rnd_h), int(rnd_w)
        img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

        # augmentation - flip, rotate
        img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'], self.opt['use_rot'])

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
