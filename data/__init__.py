'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
from data.LQGT_dataset import LQGTDataset as D


def create_dataloader(dataset, dataset_opt, opt=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
        batch_size = dataset_opt['batch_size']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, sampler=None, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    return D(dataset_opt)
