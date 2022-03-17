import os
import math
import argparse
import random
import logging
import torch
import spectral
import numpy as np
from torchvision.utils import make_grid
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

from tensorboardX import SummaryWriter


def main():

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

    #### random seed
    # seed = opt['train']['manual_seed']
    # if seed is None:
    #     seed = random.randint(1, 10000)
    # util.set_random_seed(seed)
    #
    # torch.backends.cudnn.benchmark = True
    # # torch.backends.cudnn.deterministic = True

    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # tensorboard logger
    tb_logger = SummaryWriter(log_dir='tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_loader = create_dataloader(train_set, dataset_opt, opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt)
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test(current_step)

                    visuals = model.get_current_visuals()
                    # channel_wise images
                    # sr_img = make_grid(visuals['SR'][0,0:4, ...].unsqueeze(1), normalize=True, nrow=2)
                    # gt_img = make_grid(visuals['GT'][0,0:4, ...].unsqueeze(1), normalize=True, nrow=2)
                    bottleneck_fea_img = make_grid(visuals['bottleneck_fea'][0,...].unsqueeze(1), normalize=True)
                    print("Feature maps: min=%.3f, max=%.3f"%(visuals['bottleneck_fea'].min(),visuals['bottleneck_fea'].max()))

                    # RGB images
                    SR_cube = np.transpose(visuals['SR'].detach().squeeze(dim=0).cpu().numpy(), (1, 2, 0))
                    #SR_RGB = spectral.get_rgb(SR_cube, bands=[23, 15, 5], stretch=(0.02, 0.98))
                    GT_cube = np.transpose(visuals['GT'].detach().squeeze(dim=0).cpu().numpy(), (1, 2, 0))
                    #GT_RGB = spectral.get_rgb(GT_cube, bands=[23, 15, 5], stretch=(0.02, 0.98))

                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_image('A/SR ' + str(idx), SR_cube, global_step=current_step, dataformats='HWC')
                        tb_logger.add_image('B/GT ' + str(idx), GT_cube, global_step=current_step, dataformats='HWC')
                        tb_logger.add_image('C/bottleneck_fea_img ' + str(idx), bottleneck_fea_img, global_step=current_step, dataformats='CHW')

                    if opt['save_images']:
                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(util.tensor2img(sr_img), save_img_path)
                        save_img_path = os.path.join(img_dir, 'fea_{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(util.tensor2img(bottleneck_fea_img), save_img_path)

                    # calculate PSNR
                    # avg_psnr = util.calculate_psnr(SR_cube*255, GT_cube*255)
                    # print('PSNR='+str(avg_psnr))
                    #tb_logger.add_scalar('PSNR', avg_psnr, current_step)
                #
                # avg_psnr = avg_psnr / idx
                #
                # # log
                # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                # logger_val = logging.getLogger('val')  # validation logger
                # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                #     epoch, current_step, avg_psnr))
                # # tensorboard logger
                # if opt['use_tb_logger'] and 'debug' not in opt['name']:
                #     tb_logger.add_scalar('psnr', avg_psnr, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)



if __name__ == '__main__':
    main()
