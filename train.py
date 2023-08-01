import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
from custom import optimizers as optim
from custom.caltime import RemainTime
from custom.serverLog import LogClass

parser = argparse.ArgumentParser(description='config')
## yaml configuration files
parser.add_argument('--config', type=str, default=None, help='pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help='resume training or not')
parser.add_argument('--custom', type=str, default=None, help='use custom block')
parser.add_argument('--cloudlog', type=str, default=None, help='use cloud log')


def save_model(_path, _epoch, _model, _optimizer, _scheduler, _stat_dict):
    # torch.save(model.state_dict(), saved_model_path)
    torch.save({
        'epoch': _epoch,
        'model_state_dict': _model.state_dict(),
        'optimizer_state_dict': _optimizer.state_dict(),
        'scheduler_state_dict': _scheduler.state_dict(),
        'stat_dict': _stat_dict
    }, _path)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
        opt = vars(args)
        yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[', '').replace(']', '')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_datasets

    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    ## definitions of model
    try:
        model = utils.import_module('models.{}_network'.format(args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    if args.fp == 16:
        model.half()

    ## load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load(ckpt['model_state_dict'])

    model = nn.DataParallel(model).to(device)

    ## definition of loss and optimizer
    loss_func = eval('nn.' + args.loss + '()')
    if args.fp == 16:
        eps = 1e-3
    else:
        eps = 1e-8
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=eps)
    elif args.optim == 'lamb':
        optimizer = optim.Lamb(model.parameters(), lr=args.lr, eps=eps)
    scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)

    ## resume training
    start_epoch = 1
    if args.resume is not None:
        ckpt_files = os.path.join(args.resume, 'models', "model_x{}_latest.pt".format(args.scale))
        if len(ckpt_files) != 0:
            ckpt = torch.load(ckpt_files)
            prev_epoch = ckpt['epoch']

            start_epoch = prev_epoch + 1
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            stat_dict = ckpt['stat_dict']
            ## reset folder and param
            experiment_path = args.resume
            log_name = os.path.join(experiment_path, 'log.txt')
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('select {}, resume training from epoch {}.'.format(ckpt_files, start_epoch))
    else:
        ## auto-generate the output logname
        experiment_name = None
        timestamp = utils.cur_timestamp_str()
        if args.log_name is None:
            experiment_name = '{}-x{}-{}'.format(args.model, args.scale, timestamp)
        else:
            experiment_name = '{}-{}'.format(args.log_name, timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        log_name = os.path.join(experiment_path, 'log.txt')
        stat_dict = utils.get_stat_dict()
        ## create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        ## save training paramters
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config_default.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)

    ## print architecture of model
    time.sleep(3)  # sleep 3 seconds
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    # print(model)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters:' + str(num_params // 1024) + 'k')
    sys.stdout.flush()

    ## start training
    timer_start = time.time()
    rt = RemainTime(args.epochs)
    cloudLogName = experiment_path.split(os.sep)[-1]
    log = LogClass(args.cloudlog == 'on')
    log.sendLog('start trainning', cloudLogName)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        print('##===========-fp{}-training, Epoch: {}, lr: {} =============##'.format(args.fp, epoch, opt_lr))
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            lr, hr = batch
            if args.fp == 16:
                lr, hr = lr.type(torch.HalfTensor), hr.type(torch.HalfTensor)
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = loss_func(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)

            if (iter + 1) % args.log_every == 0:
                cur_steps = (iter + 1) * args.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)
                stat_dict['losses'].append(avg_loss)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}'.format(cur_epoch, cur_steps, total_steps, avg_loss,
                                                                           duration))

        if epoch % args.test_every == 0:
            torch.set_grad_enabled(False)
            test_log = ''
            model = model.eval()
            for valid_dataloader in valid_dataloaders:
                avg_psnr, avg_ssim = 0.0, 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                for lr, hr, _ in tqdm(loader, ncols=80):
                    if args.fp == 16:
                        lr, hr = lr.type(torch.HalfTensor), hr.type(torch.HalfTensor)
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    # quantize output to [0, 255]
                    hr = hr.clamp(0, 255)
                    sr = sr.clamp(0, 255)
                    # conver to ycbcr
                    if args.colors == 3:
                        hr_ycbcr = utils.rgb_to_ycbcr(hr)
                        sr_ycbcr = utils.rgb_to_ycbcr(sr)
                        hr = hr_ycbcr[:, 0:1, :, :]
                        sr = sr_ycbcr[:, 0:1, :, :]
                    # crop image for evaluation
                    hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    # calculate psnr and ssim
                    psnr = utils.calc_psnr(sr, hr)
                    ssim = utils.calc_ssim(sr, hr)
                    avg_psnr += psnr
                    avg_ssim += ssim
                avg_psnr = round(avg_psnr / len(loader) + 5e-3, 2)
                avg_ssim = round(avg_ssim / len(loader) + 5e-5, 4)
                stat_dict[name]['psnrs'].append(avg_psnr)
                stat_dict[name]['ssims'].append(avg_ssim)
                save_model_flag = False
                if stat_dict[name]['best_psnr']['value'] < avg_psnr:
                    stat_dict[name]['best_psnr']['value'] = avg_psnr
                    stat_dict[name]['best_psnr']['epoch'] = epoch
                    save_model_flag = True
                    if name == 'set5':
                        log.sendLog('PSNR:{} epoch:{}/{}'.format(float(avg_psnr), epoch, args.epochs), cloudLogName)
                if stat_dict[name]['best_ssim']['value'] < avg_ssim:
                    stat_dict[name]['best_ssim']['value'] = avg_ssim
                    stat_dict[name]['best_ssim']['epoch'] = epoch
                    save_model_flag = True
                if save_model_flag:
                    # sava best model
                    save_model(os.path.join(experiment_model_path, 'model_x{}_{}.pt'.format(args.scale, epoch)), epoch,
                               model, optimizer, scheduler, stat_dict)
                test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f} (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n'.format(
                    name, args.scale, float(avg_psnr), float(avg_ssim),
                    stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'],
                    stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])
            # print log & flush out
            print(test_log[:-1])
            sys.stdout.flush()
            save_model(os.path.join(experiment_model_path, 'model_x{}_latest.pt'.format(args.scale)), epoch, model,
                       optimizer, scheduler, stat_dict)
            torch.set_grad_enabled(True)
            # save stat dict
            # save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
        ## update scheduler
        scheduler.step()
        rt.update(epoch)
        print()
    log.sendLog('finish trainning', cloudLogName)
