#!/usr/local/bin/python
import sys
import os
import time
import argparse
import time
import datetime
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys
src = str(os.path.split(os.path.realpath(__file__))[0]).replace("\\","/")
from PIL import Image
from PIL import ImageDraw
import glob
import math
from torchvision.transforms import ToTensor
# from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2
import logging
from einops import rearrange
import cv2
#-------------------------------------dataLoad------------------------------------------
class My_Dataset(Dataset):
    def __init__(self,args,mode='train'):
        if mode == "train":
            self.pan_images = glob.glob(args.SERVER.TRAIN_DATA+"/pan/*.tif")
            self.ms_images = glob.glob(args.SERVER.TRAIN_DATA+"/ms/*.tif") 
        elif mode == "val":
            self.pan_images = glob.glob(args.SERVER.VAL_DATA+"/pan/*.tif")
            self.ms_images = glob.glob(args.SERVER.VAL_DATA+"/ms/*.tif")
        elif mode == "test":
            self.pan_images = glob.glob(args.SERVER.TEST_DATA+"/pan/*.tif")  
            self.ms_images = glob.glob(args.SERVER.TEST_DATA+"/ms/*.tif")
        self.upscale_factor = 4

    def __len__(self):
        return len(self.pan_images)

    def __getitem__(self, index):
        pan_image = cv2.imread(self.pan_images[index],-1)
        ms_image = cv2.imread(self.ms_images[index],-1)
        
        lms_image = cv2.resize(ms_image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        pan_image = np.array(pan_image, dtype=np.uint8).astype(np.float32) / 255.0
        ms_image = np.array(ms_image, dtype=np.uint8).astype(np.float32) / 255.0
        lms_image = np.array(lms_image, dtype=np.uint8).astype(np.float32) / 255.0
        
        pan_image = ToTensor()(pan_image)
        ms_image = ToTensor()(ms_image)
        lms_image = ToTensor()(lms_image)
        samples = {
            "pan": pan_image,
            "ms": ms_image,
            "lms": lms_image,
        }
        return samples, self.pan_images[index]

def build_dataset(args, mode):
    return My_Dataset(args,mode)

#------------------------------------------------------------------------------------------------------
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, y1, y2, Weight):
        dis = torch.abs(y1-y2)
        dis = dis * Weight
        return torch.mean(dis)

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
    
    def forward(self, y1, y2, AU):
        dis = torch.pow(y1-y2, 2)
        # s = s + 0.000001
        # l = dis/(2*s) + 0.5*torch.log(s)
        l = 0.5*torch.exp(-1.0*AU)*dis + 0.5*AU
        return torch.mean(l)

#-------------------------------------Trian/validation one epoch ------------------------------------------
def train_one_epoch(args, model, dataloader_train, optimizer, lr_scheduler, epoch, logger):
    model.train()
    l1_loss = L1Loss().cuda()
    l1_loss.eval()
    U_loss = UncertaintyLoss().cuda()
    U_loss.eval()
    optimizer.zero_grad()
    num_steps = len(dataloader_train)
    start = time.time()
    totalloss = 0
    for _, samples in enumerate(dataloader_train):
        pan = samples[0]['pan'].cuda()
        ms = samples[0]['ms'].cuda()
        lms = samples[0]['lms'].cuda()
        AU, EU, hrms1, hrms2 = model(pan = pan, lms = lms)

        #result loss
        loss = l1_loss(hrms2, ms.detach(), 1)

        #uncertainty-related loss
        loss = loss + l1_loss(hrms1[0], ms.detach(), 1)*0.1 + U_loss(hrms1[0], ms.detach(), AU[0])*0.01
        loss = loss + l1_loss(hrms1[1], ms.detach(), 1)*0.1 + U_loss(hrms1[1], ms.detach(), AU[1])*0.01

        #focus loss
        U1 = 0.5*AU[0] + 0.5*EU[0]
        U2 = 0.5*AU[1] + 0.5*EU[1]
        loss  = loss + (l1_loss(hrms1[1], ms.detach(), U1) + l1_loss(hrms2, ms.detach(), U2)) * 0.01
       
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch)
        cur_lr=optimizer.param_groups[-1]['lr']
        totalloss = totalloss + loss.item()
        if _ % args.TRAIN.PRINT_FREQ == 0 and (args.distributed == False or torch.distributed.get_rank() == 0):
            logger.info(f"||train -- epoch:{epoch} / {args.TRAIN.EPOCHS} --step:{_+1} / {num_steps} --lr:{cur_lr:.7f}-----curloss: {loss.item():.5f}")
    loss_avg = totalloss / num_steps
    epoch_time = time.time() - start 
    if args.distributed == False or torch.distributed.get_rank() == 0:
        Logger_msg = f":-------------------------Train: Epoch:{epoch:04}, timespend:{datetime.timedelta(seconds=int(epoch_time))}, avg_loss:{loss_avg:.5f}"
        logger.info(Logger_msg)
        

@torch.no_grad()
def validate(args, model, dataloader_val, epoch, logger, dataset_val_len):
    model.eval()
    l1_loss = L1Loss().cuda()
    l1_loss.eval()
    num_steps = len(dataloader_val)
    start = time.time()
    totalloss = 0
    psnr = 0 
    ssim = 0
    for _, samples in enumerate(dataloader_val):
        if _ % args.TRAIN.PRINT_FREQ == 0 and (args.distributed == False or torch.distributed.get_rank() == 0):
            logger.info(f"|| Val -- epoch:{epoch} / {args.TRAIN.EPOCHS} ------step:{_+1} / {num_steps}")
        pan = samples[0]['pan'].cuda()
        ms = samples[0]['ms'].cuda()
        lms = samples[0]['lms'].cuda()
        AU, EU, hrms1, hrms2 = model(pan = pan, lms = lms)
        loss = l1_loss(hrms2, ms.detach(), 1)
        hrms2 = hrms2.clamp(0, 1)
        psnr = psnr + cal_psnr(hrms2, ms)
        ssim = ssim + cal_ssim(hrms2, ms)
        totalloss = totalloss + loss.item()
    loss_avg = totalloss / num_steps
    psnr = psnr / num_steps 
    ssim = ssim / num_steps
    epoch_time = time.time() - start 
    
    if args.distributed == False or torch.distributed.get_rank() == 0:
        Logger_msg = f":Validation: Epoch:{epoch:04}, timespend:{datetime.timedelta(seconds=int(epoch_time))}, loss_avg:{loss_avg:.4f}, psnr:{psnr:.4f}, ssim:{ssim:.4f} "
        logger.info(Logger_msg)
    return {"PSNR":psnr,"SSIM":ssim}

#-------------------------------------metrics ------------------------------------------
def cal_psnr(imag1, imag2):     # B C H W tensor [0 ~ 1]
    imag1 = imag1 * 255.0
    imag2 = imag2 * 255.0
    sum = 0
    for b in range(imag1.shape[0]):
        im1 =imag1[b:b+1,:,:,:]
        im2 =imag2[b:b+1,:,:,:]
        mse = torch.mean(torch.abs(im1 - im2) ** 2)
        if mse == 0:
            sum = sum + 100
        else:
            sum = sum + (10 * torch.log10(255 * 255 / mse))
    return sum / imag1.shape[0]

def cal_ssim(imag1,imag2):    #"b C H W"  tensor [0 ~ 1]
    imag1 = imag1 * 255.0
    imag2 = imag2 * 255.0
    ssim_sum = 0
    for i in range(imag1.shape[0]):
        im1 = imag1[i].permute(1,2,0)  #to 'H W C'
        im2 = imag2[i].permute(1,2,0)  #to 'H W C'
        im1 = im1.cpu().numpy()
        im2 = im2.cpu().numpy()
        # ssim = calculate_ssim(im1,im2)
        ssim = structural_similarity(im1, im2, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
        ssim_sum = ssim_sum + ssim
    return ssim_sum / imag1.shape[0]

#-------------------------------------log ------------------------------------------
def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=True):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root,"log/"+'UncertaintyPAN_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

#-------------------------------------log ------------------------------------------


if __name__ == '__main__':
    pass

