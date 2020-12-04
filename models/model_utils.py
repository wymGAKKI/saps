from __future__ import division
import os
import torch
import torch.nn as nn
import scipy.io as sio
import torchvision.utils as vutils
import numpy as np

def getInput(args, data):
    input_list = [data['img']]
    # print("getinput input_list:", len(input_list))  getinput input_list: 1
    input_list.append(data['mask'])
    return input_list

def parseData(args, sample, timer=None, split='train'):
    img, normal, mask = sample['img'], sample['normal'], sample['mask']
    ints = sample['ints']
    if args.in_light:
        dirs = sample['dirs'].expand_as(img)
    else: # predict lighting, prepare ground truth
        n, c, h, w = sample['dirs'].shape
        dirs_split = torch.split(sample['dirs'].view(n, c), 3, 1)
        dirs = torch.cat(dirs_split, 0)
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        img, normal, mask = img.cuda(), normal.cuda(), mask.cuda()
        dirs, ints = dirs.cuda(), ints.cuda()
        if timer: timer.updateTime('ToGPU')
    data = {'img': img, 'normal': normal, 'mask': mask, 'dirs': dirs, 'ints': ints}
    return data

def parses4Data(args, sample, timer=None, split='train'):
    img, normal, mask, reflectance = sample['img'], sample['normal'], sample['mask'], sample['reflectance']
    dirs = sample['dirs'].expand_as(img)

    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        img, normal, mask = img.cuda(), normal.cuda(), mask.cuda()
        dirs = dirs.cuda()
        if timer: timer.updateTime('ToGPU')
    data = {'img': img, 'normal': normal, 'mask': mask, 'reflectance': reflectance, 'dirs': dirs}
    return data

def parseshadowData(args, sample, timer=None, split='train'):
    normal, img, shadow ,mask = sample['normal'],sample['img'], sample['shadow'], sample['mask']
    light = sample['light'].expand_as(normal)
    mask = mask.unsqueeze(1)
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        normal, img, shadow ,mask= normal.cuda(),img.cuda(), shadow.cuda(), mask.cuda()
        light= light.cuda()
        if timer: timer.updateTime('ToGPU')
    data = {'normal': normal, 'img':img, 'shadow': shadow, 'light': light, 'mask': mask}
    return data

def parseReflectanceData(args, sample, timer=None, split='train'):
    img, mask, reflectance= sample['img'], sample['mask'], sample['reflectance']
    #print("img shape:", img.shape)
    dirs = sample['dirs'].expand_as(img)

    # else: # predict lighting, prepare ground truth
    #     n, c, h, w = sample['dirs'].shape
    #     dirs_split = torch.split(sample['dirs'].view(n, c), 3, 1)
    #     dirs = torch.cat(dirs_split, 0)
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        img, mask = img.cuda(), mask.cuda()
        reflectance = reflectance.cuda()
        dirs = dirs.cuda()
        if timer: timer.updateTime('ToGPU')
    data = {'img': img, 'mask': mask, 'dirs': dirs, 'reflectance':reflectance}
    return data

def getInputChanel(args):
    args.log.printWrite('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        args.log.printWrite('[Network Input] Adding Light direction as input')
        c_in += 3
    if args.in_mask:
        args.log.printWrite('[Network Input] Adding Mask as input')
        c_in += 1
    args.log.printWrite('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records} # 'args': args}
    torch.save(state,   os.path.join(save_path, 'checkp_{}.pth.tar'.format(epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_{}_rec.pth.tar'.format(epoch)))

def saveMultiCheckpoint(save_path, epoch=-1, models=None, optimizer=None, records=None, args=None):
    state0   = {'state_dict': models[0].state_dict(), 'model': 'LCNet'}
    torch.save(state0,   os.path.join(save_path, 'LCNet_checkp_{}.pth.tar'.format(epoch)))

    state1   = {'state_dict': models[1].state_dict(), 'model': 'NENet'}
    torch.save(state1,   os.path.join(save_path, 'NENet_checkp_{}.pth.tar'.format(epoch)))

    state2   = {'state_dict': models[2].state_dict(), 'model': 'RENet'}
    torch.save(state2,   os.path.join(save_path, 'RENet_checkp_{}.pth.tar'.format(epoch)))

    state3   = {'state_dict': models[3].state_dict(), 'model': 'CSNet'}
    torch.save(state3,   os.path.join(save_path, 'CSNet_checkp_{}.pth.tar'.format(epoch)))

    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records} # 'args': args}
    
    torch.save(records, os.path.join(save_path, 'checkp_{}_rec.pth.tar'.format(epoch)))

def conv_ReLU(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.ReLU(inplace=True)
                )

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def outputConv(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True))

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )

def upconv(cin, cout):
    return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )
