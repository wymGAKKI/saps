from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread
import scipy.io as sio
import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class MyDataset(data.Dataset):
    def __init__(self, args, root, split='train'):
        self.root  = os.path.join(root)
        self.split = split
        self.args  = args
        self.shape_list = util.readList(os.path.join(self.root, split +"objectsname.txt"))
        self.light_list = util.readList(os.path.join(self.root, "lights_dataset.txt"))

    def _getInputPath(self, index):
        #root = "/mnt/data/CyclePS/datasets/MyDataset/"
        normal_path = os.path.join(self.root, self.shape_list[index], 'normal.mat')
        reflectance_path = os.path.join(self.root, self.shape_list[index], 'Reflectance.png')
        mask_path = os.path.join(self.root, self.shape_list[index], 'mask.mat')
        img_dir     = os.path.join(self.root, self.shape_list[index])
        img_list = []
        for i in range(0, 100):
            img_list.append(os.path.join(img_dir, '%d.png' % (i)))
        #img_list    = util.readList(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))
        data = np.asarray(img_list, dtype='str')
        lights = np.genfromtxt(self.light_list, dtype='float32', delimiter=' ')
        #data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        select_idx = np.random.permutation(data.shape[0])[:self.args.in_img_num]
        #idxs = ['%03d' % (idx) for idx in select_idx]
        imgs = data[select_idx]
        #imgs = [os.path.join(img_dir, img) for img in data[:, 0]]
        dirs = lights[select_idx]
        return normal_path, imgs, dirs, reflectance_path, mask_path

    def __getitem__(self, index):
        normal_path, img_list, dirs, reflectance_path,  mask_path = self._getInputPath(index)
        normal = sio.loadmat(normal_path)['normal'].astype(np.float32)
        reflectance = imread(reflectance_path).astype(np.float32) / 255.0
        if(reflectance.shape[2] == 4):
                reflectance = reflectance[:,:,:3]
        imgs   =  []
        for i in img_list:
            img = imread(i).astype(np.float32) / 255.0
            if(img.shape[2] == 4):
                img = img[:,:,:3]
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        h, w, c = img.shape
        crop_h, crop_w = self.args.crop_h, self.args.crop_w
        if self.args.rescale and not (crop_h == h):
            sc_h = np.random.randint(crop_h, h) if self.args.rand_sc else self.args.scale_h
            sc_w = np.random.randint(crop_w, w) if self.args.rand_sc else self.args.scale_w
            img, normal = pms_transforms.rescale(img, normal, [sc_h, sc_w])

        if self.args.crop:
            img, normal = pms_transforms.randomCrop(img, normal, [crop_h, crop_w])

        # if self.args.color_aug:
        #     img = img * np.random.uniform(1, self.args.color_ratio)

        # if self.args.int_aug:
        #     ints = pms_transforms.getIntensity(len(imgs))
        #     img  = np.dot(img, np.diag(ints.reshape(-1)))
        # else:
        #     ints = np.ones(c)

        # if self.args.noise_aug:
        #     img = pms_transforms.randomNoiseAug(img, self.args.noise)

        mask = sio.loadmat(mask_path)['mask'].astype(np.float32)
        norm   = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10) # Rescale normal to unit length

        item = {'normal': normal, 'img': img, 'reflectance': reflectance}
        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['mask'] = torch.from_numpy(mask).unsqueeze(0)
        
        # normal : torch.Size([3, 128, 128])
        # img : torch.Size([6, 128, 128])
        # mask : torch.Size([1, 128, 128])
        # dirs : torch.Size([6, 1, 1])
        # ints : torch.Size([6, 1, 1])

        return item

    def __len__(self):
        return len(self.shape_list)
