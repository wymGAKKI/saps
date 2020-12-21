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

class PokemonDataset(data.Dataset):
    def __init__(self, args, root, split='train'):
        self.root  = os.path.join(root)
        self.split = split
        self.args  = args
        self.shape_list = util.readList(os.path.join(self.root, split +"objectsname.txt"))
        
    def _getInputPath(self, index):
        #root = "/mnt/data/CyclePS/datasets/MyDataset/"

        img_dir     = os.path.join(self.root, self.shape_list[index])
        img_list = []
        for i in range(1, 33):
            img_list.append(os.path.join(img_dir, '%d.jpg' % (i)))
        #img_list    = util.readList(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))
        data = np.asarray(img_list, dtype='str')
        #data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        #select_idx = np.random.permutation(data.shape[0])[:self.args.in_img_num]
        #idxs = ['%03d' % (idx) for idx in select_idx]
        imgs = data
        mask_path = os.path.join(self.root, self.shape_list[index], ' mask.jpg')
        return imgs, mask_path

    def __getitem__(self, index):
        img_list, mask_path = self._getInputPath(index)
        mask = imread(mask_path).astype(np.float32) / 255.0
        imgs   =  []
        for i in img_list:
            img = imread(i).astype(np.float32) / 255.0
            if(img.shape[2] == 4):
                img = img[:,:,:3]
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        h, w, c = img.shape
        crop_h, crop_w = self.args.crop_h, self.args.crop_w
        # if self.args.rescale and not (crop_h == h):
        #     sc_h = np.random.randint(crop_h, h) if self.args.rand_sc else self.args.scale_h
        #     sc_w = np.random.randint(crop_w, w) if self.args.rand_sc else self.args.scale_w
        #     img, mask = pms_transforms.rescale(img, mask, [sc_h, sc_w])

        # if self.args.crop:
        #     img, mask = pms_transforms.randomCrop(img, mask, [crop_h, crop_w])

        # if self.args.color_aug:
        #     img = img * np.random.uniform(1, self.args.color_ratio)

        if self.args.int_aug:
            ints = pms_transforms.getIntensity(len(imgs))
            img  = np.dot(img, np.diag(ints.reshape(-1)))
        else:
            ints = np.ones(c)

        # if self.args.noise_aug:
        #     img = pms_transforms.randomNoiseAug(img, self.args.noise)

        

        item = {'img': img}
        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])
        item['mask'] = torch.from_numpy(mask).unsqueeze(0)
        
        # normal : torch.Size([3, 128, 128])
        # img : torch.Size([6, 128, 128])
        # mask : torch.Size([1, 128, 128])
        # dirs : torch.Size([6, 1, 1])
        # ints : torch.Size([6, 1, 1])

        return item

    def __len__(self):
        return len(self.shape_list)
