from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread

import torch
import torch.utils.data as data

from datasets import pms_transforms
from datasets import util
np.random.seed(0)

class ShadowDataset(data.Dataset):
    #root = "/mnt/data/CyclePS/datasets/ShadowDataset/"
    def __init__(self, args, root, split='train'):
        self.root  = os.path.join(root)
        self.split = split
        self.args  = args
        self.shape_list = util.readList(os.path.join(self.root, split+"objectsname.txt"))
        self.light_list = util.readList(os.path.join(self.root, "lights_dataset.txt"))

    def _getInputPath(self, index):
        obj_index = index // 25
        obj = self.shape_list[obj_index]
        view_light = util.readList(os.path.join(self.root,obj, "light.txt"))
        view,light = view_light[index%25].split('-')
        normal_path = os.path.join(self.root, obj, obj + view + '_normal.png')
        mask_path = os.path.join(self.root, obj, obj + view + '_mask.png')
        shadow_path    = os.path.join(self.root, obj, obj + view +'light_'+light+'_shadow.png')
        light = self.light_list[int(light)]
        return normal_path, mask_path, shadow_path, light

    def __getitem__(self, index):
        normal_path, mask_path, shadow_path, light = self._getInputPath(index)
        normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
        shadow = imread(shadow_path).astype(np.float32)
        mask = imread(mask_path).astype(np.float32)
        norm   = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10) # Rescale normal to unit length

        item = {'normal': normal, 'shadow': shadow}
        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])
        item['shadow'] = torch.narrow(item['shadow'], 0, 0, 1)
        item['light'] = np.float32(light.split(' '))
        item['light']=torch.from_numpy(item['light']).view(-1, 1, 1).float()
        item['mask'] = torch.from_numpy(mask)
        # item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        # item['ints'] = torch.from_numpy(ints).view(-1, 1, 1).float()
        
        # normal : torch.Size([3, 128, 128])
        # img : torch.Size([6, 128, 128])
        # mask : torch.Size([1, 128, 128])
        # dirs : torch.Size([6, 1, 1])
        # ints : torch.Size([6, 1, 1])

        return item

    def __len__(self):
        return len(self.shape_list)*25

