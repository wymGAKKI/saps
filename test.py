import torch
from torch import nn
import datasets.util as util
import numpy as np
from datasets import pms_transforms
import sys
from imageio import imread

p='data/datasets/PS_Sculpture_Dataset/Images/two-wrestlers-in-combat-repost_Two_wrestlersincombat_s-0.16_x--10_y--10_000/green-metallic-paint/two-wrestlers-in-combat-repost_Two_wrestlersincombat_s-0.16_x--10_y--10_000_green-metallic-paint.txt'
a = [1, 2, 4]
img_list = util.readList(p)
data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
select_idx = np.random.permutation(data.shape[0])[:32]

bunny_path = "/home/wym/code/SDPS-Net/data/ToyPSDataset/Bunny/l_000.png"
bunny_mask_path = "/home/wym/code/SDPS-Net/data/ToyPSDataset/Bunny/mask.png"
ear_path = "/home/wym/code/SDPS-Net/data/wymdata/ear1/image1.png"
ear_mask_path = "/home/wym/code/SDPS-Net/data/wymdata/ear1/mask.png"

# #print(data[1], data[2], data[4])
# data = data[a, :]
# imgs = [img for img in data[:, 0]]
# dirs = data[:, 1:4].astype(np.float32)
# print(dirs)
# dirs = torch.from_numpy(dirs).view(-1, 1, 1).float()
# print(dirs)
def img_len():
    img = np.random.rand(2,3,4)
    print(len(img))

def con_img():
    img_a = np.random.rand(3,5,5)
    img_b = np.random.rand(3,5,5)
    imgs = [img_a, img_b]
    img = np.concatenate(imgs, 2)
    print(len(imgs))
def int():
    intensity = np.random.random((5, 1))
    color = np.ones((1,3)) # Uniform color
    
    intens = (intensity.repeat(3, 1) * color)
    ints = torch.from_numpy(intens).view(-1, 1, 1).float()
    print(ints.shape)

def ran():
    for i in range(1,5):
        print(i)

def exp():
    img = torch.randn(6, 2, 2)
    #img = torch.from_numpy(img).float()
    dirs = torch.randn(6, 1, 1)
    print(dirs)
    dirs = dirs.expand_as(img)
    print(dirs)

def split():
    img = torch.randn(4,3,1,1)
    s = img.view(4,3)
    print(s.shape)

def view():
    img = torch.randn(3, 5, 5)
    img = img.view(-1)
    print(img.shape)

def l():
    li = [[1], [2], {'s': 2}]
    print(li[2])

def fuck():
    print("--------IN:", sys._getframe().f_code.co_name,"-----------------------------------")

def maxp():
    a = torch.randn(3,3)
    print("a:",a)
    b =torch.randn(3,3)
    print('b:',b)
    feats = torch.stack([a,b], 1)
    print("feats:",feats)
    x,y = feats.max(1)
    print("x,y:", x, y)
def img():
    bunny = imread(bunny_path).astype(np.float32) / 255.0
    ear = imread(ear_path).astype(np.float32) / 255.0
    print("ear:",ear.shape)
    print("bunny:",bunny.shape)

def mask():
    bunny = imread(bunny_mask_path).astype(np.float32) / 255.0
    ear = imread(ear_mask_path).astype(np.float32) / 255.0
    if ear.ndim > 2: ear = ear[:,:,0]
    ear = ear.reshape(ear.shape[0], ear.shape[1], 1)
    if bunny.ndim > 2: bunny = bunny[:,:,0]
    bunny = bunny.reshape(bunny.shape[0], bunny.shape[1], 1)
    print("ear:",ear.shape)
    print("bunny:",bunny.shape)

def d():
    ints = np.genfromtxt("/home/wym/code/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/cowPNG/light_intensities.txt")
    print('before:', ints[1])
    ints = [np.diag(1 / ints[i]) for i in range(96)]
    print('after:', ints[1])
def mul():
    a = torch.Tensor([[1,2],[2,3]])
    b = torch.Tensor([[1,2],[2,3]])
    print((a*b))
    print((a*b).sum(1))
    print((a*b).sum(1).clamp(-1,1))

def mask():
    img = torch.randn(5, 5)
    mask = torch.Tensor([[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0]]).byte()
    print(img, '\n')
    print(img[mask])
    print(img[mask].shape)

def tup():
    t = (1,2,3)
    l = [1,2,3]
    l += t
    print(l)

def npran():
    intensity = np.random.random((2, 1)) * 1.8 + 0.2
    color = np.ones((1, 3)) # Uniform color
    intens = (intensity.repeat(3, 1) * color)
    print('intensity:', intensity)
    print('color:', color)
    print('intens:', intens)
    print('intensity:', intensity.repeat(3, 1))

npran()