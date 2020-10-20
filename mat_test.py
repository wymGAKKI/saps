import torch
import scipy.io as scio
import numpy as np

def reconstruct(normal, light):
        n, c, h, w = normal.shape
        # print('***************reflectance', pred_nr['reflectance'].max(), pred_nr['reflectance'].min())

        # print("reflectance max:", reflectance.max())
        # print("reflectance min:", reflectance.min())
        # normal, reflectance:[batch, 3, height, weight]
        # light: [batch, 3*input_img, height, weight]
        lights = torch.split(light, 3, 1)
        recons = []
        zeros = torch.zeros_like(normal)
        for idx in range(len(lights)):
            product = (lights[idx] * normal).sum(1, keepdim = True)
            zeros = torch.zeros_like(product)
            #print(product.dtype)
            #maximam = torch.max(zeros, product)
            shading = torch.max(product, zeros)
            recons.append(shading)
        recon = torch.cat(recons, 1)
        # print("recon max:", recon.max())
        # print("recon min:", recon.min())
        # print('recons:', recon.shape)
        return recon
def reconInputs(light, shape):
    lights = []
    dirs = torch.split(light, shape[0], 0)
    n, c, h, w = shape
    for i in range(len(dirs)):
        lights.append(dirs[i].expand(shape) if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1).expand(shape))
        
    #     img   = imgs[i].contiguous().view(n * c, h * w)
    #     img   = torch.mm(l_int, img).view(n, c, h, w)
    light = torch.cat(lights, 1)
    return light

def arrayToTensor(array):
    if array is None:
        return array
    array = np.transpose(array, (0, 3, 1, 2))
    tensor = torch.from_numpy(array)
    return tensor.float()

filepath = "/home/wym/code/SDPS-Net/data/models/10-16_run_model,LCNet_CVPR2019,LCNet,NENet,UPS_DiLiGenT_main,max,in_m-32,te_h-128,te_w-128,int_aug/test/Mat/"
light_filename = '1_9_light.mat'
normal_filename = '1_9_normal.mat'
light_path = filepath + light_filename
normal_path = filepath + normal_filename

lights = scio.loadmat(light_path)
normals = scio.loadmat(normal_path)


normal_gt = normals['gt']
normal_pred = normals['pred']

light_gt = lights['gt']
light_pred = lights['pred']


normal_gt.resize((normal_gt.size // 3 ,3))
normal_pred.resize((normal_pred.size // 3, 3))
# print('normal:', normal_gt.shape, normal_pred.shape)
# print('light:', light_gt.shape, light_pred.shape)
# print(np.linalg.norm(normal_pred[0]))
def ne():
    sum = 0
    count = 0
    for idx in range(normal_gt.size // 3):
        if np.linalg.norm(normal_gt[idx]) > 0:
            norm = np.linalg.norm(normal_pred[idx]) * np.linalg.norm(normal_gt[idx])
            product = (normal_pred[idx] * normal_gt[idx]).sum()
            theta = np.arccos(product / norm)
            count += 1
            sum += theta

    avg = (sum / count) / np.pi * 180

    print(avg)

def le():
    sm = 0
    count = 0
    for idx in range(light_gt.size // 3):
    # for idx in range(1):
        if np.linalg.norm(light_gt[idx]) > 0:
            # print('light1:', light_gt[0], light_pred[0])
            norm = np.linalg.norm(light_pred[idx]) * np.linalg.norm(light_gt[idx])
            product = (light_pred[idx] * light_gt[idx]).sum()
            theta = np.arccos(product / norm)
            count += 1
            sm += theta
    # idx = 4

    # norm = np.linalg.norm(light_pred[idx]) * np.linalg.norm(light_gt[idx])
    # product = (light_pred[idx] * light_gt[idx]).sum()
    # theta = np.arccos(product / norm)
    # count += 1
    # sm += theta

    avg = (sm / count) / np.pi * 180

    print(avg)

ne()