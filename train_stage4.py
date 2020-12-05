from __future__ import division
import torch
from models import model_utils
from utils  import eval_utils, time_utils
from PIL import Image, ImageFile
import numpy as np
import scipy.io as sio
from imageio import imread
import torchvision.utils as vutils
ImageFile.LOAD_TRUNCATED_IMAGES = True
def train(args, loader, models, criterion, optimizers, log, epoch, recorder):
    models[0].train()
    models[1].train()
    models[2].train()
    models[3].train()
    optimizer, optimizer_c = optimizers
    log.printWrite('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader): 
        
        data = model_utils.parsestage4Data(args, sample, timer, 'train')
        #data = {'img': img(n, 3*in_img_num, h, w), 
        # 'normal': normal(n, 3, h, w), 
        # 'mask': mask(n, 1, h, w), 
        # 'dirs': dirs(in_img_num*n, 3, h, w)
        input = model_utils.getInput(args, data)
        #input: list [data['img'], data['mask']]
        pred_c = models[0](input);  
        # pred_c: {'dirs_x', 'dirs_y', 'dirs', 'ints', 'intens'}       
        input.append(pred_c)
        pred_n = models[1](input); timer.updateTime('Forward')    

        dirs_n = torch.split(input[2]['dirs'], input[0].shape[0], 0)#长为in img num的n*3的list
        dirs_r = torch.cat(dirs_n, 1)
        input_r = {}
        input_r['img'] = input[0]
        input_r['mask'] = input[1]
        input_r['dirs'] = dirs_r
        pred_r =  models[2](input_r)

        input.append(pred_n)
        input.append(pred_r)
        shadow_input = prepareShadowInputs(input)
        shadowlist = []
        for s in shadow_input:
            shadow = models[3](s)['shadow']
            shadowlist.append(shadow)
        pred_s = torch.cat(shadowlist, 1)
        input.append(pred_s)
        #input: list [data['img'], data['mask'], 
        # pred_c{'dirs_x', 'dirs_y', 'dirs', 'ints', 'intens'},
        # pred_n{'normal'}, pred_r{'reflectance'},
        # pred_s[batch, in_img_num*1, h, w]]
        optimizer.zero_grad()

        recon_inputs = reconInputs(args, input)
        lights = recon_inputs['lights']
        ints = recon_inputs['ints']
        normal = pred_n['normal']
        shadow = pred_s
        reflectance = pred_r['reflectance']
        recon = reconstruct(normal, reflectance, lights, ints, shadow)
        loss = criterion.forward(255 * recon,255 * data['img']); 
        timer.updateTime('Crit');
        criterion.backward(); timer.updateTime('Backward')

        recorder.updateIter('train', loss.keys(), loss.values())

        optimizer.step(); timer.updateTime('Solver')
        # print('-----------------models parameters:----------\n')
        # for name, parms in models[2].named_parameters():	
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)

        if iters % args.train_save == 0:
            results, recorder, nrow = prepareSave(args, data, pred_c, pred_n, recon, pred_r, recorder, log) 
            log.saveImgResults(results, 'train', epoch, iters, nrow=nrow)
            log.plotCurves(recorder, 'train', epoch=epoch, intv=args.train_disp)

        if args.max_train_iter > 0 and iters >= args.max_train_iter: break
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareSave(args, data, pred_c, pred, recon, pred_r, recorder, log):
    input_var, mask_var = data['img'], data['mask']
    results = [input_var.data, recon, mask_var.data, (data['normal'].data+1)/2, ]
    if args.s1_est_d:
        l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred_c['dirs'].data, args.batch)
        recorder.updateIter('train', l_acc.keys(), l_acc.values())
    if args.s1_est_i:
        int_acc, data['int_err'] = eval_utils.calIntsAcc(data['ints'].data, pred_c['intens'].data, args.batch)
        recorder.updateIter('train', int_acc.keys(), int_acc.values())

    if args.s2_est_n:
        acc, error_map = eval_utils.calNormalAcc(data['normal'].data, pred['normal'].data, mask_var.data)
        pred_n = (pred['normal'].data + 1) / 2
        reflectance = (pred_r['reflectance'].data + 1) / 2
        masked_pred = pred_n * mask_var.data.expand_as(pred['normal'].data)
        masked_reflectance = reflectance * mask_var.data.expand_as(pred_r['reflectance'].data)
        res_n = [masked_pred, masked_reflectance, error_map['angular_map']]
        results += res_n
        recorder.updateIter('train', acc.keys(), acc.values())

    nrow = input_var.shape[0] if input_var.shape[0] <= 32 else 32
    return results, recorder, nrow
def prepareShadowInputs(x):
        #input: list [data['img'], data['mask'], 
        # pred_c{'dirs_x', 'dirs_y', 'dirs', 'ints', 'intens'},
        # pred_n{'normal'}, pred_r{'reflectance'}]
    normal = x[3]['normal']
    dirs = torch.split(x[2]['dirs'], x[0].shape[0], 0)
    n, c, h, w = normal.shape
    shadow_input = []
    for i in range(len(dirs)):
        l_dir = dirs[i] if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1)
        normal_light = {'normal':normal, 'light':l_dir.expand_as(normal)}
        shadow_input.append(normal_light)
    
    return shadow_input

def reconInputs(args, x):
        #input: list [data['img'], data['mask'], 
        # pred_c{'dirs_x', 'dirs_y', 'dirs', 'ints', 'intens'},
        # pred_n{'normal'}, pred_r{'reflectance'},
        # pred_s[batch, in_img_num*1, h, w]]
        imgs = torch.split(x[0], 3, 1)
        dirs = torch.split(x[2]['dirs'], x[0].shape[0], 0)
        ints = torch.split(x[idx]['intens'], 3, 1)
        #shadows = torch.split(x[5], 3, 1)
        # print("dirs:", dirs[0].shape) input_img_num (batch, 3)
        # print("ints:", ints[0].shape) input_img_num (batch, 3)
        s2_inputs = {}
        lights = []
        
        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            lights.append(dirs[i].expand_as(imgs[i]) if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1).expand_as(imgs[i]))
            
        #     img   = imgs[i].contiguous().view(n * c, h * w)
        #     img   = torch.mm(l_int, img).view(n, c, h, w)
        light = torch.cat(lights, 1)

        s2_inputs['lights'] = light
        s2_inputs['ints'] = ints
        return s2_inputs

def reconstruct(pred_n, pred_r, light, shadow):
        normal = pred_n
        n, c, h, w = normal.shape
        # print('***************reflectance', pred_nr['reflectance'].max(), pred_nr['reflectance'].min())
        reflectance = (pred_r + 1) / 2
        # print("reflectance max:", reflectance.max())
        # print("reflectance min:", reflectance.min())
        # normal, reflectance:[batch, 3, height, weight]
        # light: [batch, 3*input_img, height, weight]
        # shadow:[batch, input_img, height, weight]
        lights = torch.split(light, 3, 1)
        shadows = torch.split(shadow, 1, 1)
        recons = []
        zeros = torch.zeros_like(reflectance)
        for idx in range(len(lights)):
            product = (lights[idx] * normal).sum(1, keepdim = True).expand_as(reflectance)
            zeros = torch.zeros_like(product)
            #print(product.dtype)
            #maximam = torch.max(zeros, product)
            maximam = torch.max(product, zeros)
            #img = maximam * reflectance * shadows[idx].expand_as(reflectance)
            img = torch.max(zeros,(maximam * reflectance - shadows[idx])).expand_as(reflectance)
            recons.append(img)
        recon = torch.cat(recons, 1)
        # print("recon max:", recon.max())
        # print("recon min:", recon.min())
        # print('recons:', recon.shape)
        return recon

def testReconstruct():
    npath = "/mnt/data/CyclePS/datasets/MyDataset/4a-bust-of-demosthenes_5/normal.mat"
    rpath = "/mnt/data/CyclePS/datasets/MyDataset/4a-bust-of-demosthenes_5/Reflectance.png"
    #imgpath = "/mnt/data/CyclePS/datasets/MyDataset/4a-bust-of-demosthenes_5/79.png"
    spath = "/mnt/data/CyclePS/datasets/MyDataset/4a-bust-of-demosthenes_5/Shadow/35Shadow.png"
    # normal = sio.loadmat(npath)['normal'].astype(np.float32)
    # normal = torch.from_numpy(np.transpose(normal, (2, 0, 1))).unsqueeze(0)
    # # img = imread(imgpath).astype(np.float32) / 255.0
    # # img = torch.from_numpy(img[:,:,:3])
    # # img = np.transpose(img, (2,0,1))
    
    # reflectance = imread(rpath).astype(np.float32) / 255.0
    # reflectance = torch.from_numpy(np.transpose(reflectance, (2, 0, 1))).unsqueeze(0)
    # reflectance = reflectance[:,:3,:,:]
    
    # shadow = imread(spath).astype(np.float32) / 255.0
    # shadow = torch.from_numpy(np.transpose((shadow), (2, 0, 1))).unsqueeze(0)
    # light = torch.from_numpy(np.asarray([-0.857027, 0.075646, 0.509688], dtype='float32').reshape(1,3,1,1))
    # light = light.expand_as(normal)
    # # print(reflectance.shape)
    # # print(normal.shape)
    # # print(light.shape)
    # # print(shadow.shape)
    # recon = reconstruct(normal, reflectance, light, shadow)
    # print(recon.max())
    # vutils.save_image(recon, 'recon.png')
    reflectance = imread(rpath).astype(np.float32) / 255.0
    if(reflectance.shape[2] == 4):
            reflectance = reflectance[:,:,:3]
    reflectance = torch.from_numpy(np.transpose(reflectance, (2, 0, 1))).unsqueeze(0)
    vutils.save_image(reflectance, 'rf.png')

if __name__ == '__main__':
    testReconstruct()