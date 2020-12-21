import os
import torch
from models import model_utils
from utils import eval_utils, time_utils 
import numpy as np

def get_itervals(args, split):
    if split not in ['train', 'val', 'test']:
        split = 'test'
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    stop_iters = args_var['max_'+split+'_iter']
    return disp_intv, save_intv, stop_iters

def test(args, split, loader, models, log, epoch, recorder):
    models[0].eval()
    models[1].eval()
    models[2].eval()
    models[3].eval()
    log.printWrite('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    res = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parsePokeData(args, sample, timer, 'train')
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

            #dirs_n = torch.split(input[2]['dirs'], input[0].shape[0], 0)#长为in img num的n*3的list
            input_r = {}
            input_r['img'] = input[0]
            input_r['mask'] = input[1]
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

            recon_inputs = reconInputs(args, input)
            lights = recon_inputs['lights']
            normal = pred_n['normal']
            shadow = pred_s
            reflectance = pred_r['reflectance']
            recon = reconstruct(normal, reflectance, lights, shadow,data['mask'])

            #recoder, iter_res, error = prepareNormalRes(args, data, pred_n, recorder, log, split)
            #print("data['img'].shape:", data['img'].shape)
            #res.append(iter_res)
            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                #log.printItersSummary(opt)

            if iters % save_intv == 0:
                results, recorder, nrow = prepareSave(args, data, pred_n, pred_r, pred_s, recon, recorder, log)
                #log.saveShadowResults(pred_s, 'test', epoch, iters, nrow=nrow)
                log.saveImgResults(results, split, epoch, iters, nrow=nrow, error='')
                #log.saveMatResults(pred['normal'], data['normal'], pred_c['dirs'], data['dirs'], split, epoch, iters, nrow=nrow, error='')
                #log.plotCurves(recorder, split, epoch=epoch, intv=disp_intv)

            if stop_iters > 0 and iters >= stop_iters: break
    #res = np.vstack([np.array(res), np.array(res).mean(0)])
    save_name = '%s_res.txt' % (args.suffix)
    #np.savetxt(os.path.join(args.log_dir, split, save_name), res, fmt='%.2f')
    # if res.ndim > 1:
    #     for i in range(res.shape[1]):
    #         save_name = '%s_%d_res.txt' % (args.suffix, i)
    #         np.savetxt(os.path.join(args.log_dir, split, save_name), res[:,i], fmt='%.3f')

    # opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    # #log.printEpochSummary(opt)

def prepareSave(args, data, pred, pred_r, pred_s, recon, recorder, log):
    input_var, mask_var = data['img'], data['mask']
    results = [input_var.data, recon, mask_var.data]
    shadows = []
    for s in pred_s.split(1, 1):
        s = s.repeat(1, 3, 1, 1)
        shadows.append(s)
    shadow = torch.cat(shadows, 1)
    # if args.s1_est_d:
    #     l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred_c['dirs'].data, args.batch)
    #     recorder.updateIter('train', l_acc.keys(), l_acc.values())
    # if args.s1_est_i:
    #     int_acc, data['int_err'] = eval_utils.calIntsAcc(data['ints'].data, pred_c['intens'].data, args.batch)
    #     recorder.updateIter('train', int_acc.keys(), int_acc.values())

    if args.s2_est_n:
        #acc, error_map = eval_utils.calNormalAcc(data['normal'].data, pred['normal'].data, mask_var.data)
        pred_n = (pred['normal'].data + 1) / 2
        reflectance = pred_r['reflectance']
        
        masked_pred = pred_n * mask_var.data.expand_as(pred['normal'].data)
        masked_reflectance = reflectance * mask_var.expand_as(pred_r['reflectance'].data)
        res_n = [masked_pred, masked_reflectance.data, shadow.data]
        results += res_n
        #recorder.updateIter('train', acc.keys(), acc.values())

    nrow = input_var.shape[0] if input_var.shape[0] <= 32 else 32
    return results, recorder, nrow

def prepareShadowInputs(x):
        #input: list [data['img'], data['mask'], 
        # pred_c{'dirs_x', 'dirs_y', 'dirs', 'ints', 'intens'},
        # pred_n{'normal'}, pred_r{'reflectance'}]
    img = x[0]
    imgs = torch.split(img, 3, 1)
    normal = x[3]['normal']
    dirs = torch.split(x[2]['dirs'], x[0].shape[0], 0)
    n, c, h, w = normal.shape
    shadow_input = []
    for i in range(len(dirs)):
        l_dir = dirs[i] if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1)
        i = imgs[i]
        normal_light = {'normal':normal, 'light':l_dir.expand_as(normal), 'img':i}
        shadow_input.append(normal_light)
    return shadow_input

def reconInputs(args, x):
        #input: list [data['img'], data['mask'], 
        # pred_c{'dirs_x', 'dirs_y', 'dirs', 'ints', 'intens'},
        # pred_n{'normal'}, pred_r{'reflectance'},
        # pred_s[batch, in_img_num*1, h, w]]
        imgs = torch.split(x[0], 3, 1)
        dirs = torch.split(x[2]['dirs'], x[0].shape[0], 0)
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
        return s2_inputs

def reconstruct(pred_n, pred_r, light, shadow,mask):
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
            img = maximam * reflectance * shadows[idx].expand_as(reflectance) * mask
            recons.append(img)
        recon = torch.cat(recons, 1)
        # print("recon max:", recon.max())
        # print("recon min:", recon.min())
        # print('recons:', recon.shape)
        return recon

def prepareNormalRes(args, data, pred, recorder, log, split):
    data_batch = args.val_batch if split == 'val' else args.test_batch
    iter_res = []
    error = ''
    acc, error_map = eval_utils.calNormalAcc(data['normal'].data, pred['normal'].data, data['mask'].data)
    recorder.updateIter(split, acc.keys(), acc.values())
    iter_res.append(acc['n_err_mean'])
    error += 'N_%.3f-' % (acc['n_err_mean'])
    data['error_map'] = error_map['angular_map']

    return recorder, iter_res, error