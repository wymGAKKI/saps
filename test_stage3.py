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

def test(args, split, loader, model, log, epoch, recorder):
    model.eval()
    log.printWrite('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    res = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            #print("\nIn for:", sample['mask'].shape)
            #print("sample :", sample.keys())
            input = model_utils.parseReflectanceData(args, sample, timer, split)
            #input is a list
            pred = model(input); timer.updateTime('Forward')

            recoder, iter_res, error = prepareRes(args, input, pred, recorder, log, split)
            #print("data['img'].shape:", data['img'].shape)
            res.append(iter_res)
            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                results, nrow = prepareSave(args, input, pred, recorder, log)
                log.saveImgResults(results, split, epoch, iters, nrow=nrow, error='')
                #log.saveMatResults(pred['normal'], data['normal'], pred_c['dirs'], data['dirs'], split, epoch, iters, nrow=nrow, error='')
                log.plotCurves(recorder, split, epoch=epoch, intv=disp_intv)

            if stop_iters > 0 and iters >= stop_iters: break
    res = np.vstack([np.array(res), np.array(res).mean(0)])
    save_name = '%s_res.txt' % (args.suffix)
    np.savetxt(os.path.join(args.log_dir, split, save_name), res, fmt='%.2f')
    if res.ndim > 1:
        for i in range(res.shape[1]):
            save_name = '%s_%d_res.txt' % (args.suffix, i)
            np.savetxt(os.path.join(args.log_dir, split, save_name), res[:,i], fmt='%.3f')

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareRes(args, data, pred, recorder, log, split):
    data_batch = args.val_batch if split == 'val' else args.test_batch
    iter_res = []
    error = ''
    acc, error_map = eval_utils.calReflectanceAcc(data['reflectance'].data, pred['reflectance'].data, data['mask'].data)
    recorder.updateIter(split, acc.keys(), acc.values())
    iter_res.append(acc['reflectance_err_mean'])
    error += 'N_%.3f-' % (acc['reflectance_err_mean'])
    data['error_map'] = error_map['angular_map']

    return recorder, iter_res, error

def prepareSave(args, data, pred, recorder, log):

    masked_pred = pred['reflectance'] * data['mask'].data.expand_as(pred['reflectance'].data)
    #acc, error_map = eval_utils.calShadowAcc(data['reflectance'].data, pred['reflectance'].data,data['mask'].data)
    #recorder.updateIter('train', acc.keys(), acc.values())
    results = [data['img'].data, data['reflectance'].data, masked_pred.data]
    nrow = pred["reflectance"].shape[0]
    return results, nrow
