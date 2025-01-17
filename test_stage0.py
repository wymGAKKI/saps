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
            input = model_utils.parseshadowData(args, sample, timer, split)
            shadow_input = prepareShadowInputs(input)
            shadowlist = []
            for s in shadow_input:
                shadow = models[3](s)['shadow']
                shadowlist.append(shadow)
            pred_s = torch.cat(shadowlist, 1)
            #pred = model(input); timer.updateTime('Forward')

            #recoder, iter_res, error = prepareRes(args, input, pred, recorder, log, split)

            res.append(iter_res)
            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                results, recorder, nrow = prepareSave(args, input, pred, recorder, log)
                log.saveShadowResults(results, split, epoch, iters, nrow=nrow, error=error)
                log.plotCurves(recorder, split, epoch=epoch, intv=disp_intv)

            if stop_iters > 0 and iters >= stop_iters: break
    res = np.vstack([np.array(res), np.array(res).mean(0)])
    save_name = '%s_res.txt' % (args.suffix)
    #np.savetxt(os.path.join(args.log_dir, "shadow",split, save_name), res, fmt='%.2f')
    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def testOnBm(args, split, loader, model, log, epoch, recorder):
    model.eval()
    log.printWrite('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    res = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            input = model_utils.getShadowInput(args, sample, timer, split)
            pred = model(input); timer.updateTime('Forward')

            #recoder, iter_res, error = prepareRes(args, input, pred, recorder, log, split)

            res.append(iter_res)
            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                results, recorder, nrow = prepareSave(args, input, pred, recorder, log)
                log.saveShadowResults(results, split, epoch, iters, nrow=nrow, error=error)
                log.plotCurves(recorder, split, epoch=epoch, intv=disp_intv)

            if stop_iters > 0 and iters >= stop_iters: break
    #res = np.vstack([np.array(res), np.array(res).mean(0)])
    save_name = '%s_res.txt' % (args.suffix)
    #np.savetxt(os.path.join(args.log_dir, "shadow",split, save_name), res, fmt='%.2f')
    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareRes(args, data, pred, recorder, log, split):
    data_batch = args.val_batch if split == 'val' else args.test_batch
    iter_res = []
    error = ''
    acc, error_map = eval_utils.calShadowAcc(data['shadow'].data, pred['shadow'].data, data['mask'].data)
    recorder.updateIter(split, acc.keys(), acc.values())
    iter_res.append(acc['shadow_err_mean'])
    error += 'N_%.3f-' % (acc['shadow_err_mean'])
    data['error_map'] = error_map['angular_map']

    return recorder, iter_res, error


def prepareSave(args, data, pred, recorder, log):
    pred_out = pred["shadow"].repeat(1, 3, 1, 1)
    #gt_out = data["shadow"].repeat(1, 3, 1, 1)
    img = data['img']
    results = [gt_out.data, pred_out.data, img.data]
    acc, error_map = eval_utils.calShadowAcc(data['shadow'].data, pred['shadow'].data,data['mask'].data)
    recorder.updateIter('train', acc.keys(), acc.values())
    nrow = pred["shadow"].shape[0] 
    return results, recorder, nrow
