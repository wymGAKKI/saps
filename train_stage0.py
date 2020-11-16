from models import model_utils
from utils  import eval_utils, time_utils
from PIL import Image, ImageFile
import sys
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(args, loader, model, criterion, optimizer, log, epoch, recorder):
    model.train()
    log.printWrite('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        input = model_utils.parseshadowData(args, sample, timer, 'train')
        pred = model(input); timer.updateTime('Forward')
        optimizer.zero_grad()
        loss = criterion.forward(pred['shadow'], input['shadow']); 
        timer.updateTime('Crit');
        criterion.backward(); timer.updateTime('Backward')
        recorder.updateIter('train', loss.keys(), loss.values())
        optimizer.step(); timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)

        if iters % args.train_save == 0:
            results, recorder, nrow = prepareSave(args, input, pred, recorder, log) 
            log.saveShadowResults(results, 'train', epoch, iters, nrow=nrow)
            log.plotCurves(recorder, 'train', epoch=epoch, intv=args.train_disp)

        if args.max_train_iter > 0 and iters >= args.max_train_iter: break
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareSave(args, data, pred, recorder, log):
    pred_out = pred["shadow"].repeat(1,3,1,1)
    gt_out = data["shadow"].repeat(1,3,1,1)
    results = [gt_out.data, pred_out.data]
    acc, error_map = eval_utils.calShadowAcc(data['shadow'].data, pred['shadow'].data,data['mask'].data)
    recorder.updateIter('train', acc.keys(), acc.values())
    nrow = data['normal'].shape[0] 
    return results, recorder, nrow
