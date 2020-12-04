import torch
from models import model_utils
from utils  import eval_utils, time_utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def train(args, loader, model, criterion, optimizers, log, epoch, recorder):
    model.train()
    optimizer, optimizer_c = optimizers
    log.printWrite('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        input = model_utils.parseReflectanceData(args, sample, timer, 'train')
        pred_r =  model(input); timer.updateTime('Forward')    
        optimizer.zero_grad()

        loss = criterion.forward(255 * pred_r['reflectance'] ,255 * input['reflectance']); 
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
            results, recorder, nrow = prepareReflectanceSave(args, input, pred_r,  recorder, log) 
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

def prepareReflectanceSave(args, data, pred, recorder, log):
    pred_out = pred["reflectance"]
    gt_out = data["reflectance"]
    mask = data['mask']
    masked_pred = pred_out * mask.data.expand_as(pred['reflectance'].data)
    results = [gt_out.data, masked_pred.data]
    acc, error_map = eval_utils.calShadowAcc(data['reflectance'].data, pred['reflectance'].data,data['mask'].data)
    recorder.updateIter('train', acc.keys(), acc.values())
    nrow = data['img'].shape[0] 
    return results, recorder, nrow

def reconInputs(args, x):
        imgs = torch.split(x[0], 3, 1)
        idx = 1
        if args.in_light: idx += 1
        if args.in_mask:  idx += 1
        dirs = torch.split(x[idx]['dirs'], x[0].shape[0], 0)
        ints = torch.split(x[idx]['intens'], 3, 1)
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

def reconstruct(pred_n, pred_r, light, ints):
        normal = pred_n
        n, c, h, w = normal.shape
        # print('***************reflectance', pred_nr['reflectance'].max(), pred_nr['reflectance'].min())
        reflectance = (pred_r + 1) / 2
        # print("reflectance max:", reflectance.max())
        # print("reflectance min:", reflectance.min())
        # normal, reflectance:[batch, 3, height, weight]
        # light: [batch, 3*input_img, height, weight]
        lights = torch.split(light, 3, 1)
        recons = []
        zeros = torch.zeros_like(reflectance)
        for idx in range(len(lights)):
            product = (lights[idx] * normal).sum(1, keepdim = True).expand_as(reflectance)
            zeros = torch.zeros_like(product)
            #print(product.dtype)
            #maximam = torch.max(zeros, product)
            maximam = torch.max(product, zeros)
            img = maximam * reflectance
            l_int = torch.diag(ints[idx].contiguous().view(-1))
            img   = img.contiguous().view(n * c, h * w)
            img   = torch.mm(l_int, img).view(n, c, h, w)
            recons.append(img)
        recon = torch.cat(recons, 1)
        # print("recon max:", recon.max())
        # print("recon min:", recon.min())
        # print('recons:', recon.shape)
        return recon