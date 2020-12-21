import torch
from options  import stage4_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils

import train_poke as train_utils
import test_poke as test_utils

args = stage4_opts.TrainOpts().parse()
log  = logger.Logger(args)
args.retrain = "/home/wym/code/SDPS-Net/data/models/LCNet.pth.tar"
args.retrain_s2 = "/home/wym/code/SDPS-Net/data/models/NENet.pth.tar"
args.retrain_s3 = "/home/wym/code/SDPS-Net/data/models/skipRENetnoBatch.pth.tar"
args.retrain_s0 = "/home/wym/code/SDPS-Net/data/models/CSNetDi.pth.tar"
#args.retrain_s0 = "/home/wym/code/SDPS-Net/data/logdir/UPS_Synth_Dataset/Shadow/12-2,LCNet,max,adam,cos,ba_h-8,sc_h-128,cr_h-128,in_r-5e-05,no_w-1,di_w-1,in_w-1,in_m-32,di_s-36,in_s-20,in_mask,s1_est_d,s1_est_i,color_aug,int_aug,concat_data/checkpointdir/checkp_30.pth.tar"
def main(args):
    model_s1 = custom_model.buildModel(args)
    model_s2 = custom_model.buildModelStage2(args)
    model_s3 = custom_model.buildModelStage3(args)
    model_s0 = custom_model.buildModelStage0(args)
    models = [model_s1, model_s2, model_s3, model_s0]

    #optimizer, scheduler, records = solver_utils.configMultiOptimizer(args, models)
    optimizer, scheduler, records = solver_utils.configMultiOptimizer(args, [models[1], models[2]])
    optimizers = [optimizer, -1]
    criterion = solver_utils.Stage4Crit(args)
    recorder  = recorders.Records(args.log_dir, records)

    train_loader, val_loader = custom_data_loader.pokemonDataloader(args)

    for epoch in range(args.start_epoch, args.epochs+1):
        scheduler.step()

        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])

        train_utils.train(args, train_loader, models, criterion, optimizers, log, epoch, recorder)
        if epoch % args.save_intv == 0: 
            model_utils.saveMultiCheckpoint(args.cp_dir, epoch, models, optimizer, recorder.records, args)
        #log.plotCurves(recorder, 'train')

        if epoch % args.val_intv == 0:
            test_utils.test(args, 'val', val_loader, models, log, epoch, recorder)
            #log.plotCurves(recorder, 'val')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
