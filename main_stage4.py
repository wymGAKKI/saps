import torch
from options  import stage4_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils

import train_stage4 as train_utils
import test_stage4 as test_utils

args = stage4_opts.TrainOpts().parse()
log  = logger.Logger(args)

def main(args):
    model_s1 = custom_model.buildModel(args)
    model_s2 = custom_model.buildModelStage2(args)
    model_s3 = custom_model.buildModelStage3(args)
    model_s0 = custom_model.buildModelStage0(args)
    models = [model_s1, model_s2, model_s3, model_s0]

    optimizer, scheduler, records = solver_utils.configMultiOptimizer(args, models)
    optimizers = [optimizer, -1]
    criterion = solver_utils.Stage4Crit(args)
    recorder  = recorders.Records(args.log_dir, records)

    train_loader, val_loader = custom_data_loader.myDataloader(args)

    for epoch in range(args.start_epoch, args.epochs+1):
        scheduler.step()

        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])

        train_utils.train(args, train_loader, models, criterion, optimizers, log, epoch, recorder)
        if epoch % args.save_intv == 0: 
            model_utils.saveMultiCheckpoint(args.cp_dir, epoch, models, optimizer, recorder.records, args)
        log.plotCurves(recorder, 'train')

        if epoch % args.val_intv == 0:
            test_utils.test(args, 'val', val_loader, models, log, epoch, recorder)
            log.plotCurves(recorder, 'val')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
