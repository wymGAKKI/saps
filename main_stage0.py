import torch
from options  import stage0_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils

import train_stage0 as train_utils
import test_stage0  as test_utils

args = stage0_opts.TrainOpts().parse()
log  = logger.Logger(args)

def main(args):
    model = custom_model.buildModelStage0(args)
    optimizer, scheduler, records = solver_utils.configOptimizer(args, model)
    criterion = solver_utils.Stage0Crit(args)
    recorder  = recorders.Records(args.log_dir, records)

    train_loader, val_loader = custom_data_loader.shadowDataloader(args)

    for epoch in range(args.start_epoch, args.epochs+1):
        optimizer.step()
        scheduler.step()
        recorder.insertRecord('train', 'lr', epoch, scheduler.get_last_lr())

        train_utils.train(args, train_loader, model, criterion, optimizer, log, epoch, recorder)
        if epoch % args.save_intv == 0: 
            model_utils.saveCheckpoint(args.cp_dir, epoch, model, optimizer, recorder.records, args)
        log.plotCurves(recorder, 'train')

        if epoch % args.val_intv == 0:
            test_utils.test(args, 'val', val_loader, model, log, epoch, recorder)
            log.plotCurves(recorder, 'val')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)