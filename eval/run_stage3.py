import torch,sys
sys.path.append('.')
from options  import stage3_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils
from options  import run_model_opts
import train_stage3 as train_utils
import test_stage3 as test_utils

args = run_model_opts.RunModelOpts().parse()
args = stage3_opts.TrainOpts().parse()
log  = logger.Logger(args)

#### CUDA_VISIBLE_DEVICES=0 python main_stage3.py --retrain "/home/wym/code/SDPS-Net/data/models/LCNet_CVPR2019.pth.tar" --retrain_s2 "/home/wym/code/SDPS-Net/data/models/NENet_CVPR2019.pth.tar"
def main(args):
    model = custom_model.buildModelStage3(args)

    recorder  = recorders.Records(args.log_dir)
    val_loader = custom_data_loader.benchmarkLoader(args)
    #train_loader, val_loader = custom_data_loader.reflectanceDataloader(args)
    test_utils.testOnBm(args, 'val', val_loader, model, log, 1, recorder)
    log.plotCurves(recorder, 'val')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)