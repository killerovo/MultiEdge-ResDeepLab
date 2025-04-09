import argparse
import logging
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from config import *
from dataset import *
from models import *
from utils import *
from losses import PixelwiseContrastiveLoss
from torch.nn import CrossEntropyLoss
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power for base_lr in self.base_lrs]

def get_scheduler(optimizer, args):
    if args.lr_strategy == 'step':
        return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_strategy == 'exponential':
        return ExponentialLR(optimizer, gamma=args.gamma)
    elif args.lr_strategy == 'cyclic':
        return CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr, step_size_up=args.step_size_up, mode=args.cycle_mode)
    elif args.lr_strategy == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    elif args.lr_strategy == 'poly':
        return PolyLR(optimizer, max_iter=args.epochs, power=args.power)
    else:
        raise ValueError(f"Unknown lr_strategy: {args.lr_strategy}")


def train(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    local_rank = args.rank
    device = torch.device(args.device)
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    

   
   
   
    # Variables and logger Init
    cudnn.benchmark = True
    if local_rank == 0:
        get_logger()
    criterion = CrossEntropyLoss()
    
    
    # Initialize TensorBoard only for the main process
    if local_rank == 0:
        writer = SummaryWriter(log_dir=args.output_root)
    
    # Data Load
    trainloader,train_sampler = data_loader(args, mode='train')
    validloader,valid_sampler = data_loader(args, mode='valid')

    # Model Load
    net, optimizer, best_score, start_epoch = load_model(args, class_num=args.class_num, mode='train')
    net=net.to(device)
    checkpoint_path = os.path.join("../../output", "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if local_rank == 0:
        torch.save(net.state_dict(), checkpoint_path)

    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    net = DDP(net, device_ids=[args.gpu], find_unused_parameters=True)
    
    
    scheduler = get_scheduler(optimizer, args)

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        # Train Model
        if local_rank == 0:
            print('\n\n\nEpoch: {}\n<Train>'.format(epoch))
            print('\n\n\nBatch: {}\n<Train>'.format(args.batch_size))
        net.train()
        loss = 0
        total_dice_loss = 0
        total_contrast_loss=0
        torch.set_grad_enabled(True)
        for idx, (inputs, targets) in enumerate(trainloader):
           
            inputs, targets = inputs.to(device), targets.to(device),reshape_targets.to(device)
            
            
            outputs, out_feature= net(inputs)
                
            
            
            targets_one_hot = F.one_hot(targets, num_classes=args.class_num).permute(0, 3, 1, 2).float()
            dice_loss = dice_coef(outputs, targets_one_hot)
            ce_loss=criterion(outputs,targets)
            batch_loss =ce_loss+dice_loss
            
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_dice_loss += float(dice_loss)
            
            loss += float(batch_loss)
            
            if local_rank == 0:
                writer.add_scalar('Loss/train', batch_loss.item(), epoch * len(trainloader) + idx)
                writer.add_scalar('Contrast/train', contrast_loss.item(), epoch * len(trainloader) + idx)
                writer.add_scalar('Dice/train', 1 - dice_loss.item(), epoch * len(trainloader) + idx)

            
            if local_rank == 0:
                progress_bar(idx, len(trainloader), 'Loss: %.5f, Dice-Coef: %.5f'
                             %((loss / (idx + 1)), (1 - (total_dice_loss / (idx + 1)))))
        if local_rank == 0:
            log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'\
                             %(epoch, loss / (idx + 1), 1 - (total_dice_loss / (idx + 1)))])
            logging.info(log_msg)
            checkpoint = Checkpoint(net, optimizer, epoch)
            checkpoint.save(os.path.join(args.ckpt_root, args.model + '_latest.pt'))
            print("Saving latest weight.........")

        scheduler.step()

        # Validate Model
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            if local_rank == 0:
                print('\n\n<Validation>')
            net.eval()
            for module in net.module.modules():
                if isinstance(module, torch.nn.modules.Dropout2d):
                    module.train(True)
                elif isinstance(module, torch.nn.modules.Dropout):
                    module.train(True)
                else:
                    pass
            loss = 0
            total_dice_loss = 0
            
            torch.set_grad_enabled(False)
            for idx, (inputs, targets) in enumerate(validloader):
                inputs, targets = inputs.to(device), targets.to(device),reshape_targets.to(device)
                
                ori_inputs=inputs.detach().clone()
                outputs, out_feature= net(inputs)
                fused=None
                if type(outputs) == tuple:
                    outputs = outputs[0]
                
                targets_one_hot = F.one_hot(targets, num_classes=args.class_num).permute(0, 3, 1, 2).float()
                dice_loss = dice_coef(outputs.detach().cpu(), targets_one_hot.detach().cpu(), backprop=False)
                ce_loss=criterion(outputs,targets)
                batch_loss =ce_loss+dice_loss
                loss += float(batch_loss)
                total_dice_loss += float(dice_loss)
                
                
                if local_rank == 0:
                    writer.add_scalar('Loss/valid', batch_loss.item(), epoch * len(validloader) + idx)
                    
                    writer.add_scalar('Dice/valid', 1 - dice_loss.item(), epoch * len(validloader) + idx)
                    
                
                if local_rank == 0:
                    progress_bar(idx, len(validloader), 'Loss: %.5f, Dice-Coef: %.5f'
                                 %((loss / (idx + 1)), (1 - (total_dice_loss / (idx + 1)))))
            if local_rank == 0:
                log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'
                                %(epoch, loss / (idx + 1), 1 - (total_dice_loss / (idx + 1)))])
                logging.info(log_msg)

            # Visualize some predictions
            if epoch % args.vis_interval == 0 and local_rank == 0:
                with torch.no_grad():
                    visualize(writer, ori_inputs, fused, outputs, targets, epoch,args.fusion)
                    
        
            # Save Model
            if local_rank == 0:
                
                score = 1 - (total_dice_loss / (idx + 1))
                if score > best_score:
                    checkpoint = Checkpoint(net, optimizer, epoch, score)
                    checkpoint.save(os.path.join(args.ckpt_root, args.model + '_val_best.pt'))
                    best_score = score
                    print("Saving val best weight.........")

    if local_rank == 0:
        writer.close()
    
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=bool, default=False,
                        help="Model Training resume.")
    parser.add_argument("--mode", type=str, default="train",
                        help="The stata of Model")  
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Number of epochs between validations")
    parser.add_argument("--vis_interval", type=int, default=10,
                        help="Number of epochs between vis")

    parser.add_argument("--model", type=str, default='unet',
                        help="Model Name (unet, pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--in_channel", type=int, default=3,
                        help="A number of images to use for input")

    parser.add_argument("--class_num", type=int, default=4,
                        help="A number of classes for segmentation label")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=300,
                        help="The training epochs to run.")
    
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate to use in training")
   
    parser.add_argument("--output_root", type=str, default="../../output",
                        help="The directory containing the result predictions")
    parser.add_argument("--ckpt_root", type=str, default="../../output",
                        help="The directory containing the checkpoint files")

    parser.add_argument("--lr_strategy", type=str, default='poly',
                        help="Learning rate strategy (step, exponential, cyclic, cosine, poly)")
    parser.add_argument("--step_size", type=int, default=30,
                        help="Step size for StepLR and CyclicLR")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Gamma for StepLR and ExponentialLR")
    parser.add_argument("--base_lr", type=float, default=0.001,
                        help="Base learning rate for CyclicLR")
    parser.add_argument("--max_lr", type=float, default=0.1,
                        help="Max learning rate for CyclicLR")
    parser.add_argument("--step_size_up", type=int, default=2000,
                        help="Step size up for CyclicLR")
    parser.add_argument("--cycle_mode", type=str, default='triangular',
                        help="Mode for CyclicLR (triangular, triangular2, exp_range)")
    parser.add_argument("--t_max", type=int, default=50,
                        help="Maximum number of iterations for CosineAnnealingLR")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Power for PolyLR")

    
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--rank', default=0, type=int,
                        help='no of distributed processes')
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    train(args)
