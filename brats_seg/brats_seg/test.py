import pdb
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchmetrics.classification import Dice, JaccardIndex
import torchmetrics

from sklearn.metrics import jaccard_score, f1_score

from config import *
from collections import OrderedDict
from dataset import *
from models import *
from utils import *



def fusion(batch_input ,model_fusion,is_save=False):
    '''
    batch: shape(B, num_modals, T, H, W)
    '''
    
    B, C, H, W = batch_input.shape
    
    #flair,t1ce,t2
    x1 = batch_input[:, 0, :, :].unsqueeze(1)
    x2 = batch_input[:, 1, :, :].unsqueeze(1)
    x3 = batch_input[:, 2, :, :].unsqueeze(1)
    del batch_input
    torch.cuda.empty_cache()
    with torch.no_grad():
        
        x12 = model_fusion(x1.repeat(1, 3, 1, 1), x2.repeat(1, 3, 1, 1)).clamp(0, 1)
        x13 = model_fusion(x1.repeat(1, 3, 1, 1), x3.repeat(1, 3, 1, 1)).clamp(0, 1)
        #import pdb; pdb.set_trace()
        x23 = model_fusion(x2.repeat(1, 3, 1, 1), x3.repeat(1, 3, 1, 1)).clamp(0, 1)

    # 将RGB输出转换为灰度
    weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).cuda()
    x12 = torch.sum(x12 * weights, dim=1)
    x13 = torch.sum(x13 * weights, dim=1)
    x23 = torch.sum(x23 * weights, dim=1)

    batch_fused = torch.stack([x12, x13, x23], 1)  # (B * T, C, H, W)
    return batch_fused

def test(args):

    # Device Init
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cudnn.benchmark = True

    # Data Load
    testloader = data_loader(args, mode='test')

    # Model Load
    # Model Load
    net, _,  _,   _,  = load_model(args, class_num=args.class_num, mode='test')
    
    #net = torch.nn.DataParallel(net)
    net = net.to(device)
    checkpoint = torch.load(args.ckpt_root,map_location=device)

    # 如果是多卡训练的权重文件，需要去掉 'module.' 前缀
    state_dict = checkpoint["model_state"]
    net.load_state_dict(state_dict,strict=False)


   
    
    
    toPIL = transforms.ToPILImage() 
    # Define metrics
    total_dice_loss = 0
    dice_metric = Dice(num_classes=args.class_num).to(device)
    iou_metric = JaccardIndex(num_classes=args.class_num,task="multiclass").to(device)

    net.eval()
    torch.set_grad_enabled(False)
    for idx, (inputs, labels) in enumerate(testloader):
        inputs = inputs.to(device)
        labels=labels.to(device)
        
        outputs,_= net(inputs)
        b=int(outputs.size(0)/2)
       
        if type(outputs) == tuple:
            outputs = outputs[0]
        targets_one_hot = F.one_hot(labels, num_classes=args.class_num).permute(0, 3, 1, 2).float()        
        dice_loss = dice_coef(outputs, targets_one_hot)
        total_dice_loss += float(dice_loss)
        print(idx)
    print(1 - (total_dice_loss / (idx + 1)))
    
    # Compute final metrics
    dice_per_class = dice_metric.compute()
    iou_per_class = iou_metric.compute()
    
    avg_dice = dice_per_class.mean().item()
    avg_iou = iou_per_class.mean().item()
    
    print("Dice per class:", dice_per_class)
    print("IoU per class:", iou_per_class)
    print("Average Dice:", avg_dice)
    print("Average IoU:", avg_iou)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class_num", type=int, default=4,
                        help="A number of classes for segmentation label")
    parser.add_argument("--model", type=str, default='deeplab', # Need to be fixed
                        help="Model Name")
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument("--fusion", type=bool, default=True,
                        help="Multi-modal Fusion.")
    parser.add_argument("--batch_size", type=int, default=16, # Need to be fixed
                        help="The batch size to load the data")
    parser.add_argument("--img_root", type=str, default="../data/train/image_FLAIR",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the results.")
    parser.add_argument("--ckpt_root", type=str, default="../../model/hywang/fusionnet/94_deeplab_latest.pt",
                        help="The directory containing the trained model checkpoint")
    parser.add_argument("--resume", type=bool, default=False,
                        help="Model Training resume.")
    args = parser.parse_args()

    test(args)
