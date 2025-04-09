import pdb
import cv2
import os
import numpy as np
import nibabel as nib
import torch
import sys
import time
import logging
import logging.handlers
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from torchvision.utils import make_grid
'''
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral,\
         create_pairwise_gaussian, softmax_to_unary, unary_from_softmax
'''
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

from abc import ABC

class Configer:
    def __init__(self, config):
        self.config = config

    def get(self, section, key):
        return self.config[section][key]

    def exists(self, section, key):
        return key in self.config.get(section, {})

setting = {
    'contrast': {
        'temperature': 0.07,
        'base_temperature': 0.07,
        'max_samples': 1024,
        'max_views': 100,
    },
    'loss': {
        'params': {
            'ce_ignore_index': -1
        }
    }
}

configer = Configer(setting)

class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer=configer):
        super(PixelContrastLoss, self).__init__()
        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')
        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']
        self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_views = self.configer.get('contrast', 'max_views')

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
            classes.append(this_classes)
            total_classes += len(this_classes)
        if total_classes == 0:
            return None, None
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception('this should never be touched! {} {} {}'.format(num_hard, num_easy, n_view))
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits+ 1e-10)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)
        batch_size = feats.shape[0]
        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        
        if feats_ is None: 
            loss=0
        else:
            loss = self._contrastive(feats_, labels_)
        return loss
    
# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        
        reduction = 'mean'
        

        ignore_index = -1
        

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()
    
class ContrastCELoss(nn.Module, ABC):
    def __init__(self, configer=configer):
        super(ContrastCELoss, self).__init__()

        self.configer = configer

        
        
        

        self.loss_weight = 0.1
        self.use_rmi = False

        self.seg_criterion = FSCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, seg,embedding, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        _, predict = torch.max(seg, 1)
        #import pdb;pdb.set_trace()
        #print(embedding.shape,target.shape,predict.shape)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training
    
def dice_coef(preds, targets, backprop=True):
   
    smooth = 1e-5
    class_num = 4
    if backprop:
        for i in range(class_num):
            pred = preds[:,i,:,:]
            target = targets[:,i,:,:]
            intersection = (pred * target).sum()
            loss_ = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            if i == 0:
                loss = loss_
            else:
                #import pdb;pdb.set_trace()
                loss = loss + loss_
        loss = loss/class_num
        return loss
    else:
        # Need to generalize
        targets = np.array(targets.argmax(1))
        if len(preds.shape) > 3:
            preds = np.array(preds).argmax(1)
        for i in range(class_num):
            pred = (preds==i).astype(np.uint8)
            target= (targets==i).astype(np.uint8)
            intersection = (pred * target).sum()
            loss_ = 1-((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            if i == 0:
                loss =loss_
            else:
                loss = loss + loss_
        loss = loss/(class_num)
        return loss


def visualize(writer, inputs, fused, outputs, targets, epoch,fusion):
    input_names = ['flair', 't1c', 't2']
    fused_names = ['flair-t1c', 'flair-t2', 't1c-t2']
    
    # 拆分输入和融合的模态
    for i, name in enumerate(input_names):
        # 拆分每个模态
        modality = inputs[:, i, :, :].unsqueeze(1)
        # 创建输入图像网格
        input_grid = make_grid(modality.cpu(), nrow=4, scale_each=True)
        writer.add_image(f'Input Images/{name}', input_grid, epoch)
    
    # if fusion:
    #     for i, name in enumerate(fused_names):
    #         # 拆分每个模态
    #         modality = fused[:, i, :, :].unsqueeze(1)
    #         # 创建融合图像网格
    #         fused_grid = make_grid(modality.cpu(), nrow=4, scale_each=True)
    #         writer.add_image(f'Fused Images/{name}', fused_grid, epoch)
    
    # 创建输出图像网格
    output_grid = make_grid(outputs.argmax(dim=1, keepdim=True).cpu()*64, nrow=4, scale_each=True)
    writer.add_image('Predictions', output_grid, epoch)
    
    # 创建目标图像网格
    target_grid = make_grid(targets.cpu().unsqueeze(1)*64, nrow=4, scale_each=True)
    writer.add_image('Ground Truth', target_grid, epoch)



def erode_dilate(outputs, kernel_size=7):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    outputs = outputs.astype(np.uint8)
    for i in range(outputs.shape[0]):
        img = outputs[i]
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        outputs[i] = img
    return outputs


def post_process(args, inputs, outputs, input_path=None,
                 crf_flag=True, erode_dilate_flag=True,
                 save=True, overlap=True):
    inputs = (np.array(inputs.squeeze()).astype(np.float32)) * 255
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs,inputs,inputs), axis=3)
    outputs = np.array(outputs)

    # Conditional Random Field
    if crf_flag:
        outputs = get_crf_img(inputs, outputs)
    else:
        outputs = outputs.argmax(1)

    # Erosion and Dilation
    if erode_dilate_flag:
        outputs = erode_dilate(outputs, kernel_size=7)
    if save == False:
        return outputs

    outputs = outputs*255
    for i in range(outputs.shape[0]):
        path = input_path[i].split('/')
        output_folder = os.path.join(args.output_root, path[-2])
        try:
            os.mkdir(output_folder)
        except:
            pass
        output_path = os.path.join(output_folder, path[-1])
        if overlap:
            img = outputs[i]
            img = np.expand_dims(img, axis=2)
            zeros = np.zeros(img.shape)
            img = np.concatenate((zeros,zeros,img), axis=2)
            img = np.array(img).astype(np.float32)
            img = inputs[i] + img
            if img.max() > 0:
                img = (img/img.max())*255
            else:
                img = (img/1) * 255
            cv2.imwrite(output_path, img)
        else:
            img = outputs[i]
            cv2.imwrite(output_path, img)
    return None

'''
TODO: Need to fix
def save_img(args, inputs, outputs, input_paths, overlap=True):
    inputs = (np.array(inputs.squeeze()).astype(np.float32)) * 255
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs,inputs,inputs), axis=3)
    inputs = np.expand_dims(inputs, axis=3)
    outputs = np.array(outputs.max(1)[1])*255
    kernel = np.ones((5,5),np.uint8)

    for i, path in enumerate(input_paths):
        path = path.split('/')[-2]
        if i == 0:
            compare = path
        else:
            if compare != path:
                raise ValueError('Output Merge Fail')
            pass

    final_img = None
    output_path = os.path.join(args.output_root, path+'.nii.gz')
    for i in range(outputs.shape[0]):
        if overlap:
            img = cv2.morphologyEx(outputs[i].astype(np.uint8), cv2.MORPH_OPEN, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = np.expand_dims(img, axis=2)
            zeros = np.zeros(img.shape)
            img = np.concatenate((zeros,zeros,img), axis=2)
            img = np.expand_dims(img, axis=2)
            img = np.array(img).astype(np.float32)
            img = inputs[i] + img
            if img.max() > 0:
                img = (img/img.max())*255
            else:
                img = (img/1) * 255
            img = np.expand_dims(img, axis=3)
            if i == 0:
                final_img = img
            else:
                final_img = np.concatenate((final_img,img),axis=3)
        else:
            img = output[i]
    output_path = os.path.join(args.output_root, path)
    final_img = nib.Nifti1Pair(final_img, np.eye(4))
    nib.save(final_img, output_path)
    print(output_path)
'''


class Checkpoint:
    def __init__(self, model, optimizer=None, epoch=0, best_score=1):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.best_score = best_score

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.epoch = checkpoint["epoch"]
        self.best_score = checkpoint["best_score"]
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in self.optimizer.state.values():
                  for k, v in state.items():
                           if torch.is_tensor(v):
                                    state[k] = v.cuda()

    def save(self, path):
        state_dict = self.model.module.state_dict()
        torch.save({"model_state": state_dict,
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_score": self.best_score}, path)


def progress_bar(current, total, msg=None):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_logger(level="DEBUG", file_level="DEBUG"):
    logger = logging.getLogger(None)
    logger.setLevel(level)
    fomatter = logging.Formatter(
            '%(asctime)s  [%(levelname)s]  %(message)s  (%(filename)s:  %(lineno)s)')
    fileHandler = logging.handlers.TimedRotatingFileHandler(
            'result.log', when='d', encoding='utf-8')
    fileHandler.setLevel(file_level)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)
    return logger
