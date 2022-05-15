import torch
import torch.nn as nn
import numpy as np
import math
from core.config import cfg
import torchvision


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
            
            
def lr_cos_fun(cur_epoch):
    return 0.5 * cfg.OPTIM.BASE_LR * (1 + math.cos(cur_epoch / cfg.OPTIM.MAX_EPOCHS * math.pi))
    

def get_epoch_lr(cur_epoch):
    lr = lr_cos_fun(cur_epoch)
    # Warm Up
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS:
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        cur_warmup_factor = (1 - alpha) * cfg.OPTIM.WARMUP_FACTOR + alpha
        lr *= cur_warmup_factor
    
    return lr
    

def set_epoch_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def construct_optimizer(model):
    params = model.parameters()
    if cfg.OPTIM.NAME == 'Adam':
        optimizer = torch.optim.Adam(params, 
                              lr=cfg.OPTIM.BASE_LR, 
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY
                              )
    elif cfg.OPTIM.NAME == 'SGD':
        optimizer = torch.optim.SGD(params, 
                            lr=cfg.OPTIM.BASE_LR, 
                            momentum=cfg.OPTIM.MOMENTUM,
                            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                            dampening=cfg.OPTIM.DAMPENING,
                            nesterov=cfg.OPTIM.NESTEROV)
    elif cfg.OPTIM.NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, 
                              lr=cfg.OPTIM.BASE_LR, 
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY
                              )
    
    return optimizer