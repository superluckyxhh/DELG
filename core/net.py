import torch
import torch.nn as nn
import math
from core.config import cfg


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
        
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = cfg.BN.ZERO_INIT_FINAL_GAMMA
        zero_init_gamma = hasattr(m, 'final_bn') and m.final_bn and zero_init_gamma
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
        
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()
        

def freeze_weights(model, freeze=[]):
    for name, child in model.named_children():
        if name in freeze:
            for param in child.parameters():
                param.requires_grad = False
                
def unfreeze_weights(model, freeze=[]):
    for name, child in model.named_children():
        if name in freeze:
            for param in child.parameters():
                param.requires_grad = True