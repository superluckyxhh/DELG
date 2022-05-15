import torch
import torch.nn as nn
import torch.nn.functional as F
    

class GeMPool(nn.Module):
    def __init__(self, gem_p=3.0, trainable=False, eps=1e-6):
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * gem_p)
        else:
            self.p = gem_p
        
        self.eps = eps
        
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)