import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from logzero import logger
from core.config import cfg

from models.delg_global import DELG_Global

 
class AttnHead(nn.Module):
    def __init__(self, in_planes, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, num_classes, bias=True)
        self._init_parameters()
    
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    
class Attention(nn.Module):
    def __init__(self, hidden_dim, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1024, hidden_dim, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=kernel_size) 
        self.active_layer = nn.Softplus()
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        attn_score = self.active_layer(x)
        
        # L2-normlize the feature map before pooling
        # L2-normlization shape: [b, c, h, w]
        inputs_norm = F.normalize(inputs, p=2, dim=1)
        attn = attn_score.expand_as(inputs_norm)
        # shape: [b, c, h, w]
        feat = attn * inputs_norm
        # tmp_feat = (attn * inputs_norm).sum((2, 3))
        
        return feat, attn_score


class AutoEncoder(nn.Module):
    def __init__(self, reduced_dim=128, expand_dim=1024, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1024, reduced_dim, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(reduced_dim, expand_dim, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        reduced_feat = self.conv1(x)
        expand_feat = self.conv2(reduced_feat)
        expand_feat = self.relu(expand_feat)
        
        return reduced_feat, expand_feat 


class DELG_Local(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.global_encoder = DELG_Global(backbone, num_classes)
        self.attn = Attention(hidden_dim=cfg.MODEL.REDUCTION_DIM)
        self.auto_encoder = AutoEncoder(reduced_dim=cfg.MODEL.REDUCED_DIM,
                                        expand_dim=cfg.MODEL.EXPAND_DIM)
        self.attn_cls = AttnHead(1024, num_classes)
        
        logger.info('DELG Model Initial Done')
        
    def forward(self, x):
        # with torch.no_grad():
        #     _, _, local_featmap = self.global_encoder(x)
        _, _, local_featmap = self.global_encoder(x)
        
        local_feats, local_expand_feats = self.auto_encoder(local_featmap)
        local_attn_prelogits, attn_score = self.attn(local_expand_feats)
        
        local_attn_logits = self.attn_cls(local_attn_prelogits)
        
        return local_attn_logits, local_expand_feats, attn_score, local_featmap
            

if __name__ == '__main__':
    inps = torch.randn((1, 3, 128, 128)).cuda()
    model = DELG_Local('resneXt50', 81313).cuda()
    output = model(inps)
    for name, child in model.named_children():
        print('name:', name)
        print('child:', child)