import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from logzero import logger
from core.config import cfg

from modules.pool import GeMPool
from modules.resnext import ResNeXt


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        self._init_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine
    

class GlobalHead(nn.Module):
    def __init__(self, hidden_planes):
        super().__init__()
        self.gempool = GeMPool(gem_p=cfg.GEM.P, trainable=cfg.GEM.TRAIN)
        self.fc = nn.Linear(2048, hidden_planes, bias=True)
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
        x = self.gempool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    
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


class DELG(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.extractor = ResNeXt(name=backbone)
        self.global_head = GlobalHead(cfg.MODEL.REDUCTION_DIM)
        self.attn = Attention(hidden_dim=cfg.MODEL.REDUCTION_DIM)
        self.auto_encoder = AutoEncoder(reduced_dim=cfg.MODEL.REDUCED_DIM,
                                        expand_dim=cfg.MODEL.EXPAND_DIM)
        self.global_cls = ArcMarginProduct_subcenter(cfg.MODEL.REDUCTION_DIM, num_classes)
        self.attn_cls = AttnHead(1024, num_classes)
        
        logger.info('DELG Model Initial Done')
        
    def forward(self, x):
        local_featmap, global_featmap = self.extractor(x)
        block3 = local_featmap.detach()
        
        global_prelogits = self.global_head(global_featmap)
        
        local_feats, local_expand_feats = self.auto_encoder(block3)
        local_attn_prelogits, attn_score = self.attn(local_expand_feats)
        
        global_logits = self.global_cls(global_prelogits)
        local_attn_logits = self.attn_cls(local_attn_prelogits)
        
        return global_prelogits, local_feats, local_attn_prelogits, \
            local_expand_feats, attn_score, global_logits, \
            local_attn_logits, local_featmap, block3
            

if __name__ == '__main__':
    inps = torch.randn((8, 3, 256, 256)).cuda()
    model = DELG('resneXt101', 81313).cuda()
    output = model(inps)
    for name, child in model.named_children():
        print('name:', name)
        print('child:', child)