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


# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         # stdv = 1. / math.sqrt(self.weight.size(1))
#         # self.weight.data.uniform_(-stdv, stdv)

#     def forward(self, features):
#         cosine = F.linear(F.normalize(features), F.normalize(self.weight))
#         return cosine
    

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


class GlobalCls(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.fc = nn.Linear(512, num_class)
        
    def forward(self, x):
        return self.fc(x)
    

class DELG_Global(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.extractor = ResNeXt(name=backbone)
        self.global_head = GlobalHead(cfg.MODEL.REDUCTION_DIM)
        self.global_cls = ArcMarginProduct_subcenter(cfg.MODEL.REDUCTION_DIM, num_classes)
        # self.global_cls = GlobalCls(num_classes)
        logger.info('DELG Global Encoder Model Initial Done')
        
    def forward(self, x):
        local_featmap, global_featmap = self.extractor(x)
        global_prelogits = self.global_head(global_featmap)
        global_logits = self.global_cls(global_prelogits)
        
        return global_prelogits, global_logits, local_featmap
            

# if __name__ == '__main__':
#     inps = torch.randn((8, 3, 256, 256)).cuda()
#     model = DELG_Global('resneXt101', 81313).cuda()
#     output = model(inps)
#     for name, child in model.named_children():
#         print('name:', name)
#         print('child:', child)