import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from core.config import cfg
from modules.resnext import ResNeXt


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
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
    

class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super().__init__()
        
        self.d0 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[0],padding='same')
        self.d1 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[1],padding='same')
        self.d2 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[2],padding='same')
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        
        return x


class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm

        return x, att_score   


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fl, fg):
        bs, c, w, h = fl.shape
        
        fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
        fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
        fg_norm = torch.norm(fg, dim=1)
        
        fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
        fl_orth = fl - fl_proj
        
        f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
        
        return f_fused  


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super().__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        res = gem(x, p=self.p, eps=self.eps)   
    
        return res


class DOLG(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = ResNeXt(name=backbone)
        self.global_pool = GeM(p_trainable=cfg.GEM.TRAIN)
        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        
        self.multi_atrous = MultiAtrousModule(1024, 1024, [6, 12, 18])
        self.attention = SpatialAttention2d(1024)
        self.fusion = OrthogonalFusion()
        self.conv = nn.Conv2d(2048, 1024, kernel_size=1)
        self.bn = nn.BatchNorm2d(1024, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act =  nn.SiLU(inplace=True)
        
        self.neck = nn.Sequential(
                nn.Linear(2048, cfg.MODEL.REDUCTION_DIM, bias=True),
                nn.BatchNorm1d(cfg.MODEL.REDUCTION_DIM),
                torch.nn.PReLU()
            )
        self.head = ArcMarginProduct_subcenter(cfg.MODEL.REDUCTION_DIM, num_classes)
        
    def forward(self, x):
        f_l, f_g = self.backbone(x)
        f_l = self.multi_atrous(f_l)
        f_l, att_score = self.attention(f_l)
        
        f_g = self.act(self.bn(self.conv(f_g)))
        f_g = self.global_pool(f_g)
        f_g = f_g[:, :, 0, 0]
        
        f_fuse = self.fusion(f_l, f_g)
        f_fuse = self.fusion_pool(f_fuse)
        f_fuse = f_fuse[:, :, 0, 0]
        
        f_embed = self.neck(f_fuse)
        logits = self.head(f_embed)
        preds = logits.softmax(dim=1)
        preds_conf, preds_cls = preds.max(1)
        
        return logits, f_embed
       
    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False


    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True


# if __name__ == '__main__':
#     inps = torch.randn((64, 3, 224, 224)).cuda()
#     model = DOLG('resneXt101', 81313).cuda()
#     logits, emb = model(inps)
#     print('logits shape:', logits.shape)
#     print('embedding shape:', emb.shape)