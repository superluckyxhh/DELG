import torch
import torch.nn as nn
import  math
from core.config import cfg


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
    
    
class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
    
        return loss.mean()
    
    
# class ArcFaceLossAdaptiveMargin(nn.modules.Module):
#     def __init__(self, margins, n_classes, s=30.0):
#         super().__init__()
#         self.crit = DenseCrossEntropy()
#         self.s = s
#         self.margins = margins
#         self.out_dim =n_classes
            
#     def forward(self, logits, labels):
#         ms = []
#         ms = self.margins[labels.cpu().numpy()]
#         cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
#         sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
#         th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
#         mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
#         labels = F.one_hot(labels, self.out_dim).float()
#         logits = logits.float()
#         cosine = logits
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
#         phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
#         output = (labels * phi) + ((1.0 - labels) * cosine)
#         output *= self.s
#         loss = self.crit(output, labels)
    
#         return {'loss': loss}     


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", weight=None, 
                 reduction="mean", class_weights_norm=None):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        self.crit = nn.CrossEntropyLoss(reduction="none")   
        
        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)
        s = self.s
        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)
            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return {'loss':loss}

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return {'loss':loss}
    

class Arcface(nn.Module):
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s
        
        return pred_class_logits
    
 
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, inputs, targets):
        loss = self.criterion(inputs, targets)
        return {"loss": loss}
    
    
class DELG_Criterion(nn.Module):
    def __init__(self, arcface_s, arcface_m):
        super().__init__()
        self.arcface = ArcFaceLoss(arcface_s, arcface_m)  
        self.mse = nn.MSELoss(reduction='mean')
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, global_logits, attn_logits, reconstruct_feats, local_featmap, labels):
        global_loss = self.arcface(global_logits, labels)
        restruct_loss = self.mse(reconstruct_feats, local_featmap) * cfg.LOSS.RESTRUCT_WEIGHT 
        attn_loss = self.cross_entropy(attn_logits, labels) * cfg.LOSS.ATTN_WEIGHT
        
        return {'global_loss': global_loss['loss'], 
                'restruction_loss': restruct_loss,
                'attn_loss': attn_loss}


class DELG_Local_Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, attn_logits, reconstruct_feats, local_featmap, labels):
        restruct_loss = self.mse(reconstruct_feats, local_featmap) * cfg.LOSS.RESTRUCT_WEIGHT 
        attn_loss = self.cross_entropy(attn_logits, labels) * cfg.LOSS.ATTN_WEIGHT
        
        return {'restruction_loss': restruct_loss,
                'attn_loss': attn_loss}
        

def setup_criterion():
    return DELG_Criterion(arcface_s=cfg.LOSS.ARCFACE_SCALE,
                     arcface_m=cfg.LOSS.ARCFACE_MARGIN)

    
def set_global_criterion():
    return CrossEntropy()


def set_arcface_criterion():
    return ArcFaceLoss()


def set_local_criterion():
    return DELG_Local_Criterion()

    
def topk_errors(preds, labels, ks):
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]