import os
import torch
import numpy as np
import argparse
import json

import core.optim as optim
import core.distributed as dist
import core.net as net

from core.config import cfg
from core.utils import reduce_dict, scaled_all_reduce
from core.logger import Logger, SmoothedValue, MetricLogger

from models.delg_local import DELG_Local
from datasets.loader import _construct_train_loader, _construct_test_loader
from core.meters import topk_errors, set_local_criterion

# DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def setup_env(args):
    dist.init_dist_mode(args)
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed Used: ', seed)


def setup_model():
    model = DELG_Local(cfg.MODEL.BACKBONE, cfg.TRAIN.DATASET_NUM_CLASS)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable params:', n_params)
    
    cur_dev = torch.cuda.current_device()
    model = model.cuda(device=cur_dev)
    
    return model, n_params


def train_epoch(train_loader, model, optimizer, criterion, cur_epoch, print_freq=1, tb_logger=None):
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_epoch_lr(optimizer, lr)
    
    model.train()
    criterion.train()
    
    logger = MetricLogger(delimiter='  ')
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{cur_epoch}]'
    
    for ims, labels in logger.log_every(train_loader, print_freq, header):
        ims = ims.cuda()
        labels = labels.cuda(non_blocking=True)
        
        local_logits, local_expand_feats,\
        attn_score, local_featmap = model(ims)  
          
        loss_dict = criterion(local_logits, local_expand_feats, local_featmap, labels)
        restruct_loss = loss_dict['restruction_loss']
        attn_loss = loss_dict['attn_loss']
        
        local_loss = restruct_loss + attn_loss
        
        optimizer.zero_grad()
        local_loss.backward()
        optimizer.step()
        
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_item = {
            k: v.item() for k, v in loss_dict_reduced.items()
        }
        
        logger.update(**loss_dict_reduced_item)
        logger.update(lr=lr)
        if tb_logger is not None and dist.is_main_process():
            tb_logger.add_scalars(loss_dict_reduced, prefix='train')
            
    logger.synchronize_between_processes()
    print('Average train stats:', logger)
    return {k: meter.global_avg for k, meter in logger.meters.items()}

                
# @torch.no_grad()
# def test_epoch(test_loader, model, cur_epoch, print_freq=1, tb_logger=None):
#     model.eval()
    
#     logger = MetricLogger(delimiter='  ')
#     header = f'Epoch: [{cur_epoch}]'
    
#     for ims, labels in logger.log_every(test_loader, print_freq, header):
#         ims = ims.cuda()
#         labels = labels.cuda(non_blocking=True)
        
#         local_logits, local_expand_feats,\
#         attn_score, local_featmap = model(ims)
        
#         top1_err, top5_err  = topk_errors(global_logits, labels, [1, 5])
#         top1_err, top5_err = scaled_all_reduce([top1_err, top5_err])
#         top1_err, top5_err = top1_err.item(), top5_err.item()
        
#         topk_errs = {'top1_err': top1_err, 'top5_err': top5_err}
#         logger.update(**topk_errs)
        
#         if tb_logger is not None and dist.is_main_process():
#             tb_logger.add_scalars(topk_errs, prefix='test')
    
#     logger.synchronize_between_processes()
#     print('Average test stats:', logger)
#     return {k: meter.global_avg for k, meter in logger.meters.items()}

        

def train_local_model(args):
    setup_env(args)
    
    train_loader = _construct_train_loader()
    test_loader = _construct_test_loader()
    
    model, num_params = setup_model()
    model_without_ddp = model
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[cur_dev],
                find_unused_parameters=True)
        model_without_ddp = model.module
    
    if cfg.MODEL.LOAD is not None:
        state_dict = torch.load(cfg.MODEL.LOAD)
        model.load_state_dict(state_dict['model'])
        print('Load Model Weights Done')
        
    criterion = set_local_criterion()
    optimizer = optim.construct_optimizer(model)
    
    artifact_name = (
        f'local_backbone_{cfg.MODEL.BACKBONE}_lr_{cfg.OPTIM.BASE_LR}'
    )
    artifact_path = os.path.join(cfg.MODEL.SAVEPATH, artifact_name)
    os.makedirs(artifact_path, exist_ok=True)
    
    tb_logger = Logger(artifact_path) if dist.is_main_process() else None
    
    for cur_epoch in range(cfg.OPTIM.MAX_EPOCHS):
        train_stats = train_epoch(
            train_loader=train_loader,
            model=model, 
            optimizer=optimizer, 
            criterion=criterion, 
            cur_epoch=cur_epoch, 
            print_freq=cfg.LOG.PRINT_FREQ,
            tb_logger=tb_logger
        )
       
        # Save the model
        if cur_epoch % cfg.LOG.SAVE_INTERVAL == 0 or cur_epoch == cfg.OPTIM.MAX_EPOCHS - 1:
            if dist.is_main_process():
                torch.save({'model': model_without_ddp.state_dict()},
                           f'{artifact_path}/model-epoch{cur_epoch}.pth')
        log_state = {
            'epoch': cur_epoch,
            'num_params': num_params,
            **{f'train_{k}': v for k, v in train_stats.items()}
        }
        with open(f'{artifact_path}/train.log', 'a') as f:
            f.write(json.dumps(log_state) + '\n')
            
        # Evaluate the model
        # if cur_epoch % cfg.LOG.EVAL_PERIOD == 0 or cur_epoch == cfg.OPTIM.MAX_EPOCHS - 1:
        #     test_stats = test_epoch(test_loader, model, cur_epoch, print_freq=cfg.LOG.TEST_PRINT_FREQ, tb_logger=tb_logger)
    
    print('Training Finished')
    
        
def test_local_model(args, model_weights):
    setup_env(args)
    model = setup_model()
    
    if model_weights is not None:
        state_dict = torch.load(model_weights)
        model.load_state_dict(state_dict['model'])        
        print('Load Model Weights Done')
        
    cur_epoch=10
    test_loader = _construct_test_loader()
    test_epoch(test_loader, model, cur_epoch, print_freq=cfg.LOG.TEST_PRINT_FREQ, tb_logger=None)
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()
    
    train_model(args)