import argparse
import os
import sys

from yacs.config import CfgNode as CfgNode

_C = CfgNode()

cfg = _C

# ------------------ Train Options ------------------
_C.TRAIN = CfgNode()

_C.TRAIN.DATASET_NUM_CLASS = 81313

_C.TRAIN.DATASET_ROOT = '/home/user/dataset/gld2021'

# 80% --> train_split.txt   
# 100% --> train_relabel_list.txt
_C.TRAIN.DATASET_FILE = 'train_split.txt'

_C.TRAIN.IM_SIZE = 300

_C.TRAIN.BATCH_SIZE = 10

_C.TRAIN.MODE = 'train'

_C.TRAIN.NUM_WORKERS = 4

_C.TRAIN.PIN_MEMORY = True

# ------------------ Test Options ------------------
_C.TEST = CfgNode()

_C.TEST.DATASET_NUM_CLASS = 81313

_C.TEST.DATASET_ROOT = '/home/user/dataset/gld2021'

_C.TEST.DATASET_FILE = 'test_split.txt'

_C.TEST.IM_SIZE = 256

_C.TEST.BATCH_SIZE = 32

_C.TEST.MODE = 'test'

_C.TEST.NUM_WORKERS = 4

_C.TEST.PIN_MEMORY = True

# ----------------- Model Options -------------------
_C.MODEL = CfgNode()
# ["DELG", "DOLG"]
_C.MODEL.NAME = 'DOLG'
# ["resnet50", "resnet101", "resneXt50", "resneXt101"]
_C.MODEL.BACKBONE = 'resneXt50'

_C.MODEL.REDUCTION_DIM = 512

_C.MODEL.REDUCED_DIM = 128

_C.MODEL.EXPAND_DIM = 1024

_C.MODEL.LOAD = None

_C.MODEL.SAVEPATH = '/home/user/code/RetrievalNet/artifacts'

# ------------------ BN Options -------------------
_C.BN = CfgNode()

_C.BN.EPS = 1e-5

_C.BN.MOMENTUM = 0.1

_C.BN.ZERO_INIT_FINAL_GAMMA = False

# ------------------ GeM Options ------------------
_C.GEM = CfgNode()

_C.GEM.P = 3.0

_C.GEM.TRAIN = False

# ------------------ Loss Options ----------------------
_C.LOSS = CfgNode()
# ['adaptive_arcface'. 'arcface']
_C.LOSS.NAME = 'arcface'

_C.LOSS.ARCFACE_MARGIN = 0.1

_C.LOSS.ARCFACE_SCALE = 30.0

_C.LOSS.ATTN_WEIGHT = 1

_C.LOSS.RESTRUCT_WEIGHT = 10

# --------------------- Optim Options ---------------------
_C.OPTIM = CfgNode()
# "Adam" "SGD" "AdamW"
_C.OPTIM.NAME = 'SGD'

_C.OPTIM.NUM_GPUS = 1

_C.OPTIM.BASE_LR = 0.01

_C.OPTIM.LR_POLICY = 'cos'

_C.OPTIM.MAX_EPOCHS = 100

_C.OPTIM.MOMENTUM = 0.9

_C.OPTIM.WEIGHT_DECAY = 0.0001

_C.OPTIM.WARMUP_FACTOR = 0.1

_C.OPTIM.WARMUP_EPOCHS = 5

_C.OPTIM.DAMPENING = 0.0

_C.OPTIM.NESTEROV = True

# ------------------- Logger Options ------------------
_C.LOG = CfgNode()

_C.LOG.PATH = ''

_C.LOG.PRINT_FREQ = 8

_C.LOG.TEST_PRINT_FREQ = 10

_C.LOG.SAVE_INTERVAL = 1

_C.LOG.EVAL_PERIOD = 100

# ------------------ Train Options --------------------
