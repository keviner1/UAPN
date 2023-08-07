#!/usr/local/bin/python
import sys
import os
import yaml
from yacs.config import CfgNode as CN
import sys
sys.path.append("..")
from Registry import CONFIG_REGISTRY


_C = CN()
# -----------------------------------------------------------------------------
#                              Server settings
# -----------------------------------------------------------------------------
_C.SERVER = CN()
_C.SERVER.gpus = 1
_C.SERVER.TRAIN_DATA = ''
_C.SERVER.VAL_DATA = ''
_C.SERVER.TEST_DATA = ''
_C.SERVER.OUTPUT = 'output\\job1'
# -----------------------------------------------------------------------------
#                            Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.choice = "UAPN"
_C.MODEL.channels = 12    # UAPN-B = 12
# -----------------------------------------------------------------------------
#                           Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.EPOCHS = 1000
_C.TRAIN.PRINT_FREQ = 500
_C.TRAIN.RESUME = ""
# ----------------------------------LR scheduler-------------------------------
_C.TRAIN.LR_MODE = "step"
_C.TRAIN.BASE_LR = 0.005
_C.TRAIN.LR_DECAY = 0.5
_C.TRAIN.LR_STEP = 300
# -----------------------------------Optimizer-----------------------------
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


@CONFIG_REGISTRY.register()
def config_1(_):
    config = _C.clone()
    return config


