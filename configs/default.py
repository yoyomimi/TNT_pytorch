# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
from yacs.config import CfgNode as CN

INF = 1e8

_C = CN()

# working dir
_C.OUTPUT_ROOT = ''

# distribution
_C.DIST_BACKEND = 'nccl'

_C.WORKERS = 4

_C.PI = 'mAP'

# cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# dataset
_C.DATASET = CN()
_C.DATASET.FILE = 'kongke'
_C.DATASET.NAME = 'KongkeDataset'
_C.DATASET.ROOT = ''
_C.DATASET.MEAN = []
_C.DATASET.STD = []
_C.DATASET.IMG_NUM_PER_GPU = 2
_C.DATASET.NUM_CLASSES = 81


# model
_C.MODEL = CN()
# specific model 
_C.MODEL.FILE = ''
_C.MODEL.NAME = ''
# resume
_C.MODEL.RESUME_PATH = ''

# backbone
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.PRETRAINED = False
_C.MODEL.BACKBONE.WEIGHTS = ''
_C.MODEL.BACKBONE.CONV_BODY = 'R-50-FPN-RETINANET'
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
_C.MODEL.BACKBONE.USE_GN = False

# resnets
_C.MODEL.RESNETS = CN()
# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1
# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64
# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True
# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"
# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1
_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
# Deformable convolutions
_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1


# neck
_C.MODEL.NECK = CN()
_C.MODEL.NECK.IN_CHANNELS = [256, 512, 1024, 2048]
_C.MODEL.NECK.OUT_CHANNELS = 256
_C.MODEL.NECK.START_LEVEL = 1
_C.MODEL.NECK.ADD_EXTRA_CONVS = True
_C.MODEL.NECK.EXTRA_CONVS_ON_INPUTS = False
_C.MODEL.NECK.NUM_OUTS = 5
_C.MODEL.NECK.RELU_BEFORE_EXTRA_CONVS = True

# head
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.IN_CHANNELS = 256
_C.MODEL.HEAD.OUT_CHANNELS = 256
_C.MODEL.HEAD.STACKED_CONVS = 4
_C.MODEL.HEAD.STRIDES = (8, 16, 32, 64, 128)

# loss
_C.LOSS = CN()
_C.LOSS.INF = 1e8
_C.LOSS.REGRESS_RANGES = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))
_C.LOSS.FOCAL_ALPHA = 0.25
_C.LOSS.FOCAL_GAMMA = 2
_C.LOSS.LAMDA = 1
_C.LOSS.IOU_EPS = 1e-6
_C.LOSS.FILE = ''
_C.LOSS.NAME = ''


# trainer
_C.TRAINER = CN()
_C.TRAINER.FILE = ''
_C.TRAINER.NAME = ''

# train
_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = ''
_C.TRAIN.LR = 0.005
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0001
# optimizer SGD
_C.TRAIN.NESTEROV = False
# learning rate scheduler
_C.TRAIN.LR_SCHEDULER = 'MultiStepWithWarmup'
_C.TRAIN.LR_STEPS = [120000, ]
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.WARMUP_STEP = 500
_C.TRAIN.WARMUP_INIT_FACTOR = 1.0 / 3
_C.TRAIN.MAX_ITERATIONS = 360000
# train resume
_C.TRAIN.RESUME = False
# input size
_C.TRAIN.INPUT_MIN = 788
_C.TRAIN.INPUT_MAX = 1400
# print freq
_C.TRAIN.PRINT_FREQ = 20
# save checkpoint during train
_C.TRAIN.SAVE_INTERVAL = 5000
_C.TRAIN.SAVE_EVERY_CHECKPOINT = False
# val when train
_C.TRAIN.VAL_WHEN_TRAIN = False

# test
_C.TEST = CN()
# input size
_C.TEST.SCORE_THRESHOLD = 0.3
_C.TEST.NMS_THRESHOLD = 0.5
# test image dir
_C.TEST.IMAGE_DIR = ''
_C.TEST.TEST_SIZE = (512, 768)
_C.TEST.MAX_PER_IMG = 100
_C.TEST.NMS_PRE = 1000
_C.TEST.PR_CURVE = False
_C.TEST.OUT_DIR = ''
_C.TEST.AP_IOU = 0.5


def update_config(config, args):
    config.defrost()
    # set cfg using yaml config file
    config.merge_from_file(args.yaml_file)
    # update cfg using args
    config.merge_from_list(args.opts)
    config.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)