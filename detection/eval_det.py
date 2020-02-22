# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import argparse
import os
import os.path as osp

import torch

import _init_paths
from configs import cfg
from configs import update_config

from datasets.data_collect import objtrack_collect
from detection.utils.metrics import eval_fcos_det
from utils.utils import get_det_criterion
from utils.utils import get_dataset
from utils.utils import get_model
from utils.utils import load_eval_model

parser = argparse.ArgumentParser(description="FCOS Evaluation")
parser.add_argument(
    '--cfg',
    dest='yaml_file',
    default=None,
    help='experiment configure file name, e.g. configs/fcos_detector.yaml',
    type=str)
parser.add_argument(
    'opts',
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER)
args = parser.parse_args()

if __name__ == '__main__':
    update_config(cfg, args)
    model = get_model(cfg, cfg.MODEL.FILE, cfg.MODEL.NAME)
    resume_path = cfg.MODEL.RESUME_PATH
    _, eval_dataset = get_dataset(cfg)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.DATASET.IMG_NUM_PER_GPU,
        shuffle=False,
        drop_last=False,
        collate_fn=objtrack_collect,
    )
    criterion = get_det_criterion(cfg)
    
    model = load_eval_model(resume_path, model)

    model.cuda()
    model.eval()

    mAP, aps, pr_curves = eval_fcos_det(cfg,
        criterion,
        eval_loader,
        model,
        rescale=None)
    print(f"score_threshold:{cfg.TEST.SCORE_THRESHOLD}, nms_iou:{cfg.TEST.NMS_THRESHOLD}")
    print(f'size:{cfg.TEST.TEST_SIZE[0], cfg.TEST.TEST_SIZE[1]} mAP:{mAP}')

    # save output val_images
    if osp.exists(cfg.TEST.OUT_DIR):
        import shutil
        shutil.rmtree(cfg.TEST.OUT_DIR)
    os.makedirs(cfg.TEST.OUT_DIR)

    # save pr_curve
    if cfg.TEST.PR_CURVE:
        for class_idx, pr_curve in enumerate(pr_curves):
            pr_curve.savefig(osp.join(cfg.TEST.OUT_DIR, f'PR_class_{class_idx + 1}.jpg'))
        print(f'PR curve saved in {cfg.TEST.OUT_DIR}')