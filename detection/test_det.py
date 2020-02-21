# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import argparse
import os
import os.path as osp
import time

import cv2
import torch
from tqdm import tqdm

import _init_paths
from configs import cfg
from configs import update_config
from datasets.transform import PredictionTransform
from detection.utils.metrics import run_fcos_det_example
from utils.utils import get_criterion
from utils.utils import get_model

parser = argparse.ArgumentParser(description="run FCOS example")
parser.add_argument(
    "--jpg_path",
    type=str,
    default='',
    help='you can choose an image, or a directory contain several images')
parser.add_argument(
    '--cfg',
    dest='yaml_file',
    default='',
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

    #detect
    model = get_model(cfg, cfg.MODEL.FILE, cfg.MODEL.NAME)
    criterion = get_criterion(cfg)
    model.cuda()
    model.eval()
    
    if cfg.MODEL.RESUME_PATH != '':
        if osp.exists(cfg.MODEL.RESUME_PATH):
            print(f'==> model load from {cfg.MODEL.RESUME_PATH}')
            checkpoint = torch.load(cfg.MODEL.RESUME_PATH)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"==> checkpoint do not exists: \"{cfg.MODEL.RESUME_PATH}\"")
            raise FileNotFoundError
        
    if not osp.exists(cfg.TEST.OUT_DIR):
        os.mkdir(cfg.TEST.OUT_DIR)
    # read test images
    if osp.isfile(args.jpg_path):
        jpg_paths = [args.jpg_path]
    elif osp.isdir(args.jpg_path):
        file_names = os.listdir(args.jpg_path)
        jpg_paths = [osp.join(args.jpg_path, file_name) for file_name in file_names]
    else:
        raise FileNotFoundError
    transform = PredictionTransform(size=(768, 1280))
    timer = []
    for i, jpg_path in tqdm(enumerate(jpg_paths)):
        if jpg_path.split('.')[-1].lower() not in ['jpg', 'png', 'jpeg', 'bmp']:
            continue
        start = time.time()
        img = run_fcos_det_example(cfg,
            criterion,
            jpg_path,
            transform,
            model
        )
        timer.append(time.time() - start)
        num = jpg_path.split('.')[-2].split('/')[-1]
        cv2.imwrite(osp.join(cfg.TEST.OUT_DIR, f'run_fcos_example_{num}.jpg'), img)
    print(f'finnal output dir is {cfg.TEST.OUT_DIR}')
    print(f'average inference time: {sum(timer) / len(timer)}s')

    






