# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import argparse
import math
import os
import os.path as osp

import cv2
import torch
from tqdm import tqdm

from configs import cfg
from configs import update_config
from datasets.transform import PredictionTransform
from detection.utils.metrics import run_fcos_det_example
from utils.utils import get_det_criterion
from utils.utils import get_model

def write_crop(img, boxes, labels, frame_crop_path, frame_id=None):
    assert len(boxes) == len(labels)
    class_dict = {
        0: 'DontCare',
        1: 'Pedestrian',
        2: 'Car',
        3: 'Cyclist',
    }
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        xmin = math.floor(xmin)
        ymin = math.floor(ymin)
        xmax = math.ceil(xmax)
        ymax = math.ceil(ymax)
        if int(labels[i]) in [1, 2, 3]:
            label = class_dict[int(labels[i])]
        if frame_id is None:
            img_crop_path = osp.join(frame_crop_path, f'{label}_{xmin}_{ymin}_{xmax}_{ymax}_crop.jpg')
        else:
            img_crop_path = osp.join(frame_crop_path, f'{label}_{frame_id}_crop.jpg')
        im = img[ymin:ymax, xmin:xmax, :]
        cv2.imwrite(img_crop_path, im)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run det_crop example")
    parser.add_argument(
        "--videos_root",
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

    update_config(cfg, args)

    #detect
    model = get_model(cfg, cfg.MODEL.FILE, cfg.MODEL.NAME)
    criterion = get_det_criterion(cfg)
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

    transform = PredictionTransform(size=(352, 1216))
    
    if not osp.exists('crop_det'):
        os.mkdir('crop_det')

    video_list = os.listdir(args.videos_root)
    for video_name in video_list:
        video_crop_path = osp.join('crop_det', video_name)
        if not osp.exists(video_crop_path):
            os.mkdir(video_crop_path)
        video_path = osp.join(args.videos_root, video_name)
        jpg_paths = []
        # read test images
        if osp.isfile(video_path):
            jpg_paths = [video_path]
        elif osp.isdir(video_path):
            file_names = os.listdir(video_path)
            jpg_paths = [osp.join(video_path, file_name) for file_name in file_names]
        else:
            raise FileNotFoundError

        for i, jpg_path in tqdm(enumerate(jpg_paths)):
            if jpg_path.split('.')[-1].lower() not in ['jpg', 'png', 'jpeg', 'bmp']:
                continue
            frame_id = int(jpg_path.split('/')[-1].split('.')[-2])
            frame_crop_path = osp.join(video_crop_path, str(frame_id))
            if not osp.exists(frame_crop_path):
                os.mkdir(frame_crop_path)
            img, boxes, labels = run_fcos_det_example(cfg,
                criterion,
                jpg_path,
                transform,
                model,
                is_crop=True
            )
            write_crop(img, boxes, labels, frame_crop_path)
            

    






