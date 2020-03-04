# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-27
# ------------------------------------------------------------------------------
import argparse
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

import cv2
import torch

import _init_paths
from configs import cfg
from configs import update_config
from utils.utils import get_model
from utils.utils import load_eval_model
from utils.utils import write_dict_to_json

from datasets.transform import FacenetInferenceTransform
from datasets.transform import PredictionTransform

from appearance.backbones.inception_resnet_v1 import InceptionResnetV1

from detection.models.fcos import FCOS
from detection.loss.fcos_loss import FCOSLoss
from detection.utils.metrics import run_fcos_det_example

from tracklets.utils.utils import get_embeddings
from tracklets.utils.utils import get_tracklet_pair_input_features

from clusters.init_cluster import init_clustering
from clusters.optimal_cluster import get_optimal_cluster
from clusters.utils.cluster_utils import cluster_dict_processing
from TNT.utils.merge_det import merge_det


# not support multi processing for now

parser = argparse.ArgumentParser(description="run clusters generation")
parser.add_argument(
    "--frame_dir",
    type=str,
    default='',
    help='you can choose a directory contain several frames in one video')
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

    # detector
    detector = FCOS(cfg)
    assert cfg.MODEL.DETECTION_WEIGHTS != ''
    load_eval_model(cfg.MODEL.DETECTION_WEIGHTS, detector)
    detector.cuda().eval()

    # appearance emb
    emb_size = cfg.MODEL.APPEARANCE.EMB_SIZE
    emb = InceptionResnetV1(pretrained='vggface2', classify=False)
    assert cfg.MODEL.APPEARANCE.WEIGHTS != ''
    load_eval_model(cfg.MODEL.APPEARANCE.WEIGHTS, emb)
    emb.cuda().eval()

    # read test frame images
    if osp.isdir(args.frame_dir):
        file_names = os.listdir(args.frame_dir)
        jpg_paths = [osp.join(args.frame_dir, file_name) for file_name in file_names]
    else:
        raise FileNotFoundError
    
    criterion = FCOSLoss(cfg, cfg.DATASET.NUM_CLASSES)
    det_transform = PredictionTransform(size=cfg.TEST.TEST_SIZE)
    ap_transform = FacenetInferenceTransform(size=(cfg.TEST.CROP_SZIE))

    # get det result dict using the detector
    det_result = {}
    for i, jpg_path in tqdm(enumerate(jpg_paths)):
        if jpg_path.split('.')[-1].lower() not in ['jpg', 'png', 'jpeg', 'bmp']:
            continue
        frame_id = int(jpg_path.split('.')[-2].split('/')[-1]) # start from 0
        # img:(min_size, max_size, 3), boxes:(obj_num, 4), labels:(obj_num,) numpy type, mean_color <list> list of array(3)
        img, boxes, labels, mean_color = run_fcos_det_example(cfg,
            criterion,
            jpg_path,
            det_transform,
            detector,
            ap_transform=ap_transform,
        )
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0] # w
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1] # h
        img = img.cuda(non_blocking=True)
        img_embs = get_embeddings(emb, img).cpu().data.numpy()
        assert len(img_embs) == len(boxes) == len(labels) == len(mean_color)
        obj_num = len(img_embs)
        # {'frame_id': (obj_num, emb_size+4+1+3) emb x y w h label(float) mean_color(array(3))}
        det_result[frame_id] = np.zeros((obj_num, emb_size+4+1+3))
        det_result[frame_id][:, :emb_size] = img_embs
        det_result[frame_id][:, emb_size: emb_size+4] = boxes
        det_result[frame_id][:, emb_size+4] = labels
        det_result[frame_id][:, emb_size+5:] = mean_color
    
    # use coarse constriant to get coarse track dict
    coarse_track_dict = merge_det(det_result)

    # init cluster
    tnt_model = get_model(cfg, cfg.MODEL.FILE, cfg.MODEL.NAME)
    if cfg.MODEL.RESUME_PATH != '':
        load_eval_model(cfg.MODEL.RESUME_PATH, tnt_model)
    tnt_model.cuda().eval()
    cluster_dict, tracklet_time_range, coarse_tracklet_connects, tracklet_cost_dict = init_clustering(tnt_model, coarse_track_dict)

    # graph algorithm adjusts the cluster
    cluster_list, min_cost = get_optimal_cluster(coarse_tracklet_connects, tracklet_cost_dict)
    
    # post processing
    for cluster in cluster_list:
        cluster_id = min(cluster)
        for track_id in cluster:
            if track_id == cluster_id:
                continue
            cluster_dict.pop(track_id)
            cluster_dict[cluster_id].append(track_id)
    # write_dict_to_json(cluster_dict, 'data/new_cluster_dict.json')
    
    # generate cluster feat dict, cluster_id: np.array((frame_len, emb_size+4+1)), xmin, ymin, xmax, ymax
    cluster_feat_dict, cluster_frame_range = cluster_dict_processing(coarse_track_dict, tracklet_time_range, cluster_dict)
        
    # visualize based on cluster_id, locations and labels. One color for one cluster.
    visualize_dict = {} # frame_id: {track_id: {color: , label: , loc: [xmin, ymin, xmax, ymax]}}
    for frame_id in range(frame_len):
        visualize_dict[frame_id] = {}
    for cluster_id in cluster_feat_dict.keys():
        # color
        loc = cluster_feat[cluster_id][:, -5:-1] # (frame_len, 4)
        label = cluster_feat[cluster_id][:, -1] # (frame_len, 1), vote for the most proper label
        # visualize color label loc




    
    


    
    



    


    






