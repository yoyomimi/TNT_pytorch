# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-20
# ------------------------------------------------------------------------------
import torch

def objtrack_collect(batch):
    """Collect the objtrack data for one batch.
    """
    targets = []
    imgs = []
    frame_id = []
    indexs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        frame_id.append(sample[2])
        indexs.append(sample[3])
    return torch.stack(imgs, 0), targets, frame_id, indexs

def facenet_triplet_collect(batch):
    """Collect the facenet_triplet data for one batch.
    """
    anchor_imgs = []
    pos_imgs = []
    neg_imgs = []

    for sample in batch:
        anchor_imgs.append(sample[0])
        pos_imgs.append(sample[1])
        neg_imgs.append(sample[2])
        
    return torch.stack(anchor_imgs, 0), torch.stack(pos_imgs, 0), torch.stack(neg_imgs, 0)

def tracklet_pair_collect(batch):
    """Collect the tracklet_pair data for one batch.
    """
    imgs_1 = []
    imgs_2 = []
    loc_mat = []
    tracklet_mask_1 = []
    tracklet_mask_2 = []
    real_window_len = []
    connectivity = []

    for sample in batch:
        imgs_1.append(sample[0])
        imgs_2.append(sample[1])
        loc_mat.append(sample[2])
        tracklet_mask_1.append(sample[3])
        tracklet_mask_2.append(sample[4])
        real_window_len.append(sample[5])
        connectivity.append(sample[6])
        
    return imgs_1, imgs_2, torch.stack(loc_mat, 0),  torch.stack(tracklet_mask_1, 0),  
        torch.stack(tracklet_mask_2, 0), real_window_len, torch.stack(connectivity, 0)