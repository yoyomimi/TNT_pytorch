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
