# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-29
# ------------------------------------------------------------------------------
# delete if not debug
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch



def pred_connect_with_fusion(model, coarse_track_dict, tracklet_pair, emb_size, window_len=64, max_bs=64):
    """pred tracklet pair connectivity.
    Args:
        coarse_track_dict:{
            'track_id': (frame_num, emb_size+4+1), for one frame_num in one track_id [img_emb(512) x y w h label], np.array
        },
        tracklet_pair: <np.array((pair_num, 6)), [track_id_1, t_start_1, t_end_1, track_id_2,  t_start_2, t_end_2].

    Return:
        track_set: np.array((pair_num, 4)). [track_id_1, track_id_2, connectivity, cost]
        tracklet_cost_dict: <dict> {
            track_id_1:{
                track_id_2: <float> cost
            }
        }
    """
    track_set = []
    for i in tqdm(range(int(len(tracklet_pair) / max_bs) + 1)):
        if i == int(len(tracklet_pair) / max_bs):
            bs = len(tracklet_pair) % max_bs
        else:
            bs = max_bs
        tracklet_pair_features_input = np.zeros((bs, window_len, emb_size+4, 3)) # [loc emb]
        tracklet_pair_features = tracklet_pair_features_input[:, :, :, 0]
        det_mask_1 = tracklet_pair_features_input[:, :, :, 1]
        det_mask_2 = tracklet_pair_features_input[:, :, :, 2]
        track_set_bs = np.zeros((bs, 4))  
        for j in range(bs):
            idx = 8 * i + j
            track_id_1 = tracklet_pair[idx, 0]
            track_id_2 = tracklet_pair[idx, 1]
            t_min_1 = tracklet_pair[idx, 2]
            t_max_1 = tracklet_pair[idx, 3]
            t_min_2 = tracklet_pair[idx, 4]
            t_max_2 = tracklet_pair[idx, 5]
            # track_id
            track_set_bs[j, 0] = track_id_1
            track_set_bs[j, 1] = track_id_2
            # det_mask
            det_mask_1[j, :t_max_1-t_min_1+1, :] = np.ones((t_max_1-t_min_1+1, 1))
            det_mask_2[j, t_min_2-t_min_1: t_max_2-t_min_1+1, :] = np.ones((t_max_2-t_min_2+1, 1))
            # appearance_feature with locations
            tracklet_pair_features[j, :t_max_1-t_min_1+1, :4] = coarse_track_dict[track_id_1][t_min_1: t_max_1+1, emb_size: emb_size+4]
            tracklet_pair_features[j, :t_max_1-t_min_1+1, 4:] = coarse_track_dict[track_id_1][t_min_1: t_max_1+1, :emb_size]
            tracklet_pair_features[j, t_min_2-t_min_1: t_max_2-t_min_1+1, :4] = coarse_track_dict[track_id_2][t_min_2: t_max_2+1, emb_size: emb_size+4]
            tracklet_pair_features[j, t_min_2-t_min_1: t_max_2-t_min_1+1, 4:] = coarse_track_dict[track_id_2][t_min_2: t_max_2+1, :emb_size]
            # interpolate appearance_feature with locations
            feat_np = tracklet_pair_features[j, :t_max_2-t_min_1+1].copy()
            feat_np[0][np.where(feat_np[0]==0)] = 1e-5
            feat_np[-1][np.where(feat_np[-1]==0)] = 1e-5
            feat_pd = pd.DataFrame(data=feat_np).replace(0, np.nan, inplace=False)
            tracklet_pair_features[j, :t_max_2-t_min_1+1] = np.array(feat_pd.interpolate()).astype(np.float32)
        # (bs, window_len, emb_size+4, 3)
        tracklet_pair_features_input = torch.from_numpy(tracklet_pair_features_input).type(torch.FloatTensor).cuda(non_blocking=True)
        # use tnt net to pred connectivity
        cls_scores = model(tracklet_pair_features_input).data.cpu() # (bs, 2)
        track_set_bs[:, 2] = 1 - cls_scores.max(1, keepdim=True)[1].squeeze(-1).numpy() # (bs, 1), 0 is connected
        # TODO is that right below?
        track_set_bs[:, 3] = (cls_scores[:, 1] - cls_scores[:, 0]).squeeze(-1).numpy() # (bs, 1), cost is min when pos conf >> neg conf
        track_set.append(track_set_bs)
        # delete if not debug

    track_set = np.vstack(track_set) # (pair_num, 3)
    
    tracklet_cost_dict = {}
    for i in range(len(track_set)):
        track_id_1 = int(track_set[i][0])
        track_id_2 = int(track_set[i][1])
        cost = track_set[i][3]
        if track_id_1 not in tracklet_cost_dict.keys():
            tracklet_cost_dict[track_id_1] = {}
            if track_id_2 not in tracklet_cost_dict[track_id_1].keys():
                tracklet_cost_dict[track_id_1][track_id_2] = 0.
        tracklet_cost_dict[track_id_1][track_id_2] = cost

    return track_set, tracklet_cost_dict
