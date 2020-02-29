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

from clusters.utils.tracklet_connect import define_coarse_tracklet_connections
from tracklets.utils.trackletpair_connect import pred_connect_with_fusion

def update_neighbor_use_net(coarse_track_dict, coarse_tracklet_connects, emb_size, slide_window_len=60):
    track_set = []
    # get track set using net: (coarse_tracklet_id1, coarse_tracklet_id2, pred_connectivity), id1 is in the front
    track_set = pred_connect_with_fusion(coarse_track_dict, slide_window_len) #implement in tracklet.utils, sample, pred and save
    
    for n in range(len(track_set)):
        track_id_1 = int(track_set[n, 0])
        track_id_2 = int(track_set[n, 1])
        if track_set[n, 2] == 1: # connected tracklets, should be neighbors
            max_t_id_1 = np.max(np.where(coarse_track_dict[track_id_1][:, emb_size]!=-1)[0])
            min_t_id_2 = np.min(np.where(coarse_track_dict[track_id_2][:, emb_size]!=-1)[0])
            # TODO what about connective but much overlap? Will net be more reliable?
            if abs(min_t_id_2 - max_t_id_1) > slide_window_len: # not in a sliding window 
                continue
            if track_id_1 not in coarse_tracklet_connects[track_id_2]['neighbor']:
                coarse_tracklet_connects[track_id_2]['neighbor'].append(track_id_1)
                coarse_tracklet_connects[track_id_1]['neighbor'].append(track_id_2)
            if track_id_1 not in coarse_tracklet_connects[track_id_2]['conflict']:
                coarse_tracklet_connects[track_id_2]['conflict'].remove(track_id_1)
                coarse_tracklet_connects[track_id_1]['conflict'].remove(track_id_2)
        else: # should be conflict (neighbor in time, but not neighbor in tracklets)
            if track_id_1 not in coarse_tracklet_connects[track_id_2]['neighbor']:
                coarse_tracklet_connects[track_id_2]['neighbor'].remove(track_id_1)
                coarse_tracklet_connects[track_id_1]['neighbor'].remove(track_id_2)
            if track_id_1 not in coarse_tracklet_connects[track_id_2]['conflict']:
                coarse_tracklet_connects[track_id_2]['conflict'].append(track_id_1)
                coarse_tracklet_connects[track_id_1]['conflict'].append(track_id_2)


# add in config: time_dist_tresh, time_margin, time_cluster_dist, track_overlap_thresh, search_radius, clip_len, slide_window_len
def init_clustering(coarse_track_dict, remove_set=[], time_dist_tresh=11, time_margin=3, time_cluster_dist=24,
                    track_overlap_thresh=0.1, search_radius=1, clip_len=6, slide_window_len=60):
    """init time clusters and track clusters based on coarse_track_dict.
       
       Args: 
           coarse_track_dict <dict> {
               track_id <int>: <np.array>, (total_frame_num, emb_size+4+1)
            },
            remove_set: set of track_ids to be ignored because of not qualified.
            time_dist_tresh: .
            time_margin: .
            time_cluster_dist: .
            track_overlap_thresh: .
            search_radius: .
            clip_len: .
            slide_window_len: .

       Return: 
            time_cluster_dict <dict> {
                frame_id <int>: <list> list of track_ids at this frame
            },
            track_cluster_t_dict = <dict> {
                track_id <int>: <list> list of time_cluster_ids for this track
            },
            track_cluster_dict <dict> {
                cluster_id <int>: <list> list of track_ids in this cluster
            },
    """
    frame_num, feat_size = coarse_track_dict[0].shape
    emb_size = feat_size - 4 - 1

    # init time cluster and track_t_cluster
    time_cluster_dict = {}
    track_cluster_t_dict = {}
    time_cluster_num = int(np.ceil(frame_num/time_cluster_dist))
    for i in range(time_cluster_num):
        time_cluster_dict[i] = []
    for track_id in coarse_track_dict.keys():
        track = coarse_track_dict[track_id]
        idx = np.where(track[:, emb_size]!=-1)[0]
        min_t_idx = np.min(idx)
        max_t_idx = np.max(idx)
        if track_id in remove_set:
            track_cluster_t_dict[track_id] = [-1]
        else:
            min_time_cluster_idx = int(np.floor(max(min_t_idx-time_dist_tresh-time_margin, 0) / time_cluster_dist))
            max_time_cluster_idx = int(np.floor(min(max_t_idx+time_dist_tresh+time_margin, frame_num-1) / time_cluster_dist))
            track_cluster_t_dict[track_id] = list(range(min_time_cluster_idx, max_time_cluster_idx+1))
            for i in range(min_time_cluster_idx, max_time_cluster_idx+1):
                time_cluster_dict[i].append(track_id)
    
    # define connectivity among coarse tracklets
    coarse_tracklet_connects = define_coarse_tracklet_connections(coarse_track_dict, emb_size,
        track_overlap_thresh, search_radius, clip_len)

    coarse_tracklet_connects = update_neighbor_use_net(coarse_track_dict, coarse_tracklet_connects, emb_size, slide_window_len)

    # init track cluster
    track_cluster_dict = {}

    
    return time_cluster_dict, track_cluster_t_dict, track_cluster_dict



if __name__ == "__main__":
    import json
    import numpy as np

    # from utils.utils import write_dict_to_json
    

    # read det_results
    coarse_track_dict = {}
    with open('data/coarse_tracklet.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        coarse_track_dict[int(track_id)] = np.array(temp_dict[track_id], dtype=np.float32)

    time_cluster_dict, track_cluster_t_dict, track_cluster_dict = init_clustering(coarse_track_dict)
