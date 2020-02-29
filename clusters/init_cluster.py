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


# add in config: time_dist_tresh, time_margin, time_cluster_dist, track_overlap_thresh, search_radius, clip_len
def init_clustering(coarse_track_dict, remove_set=[], time_dist_tresh=11, time_margin=3, time_cluster_dist=24,
                    track_overlap_thresh=0.1, search_radius=1, clip_len=6):
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
