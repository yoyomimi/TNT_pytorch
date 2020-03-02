# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-29
# ------------------------------------------------------------------------------
# delete if not debug
import _init_paths
from utils.utils import load_eval_model
from tracklets.fushion_models.tracklet_connectivity import TrackletConnectivity

import numpy as np
import pickle

from clusters.utils.trackletpair_connect import pred_connect_with_fusion
from clusters.utils.tracklet_connect import define_coarse_tracklet_connections
from clusters.utils.tracklet_connect import get_trackletpair_t_range



def update_neighbor(coarse_track_dict, track_set, tracklet_pair, coarse_tracklet_connects, emb_size, slide_window_len=64): 
    """using pretrained results track_set to update the connect and trackletpair.
    Args:
        track_set: <list>, [track_id_1, track_id_2, connectivity]
        tracklet_pair: <list>, [track_id_1, track_id_2, t_min_1, t_max_1, t_min_2, t_max_2], removed the overlap part,
            crop the part out of the sliding window.
    """  
    for n in range(len(track_set)):
        track_id_1 = int(track_set[n, 0])
        track_id_2 = int(track_set[n, 1])
        if track_set[n, 2] == 1: # connected tracklets, should be neighbors
            min_t_id_1 = np.min(np.where(coarse_track_dict[track_id_1][:, emb_size]!=-1)[0])
            max_t_id_1 = np.max(np.where(coarse_track_dict[track_id_1][:, emb_size]!=-1)[0])
            min_t_id_2 = np.min(np.where(coarse_track_dict[track_id_2][:, emb_size]!=-1)[0])
            max_t_id_2 = np.max(np.where(coarse_track_dict[track_id_2][:, emb_size]!=-1)[0])
            # TODO what about connective but much overlap? Will net be more reliable?
            if abs(min_t_id_2 - max_t_id_1) > slide_window_len: # not in a sliding window 
                continue
            if track_id_1 not in coarse_tracklet_connects[track_id_2]['neighbor']:
                # generate tracklet_pair
                if min_t_id_1 <= min_t_id_2:
                    # track_id_1, track_id_2, 
                    if max_t_id_1 > min_t_id_2: # crop the overlap
                        new_t_max_1 = min_t_id_2
                        new_t_min_2 = max_t_id_1
                    else:
                        new_t_max_1 = max_t_id_1
                        new_t_min_2 = min_t_id_2
                    # generate tracklet_pair
                    pair = np.zeros((6)) - 1
                    pair[0] = track_id_1
                    pair[1] = track_id_2
                    t_range = np.array(get_trackletpair_t_range(min_t_id_1, new_t_max_1, new_t_min_2, max_t_id_2, slide_window_len))
                    pair[2:] = t_range
                    tracklet_pair = np.vstack([tracklet_pair, pair]).astype(np.int64)
                elif min_t_id_1 > min_t_id_2:
                    # track_id_2, track_id_1, crop the overlap
                    if max_t_id_2 > min_t_id_1:
                        new_t_max_2 = min_t_id_1
                        new_t_min_1 = max_t_id_2
                    else:
                        new_t_max_2 = max_t_id_2
                        new_t_min_1 = min_t_id_1
                    # generate tracklet_pair
                    pair = np.zeros((6)) - 1
                    pair[0] = track_id_2
                    pair[1] = track_id_1
                    t_range = np.array(get_trackletpair_t_range(min_t_id_2, new_t_max_2, new_t_min_1, max_t_id_1, slide_window_len))
                    pair[2:] = t_range
                    tracklet_pair = np.vstack([tracklet_pair, pair]).astype(np.int64)
                coarse_tracklet_connects[track_id_2]['neighbor'].append(track_id_1)
                coarse_tracklet_connects[track_id_1]['neighbor'].append(track_id_2)
            if track_id_1 in coarse_tracklet_connects[track_id_2]['conflict']:
                coarse_tracklet_connects[track_id_2]['conflict'].remove(track_id_1)
                coarse_tracklet_connects[track_id_1]['conflict'].remove(track_id_2)
        else: # should be conflict (neighbor in time, but not neighbor in tracklets)
            if track_id_1 in coarse_tracklet_connects[track_id_2]['neighbor']:
                index = np.where(np.logical_and(tracklet_pair[:, 0]==track_id_1, tracklet_pair[:, 1]==track_id_2))[0]
                index = np.hstack([index, np.where(np.logical_and(tracklet_pair[:, 0]==track_id_2, tracklet_pair[:, 1]==track_id_1))[0]])
                # delete conflict pair
                if len(index) > 0:
                    tracklet_pair = np.delete(tracklet_pair, index, 0)
                coarse_tracklet_connects[track_id_2]['neighbor'].remove(track_id_1)
                coarse_tracklet_connects[track_id_1]['neighbor'].remove(track_id_2)
            if track_id_1 not in coarse_tracklet_connects[track_id_2]['conflict']:
                coarse_tracklet_connects[track_id_2]['conflict'].append(track_id_1)
                coarse_tracklet_connects[track_id_1]['conflict'].append(track_id_2)
    return coarse_tracklet_connects, tracklet_pair

# add in config: time_dist_tresh, time_margin, time_cluster_dist, track_overlap_thresh, \
# search_radius, clip_len, slide_window_len, cost_bias, refine_track_set
def init_clustering(model, coarse_track_dict, remove_set=[], time_dist_tresh=11, time_margin=3, time_cluster_dist=24,
                    track_overlap_thresh=0.45, search_radius=3, clip_len=6, slide_window_len=64, cost_bias=0, refine_track_set=None):
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
            cost_bias: cost when one tracklet as a cluster.
            refine_track_set: the pred track using pretrained method.

       Return: 
            cluster_dict <dict> {
                frame_id <int>: <list> list of track_ids at this frame
            },

            track_cluster_t_dict = <dict> {
                track_id <int>: <list> list of time_cluster_ids for this track
            },

            track_cluster_dict <dict> {
                cluster_id <int>: <list> list of track_ids in this cluster
            },

            coarse_tracklet_connects <dict>{
                tracklet_id <int>:{
                    'neighbor': <list> list of track_ids,
                    'conflict': <list> list of track_ids

                }
            },

            track_cluster_index<dict> {
                track_id <int>:  <int> cluster_id,
            },

            cluster_cost_dict<dict> {
                cluster_id <int>: <float> cost
            },

            tracklet_cost_dict: <dict> {
                track_id_1:{
                    track_id_2: [<int> connectivity, <float> cost]
                }
            }
    """
    frame_num, feat_size = coarse_track_dict[0].shape
    emb_size = feat_size - 4 - 1

    track_num = len(coarse_track_dict)
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
    
    # init track cluster
    cluster_dict = {} # key-cluster_id, value-track_id_list
    track_cluster_index = {} # key-track_id, value-cluster_id
    cluster_cost_dict = {}
    for i in range(len(coarse_track_dict.keys())):
        track_id = list(coarse_track_dict.keys())[i]
        cluster_dict[i] = [int(track_id)]
        cluster_cost_dict[i] = cost_bias
        track_cluster_index[int(track_id)] = [i]
    
    # define connectivity among coarse tracklets
    coarse_tracklet_connects, tracklet_pair = define_coarse_tracklet_connections(coarse_track_dict, emb_size,
        track_overlap_thresh, search_radius, clip_len, slide_window_len)
    
    if refine_track_set:
        # using pretrained results track_set to update the connect and trackletpair
        track_set = pickle.load(open(refine_track_set,'rb'))
        coarse_tracklet_connects, tracklet_pair = update_neighbor(coarse_track_dict, track_set, tracklet_pair, coarse_tracklet_connects, 
            emb_size, slide_window_len)

    # get tracklet connect cost(<0 if connected) dicr
    tracklet_cost_dict = pred_connect_with_fusion(model, coarse_track_dict, tracklet_pair, emb_size, slide_window_len)

    return cluster_dict, cluster_cost_dict, coarse_tracklet_connects, time_cluster_dict, track_cluster_t_dict, tracklet_cost_dict
    

if __name__ == "__main__":
    import argparse
    import json
    import numpy as np

    from configs import cfg
    from configs import update_config

    parser = argparse.ArgumentParser(description="run clusters generation")
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        default='',
        help='experiment configure file name, e.g. configs/fcos_detector.yaml',
        type=str
    )
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    update_config(cfg, args)

    tnt_model = TrackletConnectivity(cfg)
    load_eval_model('work_dirs/trackletpair_connectivity/2020-03-01-12-13/TrackletConnectivity_epoch000_iter000010_checkpoint.pth', tnt_model)
    tnt_model.cuda().eval()

    # from utils.utils import write_dict_to_json
    

    # read det_results
    coarse_track_dict = {}
    with open('data/coarse_tracklet.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        coarse_track_dict[int(track_id)] = np.array(temp_dict[track_id], dtype=np.float32)
    
    cluster_dict, cluster_cost_dict, coarse_tracklet_connects, time_cluster_dict, track_cluster_t_dict, tracklet_cost_dict = init_clustering(tnt_model, coarse_track_dict)
    
