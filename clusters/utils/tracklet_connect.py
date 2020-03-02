# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-3-1
# ------------------------------------------------------------------------------
from tracklets.utils.pred_loc import linear_pred_v2

import numpy as np


def get_coarse_tarcklet_loc(coarse_track_dict, track_id, cand_t, emb_size):
    mask = coarse_track_dict[track_id][cand_t, emb_size] == -1
    x = coarse_track_dict[track_id][cand_t, emb_size]
    y = coarse_track_dict[track_id][cand_t, emb_size+1]
    w = coarse_track_dict[track_id][cand_t, emb_size+2]
    h = coarse_track_dict[track_id][cand_t, emb_size+3]
    center_x = x + w / 2
    center_y = y + h / 2
    center_x[mask] = -1 # -1 for irrelevant frames
    center_y[mask] = -1
    return center_x, center_y, w, h


def get_trackletpair_t_range(t_min_1, t_max_1, t_min_2, t_max_2, window_len):
    """get tracklet pair time range within a window_len.
       default: t_min_1 <= t_max_1 <= t_min_2 <= t_max_2.
    Return:
        t_start_1, t_end_1, t_start_2, t_end_2
    """
    assert t_min_1 <= t_max_1 <= t_min_2 <= t_max_2
    if t_max_1 == t_min_2:
        if t_max_1 - t_min_1 > t_max_2 - t_min_2 and t_max_1 > t_min_1:
            t_max_1 -= 1
        elif t_max_1 - t_min_1 <= t_max_2 - t_min_2:
            assert t_max_2 > t_min_2
            t_min_2 += 1
    if t_max_2 - t_min_1 + 1 <= window_len:
        # window covers both of the tracklets
        return t_min_1, t_max_1, t_min_2, t_max_2
    # window can't cover both of the tracklets
    mid_gap_t = int((t_max_1 + t_min_2) / 2) # the mid t point of the gap between two tracklets
    if mid_gap_t - t_min_1 + 1 >= 0.5 * window_len and t_max_2 - mid_gap_t + 1 <= 0.5 * window_len:
        # crop tracklet_1
        return t_max_2-window_len+1, t_max_1, t_min_2, t_max_2
    elif mid_gap_t - t_min_1 + 1 <= 0.5 * window_len and t_max_2 - mid_gap_t + 1 >= 0.5 * window_len:
        # crop tracklet_2
        return t_min_1, t_max_1, t_min_2, t_min_1+window_len-1
    else:
        # crop both tracklet_1 and tracklet_2
        t_start_1 = mid_gap_t - int(0.5 * window_len) + 1
        return t_start_1, t_max_1, t_min_2, t_start_1+window_len-1
    

def define_coarse_tracklet_connections(coarse_track_dict, emb_size, track_overlap_thresh=0.45, search_radius=1,
                                       time_dist_tresh=11, clip_len=6, window_len=64):
    """distinguish the neighbor and the conflict tracklets for all coarse tracklets;
       generate tracklet pair using sliding window (overlap <= 0).
       
       Args: 
           coarse_track_dict <dict> {
               track_id <int>: <np.array>, (total_frame_num, emb_size+4+1)
           },
           emb_size: embbedings size.
           track_overlap_thresh: .
           search_radius: .
           time_dist_tresh: max gap between two coarse tracklets attribute to the same tracklet.
           clip_len: clip length of the end or start of one coarse tracklet to match the pred_loc.
           window_len: .

       Return: 
            coarse_tracklet_connects <dict>{
                track_id <int>: {
                    'conflict': <list> track_id list of conflict coarse tracklets,
                    'neighbor': <list> track_id list of neighbor coarse tracklets,
                }
            },
            tracklet_pair: <np.array((pair_num, 6)), [track_id_1, t_start_1, t_end_1, track_id_2,  t_start_2, t_end_2]
    """
    track_num = len(coarse_track_dict.keys())
    coarse_tracklet_connects = {}   
    tracklet_pair = [] 
    for track_id in coarse_track_dict.keys():
        coarse_tracklet_connects[track_id] = dict(
            conflict=[],
            neighbor=[]
        )
    for track_id_1 in range(track_num-1):
        for track_id_2 in range(track_id_1+1, track_num):
            t_1 = np.where(coarse_track_dict[track_id_1][:, emb_size]!=-1)[0]
            t_min_1 = t_1[0]
            t_max_1 = t_1[-1]
            
            t_2 = np.where(coarse_track_dict[track_id_2][:, emb_size]!=-1)[0]
            t_min_2 = t_2[0]
            t_max_2 = t_2[-1]

            overlap_len = min(t_max_2,t_max_1) - max(t_min_1,t_min_2) + 1
            overlap_r = overlap_len / (t_max_1 - t_min_1 + 1 + t_max_2 - t_min_2 + 1 - overlap_len)
            
            if overlap_len > 0 and overlap_r > track_overlap_thresh:
                # two coarse tracklets overlap each other too much to possibly be one tracklet
                coarse_tracklet_connects[track_id_1]['conflict'].append(track_id_2)
                coarse_tracklet_connects[track_id_2]['conflict'].append(track_id_1)
                continue

            if overlap_len > 0 and overlap_r <= track_overlap_thresh:
                # two coarse tracklets overlap each other not too much, possible be one tracklet
                if (t_min_1 - t_min_2) * (t_max_1 - t_max_2) <= 0:
                    # one coarse tracklet is a subset of the other one, not possible to be one tracklet
                    coarse_tracklet_connects[track_id_1]['conflict'].append(track_id_2)
                    coarse_tracklet_connects[track_id_2]['conflict'].append(track_id_1)
                    continue
                """ The overlap of the front coarse tracklet may be wrong, which may need to concat the latter coarse tracklet
                if the center of latter coarse tracklet is within the search radius from the center of the front coarse tracklet 
                in the overlap time, the two coarse tracklets are neighbors for each other.
                """
                t_min = int(max(t_min_1, t_min_2))
                t_max = int(min(t_max_1, t_max_2))
                cand_t = np.array(range(t_min,t_max+1)) # overlap frame range
                center_x_1, center_y_1, w_1, h_1 = get_coarse_tarcklet_loc(coarse_track_dict, track_id_1, cand_t, emb_size)
                center_x_2, center_y_2, w_2, h_2 = get_coarse_tarcklet_loc(coarse_track_dict, track_id_2, cand_t, emb_size)
                dist_x = abs(center_x_1 - center_x_2)
                dist_y = abs(center_y_1 - center_y_2)
                min_len = np.min([np.min(w_1), np.min(h_1), np.min(w_2), np.min(h_2)])
                min_dist_x = np.min(dist_x/min_len)
                min_dist_y = np.min(dist_y/min_len)
                if min_dist_x < search_radius and min_dist_y < search_radius:
                    coarse_tracklet_connects[track_id_1]['neighbor'].append(track_id_2)
                    coarse_tracklet_connects[track_id_2]['neighbor'].append(track_id_1)
                    if t_min_1 <= t_min_2 and t_max - t_min + 1 < window_len:
                        # track_id_1, track_id_2, crop the overlap                           
                        new_t_max_1 = t_min_2
                        new_t_min_2 = t_max_1
                        # generate tracklet_pair
                        pair = np.zeros((6)) - 1
                        pair[0] = track_id_1
                        pair[1] = track_id_2
                        t_range = np.array(get_trackletpair_t_range(t_min_1, new_t_max_1, new_t_min_2, t_max_2, window_len))
                        pair[2:] = t_range
                        tracklet_pair.append(pair)
                    elif t_min_1 > t_min_2 and t_max - t_min + 1 < window_len:
                        # track_id_2, track_id_1, crop the overlap
                        new_t_max_2 = t_min_1
                        new_t_min_1 = t_max_2
                        # generate tracklet_pair
                        pair = np.zeros((6)) - 1
                        pair[0] = track_id_2
                        pair[1] = track_id_1
                        t_range = np.array(get_trackletpair_t_range(t_min_2, new_t_max_2, new_t_min_1, t_max_1, window_len))
                        pair[2:] = t_range
                        tracklet_pair.append(pair)

            if overlap_len <= 0 and min(abs(t_min_1-t_max_2), abs(t_min_2-t_max_1)) < time_dist_tresh:
                """ there is no overlap, but gap time less than the time_dist_tresh, possible to be one tracklet.
                We choose the tracklet clip in one specific length (clip_len) from the end of the front coarse tracklet, 
                and the clip in the same length from the start of the latter coarse tracklet. Using F to pred and adjust 
                location for each clip, compare and decide whether they are from the same tracklet.
                """
                track_t1 = np.array(range(int(t_min_1), int(t_max_1+1)))
                center_x_1, center_y_1, w_1, h_1 = get_coarse_tarcklet_loc(coarse_track_dict, track_id_1, track_t1, emb_size)
                assert len(center_x_1) == t_max_1 - t_min_1 + 1
                
                track_t2 = np.array(range(int(t_min_2), int(t_max_2+1)))
                center_x_2, center_y_2, w_2, h_2 = get_coarse_tarcklet_loc(coarse_track_dict, track_id_2, track_t2, emb_size)
                assert len(center_x_2) == t_max_2 - t_min_2 + 1

                t1 = None
                t2 = None
                if t_min_1 >= t_max_2:
                    t1 = int(t_min_1)
                    t2 = int(t_max_2)
                    real_center_x_t1 = center_x_1[0]
                    real_center_y_t1 = center_y_1[0]
                    w_t1 = w_1[0]
                    h_t1 = h_1[0]
                    real_center_x_t2 = center_x_2[-1]
                    real_center_y_t2 = center_y_2[-1]                   
                    w_t2 = w_2[-1]
                    h_t2 = h_2[-1]
                    # track_2, track_1
                    if len(center_x_1) > clip_len:
                        track_t1 = track_t1[:clip_len]
                        center_x_1 = center_x_1[:clip_len]
                        center_y_1 = center_y_1[:clip_len]
                    if len(center_x_2) > clip_len:
                        track_t2 = track_t2[-clip_len:]
                        center_x_2 = center_x_2[-clip_len:]
                        center_y_2 = center_y_2[-clip_len:]
                elif t_max_1 <= t_min_2:
                    t1 = int(t_max_1)
                    t2 = int(t_min_2)
                    real_center_x_t1 = center_x_1[-1]
                    real_center_y_t1 = center_y_1[-1]
                    w_t1 = w_1[-1]
                    h_t1 = h_1[-1]
                    real_center_x_t2 = center_x_2[0]
                    real_center_y_t2 = center_y_2[0]
                    w_t2 = w_2[0]
                    h_t2 = h_2[0]
                    # track_1, track_2
                    if len(center_x_1) > clip_len:
                        track_t1 = track_t1[-clip_len:]
                        center_x_1 = center_x_1[-clip_len:]
                        center_y_1 = center_y_1[-clip_len:]
                    if len(center_x_2) > clip_len:
                        track_t2 = track_t2[:clip_len]
                        center_x_2 = center_x_2[:clip_len]
                        center_y_2 = center_y_2[:clip_len]
                
                pred_center_x_t1 = linear_pred_v2(track_t1, center_x_1, np.array([t2]))[0]
                pred_center_y_t1 = linear_pred_v2(track_t1, center_y_1, np.array([t2]))[0]
                dist_x1 = abs(pred_center_x_t1 - real_center_x_t2)
                dist_y1 = abs(pred_center_y_t1 - real_center_y_t2)

                pred_center_x_t2 = linear_pred_v2(track_t2, center_x_2, np.array([t1]))[0]
                pred_center_y_t2 = linear_pred_v2(track_t2, center_y_2, np.array([t1]))[0]
                dist_x2 = abs(pred_center_x_t2 - real_center_x_t1)
                dist_y2 = abs(pred_center_y_t2 - real_center_y_t1)

                dist_x = min(dist_x1, dist_x2)
                dist_y = min(dist_y1, dist_y2)
                min_len = np.min([w_t1, w_t2, h_t1, h_t2])
                min_dist_x = dist_x / min_len
                min_dist_y = dist_y / min_len
                
                if min_dist_x < search_radius and min_dist_y < search_radius:
                    coarse_tracklet_connects[track_id_1]['neighbor'].append(track_id_2)
                    coarse_tracklet_connects[track_id_2]['neighbor'].append(track_id_1)
                    if t_min_1 >= t_max_2:
                        # track_2, track_1, enerate tracklet_pair
                        pair = np.zeros((6)) - 1
                        pair[0] = track_id_2
                        pair[1] = track_id_1
                        t_range = np.array(get_trackletpair_t_range(t_min_2, t_max_2, t_min_1, t_max_1, window_len))
                        pair[2:] = t_range
                        tracklet_pair.append(pair)
                    elif t_max_1 <= t_min_2:
                        # track_1, track_2, generate tracklet_pair
                        pair = np.zeros((6)) - 1
                        pair[0] = track_id_1
                        pair[1] = track_id_2
                        t_range = np.array(get_trackletpair_t_range(t_min_1, t_max_1, t_min_2, t_max_2, window_len))
                        pair[2:] = t_range
                        tracklet_pair.append(pair)

    tracklet_pair = np.vstack(tracklet_pair).astype(np.int64)

    return coarse_tracklet_connects, tracklet_pair
