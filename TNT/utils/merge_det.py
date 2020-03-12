# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-27
# ------------------------------------------------------------------------------

# delete if not debug
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from tracklets.utils.pred_loc import pred_bbox_by_F, linear_pred
from TNT.utils.detbbox_utils import get_overlap


def bbox_associate(overlap_mat, IOU_thresh): 
    idx1 = [] 
    idx2 = [] 
    new_overlap_mat = overlap_mat.copy()
    while True:
        # find all (obj_1_id, obj_2_id) pair which matches best to each other
        idx = np.unravel_index(np.argmax(new_overlap_mat, axis=None), new_overlap_mat.shape) 
        if new_overlap_mat[idx] < IOU_thresh: 
            # no matched pair satisfies the threshold constriant
            break 
        else: 
            idx1.append(idx[0]) 
            idx2.append(idx[1]) 
            new_overlap_mat[idx[0],:] = 0 
            new_overlap_mat[:,idx[1]] = 0
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    return idx1, idx2 # start from 0

# wait to add in cfg: linear_pred_thresh, coeff_norm_thresh, pred_loc_iou_tresh, pred_use_F
def merge_det(det_dict, crop_im, linear_pred_thresh=5, coeff_norm_thresh=0.15, pred_loc_iou_tresh=0.2, pred_F_mat=None):
    """merge det from the neighbour frame (based on frame_dist limitation) using emb_dist(?)
    
    Args:
        det_dict{
            'frame_id': (obj_num, emb_size+4+1+1), for one obj_det in one frame_id [img_emb(512) x y w h label crop_index(array(1))], np.array
        }
        crop_im{
            'frame_id': (obj_num, crop_min, crop_max, 3)
        }
        linear_pred_thresh: ,
        coeff_norm_thresh: , 
        pred_loc_iou_thresh: ,
        pred_F_mat: mat for use_F_mat,
        
    Return:
        track_dict{
            'track_id': (frame_num, emb_size+4+1), for one frame_num in one track_id [img_emb(512) x y w h label], np.array
        }

    frame_id, track_id both start from 0, <int> type.
    bbox: x, y, w, h
        
    """
    track_dict = {}
    frame_num = max(list([int(id) for id in det_dict.keys()])) + 1
    now_obj_num, feat_size = det_dict[0].shape
    feat_size -= 1
    emb_size = feat_size - 4 - 1

    # init tracklet using the first frame
    new_track_id = []
    max_track_id = -1
    for track_id in range(now_obj_num):
        track_dict[track_id] = np.zeros((frame_num, feat_size))-1 # set -1 for initial
        track_dict[track_id][0] = det_dict[0][track_id][:-1]
        new_track_id.append(track_id)
    if len(new_track_id):
        max_track_id = new_track_id[-1]
    
    # process the later frames, frame_id start from 0
    for i in range(1, frame_num):
        pre_obj_num = det_dict[i-1].shape[0]
        now_obj_num = det_dict[i].shape[0]
        pre_track_id = new_track_id.copy()
        new_track_id = []
        if now_obj_num == 0:
            continue
        if pre_obj_num == 0 and now_obj_num > 0:
            for track_id in range(max_track_id+1, max_track_id+now_obj_num+1):
                track_dict[track_id] = np.zeros((frame_num, feat_size))-1 # set -1 for initial
                track_dict[track_id][i] = det_dict[i][track_id-max_track_id-1][:-1]
                new_track_id.append(track_id)
            max_track_id = new_track_id[-1]
            continue

        track_idx1 = []
        track_idx2 = []
        pred_bbox1 = np.zeros((pre_obj_num, 4))
        if pred_F_mat:
            pred_bbox1 = pred_bbox_by_F(det_dict[i-1][:, emb_size:emb_size+4], pred_F_mat[:,:,i-1], 0, [], [])
        for obj in range(pre_obj_num):
            temp_track_id = pre_track_id[obj]
            frame_range = np.where(track_dict[temp_track_id][:, emb_size]!=-1)[0]
            if len(frame_range) == 0:
                raise Exception('Invalid Track!')
            
            # get linear_pred within tresh range for bbox1
            frame_start = np.min(frame_range)
            if frame_start < i-1-linear_pred_thresh:
                frame_start = i-1-linear_pred_thresh
            bbox = track_dict[temp_track_id][int(frame_start):i, emb_size:emb_size+4]
            center_x =  bbox[:, 0] + bbox[:, 2] / 2
            center_y = bbox[:, 1] + bbox[:, 3] / 2
            if len(center_x)==0:
                raise Exception('Invalid Frame Range!')
            pred_center_x = linear_pred(center_x)
            pred_center_y = linear_pred(center_y)
            pred_w = linear_pred(bbox[:, 2])
            pred_h = linear_pred(bbox[:, 3])

            # store the pred loc of the track for the current frame
            pred_bbox1[obj,2] = max(pred_w,1)
            pred_bbox1[obj,3] = max(pred_h,1)
            pred_bbox1[obj,0] = pred_center_x-pred_w/2
            pred_bbox1[obj,1] = pred_center_y-pred_h/2

        # overlap_mat: (pre_obj_num, now_obj_num)
        overlap_mat, _ ,_ ,_ = get_overlap(pred_bbox1, det_dict[i][:, emb_size:emb_size+4])

        # color dist coeff norm
        crop_id_1 = np.array(det_dict[i-1][:, emb_size+5]).astype(np.int64)
        crop_id_2 = np.array(det_dict[i][:, emb_size+5]).astype(np.int64)

        color_dist = np.zeros((pre_obj_num, now_obj_num))
        print(crop_id_1[n1], crop_id_2[n2], n1, n2, crop_im[i-1].shape, crop_im[i].shape)
        for n1 in range(pre_obj_num):
            for n2 in range(now_obj_num):
                color_dist[n1,n2] = np.sum(crop_im[i-1][crop_id_1[n1]] * crop_im[i][crop_id_2[n2]])
        
        if np.isnan(np.sum(color_dist)) or np.isnan(np.sum(overlap_mat)):
            raise Exception('Invalid Color Dist or Overlap Mat!')
        overlap_mat[color_dist<coeff_norm_thresh] = 0      
        track_idx1, track_idx2 = bbox_associate(overlap_mat, pred_loc_iou_tresh)
        # print('\nmat:', overlap_mat, '\n', track_idx1, track_idx2, '\n')

        # no matched pair
        if len(track_idx1) == 0:
            for track_id in range(max_track_id+1, max_track_id+now_obj_num+1):
                track_dict[track_id] = np.zeros((frame_num, feat_size))-1 # set -1 for initial
                track_dict[track_id][i] = det_dict[i][track_id-max_track_id-1][:-1]
                new_track_id.append(track_id)
            max_track_id = new_track_id[-1]
        elif len(track_idx1) > 0:
            for obj_id in range(now_obj_num):
                # find which new obj to be merged, merge to which track_idx index list
                temp_idx = np.where(track_idx2==obj_id)[0]
                if len(temp_idx) == 0:
                    # create new track
                    track_id = max_track_id + 1
                    track_dict[track_id] = np.zeros((frame_num, feat_size))-1 # set -1 for initial
                    track_dict[track_id][i] = det_dict[i][obj_id][:-1]
                    new_track_id.append(track_id)
                    max_track_id += 1
                elif len(temp_idx) > 0:
                    # mearge to existed track
                    obj_id_1 = track_idx1[temp_idx[0]]
                    track_id = pre_track_id[obj_id_1] # idx of track_idx in the front matches more
                    track_dict[track_id][i] = det_dict[i][obj_id][:-1]
                    new_track_id.append(track_id)
        # print(i-1, pre_track_id, '\n', i, new_track_id, '\n')
                    
    print('coarse_track_num', max_track_id+1)
    return track_dict
    

if __name__ == "__main__":
    import json
    import numpy as np

    from utils.utils import write_dict_to_json
    

    # read det_results
    det_dict = {}
    with open('data/det.json', 'r') as f:
        temp_dict = json.load(f)  
    for frame_id in temp_dict.keys():
        det_dict[int(frame_id)] = np.array(temp_dict[frame_id], dtype=np.float32)

    track_dict = merge_det(det_dict)
    
    write_dict_to_json(track_dict, 'data/coarse_tracklet.json')
    