# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-27
# ------------------------------------------------------------------------------
import numpy as np


def merge_det(det_dict):
    """merge det from the neighbour frame (based on frame_dist limitation) using emb_dist(?)
    
    Args:
        det_dict{
            'frame_id': (obj_num, emb_size+4+1), for one obj_det in one frame_id [img_emb(512) x y w h label], np.array
        }
        
    Return:
        track_dict{
            'track_id': (frame_num, emb_size+4+1), for one frame_num in one track_id [img_emb(512) x y w h label], np.array
        }

    frame_id, track_id both start from 0.
        
    """
    track_dict = {}

    frame_num = max(list([int(id) for id in det_dict.keys()])) + 1
    
    return track_dict
    
