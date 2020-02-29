# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-29
# ------------------------------------------------------------------------------
import numpy as np

def pred_connect_with_fusion(coarse_track_dict, window_len):
    """first generate tracklet pair using sliding window , and then use fushion net 
       to pred tracklet pair connectivity.

    Return:
        track_set: np.array((pair_num, 3)). [track_id_1, track_id_2, connectivity]


    """
    track_set = []

    return track_set