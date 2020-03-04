import numpy as np
import pandas as pd

def cluster_dict_processing(coarse_track_dict, tracklet_time_range, cluster_dict):
    """transfer cluster_dict to cluster feat array.
    Args: 
        coarse_track_dict <dict> {
            track_id <int>: <np.array>, (total_frame_num, emb_size+4+1)
        },
        tracklet_time_range <dict> {
            track_id <int>: <np.array>, (2), min_frame_id and max_frame_id
        },
        cluster_dict <dict>{
            track_id <int>: <list>, list of track_ids in one cluster
        }
    
    Return:
        cluster_feat_dict<dict> {
            cluster_id <int>: <np.array>, (total_frame_num, emb_size+4+1)
        },
        cluster_frame_range<dict> {
            cluster_id <int>: <np.array>, (2)
        },
    """
    cluster_num = len(list(cluster_dict.keys()))
    cluster_frame_range = {}
    cluster_feat_dict = {}
    
    cluster_count = 0
    for cluster_id in cluster_dict.keys():
        cluster_feat_dict[cluster_count] = -np.ones((coarse_track_dict[0].shape))
        cluster_frame_range[cluster_count] = np.zeros((2))
        track_ids = sorted(cluster_dict[cluster_id])
        min_t_list = []
        max_t_list = []
        #  merge tracklets within one cluster
        for track_id in track_ids:
            min_t, max_t = tracklet_time_range[track_id]
            # used to update cluster_t_range
            min_t_list.append(min_t)
            max_t_list.append(max_t)
            # find overlap, average at over lap
            overlap = np.where(cluster_feat_dict[cluster_count][min_t:max_t+1, -5] != -1)[0]
            cluster_feat_dict[cluster_count][min_t:max_t+1, :-1] += coarse_track_dict[track_id][min_t:max_t+1, :-1] + 1
            cluster_feat_dict[cluster_count][min_t:max_t+1, -1] = coarse_track_dict[track_id][min_t:max_t+1, -1]
            if len(overlap) > 0:
                cluster_feat_dict[cluster_count][min_t:max_t+1, :-1][overlap] = (cluster_feat_dict[cluster_count][min_t:max_t+1, :-1][overlap] - 1) / 2
        
        # update cluster_frame_range
        cluster_frame_range[cluster_count][0] = int(min(min_t_list))
        cluster_frame_range[cluster_count][1] = int(max(max_t_list))
        
        # interpolate within vancacy
        data = cluster_feat_dict[cluster_count][min(min_t_list):max(max_t_list)+1, :-1]
        feat_pd = pd.DataFrame(data=data).replace(-1, np.nan, inplace=False)
        feat_pd_np = np.array(feat_pd.interpolate()).astype(np.float32)
        cluster_feat_dict[cluster_count][min(min_t_list):max(max_t_list)+1, :-1] = feat_pd_np 

        # vote for label
        mask = np.where(cluster_feat_dict[cluster_count][min(min_t_list):max(max_t_list)+1, -1] != -1)[0]
        label_counts = np.bincount(cluster_feat_dict[cluster_count][min(min_t_list):max(max_t_list)+1, -1][mask].astype(np.int64))
        label = np.argmax(label_counts)
        cluster_feat_dict[cluster_count][min(min_t_list):max(max_t_list)+1, -1] = label

        cluster_count += 1

    
    return cluster_feat_dict, cluster_frame_range


if __name__ == '__main__':
    import json

    # read json, remember to int each key when use
    coarse_track_dict = {}
    with open('data/coarse_tracklet.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        coarse_track_dict[int(track_id)] = np.array(temp_dict[track_id], dtype=np.float32)
    
    cluster_dict = {}
    with open('data/new_cluster_dict.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        cluster_dict[int(track_id)] = temp_dict[track_id]

    tracklet_time_range = {}
    with open('data/tracklet_time_range.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        tracklet_time_range[int(track_id)] = temp_dict[track_id]

    cluster_feat_dict, cluster_frame_range = cluster_dict_processing(coarse_track_dict, tracklet_time_range, cluster_dict)
