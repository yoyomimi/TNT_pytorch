import numpy as np

def cluster_dict_processing(coarse_track_dict, cluster_dict):
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
    """
    cluster_num = len(list(cluster_dict.keys()))
    cluster_frame_range = np.zeros((cluster_num, 2))
    cluster_feat_dict = {}
    
    cluster_count = 0
    for cluster_id in cluster_dict.keys():
        cluster_feat_dict[cluster_count] = -np.ones((coarse_track_dict[0].shape))
        cluster_count += 1
        track_ids = cluster_dict[cluster_id]
        #  merge tracklets within one cluster
        for track_id in track_ids:
            min_t, max_t = tracklet_time_range[track_id]
            # compare min_t, max_t to update cluster_t_range

            # find overlap, average at over lap
            overlap = np.where(cluster_feat_dict[cluster_count][min_t:max_t+1, -5] == -1)[0]
            cluster_feat_dict[cluster_count][min_t:max_t+1, :] = coarse_track_dict[track_id][min_t:max_t+1, :]
        
        # interpolate within vancacy

        # vote for label

    
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

    cluster_dict_processing(coarse_track_dict, cluster_dict)