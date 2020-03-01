

# generate cluster dict using tracklet connect dict
def transfer_connect_to_cluster_dict(tracklet_connects, track_dict):
    """transfer connect graph of tracklets to cluster dict.
    Args: 
        tracklet_connects <dict>{
            track_id <int>: {
                'conflict': <list> track_id list of conflict coarse tracklets,
                'neighbor': <list> track_id list of neighbor coarse tracklets,
            }
        },
        track_dict <dict> {
            track_id <int>: <np.array>, (total_frame_num, emb_size+4+1)
        },
    
    Return:
        cluster_dict <dict>{
            cluster_id <int>: <np.array>, (total_frame_num, emb_size+4+1)
        }
    """
    
