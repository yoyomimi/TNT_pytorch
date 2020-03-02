# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-3-2
# ------------------------------------------------------------------------------
import numpy as np

import _init_paths

INF = 1e5

def Floyd(n, dist, path):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF and dist[i][j] > dist[i][k] + dist[k][j]:
                        path[i][j]= k
                        dist[i][j]= dist[i][k] + dist[k][j]
    print(dist, path)
    return dist, path

def get_optimal_cluster(cluster_dict, coarse_tracklet_connects, tracklet_comb_cost):
    """use graph algorithm to get optimal connections and clusters
    Args:
        cluster_dict <dict>{
            frame_id <int>: <list> list of track_ids at this frame
        },
        coarse_tracklet_connects <dict>{
            tracklet_id <int>:{
                'neighbor': <list> list of track_ids,
                'conflict': <list> list of track_ids
            }
        },
        tracklet_comb_cost <dict>{
            track_id_1:{
                track_id_2: [<int> connectivity, <float> cost]
            }
        }

    Initial cluster_dict stores all the vertices.
    Coarse_tracklet_connects stores all the connected edges as neighbors.
    Tracklet_comb_cost stores the weights/cost of all the possible edges in coarse_tracklet_connects.

    To minimize the total neighbor connected weights in clusters of the total graph

    Return:
        cluster_dict <dict>{
            frame_id <int>: <list> list of track_ids at this frame
        }
    """

    # use Floyd to minimize the total cost
    track_num = len(coarse_tracklet_connects)
    V = np.array(list(coarse_tracklet_connects.keys()), dtype=np.int64) # (track_num,)
    dist = (np.zeros((track_num, track_num)) + INF).astype(np.float32)
    path = np.ones((track_num, track_num))

    # init dist
    for track_id_1 in coarse_tracklet_connects.keys():
        for track_id_2 in coarse_tracklet_connects[track_id_1]['neighbor']:
            track_id_1 = int(track_id_1)
            track_id_2 = int(track_id_2)
            if track_id_1 in tracklet_comb_cost.keys() and track_id_2 in tracklet_comb_cost[track_id_1].keys():
                dist[track_id_1][track_id_2] = tracklet_comb_cost[track_id_1][track_id_2][1]
            elif track_id_2 in tracklet_comb_cost.keys() and track_id_1 in tracklet_comb_cost[track_id_2].keys():
                dist[track_id_1][track_id_2] = tracklet_comb_cost[track_id_2][track_id_1][1]
            else:
                print(track_id_1, track_id_2, 'not found the weight')
    
    # use Floyd Algorithm to calculate the min dist and corresponding path between each vertice pair
    dist, path = Floyd(track_num, dist, path)
    
    return cluster_dict


if __name__ == '__main__':
    import json

    # read json, remember to int each key when use
    coarse_tracklet_connects = {}
    with open('data/coarse_tracklet_connects.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        coarse_tracklet_connects[int(track_id)] = temp_dict[track_id]
    
    cluster_dict = {}
    with open('data/cluster_dict.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        cluster_dict[int(track_id)] = temp_dict[track_id]

    cluster_cost_dict = {}
    with open('data/cluster_cost_dict.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        cluster_cost_dict[int(track_id)] = temp_dict[track_id]

    tracklet_comb_cost = {}
    with open('data/tracklet_comb_cost.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        tracklet_comb_cost[int(track_id)] = {}
        for track_id_2 in temp_dict[track_id].keys():
            tracklet_comb_cost[int(track_id)][int(track_id_2)] = temp_dict[track_id][track_id_2]

    get_optimal_cluster(cluster_dict, coarse_tracklet_connects, tracklet_comb_cost)
    


