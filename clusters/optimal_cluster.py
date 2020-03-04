# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-3-2
# ------------------------------------------------------------------------------
import numpy as np

import _init_paths

INF = 100
min_cost = INF
best_cluster = []
count = 0 # used for convergence
go = 1 # used for convergence

def get_single_weight(key1, key2, d):
    if key1 in d.keys() and key2 in d[key1].keys() and d[key1][key2][1] <= 0:
        return d[key1][key2][1]
    return INF

def get_weight(path, d):
    cost = 0.0
    for i in range(len(path)-1):
        assert path[i] < path[i+1]
        cost += get_single_weight(path[i], path[i+1], d)
    return cost

def get_optimal(coarse_tracklet_connects, tracklet_comb_cost, visited, parent, cost=0, 
                cluster=[], convergence_len=1e7):
    global min_cost
    global best_cluster
    global count
    global go

    remains = np.where(visited == 0)[0]
    if count > convergence_len:
        go = 0
        return best_cluster, min_cost
    if len(remains) == 0:
        return cluster, cost
    elif len(remains) == 1:
        cluster.append([remains[0]])
        return cluster, cost
    root = int(min(remains))

    # neighbors in time sequence
    s = [root]
    parent[root] = root
    # de dealt with when flag == 0
    flag = 1
    best_path = []
    while(len(s) and go):
        current = s[0]
        # visited[current] = 1
        s = s[1:]
        neighbors = coarse_tracklet_connects[current]['neighbor']
        for neighbor in neighbors:
            if not go:
                break
            if neighbor > current and visited[neighbor] == 0 and get_single_weight(current, neighbor, tracklet_comb_cost) != INF: # direct graph
                parent[neighbor] = int(current)
                # path: cluster for now
                now = neighbor
                path = [now]
                while(parent[now]!=root and parent[now] != -1):
                    path = [parent[now]] + path
                    now = parent[now]
                path = [root] + path

                # update status
                visited[path] = 1
                new_cluster = cluster.copy()
                new_cluster.append(path)
                new_cost = cost + get_weight(path, tracklet_comb_cost)

                # get_optimal
                count += 1
                new_cluster, new_cost = get_optimal(coarse_tracklet_connects, tracklet_comb_cost, visited.copy(), 
                    parent.copy(), new_cost, new_cluster, convergence_len)
                
                if new_cost < min_cost:
                    flag = 0
                    min_cost = new_cost
                    best_cluster = new_cluster
                    best_path = path
                    # print(root, min_cost, count)
                    count = 0
    
                visited[path] = 0
                s.append(neighbor)

    if not go:
        return best_cluster, min_cost

    if flag and go:
        # no neighbor for this root satisfies the constriant, thus not be dealt with
        visited[root] = 1
        cluster.append([root])
        # get_optimal
        count += 1
        new_cluster, new_cost = get_optimal(coarse_tracklet_connects, tracklet_comb_cost, visited.copy(), 
            parent.copy(), cost, cluster, convergence_len)
        return new_cluster, new_cost
    
    visited[path] = 1

    return best_cluster, min_cost


def get_optimal_cluster(coarse_tracklet_connects, tracklet_comb_cost):
    """use graph algorithm to get optimal connections and clusters
    Args:
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
        min_cost <float>: the sum cost of all the clusters.
    """

    track_num = len(coarse_tracklet_connects)
    # use recursive algorithm to calculate the min dist and corresponding path between each vertice pair
    global min_cost
    global best_cluster
    parent = (np.zeros((track_num)) - 1).astype(np.int64)
    visited  = np.zeros((track_num))
    cost = 0.0
    import time
    start = time.time()
    get_optimal(coarse_tracklet_connects, tracklet_comb_cost, visited, parent, cost, convergence_len=1e7)
    print('best: ', best_cluster, min_cost)
    print('time: ', time.time()-start, 's')
    return best_cluster, min_cost


if __name__ == '__main__':
    import json
    from utils.utils import write_dict_to_json

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

    tracklet_comb_cost = {}
    with open('data/tracklet_comb_cost_1.json', 'r') as f:
        temp_dict = json.load(f)  
    for track_id in temp_dict.keys():
        tracklet_comb_cost[int(track_id)] = {}
        for track_id_2 in temp_dict[track_id].keys():
            tracklet_comb_cost[int(track_id)][int(track_id_2)] = temp_dict[track_id][track_id_2]
    
    cluster_list, min_cost = get_optimal_cluster(coarse_tracklet_connects, tracklet_comb_cost)

    for cluster in cluster_list:
        cluster_id = min(cluster)
        for track_id in cluster:
            if track_id == cluster_id:
                continue
            cluster_dict.pop(track_id)
            cluster_dict[cluster_id].append(track_id)
    write_dict_to_json(cluster_dict, 'data/new_cluster_dict.json')
    

    # # test sample: 

    # cluster_dict = {}
    # coarse_tracklet_connects = {}
    # tracklet_comb_cost = {}
    # track_num = 13
    # for i in range(track_num):
    #     coarse_tracklet_connects[i] = {
    #         'neighbor': [],
    #         'conflict': []
    #     }
    #     tracklet_comb_cost[i] = {}
    #     cluster_dict[i] = [i]
    
    # # track_1, 0
    # coarse_tracklet_connects[0]['neighbor'] = [2, 4, 6]
    # coarse_tracklet_connects[0]['conflict'] = []
    # tracklet_comb_cost[0][2] = [1, -0.4]
    # tracklet_comb_cost[0][4] = [1, -0.3]
    # tracklet_comb_cost[0][6] = [1, -0.5]
    
    # # track_2, 1
    # coarse_tracklet_connects[1]['neighbor'] = [3, 6, 10]
    # coarse_tracklet_connects[1]['conflict'] = []
    # tracklet_comb_cost[1][3] = [1, -0.8]
    # tracklet_comb_cost[1][6] = [1, -0.3]
    # tracklet_comb_cost[1][10] = [0, 0.9]
    
    # # track_3, 2
    # coarse_tracklet_connects[2]['neighbor'] = [0, 10]
    # coarse_tracklet_connects[2]['conflict'] = []
    # tracklet_comb_cost[2][10] = [1, -0.3]

    # # track_4, 3
    # coarse_tracklet_connects[3]['neighbor'] = [1, 5, 9, 12]
    # coarse_tracklet_connects[3]['conflict'] = []
    # tracklet_comb_cost[3][5] = [1, -0.1]
    # tracklet_comb_cost[3][9] = [0, 0.7]
    # tracklet_comb_cost[3][12] = [1, -0.3]

    # # track_5, 4
    # coarse_tracklet_connects[4]['neighbor'] = [0, 11]
    # coarse_tracklet_connects[4]['conflict'] = []
    # tracklet_comb_cost[4][11] = [1, -0.6]

    # # track_6, 5
    # coarse_tracklet_connects[5]['neighbor'] = [3, 7]
    # coarse_tracklet_connects[5]['conflict'] = []
    # tracklet_comb_cost[5][7] = [1, -0.4]

    # # track_7, 6
    # coarse_tracklet_connects[6]['neighbor'] = [0, 1, 12]
    # coarse_tracklet_connects[6]['conflict'] = []
    # tracklet_comb_cost[6][12] = [0, 0.6]


    # # track_8, 7
    # coarse_tracklet_connects[7]['neighbor'] = [5, 8]
    # coarse_tracklet_connects[7]['conflict'] = []
    # tracklet_comb_cost[7][8] = [1, -0.6]

    # # track_9, 8
    # coarse_tracklet_connects[8]['neighbor'] = [7, 9]
    # coarse_tracklet_connects[8]['conflict'] = []
    # tracklet_comb_cost[8][9] = [0, 0.2]

    # # track_10, 9
    # coarse_tracklet_connects[9]['neighbor'] = [3, 8]
    # coarse_tracklet_connects[9]['conflict'] = []

    # # track_11, 10
    # coarse_tracklet_connects[10]['neighbor'] = [1, 2, 11]
    # coarse_tracklet_connects[10]['conflict'] = []
    # tracklet_comb_cost[10][11] = [0, 0.4]

    # # track_12, 11
    # coarse_tracklet_connects[11]['neighbor'] = [4, 10]
    # coarse_tracklet_connects[11]['conflict'] = []

    # # track_13, 12
    # coarse_tracklet_connects[12]['neighbor'] = [3, 6]
    # coarse_tracklet_connects[12]['conflict'] = []

    


