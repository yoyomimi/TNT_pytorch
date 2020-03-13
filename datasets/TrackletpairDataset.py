# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-24
# ------------------------------------------------------------------------------
import cv2
import random
import json
import numpy as np
import logging
import os
import os.path as osp

import pandas as pd

import torch
from torch.utils.data import Dataset

class TrackletpairDataset(Dataset):
    """get tracklet pair batch for tracklet pair connectivity training.
       Args:
            crop_data_root: abs path for best.

       will there be only blanks between the two traklets in one pair, not at the side?
       
    """

    def __init__(self,
                 crop_data_root,
                 transform=None,
                 window_length=64,
                 stride = 30,
                 is_train=True,):

        self.transform = transform
        self.window_length = window_length # temporal window length to cover the tracklet pair.
        self.stride = stride # temporal window sliding stride

        if is_train:
            self.mode = 'train'
            self.data_root = osp.join(crop_data_root, 'training')
        else:
            self.mode = 'eval'
            self.data_root = osp.join(crop_data_root, 'eval')
        tracklet_pair_path = osp.join(self.data_root, 'tracklet_pair.txt')

        # read track json file.
        track_json_file = osp.join(self.data_root, f'gt_{self.mode}_track_dict.json')
        with open(track_json_file, 'r') as f:
            self.track_dict = json.load(f)

        if not osp.exists(tracklet_pair_path):
            pair_f = open(tracklet_pair_path, 'a')
            pair_count = 0
            # generate tracklet pairs [video_name_1, track_id_1, start_frame_id_1, end_frame_id_1, 
            #                          video_name_2, track_id_2, start_frame_id_2, end_frame_id_2, connectivity(0 or 1)]
            for video_name in self.track_dict.keys():
                for track_id in self.track_dict[video_name].keys():
                    class_name = self.track_dict[video_name][track_id][0]
                    now_frame_list_ori = sorted(list(map(int, self.track_dict[video_name][track_id][1])))
                    now_frame_list = now_frame_list_ori.copy()
                    if len(now_frame_list) < int(now_frame_list[-1] - now_frame_list[0] + 1): # not continous
                        continue
                    frame_window_list = []
                    # sliding temporal window for tracklet pair sampling
                    while len(now_frame_list) >= self.window_length:
                        frame_window_list.append([now_frame_list[0], now_frame_list[self.window_length-1]])
                        now_frame_list = now_frame_list[self.stride:]
                    same_cls_track_list = []
                    for track_id_new in self.track_dict[video_name].keys():
                        if self.track_dict[video_name][track_id_new][0] == class_name:
                            if track_id_new != track_id:
                                same_cls_track_list.append(track_id_new)

                    for frame_window in frame_window_list:
                        remain_frame_window = frame_window_list.copy()
                        remain_frame_window.remove(frame_window)
                        for i in range(int((frame_window[1]-frame_window[0]) / 4)):
                            start_frame_id_1 = frame_window[0]
                            end_frame_id_1, start_frame_id_2, end_frame_id_2 = sorted(random.sample(range(frame_window[0]+1, frame_window[1]+1), 3))
                            
                            # write connected pair to tracklet_pair_path
                            pair_f.write(f'{video_name},{track_id},{start_frame_id_1},{end_frame_id_1},{video_name},{track_id},{start_frame_id_2},{end_frame_id_2},{1}\n')
                            pair_count += 1    
 
                            for other_track_id in same_cls_track_list:
                                # non connected pair in other track
                                other_frame_window = sorted(list(map(int, self.track_dict[video_name][other_track_id][1])))
                                if len(other_frame_window) < int(other_frame_window[-1] - other_frame_window[0] + 1): # not continous
                                    continue
                                if start_frame_id_1 >= other_frame_window[0] and end_frame_id_1 <= other_frame_window[-1]:
                                    pair_f.write(f'{video_name},{other_track_id},{start_frame_id_1},{end_frame_id_1},{video_name},{track_id},{start_frame_id_2},{end_frame_id_2},{0}\n')
                                    pair_count += 1 
                                if start_frame_id_2 >= other_frame_window[0] and end_frame_id_2 <= other_frame_window[-1]:
                                    pair_f.write(f'{video_name},{track_id},{start_frame_id_1},{end_frame_id_1},{video_name},{other_track_id},{start_frame_id_2},{end_frame_id_2},{0}\n')
                                    pair_count += 1

            pair_f.close()
            print("Having written %d tracklet pairs" %(pair_count))
        
        with open(tracklet_pair_path, 'r') as f:
            self.tracklet_path_list = f.readlines()
        print(f'Loading {self.mode} tracklet pairs %d' %(len(self.tracklet_path_list)))

        
    def get_crop_path(self, video_name, track_id, frame_id, class_name):
        track_path = osp.join(self.data_root, video_name, str(track_id))
        crop_name = f'{class_name}_{frame_id}_crop.jpg'
        crop_path = osp.join(track_path, crop_name)
        return crop_path

    def __len__(self):
        return len(self.tracklet_path_list)
        

    def __getitem__(self, index):
        """
        Return:
            img_1: (frame_window_len, size, size, 3).
            img_2: (frame_window_len_2, size, size, 3).
            loc_mat: (frame_window_len, 4)
            tracklet_mask_1: (frame_window_len, 1).
            tracklet_mask_2: (frame_window_len, 1).
            real_window_len: <int>
            connectivity: (1, 1) LongTensor
        """
        tracklet_info = self.tracklet_path_list[index].split()[0].split(',')
        video_name_1,track_id_1,start_frame_id_1,end_frame_id_1 = tracklet_info[:4]
        video_name_2,track_id_2,start_frame_id_2,end_frame_id_2 = tracklet_info[4:8]
        connectivity = tracklet_info[8]
        img_1 = []
        img_2 = []
        loc_mat = np.zeros((self.window_length, 4))
        tracklet_mask_1 = np.zeros((self.window_length, 1))
        tracklet_mask_2 = np.zeros((self.window_length, 1))

        # get img_1, loc_mat for img1 and tracklet_mask_1
        tracklet_info_1 = self.track_dict[video_name_1][track_id_1]
        class_name_1 = tracklet_info_1[0]
        assert len(tracklet_info_1[1]) == len(tracklet_info_1[2]) == len(tracklet_info_1[3]) == len(tracklet_info_1[4]) == len(tracklet_info_1[5])
        start_frame_id_1 = int(start_frame_id_1)
        end_frame_id_1 = int(end_frame_id_1)
        for frame_id in range(start_frame_id_1, end_frame_id_1+1): # frame_id start from 0
            img_path = self.get_crop_path(video_name_1, track_id_1, frame_id, class_name_1)
            if not osp.exists(img_path):
                logging.error("Cannot found image data: " + img_path)
                raise FileNotFoundError
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if self.transform is not None:
                img_1.append(self.transform(img))
            frame_idx = tracklet_info_1[1].index(frame_id)
            loc_mat[frame_id-start_frame_id_1][0] = float(tracklet_info_1[2][frame_idx])
            loc_mat[frame_id-start_frame_id_1][1] = float(tracklet_info_1[3][frame_idx])
            loc_mat[frame_id-start_frame_id_1][2] = float(tracklet_info_1[4][frame_idx]) - float(tracklet_info_1[2][frame_idx])
            loc_mat[frame_id-start_frame_id_1][3] = float(tracklet_info_1[5][frame_idx]) - float(tracklet_info_1[3][frame_idx])
            tracklet_mask_1[frame_id-start_frame_id_1] = 1
        
        # get img_2, loc_mat for img2 and tracklet_mask_2
        tracklet_info_2 = self.track_dict[video_name_2][track_id_2]
        class_name_2 = tracklet_info_2[0]
        assert len(tracklet_info_2[1]) == len(tracklet_info_2[2]) == len(tracklet_info_2[3]) == len(tracklet_info_2[4]) == len(tracklet_info_2[5])
        start_frame_id_2 = int(start_frame_id_2)
        end_frame_id_2 = int(end_frame_id_2)
        for frame_id in range(start_frame_id_2, end_frame_id_2+1): # frame_id start from 0
            img_path = self.get_crop_path(video_name_2, track_id_2, frame_id, class_name_2)
            if not osp.exists(img_path):
                logging.error("Cannot found image data: " + img_path)
                raise FileNotFoundError
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if self.transform is not None:
                img_2.append(self.transform(img))
            frame_idx = tracklet_info_2[1].index(frame_id)
            loc_mat[frame_id-start_frame_id_1][0] = float(tracklet_info_2[2][frame_idx])
            loc_mat[frame_id-start_frame_id_1][1] = float(tracklet_info_2[3][frame_idx])
            loc_mat[frame_id-start_frame_id_1][2] = float(tracklet_info_2[4][frame_idx]) - float(tracklet_info_2[2][frame_idx])
            loc_mat[frame_id-start_frame_id_1][3] = float(tracklet_info_2[5][frame_idx]) - float(tracklet_info_2[3][frame_idx])
            tracklet_mask_2[frame_id-start_frame_id_1] = 1
        
        img_1 = torch.stack(img_1)
        img_2 = torch.stack(img_2)

        real_window_len = min(self.window_length, end_frame_id_2-start_frame_id_1+1)

        loc_mat[0][np.where(loc_mat[0]==0)] = 1e-3
        loc_mat[-1][np.where(loc_mat[-1]==0)] = 1e-3
        loc_mat=pd.DataFrame(data=loc_mat).replace(0, np.nan, inplace=False)
        loc_mat_np = np.array(loc_mat.interpolate()).astype(np.float32)
        if real_window_len < self.window_length:
            loc_mat_np[real_window_len:] = np.zeros((self.window_length-real_window_len, 4))
        loc_mat = torch.from_numpy(loc_mat_np)

        tracklet_mask_1 = torch.from_numpy(np.array(tracklet_mask_1).astype(np.float32))
        tracklet_mask_2 = torch.from_numpy(np.array(tracklet_mask_2).astype(np.float32))
        connectivity = torch.from_numpy(np.array(int(connectivity))).type(torch.LongTensor)
        
        return img_1, img_2, loc_mat, tracklet_mask_1, tracklet_mask_2, real_window_len, connectivity


if __name__ == "__main__":
    import sys
    sys.path.append(osp.join(osp.dirname(sys.path[0])))
    from datasets.transform import FacenetTransform

    transform = FacenetTransform(size=[182, 182])

    dataset = TrackletpairDataset('data/crop_data', transform=transform, is_train=True)
    img_1, img_2, loc_mat, tracklet_mask_1, tracklet_mask_2, real_window_len, connectivity = dataset.__getitem__(9000)
    
