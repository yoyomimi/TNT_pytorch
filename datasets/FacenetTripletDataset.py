# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-20
# ------------------------------------------------------------------------------
import cv2
import random
import json
import numpy as np
import os
import os.path as osp

import torch
from torch.utils.data import Dataset

class FacenetTripletDataset(Dataset):
    """get triplet batch for facenet training.
       Args:
            crop_data_root: abs path for best.
    """

    def __init__(self,
                 crop_data_root,
                 transform=None,
                 is_train=True,):

        self.transform = transform

        if is_train:
            self.mode = 'train'
            self.data_root = osp.join(crop_data_root, 'training')
        else:
            self.mode = 'eval'
            self.data_root = osp.join(crop_data_root, 'eval')
        triplet_pair_path = osp.join(self.data_root, 'triplet_pair.txt')

        if not osp.exists(triplet_pair_path):
            # read track json file.
            track_json_file = osp.join(self.data_root, f'gt_{self.mode}_track_dict.json')
            with open(track_json_file, 'r') as f:
                track_dict = json.load(f)
            
            pair_f = open(triplet_pair_path, 'a')
            
            pair_count = 0
            # generate triplet pairs [video_name, track_id, frame_id, class_name]
            for video_name in track_dict.keys():
                for track_id in track_dict[video_name].keys():
                    class_name = track_dict[video_name][track_id][0]
                    frame_list = track_dict[video_name][track_id][1]
                    for i in range(len(frame_list)*4):
                        anchor_frame_id, pos_frame_id = random.sample(frame_list, 2)
                        anchor_path = self.get_crop_path(video_name, track_id, anchor_frame_id, class_name)
                        pos_path = self.get_crop_path(video_name, track_id, pos_frame_id, class_name)
                        # neg
                        other_video_name = random.sample(track_dict.keys(), 1)[0]
                        other_track_id_set = set(track_dict[other_video_name].keys())
                        if other_video_name == video_name:
                            other_track_id_set = other_track_id_set - set(track_id)
                        other_track_id = random.sample(list(other_track_id_set), 1)[0]
                        neg_frame_id = random.sample(track_dict[other_video_name][other_track_id][1], 1)[0]
                        other_class_name = track_dict[other_video_name][other_track_id][0]
                        neg_path = self.get_crop_path(other_video_name, other_track_id, neg_frame_id, other_class_name)
                        
                        # write to triplet_pair_path
                        pair_f.write(f'{anchor_path},{pos_path},{neg_path}\n')
                        pair_count += 1

            pair_f.close()
            print("Having written %d triplet pairs" %(pair_count))
        
        with open(triplet_pair_path, 'r') as f:
            self.triplet_path_list = f.readlines()
        print(f'Loading {self.mode} triplet pairs %d' %(len(self.triplet_path_list)))


    def get_crop_path(self, video_name, track_id, frame_id, class_name):
        videos_root = video_name.split('/')[-2]
        video_real_name = video_name.split('/')[-1][7:]
        track_path = osp.join(self.data_root, f'{videos_root}_{video_real_name}', str(track_id))
        crop_name = f'{class_name}_{frame_id}_crop.jpg'
        crop_path = osp.join(track_path, crop_name)
        return crop_path

    def __len__(self):
        return len(self.triplet_path_list)
        

    def __getitem__(self, index):
        """
        Return:
            anchor: (size, size, 3).
            positive: within the same tracklet with anchor, (size, size, 3).
            negative: within the different tracklet with anchor, (size, size, 3).
        """
        anchor_path, pos_path, neg_path = self.triplet_path_list[index].split()[0].split(',')

        if not osp.exists(anchor_path):
            logging.error("Cannot found anchor image data: " + anchor_path)
            raise FileNotFoundError
        anchor_img = cv2.imread(anchor_path, cv2.IMREAD_COLOR)

        if not osp.exists(pos_path):
            logging.error("Cannot found positive image data: " + pos_path)
            raise FileNotFoundError
        pos_img = cv2.imread(pos_path, cv2.IMREAD_COLOR)

        if not osp.exists(neg_path):
            logging.error("Cannot found negative image data: " + neg_path)
            raise FileNotFoundError
        neg_img = cv2.imread(neg_path, cv2.IMREAD_COLOR)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img


if __name__ == "__main__":
    import sys
    sys.path.append(osp.join(osp.dirname(sys.path[0])))
    from datasets.transform import FacenetTransform

    transform = FacenetTransform(size=[182, 182])

    dataset = FacenetTripletDataset('data/crop_data', transform=transform, is_train=True)
    dataset.__getitem__(12)
