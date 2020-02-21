# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-20
# ------------------------------------------------------------------------------
import cv2
import json
import numpy as np
import os
import os.path as osp

import torch
from torch.utils.data import Dataset

class ObjtrackDataset(Dataset):

    def __init__(self,
                 data_root,
                 transform=None,
                 ):
        """
        info
        {
            'filename': 'a.jpg', # abs path for the image
            'frame_id': 1 # <int> start from 0
            'width': 1280,
            'height': 720,
            'ann': {
                'track_id': <np.ndarray> (n, ), # start from 0, -1 DontCare
                'bboxes': <np.ndarray> (n, 4), # 0-based, xmin, ymin, xmax, ymax
                'labels': <np.ndarray> (n, ), # DontCare(ignore), car, pedestrain, cyclist(optioanl)
                'occluded': <np.ndarray> (n, ), # optional
                'truncated': <np.ndarray> (n, ), # optional
                'alpha': <np.ndarray> (n, ), # 2D optional
                'dimensions': <np.ndarray> (n, 3), # 2D optional
                'location': <np.ndarray> (n, 3), # 2D optional
                'rotation_y': <np.ndarray> (n, ), # 2D optional
            }
        }
        
        Args:
            data_root: absolute root path for train or val data folder
            transform: train_transform or eval_transform or prediction_transform
        """
        self.class_dict = {
            'DontCare': -1,
            'Pedestrian': 0,
            'Car': 1,
            'Cyclist': 2
        }
        self.images_dir = os.path.join(data_root, 'images')
        self.labels_dir = os.path.join(data_root, 'labels')
        self.transform = transform
        self.track_infos = []
        
        img_infos = {}

        #scan the images_dir
        images_dir_list = os.listdir(self.images_dir)
        for frames_dir in images_dir_list:
            img_infos[frames_dir] = []
            frames_dir_path = os.path.join(self.images_dir, frames_dir)
            gt_path = os.path.join(self.labels_dir, frames_dir+'.txt')
            with open(gt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    labels = line.split()
                    if labels[1] == '-1': # DontCare
                        continue
                    frame_id = labels[0]
                    # already has info for this frame
                    if len(img_infos[frames_dir]) >= int(frame_id) + 1:
                        info = img_infos[frames_dir][int(frame_id)]
                        info['ann']['track_id'] = np.append(info['ann']['track_id'], int(labels[1]))
                        info['ann']['bboxes'] = np.vstack((info['ann']['bboxes'], np.array(labels[6:10], dtype=np.float32)))
                        info['ann']['labels'] = np.append(info['ann']['labels'], int(self.class_dict[labels[2]]))
                        info['ann']['truncated'] = np.append(info['ann']['truncated'], int(labels[3]))
                        info['ann']['occluded'] = np.append(info['ann']['occluded'], int(labels[4]))
                        info['ann']['alpha'] = np.append(info['ann']['alpha'], float(labels[5]))
                        info['ann']['dimensions'] = np.vstack((info['ann']['dimensions'], np.array(labels[10:13], dtype=np.float32)))
                        info['ann']['location'] = np.vstack((info['ann']['location'], np.array(labels[13:16], dtype=np.float32)))
                        info['ann']['rotation_y'] = np.append(info['ann']['rotation_y'], float(labels[16]))

                    else:
                        info = {}
                        info['frame_id'] = int(frame_id)
                        info['filename'] = os.path.join(frames_dir_path, frame_id.zfill(6)+'.png')
                        info['ann'] = dict(
                            track_id=np.array(labels[1], dtype=np.int64),
                            bboxes=np.array(labels[6:10], dtype=np.float32),
                            labels=np.array(self.class_dict[labels[2]], dtype=np.int64),                           
                            truncated=np.array(labels[3], dtype=np.int64),
                            occluded=np.array(labels[4], dtype=np.int64),
                            alpha=np.array(labels[5], dtype=np.float32),
                            dimensions=np.array(labels[10:13], dtype=np.float32),
                            location=np.array(labels[13:16], dtype=np.float32),
                            rotation_y=np.array(labels[16], dtype=np.float32),    
                            )
                        img_infos[frames_dir].append(info)
        
        for frames_dir in img_infos.keys():
            self.track_infos += [frame_info for frame_info in img_infos[frames_dir]]

    def __len__(self):
        return len(self.track_infos)
        

    def __getitem__(self, index):
        """
        Return:
            data (tensor): a image
            bboxes (tensor): shape: `(num_object, 4)`
                box = bboxes[:, :4]， label = bboxes[:, 4]
            index (int): image index
        """

        img_path = self.track_infos[index]['filename']
        if not osp.exists(img_path):
            logging.error("Cannot found image data: " + img_path)
            raise FileNotFoundError
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.track_infos[index]['height'], self.track_infos[index]['width'] = img.shape[:2]

        frame_id = self.track_infos[index]['frame_id']
        track_id = self.track_infos[index]['ann']['track_id']
        bboxes = self.track_infos[index]['ann']['bboxes']
        labels = self.track_infos[index]['ann']['labels']
        num_object = len(bboxes)
        no_object = False
        if num_object == 0:
            # no gt boxes
            no_object = True
            bboxes = np.array([0, 0, 0, 0]).reshape(-1, 4)
            labels = np.array([0]).reshape(-1, 1)
            track_id = np.array([0]).reshape(-1, 1)
        else:
            bboxes = bboxes.reshape(-1, 4)
            labels = labels.reshape(-1, 1)
            track_id = track_id.reshape(-1, 1)
        if self.transform is not None:
            img, bboxes, labels = self.transform(
                img, bboxes, labels, no_object
            )
        bboxes = np.hstack((bboxes, labels, track_id)) #labels, track_id right after bboxes
        bboxes = torch.from_numpy(bboxes.astype(np.float32))

        return img, bboxes, frame_id, index


if __name__ == "__main__":
    dataset = ObjtrackDataset('/Users/chenmingfei/Downloads/Hwang_Papers/TNT_pytorch/data/training')
    data = dataset.__getitem__(154)
    print(data)