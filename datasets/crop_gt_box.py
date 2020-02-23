import argparse
import cv2
import json
import numpy as np
import os
import os.path as osp

import sys
sys.path.append(osp.join(osp.dirname(sys.path[0])))

from detection.crop_det import write_crop

def gen_track_gt_dict(data_root, save_dirs):
    """generate track gt dict using gt label files.
        gt{
            video_name (f'{videos_root_path}_{video_name}'):{
                track_id_0: [class_name <str>,  frame_id_list <list>, containing all the involved frame ids in sequence.],
                track_id_1: [class_name <str>,  frame_id_list <list>, containing all the involved frame ids in sequence.]
            }
        }
        e.g.
        gt{
            "data/eval/objtrack/images_0012": {
                "0": ["Cyclist", [0, 1, 2, 3, 4, 5]],
                "1": ["Pedestrian", [1, 2, 3]]
            }
        }

        new_train_crop_img path: data_root/crop_data/training/video_name/track_id/f'{class_name}_{frame_id}_crop.jpg'
    """

    gt = {}

    class_dict = {
        'DontCare': -1,
        'Pedestrian': 0,
        'Car': 1,
        'Cyclist': 2
    }
    videos_dir_list = os.listdir(data_root)
    for videos in videos_dir_list:
        videos_dir = osp.join(data_root, videos, 'images')
        labels_dir = osp.join(data_root, videos, 'labels')
        images_dir_list = os.listdir(videos_dir)
        for frames_dir in images_dir_list:
            # create video_dir
            video_save_path = osp.join(save_dirs, f'{videos}_{frames_dir}')
            gt[f'{videos_dir}_{frames_dir}'] = {}
            gt_video = gt[f'{videos_dir}_{frames_dir}']

            if not osp.exists(video_save_path):
                os.mkdir(video_save_path)
            frames_dir_path = osp.join(videos_dir, frames_dir)
            gt_path = osp.join(labels_dir, frames_dir+'.txt')
            with open(gt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    labels = line.split()
                    if labels[1] == '-1': # DontCare
                        continue
                    
                    frame_id = labels[0]
                    track_id = labels[1]
                    track_save_path = osp.join(video_save_path, track_id)
                    if int(track_id) not in gt_video.keys(): 
                        gt_video[int(track_id)] = [labels[2],[]]
                    gt_track = gt_video[int(track_id)]
                    gt_track[1].append(int(frame_id))

                    if not osp.exists(track_save_path):
                        os.mkdir(track_save_path)
                    img_path = osp.join(frames_dir_path, frame_id.zfill(6)+'.png')
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    class_label = np.array(class_dict[labels[2]]+1, dtype=np.int64).reshape(-1, 1)
                    box = np.array(labels[6:10], dtype=np.float32).reshape(-1, 4)
                    write_crop(img, box, class_label, track_save_path, frame_id=frame_id)

    return gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run gt_crop")
    parser.add_argument(
        "--data_root",
        type=str,
        default='',
        help='you can choose a directory contain training and eval directories')
    parser.add_argument(
        "--dst_root",
        type=str,
        default='data/crop_data',
         help='you can choose a directory root to save new training and eval directories')
    args = parser.parse_args()


    # create crop_dir training/trackid/frameid.jpg eval/trackid/frameid.jpg
    if not osp.exists(args.dst_root):
        os.mkdir(args.dst_root)
    dst_train_root = osp.join(args.dst_root, 'training')
    if not osp.exists(dst_train_root):
        os.mkdir(dst_train_root)
    dst_eval_root = osp.join(args.dst_root, 'eval')
    if not osp.exists(dst_eval_root):
        os.mkdir(dst_eval_root)


    train_gt_dict = gen_track_gt_dict(osp.join(args.data_root, 'training'), dst_train_root)
    gt_save_path_train = osp.join(dst_train_root, 'gt_train_track_dict.json')
    with open(gt_save_path_train, 'a') as f_train:
        json.dump(train_gt_dict, f_train, ensure_ascii=False)
        f_train.write('\n')

    eval_gt_dict = gen_track_gt_dict(osp.join(args.data_root, 'eval'), dst_eval_root)
    gt_save_path_eval = osp.join(dst_eval_root, 'gt_eval_track_dict.json')
    with open(gt_save_path_eval, 'a') as f_eval:
        json.dump(eval_gt_dict, f_eval, ensure_ascii=False)
        f_eval.write('\n')
    
    # scan the crop_imgs, generate and save anchor-pos-neg triplet pair list 


