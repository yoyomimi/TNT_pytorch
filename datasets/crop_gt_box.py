import argparse
import os
import os.path as osp

from detection.crop_det import write_crop

def gen_track_gt_dict(data_root, save_dirs)
    # gt_dict[video_path][trackid] = [frameid1, frameid2...] 
    gt_dict = {}

    # images_dir_list = os.listdir(self.images_dir)
    #     for frames_dir in images_dir_list:
    #         img_infos[frames_dir] = []
    #         frames_dir_path = os.path.join(self.images_dir, frames_dir)
    #         gt_path = os.path.join(self.labels_dir, frames_dir+'.txt')
    #         with open(gt_path, 'r') as f:
    #             lines = f.readlines()
    #             for line in lines:
    #                 labels = line.split()
    #                 if labels[1] == '-1': # DontCare
    #                     continue
    #                 frame_id = labels[0]

    # write_crop(img, boxes, labels, frame_crop_path)

    return gt_dict

parser = argparse.ArgumentParser(description="run gt_crop")
parser.add_argument(
    "--data_root",
    type=str,
    default='',
    help='you can choose a directory contain training and eval directories')
parser.add_argument(
    "--dst_root",
    type=str,
    default='',
    help='you can choose a directory root to save new training and eval directories')
args = parser.parse_args()


# create crop_dir training/trackid/frameid.jpg eval/trackid/frameid.jpg
train_gt_dict = gen_track_gt_dict(osp.join(args.data_root, 'training'), osp.join(args.dst_root, 'training'))
eval_gt_dict = gen_track_gt_dict(osp.join(args.data_root, 'eval'), osp.join(args.dst_root, 'eval'))

# scan the crop_imgs, generate and save anchor-pos-neg triplet pair list 

