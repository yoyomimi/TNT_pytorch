# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch

from detection.utils.box_utils import iou_of

def compute_average_precision(precision, recall):
    """It computes average precision based on the definition of Pascal Competition.
    
    It computes the under curve area of precision and recall.
    Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """

    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] -
             recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()

def calculate_ap(id_prob_box, gt_boxes, ap_iou=0.5):
    """calculate average precision for every class over the whole dataset

    Args:
        id_prob_box (tensor): shape: `(total_box, 6)`
        id_prob_box[:, 0]: image_id in dataset
        id_prob_box[:, 1]: box score in this class
        id_prob_box[:, 2:6]: corner box
        gt_boxes (dict {int: tensor}): shape: `image_id: (total_gt_object, 4)`
            corner gt box of this class over whole dataset
        ap_iou (float): the anchor overlap iou with gt > `ap_iou` will be
                treated as True Positive. default: 0.5, AP@50
        Returns:
            ap:
            precision: TP / (TP + FP)
            recall: TP / (all gt)
        """
    total_gt_object = 0
    for k, v in gt_boxes.items():
        total_gt_object += v.size(0)

    image_ids = id_prob_box[:, 0].numpy()
    scores = id_prob_box[:, 1].numpy()
    boxes = id_prob_box[:, 2:6] # to calculate iou

    sorted_indexes = np.argsort(-scores)
    boxes = [boxes[i] for i in sorted_indexes]
    image_ids = [image_ids[i] for i in sorted_indexes]

    tp = np.zeros(id_prob_box.shape[0])
    fp = np.zeros(id_prob_box.shape[0])
    matched = set()
    for i, image_id in enumerate(image_ids):
        box = boxes[i]
        # false positive
        if image_id not in gt_boxes:
            fp[i] = 1
            continue
        gt_box = gt_boxes[image_id]  
        ious = iou_of(box, gt_box)  # (1, num_gt)
        # match the gt with max iou
        max_iou = torch.max(ious).item()
        max_arg = torch.argmax(ious).item()
        #  多个predict box 和一个gt重合，confidence最高的为TP，其他为FP
        if max_iou > ap_iou:
            if (image_id, max_arg) not in matched:
                tp[i] = 1
                matched.add((image_id, max_arg))
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    true_positive = tp.cumsum()
    false_positive = fp.cumsum()

    recall = true_positive / total_gt_object

    precision = true_positive / (true_positive + false_positive)

    return compute_average_precision(precision, recall), \
                   precision, recall

@torch.no_grad()
def eval_fcos_det(cfg, criterion, eval_loader,
                  model, rescale=None):
    model.eval()
    results = []
    img_start_id = 0
    gt_box_label = {}
    for data in tqdm(eval_loader):
        # imgs.shape = (B, H, W), targets: list [(num_object, 5)...]
        # gt_box: target[0: 4], gt_label: target[4]
        imgs, targets, _, _ = data
        batch_size = imgs.size(0)
        imgs = imgs.cuda(non_blocking=True)
        # output shape: (B, num_prior, 4), (B, num_prior, c), (num_prior, 4)
        cls_score, bbox_pred, centerness = model(imgs)
        img_num = len(imgs)
        result_list, img_start_id, gt_box_label = criterion.get_bboxes(cfg,
            cls_score,
            bbox_pred,
            centerness,
            img_num,
            img_start_id,
            targets,
            gt_box_label,
            rescale=rescale)
        results.extend(result_list)
    if results:
        results = torch.cat(results)  # shape = (dataset_num, 7)
    else:
        import warnings
        warnings.warn('no object detected, your code may exist bug')
        return 0.0, [0.0]

    aps = []
    pr_curves = []
    print("Average Precision Per-class:")
        
    for class_index in range(1, cfg.DATASET.NUM_CLASSES):
        # col-1: label, col-2: img_id, col-3~n: prob & box
        class_mask = (results[:, 0] == class_index)
        id_prob_box = results[class_mask][:, 1:]
        # gt box, key:image_id, value: object box
        gt_boxes = {}
        for img_id, target in gt_box_label.items():
            class_mask = (target[:, 4] == class_index)
            box_per_class = target[class_mask][:, :4]
            if box_per_class.shape[0] == 0:
                continue
            gt_boxes[img_id] = box_per_class
        
        if id_prob_box.shape[0] == 0:
            print(f'class_index:{class_index}, no object detect, ap=0.0')
            aps.append(0.0)
            continue
        elif len(gt_boxes) == 0:
            print(f'class_index:{class_index}, no object detect right, ap=0.0')
            aps.append(0.0)
            continue
            
        ap, precision, recall = calculate_ap(
            id_prob_box.cpu(), gt_boxes, cfg.TEST.AP_IOU)
        aps.append(ap)
        print(f"class_index:{class_index}, ap={ap},"
             f"precision={precision[-1]}, recall={recall[-1]}")

        fig = plt.figure()
        plt.plot(recall, precision, color='red')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Class {class_index}')
        pr_curves.append(fig)
    
    mAP = sum(aps) / len(aps)
    print(f"mAP : {mAP}")
    return mAP, aps, pr_curves

@torch.no_grad()
def run_fcos_det_example(cfg, criterion, jpg_path, transform, model, demo_frame=None, is_crop=False):
    model.eval()
    if demo_frame is None:
        orig_image = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    else:
        orig_image = demo_frame
    h, w, c = orig_image.shape
    img = transform(orig_image)
    _, new_h, new_w = img.shape
    img = img.unsqueeze(0).cuda()
    cls_score, bbox_pred, centerness = model(img)
    result_list, _, _ = criterion.get_bboxes(cfg,
        cls_score,
        bbox_pred,
        centerness,
        len(img))
    result_list = torch.cat(result_list)
    boxes = result_list[:,3:]
    labels = result_list[:,0]
    probs =  result_list[:,2]
    new_boxes = []
    class_dict = {
        0: 'DontCare',
        1: 'Pedestrian',
        2: 'Car',
        3: 'Cyclist',
    }
    for i in range(boxes.size(0)):
        box = boxes[i]
        box[0:4:2] = box[0:4:2] / new_w * w
        box[1:4:2] = box[1:4:2] / new_h * h
        new_boxes.append(box.cpu().data.numpy())
        if demo_frame is None and is_crop == False:
            cv2.rectangle(orig_image, (box[0], box[1]),
                         (box[2], box[3]), (0, 255, 9), 4)
            label_idx = int(labels[i])
            if label_idx in [0, 1, 2, 3]:
                label = class_dict[label_idx]
            else:
                label = 'other'
            cv2.putText(
                orig_image,
                label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 0),
                2)  # line type

    if demo_frame is not None or is_crop:
        return orig_image, np.array(new_boxes), labels.cpu().data.numpy()

    return orig_image

