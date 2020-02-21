import torch

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)

def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (..., 2): left top corner.
        right_bottom (..., 2): right bottom corner.
        the same size
        
    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)  # clamp把负值变成0
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """Compute the iou_of overlap of two sets of corner boxes.  The iou_of overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        
    Args:
        boxes0: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        boxes1: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        
    Return:
        iou_of overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """

    overlap_left_top = torch.max(
        boxes0[..., :2], boxes1[..., :2])  # (b, a, 2) 
    overlap_right_bottom = torch.min(boxes0[..., 2:],
                                     boxes1[..., 2:])  # (b,a,2)

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores(tensor): shape: `(N, 5)`: boxes in corner-form and probabilities.
        iou_threshold(float): intersection over union threshold.
        top_k(int): keep top_k results. If k <= 0, keep all the results.
        candidate_size(int): only consider the candidates with the highest scores.
        
    Returns:
         picked: a list of indexes of the kept boxes
    """

    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]  
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < (top_k == len(picked)) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]
        
    return box_scores[picked, :]

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_thr,
                   max_num=-1,
                   score_factors=None,
                   pre_nms=1000):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """

    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    #编号1-80 num_classes=81 不过阈值再判为背景0
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        #乘上centerness
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1) #(n, 5)

        cls_dets = hard_nms(  # (y, 5) y是每一类经过nms后剩下的box数
                cls_dets, nms_thr, max_num, pre_nms)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i,
                                           dtype=torch.long)
        
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        
    return bboxes, labels
