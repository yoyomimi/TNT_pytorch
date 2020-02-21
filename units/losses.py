import torch
import torch.nn.functional as F   

def iou_loss(confidence, label, reduction='sum', weight=1.0, eps=1e-6):
        """IoU loss, Computing the IoU loss between a set of predicted bboxes and target bboxes.
        The loss is calculated as negative log of IoU.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            eps (float): Eps to avoid log(0).

        """

        rows = confidence.size(0)
        cols = label.size(0)
        assert rows == cols
        if rows * cols == 0:
            return confidence.new(rows, 1)
        lt = torch.max(confidence[:, :2], label[:, :2])  # [rows, 2]
        rb = torch.min(confidence[:, 2:], label[:, 2:])  # [rows, 2]
        wh = (rb - lt + 1).clamp(min=0)                  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (confidence[:, 2] - confidence[:, 0] + 1) * (
            confidence[:, 3] - confidence[:, 1] + 1)
        area2 = (label[:, 2] - label[:, 0] + 1) * (
            label[:, 3] - label[:, 1] + 1)
        ious = overlap / (area1 + area2 - overlap)
        safe_ious = ious.clamp(min=eps)
        loss = -safe_ious.log() * weight
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('reduction can only be `mean` or `sum`')
        
def sigmoid_crossentropy_loss(confidence, label, reduction='sum'):
    """CrossEntropy Loss.

    Args:
        confidence (tensor): `(batch_size, num_priors, num_classes)`
        label (tensor): `(batch_size, num_priors)`

    """
    loss = F.binary_cross_entropy_with_logits(
        confidence, label.float(), reduction=reduction)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError('reduction can only be `mean` or `sum`')

def sigmoid_focal_loss(confidence, 
                       label, 
                       num_classes,
                       alpha=0.25, 
                       gamma=2, 
                       reduction='sum'):
    """Focal Loss, normalized by the number of positive anchor.

    Args:
            confidence (tensor): `(batch_size, num_priors, num_classes)` 0-79
            label (tensor): `(batch_size, num_priors)` 0-80

    """
    assert reduction in ['mean', 'sum']

    pred_sigmoid = confidence.sigmoid()
    #one_hot码左移一位，背景的情况直接做负样本，confidence只有1-80的类
    label = F.one_hot(label.long(), num_classes=num_classes)[:,1:].type_as(confidence)
    pt = (1 - pred_sigmoid) * label + pred_sigmoid * (1 - label)  # 1 - pt in paper
    focal_weight = (alpha *label + (1 - alpha) * (1 - label)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        confidence, label, reduction='none') * focal_weight

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError('reduction can only be `mean` or `sum`')