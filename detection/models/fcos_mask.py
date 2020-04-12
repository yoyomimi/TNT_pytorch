# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-20
# ------------------------------------------------------------------------------
import torch
from torch import nn

from detection.heads.fcos_head import FCOSHead
from detection.heads.mask_head import MaskFCHead
from detection.heads.mask_iou_head import MaskIoUHead
from detection.necks.FPN import FPN
from detection.roi_aligns.single_roi_align import SingleRoIExtractor
from detection.utils.box_utils import bbox2roi

from appearance.backbones.resnet import ResNet
from appearance.heads.mask_emb_head import MaskEmbHead
from appearance.utils.resnet_load_pkl import load_pkl, load_state_dict

from extensions import RoIAlign

from utils.utils import get_det_criterion

class FCOSMask(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg
        self.base = ResNet(cfg)
        if cfg.MODEL.BACKBONE.PRETRAINED:
            assert cfg.MODEL.BACKBONE.WEIGHTS != ''
            loaded = load_pkl(cfg, cfg.MODEL.BACKBONE.WEIGHTS)
            if "model" not in loaded:
                loaded = dict(model=loaded)
            load_state_dict(self.base, loaded.pop("model"))
            
        self.neck = FPN(
            in_channels=cfg.MODEL.NECK.IN_CHANNELS,
            out_channels=cfg.MODEL.NECK.OUT_CHANNELS,
            start_level=cfg.MODEL.NECK.START_LEVEL,
            add_extra_convs=cfg.MODEL.NECK.ADD_EXTRA_CONVS,
            extra_convs_on_inputs=cfg.MODEL.NECK.EXTRA_CONVS_ON_INPUTS,
            num_outs=cfg.MODEL.NECK.NUM_OUTS,
            relu_before_extra_convs=cfg.MODEL.NECK.RELU_BEFORE_EXTRA_CONVS
        )

        self.head = FCOSHead(
            num_classes=cfg.DATASET.NUM_CLASSES, 
            strides=cfg.MODEL.HEAD.STRIDES,
            in_channels=cfg.MODEL.HEAD.IN_CHANNELS,
            feat_channels=cfg.MODEL.HEAD.OUT_CHANNELS,
            stacked_convs=cfg.MODEL.HEAD.STACKED_CONVS
        )
        
        self.criterion = get_det_criterion(cfg)

        self.mask_roi_align = SingleRoIExtractor(
            roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
            out_channels=256,
            featmap_strides=[8, 16],
        )
        self.emb_roi_align = SingleRoIExtractor(
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[8, 16, 32],
        )

        self.mask_head = MaskFCHead(8)
        self.mask_iou_head = MaskIoUHead(8)
        self.mask_emb_head = MaskEmbHead(128, roi_feat_size=7)


    def forward(self, x):
        fpn_in = self.base(x)
        head_in = self.neck(fpn_in)
        pred_det = self.head(head_in) # list 5 level: (cls-8, box-4, centerness-1)
        cls_score, bbox_pred, centerness = pred_det
        result_list, _, _ = self.criterion.get_bboxes(self.cfg,
            cls_score,
            bbox_pred,
            centerness,
            len(x))
        rois = bbox2roi([result[:,3:] for result in result_list]) # [bbox_num, 5]
        mask_feat = self.mask_roi_align(head_in[:self.mask_roi_align.num_inputs], rois) # [bs, c, 14, 14]
        emb_feat = self.emb_roi_align(head_in[:self.emb_roi_align.num_inputs], rois) # [bs, c, 7, 7]
        # mask 
        pred_mask = self.mask_head(mask_feat) # [obj_num, num_class, mask_roi_feat_size, mask_roi_feat_size]
        
        # mask_iou 
        pos_labels = torch.cat([result[:,0] for result in result_list]).long()
        pos_pred_mask = pred_mask[range(pred_mask.size(0)), pos_labels]
        pred_mask_iou = self.mask_iou_head(mask_feat, pos_pred_mask)

        # associate embedding
        pred_emb = self.mask_emb_head(emb_feat)

        return pred_det, pred_mask, pred_mask_iou, pred_emb
