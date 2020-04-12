# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2=4-11
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np


class MaskIoUHead(nn.Module):
    def __init__(self,
                 num_classes=8,
                 in_channels=256, 
                 feat_channels=256,
                 stacked_convs=4,
                 stacked_fcs=2,
                 roi_feat_size=14,
                 fc_out_channels=1024
                ):
        """ MaskIoUHead. After ROI Align, on roi mask feats.

        Return:
            pred_mask_iou: (obj_num, num_class)
            
        """

        super().__init__()
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.stacked_fcs = stacked_fcs
        self.fc_out_channels = fc_out_channels
        self.roi_feat_size = roi_feat_size
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.mask_iou_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_dim = self.in_channels + 1 if i == 0 else self.feat_channels
            stride = 2 if i == self.stacked_convs - 1 else 1
            self.mask_iou_convs.append(
                nn.Conv2d(
                    in_dim,
                    self.feat_channels,
                    3,
                    stride=stride,
                    padding=1))
        
        flatten_area = (self.roi_feat_size // 2) *  (self.roi_feat_size // 2)
        self.mask_iou_fcs = nn.ModuleList()
        for i in range(self.stacked_fcs):
            in_channels = self.feat_channels * flatten_area if i == 0 else self.fc_out_channels
            self.mask_iou_fcs.append(nn.Linear(in_channels, self.fc_out_channels))

        self.pred_iou_fc = nn.Linear(self.fc_out_channels, self.cls_out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
    
    def init_weights(self):
        for m in self.mask_iou_convs:
            nn.init.normal_(m.weight, std=0.01)
        for m in self.mask_iou_fcs:
            nn.init.kaiming_normal_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.pred_iou_fc.weight, std=0.01)
        nn.init.constant_(self.pred_iou_fc.bias, 0)

    def forward(self, mask_feat, pred_mask):
        pred_mask = pred_mask.sigmoid()
        pred_mask_pooled = self.max_pool(pred_mask.unsqueeze(1))

        x = torch.cat((mask_feat, pred_mask_pooled), 1)

        for conv in self.mask_iou_convs:
            x = self.relu(conv(x))
        x = x.view(x.size(0), -1)
        for fc in self.mask_iou_fcs:
            x = self.relu(fc(x))
        mask_iou = self.pred_iou_fc(x)
        return mask_iou
