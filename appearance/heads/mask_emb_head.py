# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2=4-11
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from units.units import fcoshead_conv2d


class MaskEmbHead(nn.Module):
    def __init__(self,
                 emb_size=128,
                 in_channels=256, 
                 feat_channels=256,
                 stacked_convs=4,
                 roi_feat_size=7,
                 fc_out_channels=1024,
                 stacked_fcs=2
                ):
        """ MaskEmbHead. After ROI Align, on emb roi feats.

        Return:
            emb: (obj_num, emb_size)
            
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.stacked_fcs = stacked_fcs
        self.fc_out_channels = fc_out_channels
        self.roi_feat_size = roi_feat_size
        self.emb_size = emb_size
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.mask_emb_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_dim = self.in_channels if i == 0 else self.feat_channels
            self.mask_emb_convs.append(
                fcoshead_conv2d(
                    in_dim,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
        
        flatten_area = self.roi_feat_size *  self.roi_feat_size
        self.mask_emb_fcs = nn.ModuleList()
        for i in range(self.stacked_fcs):
            in_channels = self.feat_channels * flatten_area if i == 0 else self.fc_out_channels
            self.mask_emb_fcs.append(nn.Linear(in_channels, self.fc_out_channels))

        self.emb_fc = nn.Linear(self.fc_out_channels, self.emb_size)
        self.relu = nn.ReLU()
    
    def init_weights(self):
        for m in self.mask_emb_convs:
            nn.init.normal_(m[0].weight, std=0.01)
        for m in self.mask_emb_fcs:
            nn.init.kaiming_normal_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.emb_fc.weight, std=0.01)
        nn.init.constant_(self.emb_fc.bias, 0)

    def forward(self, x):
        for conv in self.mask_emb_convs:
            x = self.relu(conv(x))
        x = x.view(x.size(0), -1)
        for fc in self.mask_emb_fcs:
            x = self.relu(fc(x))
        pred_emb = self.emb_fc(x)
        return pred_emb
