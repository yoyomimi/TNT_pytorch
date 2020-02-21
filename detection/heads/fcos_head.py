# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-20
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from units.units import Scale, fcoshead_conv2d
from utils.utils import multi_apply


class FCOSHead(nn.Module):
    def __init__(self,
                 num_classes=2,
                 in_channels=256, 
                 strides=(16, 32, 64),
                 feat_channels=256,
                 stacked_convs=4,
                ):
        """ FCOSHead. After fpn, predict cls, reg, centerness based on pyramid feats.

        Args:
            num_classes (int): num of classes;
            in_channels (int)

        Forward:
            fpn_out(list): len FPN_Layer_Num(5)  `(BS X In_Channel X W X H)`

        Return:
            cls_score (list): len FPN_Layer_Num  `(BS X Channel X W X H)`
            bbox_pred (list): len FPN_Layer_Num  `(BS X 4 X W X H)`
            centerness (list): len FPN_Layer_Num `(BS X 1 X W X H)`
            
        """

        super().__init__()
        self.strides = strides
        #TODO num_classes-1?
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_dim = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                fcoshead_conv2d(
                    in_dim,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.reg_convs.append(
                fcoshead_conv2d(
                    in_dim,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
    
    def init_weights(self):
        for m in self.cls_convs:
            nn.init.normal_(m[0].weight, std=0.01)
        for m in self.reg_convs:
            nn.init.normal_(m[0].weight, std=0.01)
        prior_prob = 0.01    
        bias_cls = float(-np.log((1 - prior_prob) / prior_prob))
        nn.init.normal_(self.fcos_cls.weight, std=0.01)
        nn.init.constant_(self.fcos_cls.bias, bias_cls)
        nn.init.normal_(self.fcos_reg.weight, std=0.01)
        nn.init.normal_(self.fcos_centerness.weight, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness

if __name__ == "__main__":
    from torch.autograd import Variable
    head = FCOSHead(3, 512)
    x = Variable(torch.randn(5, 2, 512, 32, 32))
    cls_score, bbox_pred, centerness = head(x)
    print(cls_score[0].shape, bbox_pred[0].shape, centerness[0].shape, len(cls_score))

