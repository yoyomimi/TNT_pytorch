# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-20
# ------------------------------------------------------------------------------
import torch
from torch import nn

from detection.heads.fcos_head import FCOSHead
from detection.necks.FPN import FPN
from appearance.backbones.resnet import ResNet
from appearance.utils.resnet_load_pkl import load_pkl, load_state_dict

class FCOS(nn.Module):

    def __init__(self, cfg):

        super().__init__()
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
        
    def forward(self, x):
        fpn_in = self.base(x)
        head_in = self.neck(fpn_in)
        out = self.head(head_in)
        return out