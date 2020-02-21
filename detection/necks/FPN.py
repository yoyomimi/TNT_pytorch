# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (chenmingfei.lasia@bytedance.com)
# Created On: 2020-1-8
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from units.units import Scale, fcosfpn_conv2d

class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=True,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=True,
                 ):
        """ FCOSFPN. FPN based on lateral backbone, top-down paramid with extra pyramid layers.

        Args:
            in_channels (list): Channel num of input from the backbone
            out_channels (int): FPN output channel num
            num_outs (int): FPN output layer num
            end_level (int, -1): FPN layer num based on in_channel list length
            add_extra_convs (bool): Extra FPN layers
            extra_convs_on_inputs (bool): Extra FPN layers based on backbone input or FPN P5 layer
            relu_before_extra_convs (bool): ReLU activation before each extra layer

        Forward:
            fpn_in(list): len FPN_In_Layer_Num(4)  `([BS X In_Channel_1 X W_1 X H_1, ...])`

        Return:
            fpn_out(list): len FPN_Out_Layer_Num(5)  `([BS X Out_Channel X W_1 X H_1, ...])`
            
        """
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # backbone for top features
            l_conv = fcosfpn_conv2d(
                in_channels[i],
                out_channels,
                1)
            # fpn backbone for bottom features
            fpn_conv = fcosfpn_conv2d(
                out_channels,
                out_channels,
                3,
                padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = fcosfpn_conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1)
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # build laterals C3 C4 C5
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # upsample P4 P5
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')
        # build outputs
        # append the bottom two outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # append P6 P7
        if self.num_outs > len(outs):
            # when no extra layers, use max pool to get more levels on top of outputs
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                # append the mid layer
                if self.extra_convs_on_inputs:
                    # apply the top input layer as base layer for extra convs
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    # appy P5 as base layer for extra convs
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                # append two top layers
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # apply ReLU before extra convs
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

if __name__ == "__main__":
    from torch.autograd import Variable
    neck = FPN(in_channels=[256, 512, 1024, 2048],
               out_channels=256,
               start_level=1,
               add_extra_convs=True,
               extra_convs_on_inputs=False,
               num_outs=5,
               relu_before_extra_convs=True)
    x = []
    in_dim = 256
    w = 200
    h = 256
    for i in range (4):
        x.append(Variable(torch.randn(2, in_dim, w, h)))
        in_dim = in_dim * 2
        w = int(w / 2)
        h = int(h / 2)
    outs = neck(x)
    for i in range (len(outs)):
        print(outs[i].shape)