import torch
import torch.nn as nn
import numpy as np

from units.units import fcoshead_conv2d


class MaskFCHead(nn.Module):
    def __init__(self,
                 num_classes=8,
                 in_channels=256, 
                 feat_channels=256,
                 stacked_convs=4,
                 upsample_ratio=2,
                ):
        """ MaskFCNHead. After ROI Align, on roi feats.

        Return:
            pred_mask: (obj_num, num_class, upsample_ratio*roi_feat_size, upsample_ratio*roi_feat_size)
            
        """

        super().__init__()
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.upsample_ratio = upsample_ratio
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_dim = self.in_channels if i == 0 else self.feat_channels
            self.mask_convs.append(
                fcoshead_conv2d(
                    in_dim,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
        
        # deconv to upsample
        upsample_in_channels = self.feat_channels if self.stacked_convs > 0 else self.in_channels
        self.upsample = nn.ConvTranspose2d(
            upsample_in_channels,
            self.feat_channels,
            self.upsample_ratio,
            stride=self.upsample_ratio)

        logits_in_channel = self.feat_channels
        self.logits = nn.Conv2d(logits_in_channel, self.cls_out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def init_weights(self):
        for m in self.mask_convs:
            nn.init.normal_(m[0].weight, std=0.01)
        for m in [self.upsample, self.logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for mask_layer in self.mask_convs:
            x = mask_layer(x)
        x = self.upsample(x)
        x = self.relu(x)
        pred_mask = self.logits(x)
        return pred_mask
        
        
if __name__ == "__main__":
    from torch.autograd import Variable
    head = MaskFCHead(8)
    x = Variable(torch.randn(20, 256, 14, 14))
    pred_mask = head(x)
    print(pred_mask.shape)

