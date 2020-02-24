# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-24
# ------------------------------------------------------------------------------
import torch
from torch import nn
from torch.autograd import Variable

from appearance.backbones.inception_resnet_v1 import InceptionResnetV1
from utils.utils import load_eval_model

class TrackletConnectivity(nn.Module):
    def __init__(self, cfg, max_emb_bs=2, emb_dim=512):

        super().__init__()
        self.emb = InceptionResnetV1(pretrained='vggface2', classify=False)
        assert cfg.MODEL.APPEARANCE.WEIGHTS != ''
        load_eval_model(cfg.MODEL.APPEARANCE.WEIGHTS, self.emb)
        self.emb.eval()

        self.max_emb_bs = max_emb_bs
        self.emb_dim = 512

    def forward(self, img_1, img_2, loc_mat, tracklet_mask_1, tracklet_mask_2):
        assert len(img_1) == len(img_2)
        bs = len(img_1)
        window_len = loc_mat.shape[1]
        img_emb = Variable(torch.zeros(bs, window_len, self.emb_dim)).cuda(non_blocking=True)
        for i in range(bs):
            img_bs = img_1[i].shape[0]
            p_frame = 0
            while img_bs > self.max_emb_bs:
                img_emb[i][p_frame:p_frame+self.max_emb_bs] = self.emb(img_1[i][p_frame:p_frame+self.max_emb_bs])
                p_frame += self.max_emb_bs
                img_bs -= self.max_emb_bs
            if img_bs == 1:
                new_input = torch.stack([img_1[i][p_frame:p_frame+img_bs][0], img_1[i][p_frame:p_frame+img_bs][0]])
                img_emb[i][p_frame:p_frame+img_bs] = self.emb(new_input)[-1]
            else:
                img_emb[i][p_frame:p_frame+img_bs] = self.emb(img_1[i][p_frame:p_frame+img_bs])

            img_bs = img_2[i].shape[0]
            p_frame = 0
            while img_bs > self.max_emb_bs:
                img_emb[i][window_len-p_frame-self.max_emb_bs:window_len-p_frame] = self.emb(img_2[i][p_frame:p_frame+self.max_emb_bs])
                p_frame += self.max_emb_bs
                img_bs -= self.max_emb_bs
            if img_bs == 1:
                new_input = torch.stack([img_2[i][p_frame:p_frame+img_bs][0], img_2[i][p_frame:p_frame+img_bs][0]])
                img_emb[i][window_len-p_frame-img_bs:window_len-p_frame] = self.emb(new_input)[-1]
            else:
                img_emb[i][window_len-p_frame-img_bs:window_len-p_frame] = self.emb(img_2[i][p_frame:p_frame+img_bs])
            