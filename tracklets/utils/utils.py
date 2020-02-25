# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-25
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd

import torch


def get_embeddings(model, frame_feat, max_emb_bs=16):
    """use appearance inference model to get embeddings from frames in fram_list.
       Args:
           model: pretrained appearance model, eval mode for inference.
           frame_feat: (frame_num, 3, size, size).
           max_emb_bs: the maximun of the frames fed to the appearance model one time as a batch.

       Return:
           img_emb_output: (frame_num, emb_dim).
    """
    img_bs = frame_feat.shape[0]
    p_frame = 0
    img_emb = []
    while img_bs > max_emb_bs:
        img_input = frame_feat[p_frame:p_frame+max_emb_bs]
        img_emb.append(model(img_input))
        p_frame += max_emb_bs
        img_bs -= max_emb_bs
    img_input = frame_feat[p_frame:p_frame+img_bs]
    if img_bs == 1:
        new_input = torch.stack([img_input[0], img_input[0]])
        img_emb.append(model(new_input)[-1].view(1, -1))
    else:
        img_emb.append(model(img_input))
    img_emb_output = torch.cat(img_emb)

    return img_emb_output


def get_tracklet_pair_input_features(model, img_1, img_2, loc_mat, tracklet_mask_1, tracklet_mask_2, real_window_len, tracklet_pair_features):
    """Reference to the paper "https://arxiv.org/pdf/1811.07258.pdf" Section 4.2 Multi-Scale TrackletNet.
       Args:
           model: pretrained appearance model, eval mode for inference.
           img_1: one batch of tracklet_1_frames, <list>, len eqs batchsize; for each item, <torch.Tensor>, (frame_num, 3, size, size)
           img_2: one batch of tracklet_2_frames, <list>, len eqs batchsize; for each item, <torch.Tensor>, (frame_num, 3, size, size)
           loc_mat: (bs, window_len, 4), interpolated in the dataset file, indicates the location for each tracklet.
           tracklet_mask_1: (bs, window_len, 1), detection status, 1 when there is an object detected.
           tracklet_mask_2: (bs, window_len, 1), detection status, 1 when there is an object detected.
           tracklet_pair_features: (bs, window_len, emb_dim)
        
       Return:
           tracklet_pair_input: (bs, 4+emb_dim, window_len, 3)
    """
    assert len(img_1) == len(img_2) == len(tracklet_pair_features)
    bs, window_len, emb_dim = tracklet_pair_features.shape

    # extract embedding using pretrained appearance model
    for i in range(bs):
        real_win_len_now = real_window_len[i]
        emb_1 = get_embeddings(model, img_1[i])
        frame_len_1 = len(emb_1)
        emb_2 = get_embeddings(model, img_2[i])
        frame_len_2 = len(emb_2)
        tracklet_pair_features[i][:frame_len_1] = emb_1.detach()
        tracklet_pair_features[i][real_win_len_now-frame_len_2:real_win_len_now] = emb_2.detach()

    tracklet_pair_features_np = tracklet_pair_features.cpu().numpy()

    # interpolate the embeddings
    for i in range(bs):
        real_win_len_now = real_window_len[i]
        feat_np = tracklet_pair_features_np[i]
        feat_np[0][np.where(feat_np[0]==0)] = 1e-5
        feat_np[-1][np.where(feat_np[-1]==0)] = 1e-5
        feat_pd = pd.DataFrame(data=feat_np).replace(0, np.nan, inplace=False)
        feat_pd_np = np.array(feat_pd.interpolate()).astype(np.float32)
        if real_win_len_now < window_len:
            feat_pd_np[real_win_len_now:] = np.zeros((window_len-real_win_len_now, emb_dim)).astype(np.float32)
        tracklet_pair_features[i] = torch.from_numpy(feat_pd_np)
        
    tracklet_pair_features = tracklet_pair_features.cuda(non_blocking=True) # (bs, window_len, emb_dim)

    # cat loc_mat (bs, window_len, emb_dim+4, 1)
    tracklet_pair_features_with_loc = torch.cat([loc_mat, tracklet_pair_features], dim=-1).unsqueeze(-1)

    # expand det mask (bs, window_len, emb_dim+4, 1)
    
    tracklet_mask_1_input = tracklet_mask_1.expand(-1, -1, emb_dim+4).unsqueeze(-1)
    tracklet_mask_2_input = tracklet_mask_2.expand(-1, -1, emb_dim+4).unsqueeze(-1)

    # cat pair_input (bs, window_len, emb_dim+4, 3)
    tracklet_pair_input = torch.cat([tracklet_pair_features_with_loc, tracklet_mask_1_input, tracklet_mask_2_input], dim=-1)
    
    return tracklet_pair_input




    
        
       

    
    