# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F  

from tracklets.utils.utils import get_tracklet_pair_input_features
from units.trainer import BaseTrainer


class trackletpairConnectTrainer(BaseTrainer):

    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 lr_scheduler,
                 criterion,
                 log_dir='output',
                 performance_indicator='acc',
                 last_iter=-1,
                 rank=0,
                 replace_iter=None,
                 replace_model_name=None,
                 pre_ap_model=None):
        super().__init__(cfg, model, optimizer, lr_scheduler, criterion, log_dir, 
            performance_indicator, last_iter, rank, replace_iter, replace_model_name)

        self.pre_ap_model = pre_ap_model

    def _read_inputs(self, inputs):
        img_1, img_2, loc_mat, tracklet_mask_1, tracklet_mask_2, real_window_len, targets = inputs
        tracklet_pair_features = torch.zeros(len(targets), self.cfg.TRACKLET.WINDOW_lEN, self.cfg.MODEL.APPEARANCE.EMB_SIZE).cuda(non_blocking=True)
        img_1 = [img.cuda(non_blocking=True) for img in img_1]
        img_2 = [img.cuda(non_blocking=True) for img in img_2]
        loc_mat = loc_mat.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        tracklet_mask_1 = tracklet_mask_1.cuda(non_blocking=True)
        tracklet_mask_2 = tracklet_mask_2.cuda(non_blocking=True)
        if self.pre_ap_model:
            self.pre_ap_model.eval()
            tracklet_pair_features = get_tracklet_pair_input_features(self.pre_ap_model, img_1, img_2, loc_mat,
                tracklet_mask_1, tracklet_mask_2, real_window_len, tracklet_pair_features)
        return tracklet_pair_features, targets

    def _forward(self, data):
        self.model.train()
        tracklet_pair_features, targets = data
        cls_score = self.model(tracklet_pair_features)
        loss = self.criterion(cls_score, targets)

        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            logging.info(f"Iterations {self.iter:d} ")

        return loss, None
    
    def evaluate(self, eval_loader):
        self.model.eval()
        acc = 0.0
        count = 0
        for data in tqdm(eval_loader):
            count = count + 1
            if count > 1:
                break
            tracklet_pair_features, targets = self._read_inputs(data)
            cls_score = self.model(tracklet_pair_features).data.cpu()
            pred = cls_score.ge(0.5).type(torch.FloatTensor)
            true = pred.eq(targets.data.cpu().view_as(pred)).numpy()
            acc += np.sum(true) / len(pred)
        acc /= min(count, len(eval_loader))
        logging.info(f'Pred tracklet pair connectivity accuracy:{acc}')
        print(f"acc : %.3f" %(acc))

        return acc




        


        
