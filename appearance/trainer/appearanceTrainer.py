# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import logging
from tqdm import tqdm

import torch.nn.functional as F  

from units.trainer import BaseTrainer


class appearanceTrainer(BaseTrainer):

    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 lr_scheduler,
                 criterion,
                 log_dir='output',
                 performance_indicator='triplet_loss',
                 last_iter=-1,
                 rank=0,
                 replace_iter=None,
                 replace_model_name=None,
                 use_triplet=True):
        super().__init__(cfg, model, optimizer, lr_scheduler, criterion, log_dir, 
            performance_indicator, last_iter, rank, replace_iter, replace_model_name)

        self.use_triplet = use_triplet

    def _read_inputs(self, inputs):
        # post transform for identifier
        anchor_imgs, pos_imgs, neg_imgs = inputs
        anchor_imgs = anchor_imgs.cuda(non_blocking=True)
        pos_imgs = pos_imgs.cuda(non_blocking=True)
        neg_imgs = neg_imgs.cuda(non_blocking=True)
        return anchor_imgs, pos_imgs, neg_imgs 

    def _forward(self, data):
        self.model.train()
        anchor_imgs, pos_imgs, neg_imgs = data
        anchor_embeddings = self.model(anchor_imgs)
        pos_embeddings = self.model(pos_imgs)
        neg_embeddings = self.model(neg_imgs)
        if self.use_triplet:
            loss = self.criterion(anchor_embeddings, pos_embeddings, neg_embeddings)
        else:
            loss = self.criterion(cls_score, targets)

        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            logging.info(f"Iterations {self.iter:d} ")

        return loss, None
    
    def evaluate(self, eval_loader):
        self.model.eval()
        sum_loss = 0.0
        for data in tqdm(eval_loader):
            anchor_imgs, pos_imgs, neg_imgs = data
            anchor_imgs = anchor_imgs.cuda(non_blocking=True)
            pos_imgs = pos_imgs.cuda(non_blocking=True)
            neg_imgs = neg_imgs.cuda(non_blocking=True)
            anchor_embeddings = self.model(anchor_imgs)
            pos_embeddings = self.model(pos_imgs)
            neg_embeddings = self.model(neg_imgs)
            sum_loss += self.criterion(anchor_embeddings, pos_embeddings, neg_embeddings).data.cpu()
              
        avg_loss = sum_loss / len(eval_loader)
        logging.info(f'Facenet triplet loss: {avg_loss}')

        return avg_loss



        


        
