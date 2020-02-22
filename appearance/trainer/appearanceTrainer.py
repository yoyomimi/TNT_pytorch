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
                 performance_indicator='loss_delta',
                 last_iter=-1,
                 rank=0,
                 replace_iter=None,
                 replace_model_name=None):

        super().__init__(cfg, model, optimizer, lr_scheduler, criterion, log_dir, 
            performance_indicator, last_iter, rank, replace_iter, replace_model_name)
        
    def _read_inputs(self, inputs):
        # post transform for identifier
        imgs, labels, index = inputs
        imgs = imgs.cuda(non_blocking=True)
        labels = F.one_hot(labels.long(), num_classes=self.cfg.IDENTIFIER.NUM_CLASSES)
        targets = labels.cuda(non_blocking=True)
        return imgs, targets  

    def _forward(self, data):
        self.model.train()
        imgs = data[0]
        targets = data[1]
        cls_score = self.model(imgs)
        loss = self.criterion(cls_score, targets)

        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            logging.info(f"Iterations {self.iter:d} ")


        return loss, None
    
    def evaluate(self, eval_loader):
        self.model.eval()
        acc = 0.0
        count = 0
        for data in tqdm(eval_loader):
            if count > 50:
                break
            count = count + 1
            imgs = data[0]
            imgs = imgs.cuda(non_blocking=True)
            cls_score = model(imgs).sigmoid()
            pred = cls_score.data.cpu().max(1, keepdim=True)[1]
            targets = data[1].data
            true = pred.eq(targets.view_as(pred))
            acc += true.sum() / len(pred)
        acc /= min(count, len(eval_loader))
        logging.info(f'Identifier accuracy:{acc}')
        print(f"acc : {acc}")

        return acc



        


        
