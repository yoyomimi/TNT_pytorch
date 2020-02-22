# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import logging

from datasets.transform import PredictionTransform
from detection.utils.metrics import eval_fcos_det
from detection.utils.metrics import run_fcos_det_example
from units.trainer import BaseTrainer

class FCOSTrainer(BaseTrainer):

    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 lr_scheduler,
                 criterion,
                 log_dir='output',
                 performance_indicator='mAP',
                 last_iter=-1,
                 rank=0):

        super().__init__(cfg, model, optimizer, lr_scheduler, criterion,
                         log_dir, performance_indicator, last_iter, rank)
        
    def _read_inputs(self, inputs):
        imgs, targets, track_id, index = inputs
        imgs = imgs.cuda(non_blocking=True)
        # targets are list type in det tasks
        targets = [target.cuda(non_blocking=True) for target in targets]
        return imgs, targets  

    def _forward(self, data):
        self.model.train()
        imgs = data[0]
        targets = data[1]
        cls_score, bbox_pred, centerness = self.model(imgs)

        loss = self.criterion(cls_score, bbox_pred, centerness, targets)
        num_pos = loss['num_pos']
        if num_pos > 0:
            cls_loss = loss['loss_cls'] / num_pos
            reg_loss = loss['loss_bbox'] / num_pos
            centerness_loss = loss['loss_centerness'] / num_pos

        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            logging.info(f"Iterations {self.iter:d} "
                         f"lr {self.lr_scheduler.get_lr()[0]} "
                         f"cls_loss = {cls_loss.item()} "
                         f"box_loss = {reg_loss.item()} "
                         f"centerness_loss = {centerness_loss.item()}")

        return (cls_loss + self.cfg.LOSS.LAMDA * reg_loss + centerness_loss), None
    
    def evaluate(self, eval_loader):
        mAP, aps, pr_curves = eval_fcos_det(self.cfg,
            self.criterion,
            eval_loader,
            self.model,
            rescale=None)

        logging.info(f"score_threshold:{self.cfg.TEST.SCORE_THRESHOLD}, "
                     f"nms_iou:{self.cfg.TEST.NMS_THRESHOLD}")
        logging.info(f'mean Average Precision Across All Classes:{mAP}')
        
        if mAP > self.best_performance or True:
            # save PR curve
            if self.cfg.TEST.PR_CURVE:
                for class_idx, pr_curve in enumerate(pr_curves):
                    self.writer.add_figure(f'PR_curve/class_{class_idx + 1}',
                        pr_curve,
                        self.iter
                    )
                logging.info('PR curves saved')
            # save detected result
            if self.cfg.TEST.IMAGE_DIR != '':
                transform = PredictionTransform(resize=cfg.TEST.TEST_SIZE)
                run_fcos_det_example(self.cfg,
                    self.criterion,
                    self.cfg.TEST.IMAGE_DIR,
                    transform,
                    self.model
                )
                logging.info('test detected result saved')
                
        return mAP



        
