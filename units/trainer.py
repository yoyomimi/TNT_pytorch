import logging
import sys
import time

import torch
from tensorboardX import SummaryWriter

from utils.utils import AverageMeter
from utils.utils import save_checkpoint
from torch import autograd

class BaseTrainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 lr_scheduler,
                 criterion,
                 log_dir,
                 performance_indicator='mAP',
                 last_iter=-1,
                 rank=0,
                 replace_iter=None,
                 replace_model_name=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.log_dir = log_dir
        self.PI = performance_indicator
        self.rank = rank
        self.iter = last_iter + 1
        self.epoch = 0
        self.best_performance = 0.0
        self.is_best = False
        self.writer = SummaryWriter(log_dir, comment=f'_rank{rank}')
        self.max_iter = replace_iter if replace_iter else self.cfg.TRAIN.MAX_ITERATIONS
        self.replace_model_name = replace_model_name
        self.model_name = replace_model_name if replace_model_name else self.cfg.MODEL.NAME
        logging.info(f"max iterations = {self.max_iter} ")

        if self.PI == 'triplet_loss':
            self.best_performance = 1e10
        # TODO init model weights

    def _read_inputs(self, inputs):
        imgs, targets, index = inputs
        imgs = imgs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        return imgs, targets    

    def _forward(self, data):
        imgs = data[0]
        targets = data[1]
        pred = self.model(imgs)

        loss, train_prec = self.criterion(pred, targets)
        return loss, train_prec

    def train(self, train_loader, eval_loader):
        losses_1 = AverageMeter()
        losses_2 = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        end_time = time.time()
        if self.iter > self.max_iter:
            logging.info("Optimization is done !")
            sys.exit(0)
        for data in train_loader:
            self.model.train()
            # forward
            data_time.update(time.time() - end_time)
            data = self._read_inputs(data)
            # get loss
            loss, train_prec = self._forward(data)
            if isinstance(loss, tuple):
                losses_1.update(loss[0].item())
                losses_2.update(loss[1].item())
                total_loss = sum(loss)
            else:
                losses_1.update(loss.item())
                total_loss = loss
            # optimization
            self.optimizer.zero_grad()
            with autograd.detect_anomaly():
                total_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            # time for training(forward & loss computation & optimization) on one batch
            batch_time.update(time.time() - end_time)

            # log avg loss
            if self.iter > 0 and self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
                if isinstance(loss, tuple):
                    self.writer.add_scalar('loss/cls', losses_1.avg, self.iter)
                    self.writer.add_scalar('loss/box', losses_2.avg, self.iter)
                    loss_msg = f'avg_cls_loss:{losses_1.avg:.04f} avg_box_loss:{losses_2.avg:.04f}'
                else:
                    if self.replace_model_name:
                        self.writer.add_scalar(f'{self.model_name}_loss', losses_1.avg, self.iter)
                        loss_msg = f'avg_{self.model_name}_loss:{losses_1.avg:.04f}'  
                    else:
                        self.writer.add_scalar('loss', losses_1.avg, self.iter)
                        loss_msg = f'avg_loss:{losses_1.avg:.04f}'  

                logging.info(
                    f'epoch:{self.epoch:03d} '
                    f'{loss_msg:s} '
                    f'io_rate:{data_time.avg / batch_time.avg:.04f} '
                    f'samples/(gpu*s):{self.cfg.DATASET.IMG_NUM_PER_GPU / batch_time.avg:.02f}'
                )

                self.writer.add_scalar('speed/samples_per_second_per_gpu',
                                        self.cfg.DATASET.IMG_NUM_PER_GPU / batch_time.avg,
                                        self.iter)
                self.writer.add_scalar('speed/io_rate',
                                        data_time.avg / batch_time.avg,
                                        self.iter)
                if train_prec is not None:
                    logging.info(f'train precision: {train_prec}')
                losses_1.reset()
                losses_2.reset()
            
            # save checkpoint
            if self.iter > 0 and self.iter % self.cfg.TRAIN.SAVE_INTERVAL == 0:
                # evaluation
                if self.cfg.TRAIN.VAL_WHEN_TRAIN:
                    self.model.eval()
                    performance = self.evaluate(eval_loader)
                    self.writer.add_scalar(self.PI, performance, self.iter)
                    if self.PI == 'triplet_loss' and performance < self.best_performance:
                        self.is_best = True
                        self.best_performance = performance
                    elif performance > self.best_performance:
                        self.is_best = True
                        self.best_performance = performance
                    else:
                        self.is_best = False
                    logging.info(f'Now: best {self.PI} is {self.best_performance}')
                else:
                    performance = -1

                # save checkpoint
                try:
                    state_dict = self.model.module.state_dict() # remove prefix of multi GPUs
                except AttributeError:
                    state_dict = self.model.state_dict()

                if self.rank == 0:
                    if self.cfg.TRAIN.SAVE_EVERY_CHECKPOINT:
                        filename = f"{self.model_name}_epoch{self.epoch:03d}_iter{self.iter:06d}_checkpoint.pth"
                    else:
                        filename = "checkpoint.pth"
                    save_checkpoint(
                        {
                            'iter': self.iter,
                            'model': self.model_name,
                            f'performance/{self.PI}': performance,
                            'state_dict': state_dict,
                            'optimizer': self.optimizer.state_dict(),
                        },
                        self.is_best,
                        self.log_dir,
                        filename=filename
                    )
            
            self.iter += 1
            end_time = time.time()
        self.epoch += 1

    def evaluate(self, eval_loader):
        raise NotImplementedError