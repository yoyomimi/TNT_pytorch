import importlib
import logging
import numpy as np
import os
import os.path as osp
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import ConcatDataset

from bisect import bisect_right
from functools import partial
from six.moves import map, zip

from datasets.transform import TrainTransform
from datasets.transform import EvalTransform
from units.losses import sigmoid_crossentropy_loss

class AverageMeter(object):
    """Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def resource_path(relative_path):
    """To get the absolute path"""
    base_path = osp.abspath(".")

    return osp.join(base_path, relative_path)


def ensure_dir(root_dir, rank=0):
    if not osp.exists(root_dir) and rank == 0:
        print(f'=> creating {root_dir}')
        os.mkdir(root_dir)
    else:
        while not osp.exists(root_dir):
            print(f'=> wait for {root_dir} created')
            time.sleep(10)

    return root_dir


def create_logger(cfg, rank=0):
    # working_dir root
    abs_working_dir = resource_path('work_dirs')
    working_dir = ensure_dir(abs_working_dir, rank)
    # output_dir root
    output_root_dir = ensure_dir(os.path.join(working_dir, cfg.OUTPUT_ROOT), rank)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    final_output_dir = ensure_dir(os.path.join(output_root_dir, time_str), rank)
    # set up logger
    logger = setup_logger(final_output_dir, time_str, rank)

    return logger, final_output_dir


def setup_logger(final_output_dir, time_str, rank, phase='train'):
    log_file = f'{phase}_{time_str}_rank{rank}.log'
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def get_model(cfg, file, name):
    module = importlib.import_module(file)
    model = getattr(module, name)(cfg)

    return model

    
def get_optimizer(cfg, model):
    """Support two types of optimizers: SGD, Adam.
    """
    assert (cfg.TRAIN.OPTIMIZER in [
        'sgd',
        'adam',
    ])
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            nesterov=cfg.TRAIN.NESTEROV)
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    return optimizer


def load_checkpoint(cfg, model, optimizer, replace_path=None):
    last_iter = -1
    resume_path = replace_path if replace_path else cfg.MODEL.RESUME_PATH
    resume = True if replace_path else cfg.TRAIN.RESUME
    if resume_path and resume:
        if osp.exists(resume_path):
            checkpoint = torch.load(resume_path)
            # resume
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                logging.info(f'==> model resumed from {resume_path}')
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info(f'==> optimizer resumed, continue training')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            if 'iter' in checkpoint:
                last_iter = checkpoint['iter']
                logging.info(f'==> last_iter = {last_iter}')
            # pre-train
            else:
                model.load_state_dict(checkpoint)
                logging.info(f'==> model pretrained from {resume_path} \n'
                             f'==> optimizer start from scratch, fine-tuning')
        else:
            logging.error(f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    else:
        logging.info("==> train model without resume")

    return model, optimizer, last_iter


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def get_lr_scheduler(cfg, optimizer, last_epoch=-1):
    """Support three types of optimizers: StepLR, MultiStepLR, MultiStepWithWarmup.
    """
    assert (cfg.TRAIN.LR_SCHEDULER in [
        'StepLR',
        'MultiStepLR',
        'MultiStepWithWarmup',
    ])
    if cfg.TRAIN.LR_SCHEDULER == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.TRAIN.LR_STEPS[0],
            cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch)
    elif cfg.TRAIN.LR_SCHEDULER == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_STEPS,
            cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch)
    elif cfg.TRAIN.LR_SCHEDULER == 'MultiStepWithWarmup':
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.TRAIN.LR_STEPS,
            cfg.TRAIN.LR_FACTOR,
            cfg.TRAIN.WARMUP_INIT_FACTOR,
            cfg.TRAIN.WARMUP_STEP,
            last_epoch)
    else:
        raise AttributeError(f'{cfg.TRAIN.LR_SCHEDULER} is not implemented')
    
    return lr_scheduler

def get_criterion(cfg):
    module = importlib.import_module(cfg.LOSS.FILE)
    critertion = getattr(module, cfg.LOSS.NAME)(cfg,
        cfg.DATASET.NUM_CLASSES,
        cfg.MODEL.HEAD.STRIDES,
    )
    return critertion

def get_det_trainer(cfg, model, optimizer, lr_scheduler,
                criterion, log_dir, last_iter, rank):
    module = importlib.import_module(cfg.TRAINER.FILE)
    Trainer = getattr(module, cfg.TRAINER.NAME)(
        cfg,
        model,
        optimizer,
        lr_scheduler,
        criterion,
        log_dir,
        performance_indicator=cfg.PI,
        last_iter=last_iter,
        rank=rank
    )
    return Trainer

def list_to_set(data_list, name='train'):
    if len(data_list) == 0:
        dataset = None
        logging.warning(f"{name} dataset is None")
    elif len(data_list) == 1:
        dataset = data_list[0]
    else:
        dataset = ConcatDataset(data_list)
        
    if dataset is not None:
        logging.info(f'==> the size of {name} dataset is {len(dataset)}')
    return dataset

    
def get_dataset(cfg, is_ident=False):
    if is_ident:
        train_transform = PersonTrainTransform(cfg.IDENTIFIER.SIZE, cfg.DATASET.MEAN)
        eval_transform = PersonEvalTransform(cfg.IDENTIFIER.SIZE, cfg.DATASET.MEAN)
        module = importlib.import_module(cfg.IDENTIFIER.DATASET_FILE)
        Dataset = getattr(module, cfg.IDENTIFIER.DATASET_NAME)
    else:
        train_transform = TrainTransform([cfg.TRAIN.INPUT_MIN, cfg.TRAIN.INPUT_MAX], cfg.DATASET.MEAN)
        eval_transform = EvalTransform(cfg.TEST.TEST_SIZE, cfg.DATASET.MEAN)
        module = importlib.import_module(cfg.DATASET.FILE)
        Dataset = getattr(module, cfg.DATASET.NAME)

    data_root = cfg.DATASET.ROOT # abs path in yaml
    # get train data list
    train_root = osp.join(data_root, 'train')
    train_set = [d for d in os.listdir(train_root) if osp.isdir(osp.join(train_root, d))]  
    train_list = []
    for sub_set in train_set:
        train_sub_root = osp.join(train_root, sub_set)
        logging.info(f'==> load train sub set: {train_sub_root}')
        train_sub_set = Dataset(train_sub_root, train_transform)
        train_list.append(train_sub_set)
    # get eval data list
    eval_root = osp.join(data_root, 'val')
    eval_set = [d for d in os.listdir(eval_root) if osp.isdir(osp.join(eval_root, d))]
    eval_list = []      
    for sub_set in eval_set:
        eval_sub_root = osp.join(eval_root, sub_set)
        logging.info(f'==> load val sub set: {eval_sub_root}')
        eval_sub_set = Dataset(eval_sub_root, eval_transform)
        eval_list.append(eval_sub_set)
    # concat dataset list
    train_dataset = list_to_set(train_list, 'train')
    eval_dataset = list_to_set(eval_list, 'eval')
    
    return train_dataset, eval_dataset

def get_identifier_trainer_args(cfg, model, output_dir, proc_rank):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.IDENTIFIER.LR,
        weight_decay=cfg.IDENTIFIER.WEIGHT_DECAY)

    replace_path = cfg.IDENTIFIER.WEIGHTS if cfg.IDENTIFIER.RESUME else None
    model, optimizer, last_iter = load_checkpoint(cfg, model, optimizer, replace_path=replace_path)

    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.IDENTIFIER.LR_STEPS,
        cfg.IDENTIFIER.LR_FACTOR,
        cfg.IDENTIFIER.WARMUP_INIT_FACTOR,
        cfg.IDENTIFIER.WARMUP_STEP,
        last_iter)
    
    criterion = sigmoid_crossentropy_loss

    return dict(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        log_dir=output_dir,
        performance_indicator='acc',
        last_iter=last_iter,
        rank=proc_rank,
        replace_iter=cfg.IDENTIFIER.MAX_ITERATIONS,
        replace_model_name=cfg.IDENTIFIER.FILE
    )

def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    logging.info(f'save model to {output_dir}')
    if is_best:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))

def load_eval_model(resume_path, model):
    if resume_path != '':
        if osp.exists(resume_path):
            print(f'==> model load from {resume_path}')
            checkpoint = torch.load(resume_path)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    return model

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def naive_np_nms(dets, thresh):  
    """Pure Python NMS baseline."""  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = x1.argsort()[::-1]  
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  
    return dets[keep]