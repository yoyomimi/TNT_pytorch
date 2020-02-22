# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import argparse
import random

import pprint
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import _init_paths
from configs import cfg
from configs import update_config

# from datasets.data_collect import triplet_collect #TODO
from appearance.backbones.inception_resnet_v1 import InceptionResnetV1
from utils.utils import create_logger
from utils.utils import get_model
from utils.utils import get_optimizer
from utils.utils import get_lr_scheduler
from utils.utils import get_criterion
from utils.utils import get_trainer
from utils.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Appearance embedding extraction.')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/facenet_triplet_appearance.yaml',
        required=True,
        type=str)
    
    # default distributed training
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')
    parser.add_argument(
        '--world-size',
        dest='world_size',
        default=1,
        type=int,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--rank',
        default=0,
        type=int,
        help='node rank for distributed training, machine level')
    parser.add_argument(
        '--dist-url',
        dest='dist_url',
        default='tcp://127.0.0.1:23456',
        type=str,
        help='url used to set up distributed training')
    
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
          
    return args


def main_per_worker(process_index, ngpus_per_node, args):
    update_config(cfg, args)
    
    # torch seed
    torch.cuda.manual_seed(random.random())

    # cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    #proc_rank
    proc_rank = args.rank * ngpus_per_node + process_index

    #create logger
    logger, output_dir = create_logger(cfg, proc_rank)
    # logger.info(pprint.pformat(args))
    # logger.info(cfg)
    
    model = InceptionResnetV1(pretrained='vggface2', classify=False, path=[cfg.MODEL.FEATURE_PATH, cfg.MODEL.LOGITS_PATH])
    optimizer = get_optimizer(cfg, model)
    model, optimizer, last_iter = load_checkpoint(cfg, model, optimizer)
    lr_scheduler = get_lr_scheduler(cfg, optimizer, last_iter)
    # train_dataset, eval_dataset = get_dataset(cfg)

    # # distribution
    # if args.distributed:
    #     logger.info(
    #         f'Init process group: dist_url: {args.dist_url},  '
    #         f'world_size: {args.world_size}, '
    #         f'machine: {args.rank}, '
    #         f'rank:{proc_rank}'
    #     )
    #     dist.init_process_group(
    #         backend=cfg.DIST_BACKEND,
    #         init_method=args.dist_url,
    #         world_size=args.world_size,
    #         rank=proc_rank
    #     )
    #     torch.cuda.set_device(process_index)
    #     model.cuda()
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[process_index]
    #     )
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         train_dataset
    #     )
    #     batch_size = cfg.DATASET.IMG_NUM_PER_GPU

    # else:
    #     assert proc_rank == 0, ('proc_rank != 0, it will influence '
    #                             'the evaluation procedure')
    #     model = torch.nn.DataParallel(model).cuda()
    #     train_sampler = None
    #     batch_size = cfg.DATASET.IMG_NUM_PER_GPU * ngpus_per_node
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=(train_sampler is None),
    #     drop_last=True,
    #     collate_fn=objtrack_collect,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=True,
    #     sampler=train_sampler
    # )

    # eval_loader = torch.utils.data.DataLoader(
    #     eval_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     collate_fn=objtrack_collect,
    #     num_workers=cfg.WORKERS
    # )
    
    # criterion = get_criterion(cfg)

    # Trainer = get_trainer(
    #     cfg,
    #     model,
    #     optimizer,
    #     lr_scheduler,
    #     criterion,
    #     output_dir,
    #     last_iter,
    #     proc_rank,
    # )

    # while True:
    #     Trainer.train(train_loader, eval_loader)


if __name__ == '__main__':
    args = parse_args()
    args.distributed = (args.world_size > 1 or args.distributed)
    ngpus_per_node = torch.cuda.device_count()
    print(f"Using {ngpus_per_node} gpus in machine {args.rank}")
    print(f"Distributed training = {args.distributed}")

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(
            main_per_worker,
            args=(ngpus_per_node, args),
            nprocs=ngpus_per_node
        )
    else:
        assert args.rank == 0, ('if you do not use distributed training, '
                                'machine rank must be set to 0')
        main_per_worker(
            0,
            ngpus_per_node,
            args,
        )