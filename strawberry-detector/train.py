#!/usr/bin/env python3

import argparse
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmcv import Config
import mmcv
import os.path as osp


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true',
                        help='whether not to evaluate during training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    
    # Build model
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    # Add meta information
    model.CLASSES = datasets[0].CLASSES
    
    # Train model
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=(not args.no_validate),
        timestamp=None,
        meta=dict()
    )


if __name__ == '__main__':
    main()
