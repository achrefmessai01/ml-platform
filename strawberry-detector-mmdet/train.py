#!/usr/bin/env python3

import argparse
import os.path as osp
from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', help='resume from checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    
    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    # Set resume
    if args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    
    # Set random seed
    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed)
    
    # Build runner and start training
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
