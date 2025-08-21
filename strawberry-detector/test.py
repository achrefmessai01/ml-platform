#!/usr/bin/env python3

import argparse
import torch
import mmcv
from mmdet.apis import DetInferencer

# Temporarily monkey patch torch.load for MMEngine checkpoints
# This is the recommended approach for trusted checkpoints in PyTorch 2.6+
_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    # For MMEngine checkpoints, we need to set weights_only=False as they contain
    # custom objects like HistoryBuffer that are safe but not in the default allowlist
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load


def main():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img', help='image file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='bbox score threshold')
    args = parser.parse_args()

    # Initialize the DetInferencer with config and checkpoint
    inferencer = DetInferencer(
        model=args.config, 
        weights=args.checkpoint, 
        device='cuda:0'
    )
    
    # Run inference
    results = inferencer(
        args.img,
        pred_score_thr=args.score_thr,
        no_save_vis=True,  # Don't auto-save
        return_vis=True,  # Return visualization
        show=False
    )
    
    # Save visualization if output path specified
    if args.out:
        visualization = results['visualization'][0]  # Get first (and only) image
        # Convert RGB to BGR for correct color display (mmcv.imwrite expects BGR)
        visualization_bgr = mmcv.rgb2bgr(visualization)
        mmcv.imwrite(visualization_bgr, args.out)
        print(f"Result saved to {args.out}")


if __name__ == '__main__':
    main()
