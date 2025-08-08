#!/usr/bin/env python3

import argparse
from mmdet.apis import inference_detector, init_detector
import mmcv


def main():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img', help='image file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='bbox score threshold')
    args = parser.parse_args()

    # Build model
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    
    # Test image
    result = inference_detector(model, args.img)
    
    # Show result
    img = mmcv.imread(args.img)
    model.show_result(img, result, score_thr=args.score_thr, out_file=args.out)


if __name__ == '__main__':
    main()
