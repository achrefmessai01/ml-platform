# Configuration for strawberry detection fine-tuning
# Inherits from Faster R-CNN R50 FPN 1x COCO base config

_base_ = '../venv/lib/python3.10/site-packages/mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# Model configuration - adjust for single class detection
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)  # Single "strawberry" class
    )
)

# Dataset configuration - point to your actual dataset
dataset_type = 'CocoDataset'
data_root = '/home/vgreens/SSD_drive_1.8T/datasets/crop_cell/detection/coco_organized/'

# Override dataset paths to use proper splits with separate image folders
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        data_root=data_root,
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=('strawberry',))),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        data_root=data_root,
        data_prefix=dict(img='val/'),
        metainfo=dict(classes=('strawberry',))),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        data_root=data_root,
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=('strawberry',)))
)

# Override evaluation to use validation split
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox')

# Training hyperparameters optimized for strawberry detection
# Learning rate and schedule
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))  # Lower LR for fine-tuning

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,  # Shorter training for initial test
        by_epoch=True,
        milestones=[12, 16],
        gamma=0.1)
]

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=1)  # Validate every 1 epoch
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Checkpoint and logging with early stopping
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),  # Log every 20 iterations (more frequent)
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5, 
        save_best='auto', 
        rule='greater',
        max_keep_ckpts=3  # Keep only best 3 checkpoints to save space
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=False),  # Enable visualization hooks
)

# Add early stopping
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=8,  # Stop if no improvement for 8 validation cycles (16 epochs)
        min_delta=0.001  # Minimum improvement threshold
    )
]

# Enhanced logging
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# Visualization backend (MLflow for experiment tracking)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        exp_name='strawberry_detection',
        run_name='faster_rcnn_r50_fpn',
        tags={'model': 'faster_rcnn', 'dataset': 'strawberry', 'classes': '1'},
        params={'lr': 0.005, 'batch_size': 2, 'epochs': 20},
        save_dir='work_dirs/strawberry_detector/mlflow_logs'
    )
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Completely override train_dataloader to avoid inheritance issues
train_dataloader = dict(
    _delete_=True,  # Delete inherited configuration
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoDataset',
                ann_file='/home/vgreens/SSD_drive_1.8T/datasets/crop_cell/detection/coco_organized/annotations/instances_train.json',
                data_root='/home/vgreens/SSD_drive_1.8T/datasets/crop_cell/detection/coco_organized/',
                data_prefix=dict(img='train/'),
                metainfo=dict(classes=('strawberry',)),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                    dict(type='RandomFlip', prob=0.5),
                    dict(type='PackDetInputs'),
                ]
            ),
            dict(
                type='CocoDataset',
                ann_file='/home/vgreens/SSD_drive_1.8T/datasets/disease_data/detection/coco_strawberries/train/_annotations.coco.json',
                data_root='/home/vgreens/SSD_drive_1.8T/datasets/disease_data/detection/coco_strawberries/',
                data_prefix=dict(img='train/'),
                metainfo=dict(classes=('strawberry',)),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                    dict(type='RandomFlip', prob=0.5),
                    dict(type='PackDetInputs'),
                ]
            )
        ]
    ))

val_dataloader = dict(
    _delete_=True,  # Delete inherited configuration
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoDataset',
                ann_file='/home/vgreens/SSD_drive_1.8T/datasets/crop_cell/detection/coco_organized/annotations/instances_val.json',
                data_root='/home/vgreens/SSD_drive_1.8T/datasets/crop_cell/detection/coco_organized/',
                data_prefix=dict(img='val/'),
                metainfo=dict(classes=('strawberry',)),
                test_mode=True,
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
                ]
            ),
            dict(
                type='CocoDataset',
                ann_file='/home/vgreens/SSD_drive_1.8T/datasets/disease_data/detection/coco_strawberries/valid/_annotations.coco.json',
                data_root='/home/vgreens/SSD_drive_1.8T/datasets/disease_data/detection/coco_strawberries/',
                data_prefix=dict(img='valid/'),
                metainfo=dict(classes=('strawberry',)),
                test_mode=True,
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
                ]
            )
        ]
    ))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        data_root=data_root,
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=('strawberry',)),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
        ]))

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox')

# Work directory
work_dir = './work_dirs/strawberry_detector'

# Load from pretrained COCO model
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
