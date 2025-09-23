# Configuration for strawberry detection fine-tuning
# Inherits from Faster R-CNN R50 FPN 1x COCO base config

_base_ = "../venv/lib/python3.10/site-packages/mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"

# Model configuration - adjust for actual number of classes
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=1))  # 1 class for strawberry detection
)

# Dataset configuration - point to your actual dataset
dataset_type = "CocoDataset"
data_root = "/home/vgreens/SSD_drive_1.8T/datasets/merged_data/detection/combined_coco/"

single_class_metainfo = dict(classes=("object",))

# Override dataset paths to use proper splits with separate image folders
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_train.json",
        data_root=data_root,
        data_prefix=dict(img="train/"),
        metainfo=single_class_metainfo,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val.json",
        data_root=data_root,
        data_prefix=dict(img="val/"),
        metainfo=single_class_metainfo,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_test.json",
        data_root=data_root,
        data_prefix=dict(img="test/"),
        metainfo=single_class_metainfo,
    ),
)

# Override evaluation to use validation split
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val.json",
    metric="bbox",
)

# Training hyperparameters optimized for strawberry detection
# Learning rate and schedule
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001),
)  # Lower LR for fine-tuning

param_scheduler = [
    # Shorter warmup period (2 epochs instead of 5)
    dict(
        type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=200
    ),  # ~2 epochs
    dict(
        type="MultiStepLR",
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[30, 40, 45],  # 60%, 80%, 90% of training
        gamma=0.1,
    ),
]

# Training settings
train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=50, val_interval=1
)  # Validate every epoch
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Checkpoint and logging
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(
        type="LoggerHook", interval=20
    ),  # Log every 20 iterations (more frequent)
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        save_best="auto",
        rule="greater",
        max_keep_ckpts=3,
    ),  # Save every 5 epochs
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(
        type="DetVisualizationHook", draw=False
    ),  # Enable visualization hooks
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP_50",  # Metric to monitor
        patience=5,  # Stop if no improvement for 5 validation intervals
        rule="greater",  # Higher mAP is better
        min_delta=0.005,  # Minimum change to qualify as improvement
        strict=False,  # Don't stop on first epoch
        check_finite=True,  # Stop if metric becomes NaN/inf
        stopping_threshold=None,  # Optional: stop if metric reaches threshold
    ),
)

# Enhanced logging
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"

# Visualization backend (MLflow for experiment tracking)
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="MLflowVisBackend",
        exp_name="strawberry_detection",
        run_name="faster_rcnn_r50_fpn_single_class",
        tags={"model": "faster_rcnn", "dataset": "strawberry", "classes": "1"},
        params={"lr": 0.005, "batch_size": 4, "epochs": 50},
        save_dir="work_dirs/strawberry_detector/mlflow_logs",
    ),
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# IMPROVED: Enhanced data augmentation pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # Additional augmentations for better generalization
    dict(type='PhotoMetricDistortion'),  # Color augmentation
    dict(type='Expand', mean=[103.53, 116.28, 123.675], to_rgb=False, ratio_range=(1, 2)),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(type='PackDetInputs'),
]

val_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# Data loading - Override dataloaders to use our strawberry dataset
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_train.json",
        data_root=data_root,
        data_prefix=dict(img="train/"),
        metainfo=single_class_metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=None),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="Resize", scale=(1333, 800), keep_ratio=True),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PackDetInputs"),
        ],
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val.json",
        data_root=data_root,
        data_prefix=dict(img="val/"),
        metainfo=single_class_metainfo,
        test_mode=True,
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=None),
            dict(type="Resize", scale=(1333, 800), keep_ratio=True),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="PackDetInputs",
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
            ),
        ],
    ),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_test.json",
        data_root=data_root,
        data_prefix=dict(img="test/"),
        metainfo=single_class_metainfo,
        test_mode=True,
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=None),
            dict(type="Resize", scale=(1333, 800), keep_ratio=True),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="PackDetInputs",
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
            ),
        ],
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
    format_only=False
)

# Work directory
work_dir = "./work_dirs/strawberry_detector"

# Load from pretrained COCO model
load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
