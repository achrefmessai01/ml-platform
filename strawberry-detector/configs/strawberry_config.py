# Configuration for strawberry detection fine-tuning
# Inherits from Faster R-CNN R50 FPN 1x COCO base config

_base_ = '../venv/lib/python3.10/site-packages/mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# Model configuration - adjust for actual number of classes
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=7)  # 7 classes in your dataset
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
        metainfo=dict(classes=('Flower', 'Flower_Fruit', 'Green_Fruit', 'Green_White_Fruit', 'Red_Fruit', 'UNSURE', 'White_Red_Fruit'))),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        data_root=data_root,
        data_prefix=dict(img='val/'),
        metainfo=dict(classes=('Flower', 'Flower_Fruit', 'Green_Fruit', 'Green_White_Fruit', 'Red_Fruit', 'UNSURE', 'White_Red_Fruit'))),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        data_root=data_root,
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=('Flower', 'Flower_Fruit', 'Green_Fruit', 'Green_White_Fruit', 'Red_Fruit', 'UNSURE', 'White_Red_Fruit')))
)

# Override evaluation to use validation split
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox')

# Work directory
work_dir = './work_dirs/strawberry_detector'
