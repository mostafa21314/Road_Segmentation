# dataset settings for CULane (binary: background + lane)
dataset_type = 'CULaneDataset'
data_root = 'data/culane/'
gt_txt_dir = 'CULane_Rural_Subset(1)/CULane_Rural_Subset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
classes = ('background', 'lane')
palette = [[0, 0, 0], [255, 255, 255]]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1640, 590), ratio_range=(0.8, 1.5)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1664, 576),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=20,
        dataset=dict(
            type='CustomDataset',
            data_root=data_root,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        gt_txt_dir=gt_txt_dir + '/val',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        gt_txt_dir=gt_txt_dir + '/test',
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))
