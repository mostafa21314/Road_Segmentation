"""SegMAN-Tiny — CARLA-only fine-tune, frozen early backbone, 4-way eval.
Train on CARLA train split (80% of frames). Validate during training on IDD val
so best_mIoU = best real-world generalisation. The training script also runs
post-training eval on CARLA val (20%) and IDD val. Uses a separate data root
(`data/carla_split/`) to avoid colliding with the un-split `data/carla/` used
by other CARLA experiments.
"""

_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_adamw.py',
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_t',
        pretrained=None,        # weights come from cfg.load_from (set by script)
        style='pytorch',
    ),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                 class_weight=[1.0, 3.0]),
            dict(type='DiceLoss', loss_weight=3.0, ignore_index=255),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
classes  = ('background', 'road')
palette  = [[0, 0, 0], [128, 64, 128]]

# CARLA frames are 1280x720
carla_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

idd_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1024, 576),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ]),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_root='data/carla_split/',
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=carla_train_pipeline,
    ),
    # Validate during training on IDD val → "best ckpt" tracks real-world quality
    val=dict(
        type='CustomDataset',
        data_root='data/idd20k/',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=idd_test_pipeline,
    ),
    # Placeholder — the script overrides at eval time with both IDD and CARLA val
    test=dict(
        type='CustomDataset',
        data_root='data/idd20k/',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=idd_test_pipeline,
    ),
)

# Lower LR than from-scratch — fine-tune from a converged IDD checkpoint
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm':      dict(decay_mult=0.),
        'head':      dict(lr_mult=10.),
    }),
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# 4000 CARLA train images, batch 4 → 1000 iters/epoch. 16k iters ≈ 16 epochs.
runner            = dict(type='IterBasedRunner', max_iters=16000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation        = dict(interval=2000, metric='mIoU', save_best='mIoU')
