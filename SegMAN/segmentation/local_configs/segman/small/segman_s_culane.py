_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/culane_590x590.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_adamw.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_s',
        pretrained='pretrained/SegMAN_Encoder_s.pth.tar',
        style='pytorch',
    ),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[64, 144, 288, 512],
        in_index=[0, 1, 2, 3],
        channels=152,
        feat_proj_dim=288,
        dropout_ratio=0.1,
        num_classes=2,  # background + lane
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            _delete_=True,
            type='ProximityWeightedCELoss',
            lane_weight=15.0,
            proximity_weight=8.0,
            proximity_radius=10,
            edge_multiplier=4.0,    # edge pixels get their weight * 4
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

# optimizer
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'head': dict(lr_mult=10.)
    }))

lr_config = dict(
    _delete_=True, policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=4, workers_per_gpu=4)
evaluation = dict(interval=2000, metric=['mIoU', 'mFscore'], save_best='mIoU')

custom_hooks = [
    dict(
        type='VisualizationHook',
        img_path='data/culane/img_dir/val/driver_100_30frame_05250434_0295.MP4_00060.jpg',
        ann_path='data/culane/ann_dir/val/driver_100_30frame_05250434_0295.MP4_00060.png',
        out_dir='work_dirs/segman_s_culane/visualizations',
        palette=[[0, 0, 0], [255, 0, 0]],
        interval=2000,
        priority='LOW',
    ),
]
