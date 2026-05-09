"""SegMAN-Tiny — binary road segmentation on IDD-20k-II."""

_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/idd20k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_adamw.py',
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_t',
        pretrained='pretrained/SegMAN_Encoder_t.pth.tar',
        style='pytorch',
    ),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
        dropout_ratio=0.1,
        num_classes=2,                   # background + road
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            # Up-weight road class because it covers less area than background
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                 class_weight=[1.0, 3.0]),
            dict(type='DiceLoss', loss_weight=3.0, ignore_index=255),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)),
)

# Fine-tune AdamW — slightly higher LR than CityScapes since we start from
# the pre-trained encoder and fine-tune on a smaller domain-specific dataset.
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
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
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

data = dict(samples_per_gpu=4, workers_per_gpu=4)

evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
