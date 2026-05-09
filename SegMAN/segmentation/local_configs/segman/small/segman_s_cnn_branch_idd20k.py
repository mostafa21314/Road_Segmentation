"""
segman_s_cnn_branch_idd20k.py
SegMAN-Small on IDD-20k-II — binary road segmentation (background / road).

This config mirrors the segman_s_culane experiment (Small encoder whose
SegMANDecoder has explicit CNN convolutional branches: conv_downsample_2 and
conv_downsample_4) but targets the Indian Driving Dataset (IDD-20k-II) and
trains for 20 000 iterations rather than 40 000.

CNN-branch detail inside SegMANDecoder
---------------------------------------
  forward_winssm() runs two parallel processing paths:
    · SSM path  – VSSM block over multi-scale unshuffled features
    · CNN path  – conv_downsample_2  (3×3, stride 2)
                  conv_downsample_4  (5×5, stride 4)
                  pixel_unshuffle    (bring all to 1/4 resolution)
                  reduce_channels    (1×1 convs, one per scale)
  The CNN-path features are concatenated with SSM-path features before
  final fusion, giving the model both local-texture (CNN) and long-range
  sequential (SSM) representations.
"""

_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/idd20k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_adamw.py',  # overridden below
]

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
        # SegMAN-Small encoder produces 4-scale features at these channel widths
        in_channels=[64, 144, 288, 512],
        in_index=[0, 1, 2, 3],
        channels=152,           # decoder hidden dim (matches CULane-Small run)
        feat_proj_dim=288,      # projection dim for MLP projections
        dropout_ratio=0.1,
        num_classes=2,          # 0 = background, 1 = road
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            # CE: strong recall signal; up-weight road (class 1) — road occupies
            # less area than background in IDD scenes
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                 class_weight=[1.0, 3.0]),
            # Dice: directly penalises FP → tighter road mask predictions
            dict(type='DiceLoss', loss_weight=3.0, ignore_index=255),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)),
)

# ── Optimiser ─────────────────────────────────────────────────────────────────
# Same AdamW recipe as the CULane-Small run; slightly higher LR than the
# standard ADE20k schedule because we fine-tune from a pretrained encoder on a
# domain-specific, smaller dataset.
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
    warmup_iters=1000,    # shorter warm-up for 20 k schedule
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# ── Training schedule ─────────────────────────────────────────────────────────
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')

data = dict(samples_per_gpu=4, workers_per_gpu=4)
