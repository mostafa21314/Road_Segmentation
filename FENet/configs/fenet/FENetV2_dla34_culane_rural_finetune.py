net = dict(type='Detector', )

backbone = dict(
    type='DLAWrapper',
    dla='dla34',
    pretrained=False,  # weights come from --load_from checkpoint, not ImageNet
)

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)

heads = dict(type='FEHeadV2',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

Piou_loss_weight = 4.
cls_loss_weight = 4.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
Dliou_loss_weight = 1
Driou_loss_weight = 1

work_dirs = "work_dirs/fenetv2/dla34_culane_rural_finetune"

neck = dict(type='FEFPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

epochs = 15
batch_size = 8  # smaller batch -> more gradient steps per epoch on a small dataset

# Rural subset: 2440 training images
# Lower LR for finetuning (original full-training LR was 0.6e-3 for batch 24,
# 3e-4 for batch 8 — we use 10x smaller for finetuning)
train_images = 2440
optimizer = dict(type='AdamW', lr=2e-4)
total_iter = (train_images // batch_size) * epochs  # ~4575 iters
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 1   # evaluate every epoch (short run, don't want to miss the best)
save_ep = 5   # save checkpoint every 5 epochs

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/CULane'
dataset_type = 'CULane'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 4
log_interval = 50
seed = 0
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
