"""
train_segman_carla_freeze_finetune.py
=====================================
CARLA-only fine-tune of SegMAN-Tiny with frozen early backbone stages,
plus 4-way before/after evaluation on CARLA val + IDD val.

Strategy
--------
  - 80/20 deterministic split of the 5000 CARLA frames by sorted filename
    → 4000 train / 1000 val. Stored at  data/carla_split/  to avoid clashing
    with the un-split  data/carla/  used by the other CARLA experiments.
  - Freeze the first N backbone stages (default 3 → ~23% of params frozen,
    77% trainable). Override with --freeze-stages.
  - Start from the best IDD checkpoint.
  - Validate during training on IDD val so the "best" checkpoint is the one
    that generalises best to real-world data, not the one that overfits CARLA.
  - After training, evaluate on both IDD val and CARLA val. Combined with the
    "before" eval on the same two sets this yields a 4-cell table:
        IDD    before / after   → real-world drift (catastrophic forgetting?)
        CARLA  before / after   → synthetic-domain learning
        delta(IDD) vs delta(CARLA) → over/underfit signal

Run from the SegMAN root:
    conda run -n segmanaser python train_segman_carla_freeze_finetune.py
Flags:
    --baseline-ckpt  IDD checkpoint to start from
    --carla-dir      CARLA dataset root (rgb/, seg/)
    --work-dir       Output dir (default: work_dirs/segman_t_carla_freeze_finetune)
    --config         Config path
    --freeze-stages  N in {0..4}, default 3
    --train-frac     Fraction of CARLA used for training (default 0.8)
    --gpu-id, --seed, --iters, --resume-from
    --eval-only      Skip training; needs --after-ckpt
    --after-ckpt     Checkpoint for after-eval (with --eval-only)
    --no-before      Skip the baseline eval step
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEG_DIR    = os.path.join(SCRIPT_DIR, 'segmentation')
sys.path.insert(0, SEG_DIR)
os.chdir(SEG_DIR)

# Pre-load PyTorch native libs so the selective_scan CUDA ext can resolve them
import ctypes, glob as _glob
_torch_lib = os.path.join(sys.prefix, 'lib', 'python3.10',
                          'site-packages', 'torch', 'lib')
for _so in sorted(_glob.glob(os.path.join(_torch_lib, 'lib*.so*'))):
    try: ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
    except OSError: pass

import numpy as np
import torch
import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmcv.utils import Config, get_git_hash

from mmseg import __version__
from mmseg.apis import (init_random_seed, set_random_seed,
                        single_gpu_test, train_segmentor)
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (build_dp, get_device, get_root_logger,
                         setup_multi_processes)
from mmseg.core.evaluation.metrics import pre_eval_to_metrics


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-ckpt',
                   default='work_dirs/segman_t_idd20k/best_mIoU_iter_16000.pth')
    p.add_argument('--carla-dir',
                   default=os.path.join(SCRIPT_DIR, 'Carladataset'))
    p.add_argument('--work-dir',
                   default='work_dirs/segman_t_carla_freeze_finetune')
    p.add_argument('--config',
                   default='local_configs/segman/tiny/segman_t_carla_freeze_finetune.py')
    p.add_argument('--freeze-stages', type=int, default=3,
                   help='Freeze first N backbone stages (0..4). '
                        'Default 3 → ~23%% frozen, 77%% trainable.')
    p.add_argument('--train-frac', type=float, default=0.8,
                   help='Fraction of CARLA frames used for training (rest = val).')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--iters',  type=int, default=None)
    p.add_argument('--resume-from', default=None)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--after-ckpt', default=None)
    p.add_argument('--no-before', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# CARLA prep with deterministic 80/20 split (sorted by filename)
# ---------------------------------------------------------------------------
def prepare_carla_split(carla_dir, train_frac, logger):
    """Symlink RGB + remap mask 255→1 into  data/carla_split/{img,ann}_dir/{train,val}/."""
    rgb_src = os.path.join(carla_dir, 'rgb')
    seg_src = os.path.join(carla_dir, 'seg')
    abs_root = os.path.join(SEG_DIR, 'data', 'carla_split')

    rgb_files = sorted(f for f in os.listdir(rgb_src) if f.endswith('.png'))
    n = len(rgb_files)
    n_train = int(n * train_frac)
    train_files = rgb_files[:n_train]
    val_files   = rgb_files[n_train:]
    logger.info(f'CARLA total {n} → train {len(train_files)} / val {len(val_files)} '
                f'({train_frac:.0%}/{1-train_frac:.0%})')

    from PIL import Image
    n_sym = n_mask = 0
    for split, files in (('train', train_files), ('val', val_files)):
        img_dst = os.path.join(abs_root, 'img_dir', split)
        ann_dst = os.path.join(abs_root, 'ann_dir', split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(ann_dst, exist_ok=True)
        for fname in files:
            di = os.path.join(img_dst, fname)
            if not os.path.exists(di):
                os.symlink(os.path.join(rgb_src, fname), di)
                n_sym += 1
            da = os.path.join(ann_dst, fname)
            if not os.path.exists(da):
                mask = np.array(Image.open(os.path.join(seg_src, fname)))
                Image.fromarray((mask > 127).astype(np.uint8)).save(da)
                n_mask += 1
    logger.info(f'CARLA split prepared: {n_sym} new symlinks, {n_mask} new masks at {abs_root}')


# ---------------------------------------------------------------------------
# Freeze early backbone stages
# ---------------------------------------------------------------------------
def freeze_backbone_stages(model, n_stages, logger):
    """Backbone layers list (Tiny):
        patch_embed
        layers[0] stage1 / layers[1] downsample
        layers[2] stage2 / layers[3] downsample
        layers[4] stage3 / layers[5] downsample
        layers[6] stage4 / layers[7] identity
    n_stages=k freezes patch_embed + layers[0..2k-1].
    """
    if n_stages <= 0:
        logger.info('No backbone layers frozen.')
        return
    bb = model.backbone
    frozen = 0
    for p in bb.patch_embed.parameters():
        p.requires_grad = False; frozen += p.numel()
    n_entries = min(n_stages * 2, len(bb.layers))
    for i in range(n_entries):
        for p in bb.layers[i].parameters():
            p.requires_grad = False; frozen += p.numel()
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Freeze stages={n_stages}  frozen {frozen/1e6:.2f}M ({100*frozen/total:.1f}%)  '
                f'trainable {train/1e6:.2f}M ({100*train/total:.1f}%)')


# ---------------------------------------------------------------------------
# Generic eval — accepts a dataset config dict so we can swap IDD/CARLA
# ---------------------------------------------------------------------------
def run_eval(cfg, ds_cfg, checkpoint_path, gpu_id, logger, label):
    logger.info('=' * 60)
    logger.info(f'EVAL [{label}]   ckpt={checkpoint_path}')
    logger.info('=' * 60)

    ds_cfg = copy.deepcopy(ds_cfg)
    if isinstance(ds_cfg, dict):
        ds_cfg['test_mode'] = True
    else:
        ds_cfg.test_mode = True
    dataset = build_dataset(ds_cfg)
    loader  = build_dataloader(dataset, samples_per_gpu=1,
                               workers_per_gpu=4, dist=False, shuffle=False)

    mc = copy.deepcopy(cfg)
    mc.model.pretrained = None
    mc.model.train_cfg  = None
    model = build_segmentor(mc.model, test_cfg=mc.get('test_cfg'))

    ckpt = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = ckpt['meta'].get('CLASSES', dataset.CLASSES)
    model.PALETTE = ckpt['meta'].get('PALETTE', dataset.PALETTE)
    model = revert_sync_batchnorm(model)
    model = build_dp(model, get_device(), device_ids=[gpu_id])
    model.eval()

    res = single_gpu_test(model, loader, show=False, pre_eval=True)
    ret = pre_eval_to_metrics(res, metrics=['mIoU', 'mDice', 'mFscore'],
                              nan_to_num=0, beta=1)

    iou_bg, iou_road = float(ret['IoU'][0]), float(ret['IoU'][1])
    metrics = dict(
        mIoU           = round((iou_bg + iou_road) / 2, 4),
        IoU_road       = round(iou_road, 4),
        IoU_bg         = round(iou_bg,   4),
        Dice_road      = round(float(ret['Dice'][1]),      4),
        F1_road        = round(float(ret['Fscore'][1]),    4),
        Precision_road = round(float(ret['Precision'][1]), 4),
        Recall_road    = round(float(ret['Recall'][1]),    4),
        PixelAcc       = round(float(ret['aAcc']),         4),
    )
    logger.info(f'--- [{label}] ---')
    for k, v in metrics.items():
        logger.info(f'  {k:<18s}: {v:.4f}')
    return metrics


# ---------------------------------------------------------------------------
# Build the two eval dataset configs (IDD val + CARLA val 20% split)
# ---------------------------------------------------------------------------
def build_eval_dataset_cfgs():
    img_norm = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
    classes = ('background', 'road')
    palette = [[0, 0, 0], [128, 64, 128]]

    idd_pipe = [
        dict(type='LoadImageFromFile'),
        dict(type='MultiScaleFlipAug', img_scale=(1024, 576), flip=False, transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
    ]
    carla_pipe = [
        dict(type='LoadImageFromFile'),
        dict(type='MultiScaleFlipAug', img_scale=(1280, 720), flip=False, transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
    ]
    idd = dict(
        type='CustomDataset', data_root='data/idd20k/',
        img_dir='img_dir/val', ann_dir='ann_dir/val',
        img_suffix='.jpg', seg_map_suffix='.png',
        classes=classes, palette=palette, pipeline=idd_pipe,
    )
    carla = dict(
        type='CustomDataset', data_root='data/carla_split/',
        img_dir='img_dir/val', ann_dir='ann_dir/val',
        img_suffix='.png', seg_map_suffix='.png',
        classes=classes, palette=palette, pipeline=carla_pipe,
    )
    return idd, carla


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(cfg, args, timestamp, logger):
    cfg.work_dir   = args.work_dir
    cfg.gpu_ids    = [args.gpu_id]
    cfg.device     = get_device()
    cfg.auto_resume = False

    if args.resume_from:
        cfg.resume_from = args.resume_from
        cfg.load_from   = None
        logger.info(f'Resuming from: {args.resume_from}')
    else:
        cfg.load_from = args.baseline_ckpt
        logger.info(f'Loading weights from: {args.baseline_ckpt}')

    if args.iters:
        cfg.runner.max_iters = args.iters

    setup_multi_processes(cfg)
    seed = init_random_seed(args.seed, device=cfg.device)
    set_random_seed(seed); cfg.seed = seed

    model = build_segmentor(cfg.model,
                            train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    warnings.filterwarnings('ignore', message='SyncBN is only supported.*')
    model = revert_sync_batchnorm(model)

    freeze_backbone_stages(model, args.freeze_stages, logger)

    datasets = [build_dataset(cfg.data.train)]
    logger.info(f'CARLA training samples: {len(datasets[0])}')

    cfg.dump(os.path.join(args.work_dir, 'segman_t_carla_freeze_finetune.py'))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=('background', 'road'),
            PALETTE=[[0, 0, 0], [128, 64, 128]],
        )
    model.CLASSES = ('background', 'road')
    meta = dict(seed=seed, exp_name='segman_t_carla_freeze_finetune')
    meta.update(cfg.checkpoint_config.meta)

    logger.info(f'Training {cfg.runner.max_iters} iters with freeze_stages={args.freeze_stages}')
    t0 = time.time()
    train_segmentor(model, datasets, cfg, distributed=False,
                    validate=True, timestamp=timestamp, meta=meta)
    logger.info(f'Training done in {(time.time()-t0)/3600:.2f} h')


def find_best_ckpt(work_dir):
    best = None
    for f in sorted(os.listdir(work_dir)):
        if f.startswith('best_mIoU') and f.endswith('.pth'):
            best = os.path.join(work_dir, f)
    if best is None:
        latest = os.path.join(work_dir, 'latest.pth')
        if os.path.exists(latest):
            best = os.path.realpath(latest)
    return best


# ---------------------------------------------------------------------------
# 4-way comparison table
# ---------------------------------------------------------------------------
def print_4way(b_idd, b_car, a_idd, a_car, logger):
    keys = [('mIoU', 'mIoU'),
            ('IoU_road', 'Road IoU'),
            ('Dice_road', 'Road Dice'),
            ('F1_road', 'Road F1'),
            ('Precision_road', 'Road P'),
            ('Recall_road', 'Road R'),
            ('PixelAcc', 'Pixel Acc')]
    sep = '-' * 92
    rows = [sep,
            'BEFORE / AFTER  —  IDD val  &  CARLA val',
            sep,
            f"{'Metric':<12}{'IDD before':>12}{'IDD after':>12}{'ΔIDD':>10}"
            f"{'CARLA before':>14}{'CARLA after':>14}{'ΔCARLA':>10}",
            sep]
    for k, lbl in keys:
        bi, ai = b_idd.get(k, float('nan')), a_idd.get(k, float('nan'))
        bc, ac = b_car.get(k, float('nan')), a_car.get(k, float('nan'))
        di, dc = ai - bi, ac - bc
        rows.append(f"{lbl:<12}{bi:>12.4f}{ai:>12.4f}{di:>+10.4f}"
                    f"{bc:>14.4f}{ac:>14.4f}{dc:>+10.4f}")
    rows.append(sep)
    out = '\n'.join(rows)
    logger.info('\n' + out); print('\n' + out)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg  = Config.fromfile(args.config)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    mmcv.mkdir_or_exist(os.path.abspath(args.work_dir))
    log_file = os.path.join(args.work_dir, f'{timestamp}.log')
    logger   = get_root_logger(log_file=log_file, log_level='INFO')

    logger.info('=' * 60)
    logger.info('SegMAN-Tiny | CARLA-only fine-tune | frozen early backbone | 4-way eval')
    logger.info('=' * 60)
    logger.info(f'Baseline ckpt : {args.baseline_ckpt}')
    logger.info(f'CARLA dir     : {args.carla_dir}')
    logger.info(f'Work dir      : {args.work_dir}')
    logger.info(f'Freeze stages : {args.freeze_stages}')
    logger.info(f'Train fraction: {args.train_frac}')

    # Need the CARLA val split before we can run "before" eval on it
    prepare_carla_split(args.carla_dir, args.train_frac, logger)
    idd_ds_cfg, carla_ds_cfg = build_eval_dataset_cfgs()

    # ── Before ───────────────────────────────────────────────────────────
    before_idd, before_carla = {}, {}
    if not args.no_before:
        before_idd   = run_eval(cfg, idd_ds_cfg,   args.baseline_ckpt,
                                args.gpu_id, logger, 'BEFORE-IDD (baseline)')
        before_carla = run_eval(cfg, carla_ds_cfg, args.baseline_ckpt,
                                args.gpu_id, logger, 'BEFORE-CARLA (baseline)')
        with open(os.path.join(args.work_dir, 'before_metrics.json'), 'w') as f:
            json.dump({'checkpoint': args.baseline_ckpt,
                       'idd_val': before_idd, 'carla_val': before_carla}, f, indent=2)

    # ── Eval-only short-circuit ─────────────────────────────────────────
    if args.eval_only:
        if not args.after_ckpt:
            logger.error('--eval-only requires --after-ckpt'); return
        ai = run_eval(cfg, idd_ds_cfg,   args.after_ckpt, args.gpu_id, logger, 'AFTER-IDD')
        ac = run_eval(cfg, carla_ds_cfg, args.after_ckpt, args.gpu_id, logger, 'AFTER-CARLA')
        with open(os.path.join(args.work_dir, 'after_metrics.json'), 'w') as f:
            json.dump({'checkpoint': args.after_ckpt,
                       'idd_val': ai, 'carla_val': ac}, f, indent=2)
        if before_idd:
            print_4way(before_idd, before_carla, ai, ac, logger)
        return

    # ── Train ────────────────────────────────────────────────────────────
    run_training(cfg, args, timestamp, logger)

    # ── After ────────────────────────────────────────────────────────────
    best = find_best_ckpt(args.work_dir)
    if not best:
        logger.warning('No checkpoint found after training — skipping after eval.')
        return
    logger.info(f'Best ckpt: {best}')

    after_idd   = run_eval(cfg, idd_ds_cfg,   best, args.gpu_id, logger, 'AFTER-IDD')
    after_carla = run_eval(cfg, carla_ds_cfg, best, args.gpu_id, logger, 'AFTER-CARLA')
    with open(os.path.join(args.work_dir, 'after_metrics.json'), 'w') as f:
        json.dump({'checkpoint': best,
                   'idd_val': after_idd, 'carla_val': after_carla}, f, indent=2)

    if before_idd:
        print_4way(before_idd, before_carla, after_idd, after_carla, logger)
        comb = dict(
            approach=f'CARLA-only fine-tune, freeze_stages={args.freeze_stages}, '
                     f'train_frac={args.train_frac}',
            baseline_ckpt=args.baseline_ckpt,
            best_ckpt=best,
            before=dict(idd_val=before_idd, carla_val=before_carla),
            after =dict(idd_val=after_idd,  carla_val=after_carla),
            delta =dict(
                idd  ={k: round(after_idd.get(k, 0)   - before_idd.get(k, 0), 4)
                       for k in before_idd},
                carla={k: round(after_carla.get(k, 0) - before_carla.get(k, 0), 4)
                       for k in before_carla},
            ),
        )
        with open(os.path.join(args.work_dir, 'comparison.json'), 'w') as f:
            json.dump(comb, f, indent=2)


if __name__ == '__main__':
    main()
