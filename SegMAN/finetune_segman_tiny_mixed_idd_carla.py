"""
train_segman_mixed_finetune.py
==============================
Mixed IDD + CARLA fine-tuning with frozen early backbone layers.

Strategy
--------
  - Freeze backbone stages 0+1 (patch_embed + layers[0..3]) — preserve
    the low-level features learned on IDD real data.
  - Train backbone stages 2+3 (layers[4..7]) + full decoder.
  - Train jointly on IDD (7034 imgs) + CARLA (5000 imgs) in one pass.
  - Start from the best IDD checkpoint (best_mIoU_iter_16000.pth).
  - Validate on IDD val throughout so we can track real-world performance.

Run from SegMAN root:
    conda activate segmanaser
    python train_segman_mixed_finetune.py

Options:
    --baseline-ckpt   IDD checkpoint to start from (default: best_mIoU_iter_16000.pth)
    --work-dir        Output directory  (default: segmentation/work_dirs/segman_t_mixed_finetune)
    --config          Config path       (default: local_configs/segman/tiny/segman_t_mixed_finetune.py)
    --gpu-id          GPU index         (default: 0)
    --seed            Random seed       (default: 42)
    --iters           Override max_iters
    --freeze-stages   Number of backbone stages to freeze: 0,1,2  (default: 2)
    --no-before       Skip baseline eval (use if before_metrics.json already exists)
    --resume-from     Resume training from a checkpoint
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
    p.add_argument('--work-dir',
                   default='work_dirs/segman_t_mixed_finetune')
    p.add_argument('--config',
                   default='local_configs/segman/tiny/segman_t_mixed_finetune.py')
    p.add_argument('--gpu-id',      type=int,   default=0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--iters',       type=int,   default=None)
    p.add_argument('--freeze-stages', type=int, default=2,
                   help='Freeze first N backbone stages (0=none, 1=stage0, 2=stages0+1)')
    p.add_argument('--no-before',   action='store_true')
    p.add_argument('--resume-from', default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# CARLA data preparation (idempotent)
# ---------------------------------------------------------------------------
def prepare_carla_data(logger):
    carla_dir = os.path.join(SCRIPT_DIR, 'Carladataset')
    rgb_src   = os.path.join(carla_dir, 'rgb')
    seg_src   = os.path.join(carla_dir, 'seg')
    abs_root  = os.path.join(SEG_DIR, 'data', 'carla')
    img_dst   = os.path.join(abs_root, 'img_dir', 'train')
    ann_dst   = os.path.join(abs_root, 'ann_dir', 'train')
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(ann_dst, exist_ok=True)

    from PIL import Image
    rgb_files = sorted(f for f in os.listdir(rgb_src) if f.endswith('.png'))
    new_sym, new_mask = 0, 0
    for fname in rgb_files:
        dst_img = os.path.join(img_dst, fname)
        if not os.path.exists(dst_img):
            os.symlink(os.path.join(rgb_src, fname), dst_img)
            new_sym += 1
        dst_ann = os.path.join(ann_dst, fname)
        if not os.path.exists(dst_ann):
            mask = np.array(Image.open(os.path.join(seg_src, fname)))
            Image.fromarray((mask > 127).astype(np.uint8)).save(dst_ann)
            new_mask += 1

    logger.info(f'CARLA data ready: {len(rgb_files)} images '
                f'({new_sym} new symlinks, {new_mask} new masks)')


# ---------------------------------------------------------------------------
# Freeze early backbone layers
# ---------------------------------------------------------------------------
def freeze_backbone_stages(model, n_stages: int, logger):
    """
    Freeze the first n_stages of the backbone.
    Backbone structure (Tiny):
        patch_embed          → always frozen when n_stages >= 1
        layers[0] stage-0 (32-dim)   \  frozen when n_stages >= 1
        layers[1] downsample 0→1     /
        layers[2] stage-1 (64-dim)   \  frozen when n_stages >= 2
        layers[3] downsample 1→2     /
        layers[4] stage-2 (144-dim)  \  frozen when n_stages >= 3
        layers[5] downsample 2→3     /
        layers[6] stage-3 (192-dim)    frozen when n_stages >= 4
        layers[7] identity
    """
    if n_stages == 0:
        logger.info('No backbone layers frozen.')
        return

    backbone = model.backbone
    frozen_params = 0

    # Always freeze patch_embed if any stages are frozen
    for p in backbone.patch_embed.parameters():
        p.requires_grad = False
        frozen_params += p.numel()

    # Each stage occupies 2 entries in layers: [layer, downsample]
    n_layer_entries = n_stages * 2
    for i in range(n_layer_entries):
        for p in backbone.layers[i].parameters():
            p.requires_grad = False
            frozen_params += p.numel()

    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Frozen backbone stages : {n_stages}  '
                f'({frozen_params/1e6:.1f}M params frozen)')
    logger.info(f'Trainable / Total      : {trainable/1e6:.1f}M / {total_params/1e6:.1f}M')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_eval(cfg, checkpoint_path, gpu_id, logger, label='eval'):
    logger.info('=' * 60)
    logger.info(f'EVALUATION  [{label}]')
    logger.info(f'Checkpoint : {checkpoint_path}')
    logger.info('=' * 60)

    ds_cfg           = copy.deepcopy(cfg.data.test)
    ds_cfg.test_mode = True
    dataset = build_dataset(ds_cfg)
    loader  = build_dataloader(dataset, samples_per_gpu=1,
                               workers_per_gpu=4, dist=False, shuffle=False)

    model_cfg = copy.deepcopy(cfg)
    model_cfg.model.pretrained = None
    model_cfg.model.train_cfg  = None
    model = build_segmentor(model_cfg.model, test_cfg=model_cfg.get('test_cfg'))

    ckpt = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = ckpt['meta'].get('CLASSES', dataset.CLASSES)
    model.PALETTE = ckpt['meta'].get('PALETTE', dataset.PALETTE)
    model = revert_sync_batchnorm(model)
    model = build_dp(model, get_device(), device_ids=[gpu_id])
    model.eval()

    results = single_gpu_test(model, loader, show=False, pre_eval=True)
    ret = pre_eval_to_metrics(results, metrics=['mIoU', 'mDice', 'mFscore'],
                              nan_to_num=0, beta=1)

    iou_bg,  iou_road  = float(ret['IoU'][0]),       float(ret['IoU'][1])
    f1_bg,   f1_road   = float(ret['Fscore'][0]),     float(ret['Fscore'][1])
    prec_bg, prec_road = float(ret['Precision'][0]),  float(ret['Precision'][1])
    rec_bg,  rec_road  = float(ret['Recall'][0]),     float(ret['Recall'][1])
    dice_bg, dice_road = float(ret['Dice'][0]),       float(ret['Dice'][1])
    macc  = float(ret['aAcc'])
    miou  = (iou_bg + iou_road) / 2

    metrics = dict(
        mIoU           = round(miou,       4),
        IoU_road       = round(iou_road,   4),
        IoU_bg         = round(iou_bg,     4),
        Dice_road      = round(dice_road,  4),
        F1_road        = round(f1_road,    4),
        Precision_road = round(prec_road,  4),
        Recall_road    = round(rec_road,   4),
        PixelAcc       = round(macc,       4),
    )
    logger.info(f'--- Results [{label}] ---')
    for k, v in metrics.items():
        logger.info(f'  {k:<22}: {v:.4f}')
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(cfg, args, timestamp, logger):
    mmcv.mkdir_or_exist(os.path.abspath(args.work_dir))
    cfg.work_dir  = args.work_dir
    cfg.gpu_ids   = [args.gpu_id]
    cfg.device    = get_device()
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
    set_random_seed(seed)
    cfg.seed = seed

    model = build_segmentor(cfg.model,
                            train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    warnings.filterwarnings('ignore', message='SyncBN is only supported.*')
    model = revert_sync_batchnorm(model)

    # ── Freeze early backbone stages ──────────────────────────────────────
    freeze_backbone_stages(model, args.freeze_stages, logger)

    # Build dataset — cfg.data.train is a list → ConcatDataset
    datasets = [build_dataset(cfg.data.train)]
    logger.info(f'Total training samples: {len(datasets[0])}')

    meta = dict(seed=seed, exp_name='segman_t_mixed_finetune')
    cfg.dump(os.path.join(args.work_dir, 'segman_t_mixed_finetune.py'))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=('background', 'road'),
            PALETTE=[[0, 0, 0], [128, 64, 128]],
        )
    model.CLASSES = ('background', 'road')
    meta.update(cfg.checkpoint_config.meta)

    logger.info(f'Starting mixed fine-tuning ({cfg.runner.max_iters} iters)...')
    t0 = time.time()
    train_segmentor(model, datasets, cfg, distributed=False,
                    validate=True, timestamp=timestamp, meta=meta)
    logger.info(f'Training done in {(time.time()-t0)/3600:.2f} h')


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(before, after, logger):
    keys = [
        ('mIoU',           'mIoU'),
        ('IoU_road',       'Road IoU'),
        ('Dice_road',      'Road Dice'),
        ('F1_road',        'Road F1'),
        ('Precision_road', 'Road Precision'),
        ('Recall_road',    'Road Recall'),
        ('PixelAcc',       'Pixel Accuracy'),
    ]
    sep = '-' * 62
    lines = [sep,
             'RESULTS  —  IDD val  (before vs after mixed IDD+CARLA fine-tune)',
             sep,
             f"{'Metric':<22} {'Before':>10} {'After':>10} {'Delta':>8}",
             sep]
    for key, label in keys:
        b = before.get(key, float('nan'))
        a = after.get(key, float('nan'))
        d = a - b
        lines.append(f'{label:<22} {b:>10.4f} {a:>10.4f} {("+"+f"{d:.4f}" if d>=0 else f"{d:.4f}"):>8}')
    lines.append(sep)
    table = '\n'.join(lines)
    logger.info('\n' + table)
    print('\n' + table)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg  = Config.fromfile(args.config)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    mmcv.mkdir_or_exist(os.path.abspath(args.work_dir))
    log_file = os.path.join(args.work_dir, f'{timestamp}.log')
    logger   = get_root_logger(log_file=log_file, log_level='INFO')

    logger.info('=' * 60)
    logger.info('SegMAN-Tiny  |  Mixed IDD+CARLA Fine-Tune  (frozen early backbone)')
    logger.info('=' * 60)
    logger.info(f'Baseline ckpt  : {args.baseline_ckpt}')
    logger.info(f'Work dir       : {args.work_dir}')
    logger.info(f'Freeze stages  : {args.freeze_stages}')

    # ── Before eval ───────────────────────────────────────────────────────
    before_path = os.path.join(args.work_dir, 'before_metrics.json')
    if args.no_before and os.path.exists(before_path):
        with open(before_path) as f:
            before_metrics = json.load(f)['metrics']
        logger.info(f'Loaded existing before metrics from {before_path}')
        logger.info(f'  mIoU={before_metrics["mIoU"]:.4f}  '
                    f'Road IoU={before_metrics["IoU_road"]:.4f}  '
                    f'F1={before_metrics["F1_road"]:.4f}')
    else:
        before_metrics = run_eval(cfg, args.baseline_ckpt,
                                  args.gpu_id, logger, 'BEFORE (IDD baseline)')
        with open(before_path, 'w') as f:
            json.dump({'checkpoint': args.baseline_ckpt,
                       'metrics': before_metrics}, f, indent=2)
        logger.info(f'Before metrics saved → {before_path}')

    # ── Prepare CARLA data ────────────────────────────────────────────────
    prepare_carla_data(logger)

    # ── Train ─────────────────────────────────────────────────────────────
    run_training(cfg, args, timestamp, logger)

    # ── Find best checkpoint ──────────────────────────────────────────────
    best_ckpt = None
    for fname in sorted(os.listdir(args.work_dir)):
        if fname.startswith('best_mIoU') and fname.endswith('.pth'):
            best_ckpt = os.path.join(args.work_dir, fname)
    if best_ckpt is None:
        latest = os.path.join(args.work_dir, 'latest.pth')
        if os.path.exists(latest):
            best_ckpt = os.path.realpath(latest)

    if not best_ckpt:
        logger.warning('No checkpoint found — skipping after eval.')
        return

    logger.info(f'Best checkpoint: {best_ckpt}')

    # ── After eval ────────────────────────────────────────────────────────
    after_metrics = run_eval(cfg, best_ckpt, args.gpu_id, logger,
                             'AFTER (mixed IDD+CARLA, frozen early backbone)')
    after_path = os.path.join(args.work_dir, 'after_metrics.json')
    with open(after_path, 'w') as f:
        json.dump({'checkpoint': best_ckpt, 'metrics': after_metrics}, f, indent=2)

    # ── Comparison ────────────────────────────────────────────────────────
    print_comparison(before_metrics, after_metrics, logger)

    combined = {
        'approach': f'mixed IDD+CARLA, freeze_stages={args.freeze_stages}',
        'baseline_ckpt': args.baseline_ckpt,
        'best_ckpt': best_ckpt,
        'before': before_metrics,
        'after': after_metrics,
        'delta': {k: round(after_metrics.get(k, 0) - before_metrics.get(k, 0), 4)
                  for k in before_metrics},
    }
    comb_path = os.path.join(args.work_dir, 'comparison.json')
    with open(comb_path, 'w') as f:
        json.dump(combined, f, indent=2)
    logger.info(f'Full comparison → {comb_path}')


if __name__ == '__main__':
    main()
