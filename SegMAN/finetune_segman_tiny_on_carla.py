"""
train_segman_carla_finetune.py
==============================
Full pipeline:
  1. Evaluate the IDD-trained baseline (best_mIoU_iter_16000.pth) on IDD val  →  "before"
  2. Prepare the CARLA dataset (convert masks 255→1, create directory structure)
  3. Fine-tune SegMAN-Tiny on CARLA starting from the IDD checkpoint
  4. Evaluate the CARLA fine-tuned model on IDD val                            →  "after"
  5. Print a side-by-side comparison table

Run from the SegMAN root:
    conda run -n segmanaser python train_segman_carla_finetune.py [options]

Options:
    --baseline-ckpt   Path to baseline IDD checkpoint   (default: segmentation/work_dirs/segman_t_idd20k/best_mIoU_iter_16000.pth)
    --carla-dir       Path to CARLA dataset root         (default: Carladataset/)
    --work-dir        Output dir for this fine-tuning run (default: segmentation/work_dirs/segman_t_carla_finetune)
    --config          Fine-tune config path              (default: local_configs/segman/tiny/segman_t_carla_finetune.py)
    --gpu-id          GPU index                          (default: 0)
    --seed            Random seed                        (default: 42)
    --iters           Override max_iters                 (default: from config)
    --eval-only       Skip training; only run before/after eval (requires --after-ckpt)
    --after-ckpt      Checkpoint for "after" eval (used with --eval-only)
    --no-before       Skip the "before" eval step
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: point Python at the segmentation package
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEG_DIR    = os.path.join(SCRIPT_DIR, 'segmentation')
sys.path.insert(0, SEG_DIR)
os.chdir(SEG_DIR)

# Pre-load PyTorch native libs so selective_scan CUDA ext can resolve them
import ctypes
import glob as _glob

_torch_lib = os.path.join(sys.prefix, 'lib', 'python3.10',
                          'site-packages', 'torch', 'lib')
for _so in sorted(_glob.glob(os.path.join(_torch_lib, 'lib*.so*'))):
    try:
        ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

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
from mmseg.utils import (build_dp, collect_env, get_device, get_root_logger,
                         setup_multi_processes)
from mmseg.core.evaluation.metrics import pre_eval_to_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='CARLA fine-tune pipeline for SegMAN-Tiny')
    p.add_argument('--baseline-ckpt',
                   default='work_dirs/segman_t_idd20k/best_mIoU_iter_16000.pth')
    p.add_argument('--carla-dir',
                   default=os.path.join(SCRIPT_DIR, 'Carladataset'))
    p.add_argument('--work-dir',
                   default='work_dirs/segman_t_carla_finetune')
    p.add_argument('--config',
                   default='local_configs/segman/tiny/segman_t_carla_finetune.py')
    p.add_argument('--gpu-id',   type=int, default=0)
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--iters',    type=int, default=None)
    p.add_argument('--resume-from', default=None,
                   help='Checkpoint to resume training from')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--after-ckpt', default=None)
    p.add_argument('--no-before', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# CARLA data preparation
# ---------------------------------------------------------------------------

def prepare_carla_data(carla_dir: str, logger) -> str:
    """
    Convert CARLA seg masks (0/255) to (0/1) and create a dataset directory
    at  segmentation/data/carla/  compatible with MMSeg CustomDataset.

    Returns the data_root string used in the config ('data/carla/').
    """
    carla_dir = os.path.abspath(carla_dir)
    rgb_src   = os.path.join(carla_dir, 'rgb')
    seg_src   = os.path.join(carla_dir, 'seg')

    # Destination inside segmentation/
    data_root  = 'data/carla'          # relative to SEG_DIR
    abs_root   = os.path.join(SEG_DIR, data_root)
    img_dst    = os.path.join(abs_root, 'img_dir', 'train')
    ann_dst    = os.path.join(abs_root, 'ann_dir', 'train')

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(ann_dst, exist_ok=True)

    rgb_files = sorted(f for f in os.listdir(rgb_src) if f.endswith('.png'))
    logger.info(f'CARLA RGB images  : {len(rgb_files)}')

    converted = 0
    symlinked = 0
    for fname in rgb_files:
        # --- image: symlink ---
        src_img = os.path.join(rgb_src, fname)
        dst_img = os.path.join(img_dst, fname)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
            symlinked += 1

        # --- mask: convert 255 → 1 ---
        dst_ann = os.path.join(ann_dst, fname)
        if not os.path.exists(dst_ann):
            from PIL import Image
            mask = np.array(Image.open(os.path.join(seg_src, fname)))
            mask_bin = (mask > 127).astype(np.uint8)   # 255→1, 0→0
            Image.fromarray(mask_bin).save(dst_ann)
            converted += 1

    logger.info(f'Images symlinked  : {symlinked}  (already existed: {len(rgb_files)-symlinked})')
    logger.info(f'Masks converted   : {converted}  (already existed: {len(rgb_files)-converted})')
    logger.info(f'CARLA data root   : {abs_root}')
    return data_root


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def run_eval_on_idd(cfg, checkpoint_path, gpu_id, logger, label='eval'):
    """
    Evaluate a checkpoint on the IDD val split (cfg.data.test).
    Returns a flat metrics dict.
    """
    logger.info('=' * 60)
    logger.info(f'EVALUATION  [{label}]')
    logger.info(f'Checkpoint : {checkpoint_path}')
    logger.info('=' * 60)

    ds_cfg          = copy.deepcopy(cfg.data.test)
    ds_cfg.test_mode = True
    dataset = build_dataset(ds_cfg)

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        dist=False,
        shuffle=False,
    )

    model_cfg             = copy.deepcopy(cfg)
    model_cfg.model.pretrained = None
    model_cfg.model.train_cfg  = None
    model = build_segmentor(model_cfg.model, test_cfg=model_cfg.get('test_cfg'))

    ckpt  = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = ckpt['meta'].get('CLASSES', dataset.CLASSES)
    model.PALETTE = ckpt['meta'].get('PALETTE', dataset.PALETTE)

    model = revert_sync_batchnorm(model)
    model = build_dp(model, get_device(), device_ids=[gpu_id])
    model.eval()

    results = single_gpu_test(model, loader, show=False, pre_eval=True)

    ret = pre_eval_to_metrics(
        results,
        metrics=['mIoU', 'mDice', 'mFscore'],
        nan_to_num=0,
        beta=1,
    )

    iou_bg,   iou_road  = float(ret['IoU'][0]),       float(ret['IoU'][1])
    dice_bg,  dice_road = float(ret['Dice'][0]),       float(ret['Dice'][1])
    f1_bg,    f1_road   = float(ret['Fscore'][0]),     float(ret['Fscore'][1])
    prec_bg,  prec_road = float(ret['Precision'][0]),  float(ret['Precision'][1])
    rec_bg,   rec_road  = float(ret['Recall'][0]),     float(ret['Recall'][1])
    macc = float(ret['aAcc'])
    miou = (iou_bg + iou_road) / 2

    metrics = dict(
        mIoU          = round(miou,      4),
        IoU_road      = round(iou_road,  4),
        IoU_bg        = round(iou_bg,    4),
        Dice_road     = round(dice_road, 4),
        F1_road       = round(f1_road,   4),
        Precision_road= round(prec_road, 4),
        Recall_road   = round(rec_road,  4),
        PixelAcc      = round(macc,      4),
    )

    logger.info(f'--- Metrics [{label}] ---')
    for k, v in metrics.items():
        logger.info(f'  {k:<20s}: {v:.4f}')

    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(cfg, work_dir, gpu_id, seed, timestamp, logger):
    mmcv.mkdir_or_exist(os.path.abspath(work_dir))
    cfg.work_dir = work_dir
    cfg.gpu_ids  = [gpu_id]
    cfg.device   = get_device()
    cfg.auto_resume = False

    setup_multi_processes(cfg)

    seed_val = init_random_seed(seed, device=cfg.device)
    set_random_seed(seed_val)
    cfg.seed = seed_val
    meta = dict(seed=seed_val, exp_name='segman_t_carla_finetune')

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )
    model.init_weights()
    warnings.filterwarnings('ignore', message='SyncBN is only supported.*')
    model = revert_sync_batchnorm(model)

    datasets = [build_dataset(cfg.data.train)]
    logger.info(f'Training samples: {len(datasets[0])}')

    cfg.dump(os.path.join(work_dir, 'segman_t_carla_finetune.py'))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE,
        )
    model.CLASSES = datasets[0].CLASSES
    meta.update(cfg.checkpoint_config.meta)

    logger.info(f'Starting CARLA fine-tuning  ({cfg.runner.max_iters} iters)...')
    t0 = time.time()
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta=meta,
    )
    elapsed = time.time() - t0
    logger.info(f'Training done in {elapsed/3600:.2f} h ({elapsed:.0f} s)')


# ---------------------------------------------------------------------------
# Pretty comparison table
# ---------------------------------------------------------------------------

def print_comparison(before: dict, after: dict, logger):
    metrics = [
        ('mIoU',           'mIoU'),
        ('IoU_road',       'Road IoU'),
        ('Dice_road',      'Road Dice'),
        ('F1_road',        'Road F1'),
        ('Precision_road', 'Road Precision'),
        ('Recall_road',    'Road Recall'),
        ('PixelAcc',       'Pixel Accuracy'),
    ]
    sep  = '-' * 62
    hdr  = f"{'Metric':<22} {'Before (IDD)':>12} {'After (CARLA)':>14} {'Delta':>8}"
    rows = [sep, 'BEFORE vs AFTER  —  IDD val performance', sep, hdr, sep]
    for key, label in metrics:
        b = before.get(key, float('nan'))
        a = after.get(key,  float('nan'))
        d = a - b
        sign = '+' if d >= 0 else ''
        rows.append(f'{label:<22} {b:>12.4f} {a:>14.4f} {sign}{d:>7.4f}')
    rows.append(sep)
    table = '\n'.join(rows)
    logger.info('\n' + table)
    print('\n' + table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.iters is not None:
        cfg.runner.max_iters = args.iters

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    mmcv.mkdir_or_exist(os.path.abspath(args.work_dir))
    log_file = os.path.join(args.work_dir, f'{timestamp}.log')
    logger   = get_root_logger(log_file=log_file, log_level='INFO')

    logger.info('=' * 60)
    logger.info('SegMAN-Tiny  |  CARLA Fine-Tune  →  IDD Eval')
    logger.info('=' * 60)
    logger.info(f'Baseline ckpt : {args.baseline_ckpt}')
    logger.info(f'CARLA dir     : {args.carla_dir}')
    logger.info(f'Work dir      : {args.work_dir}')
    logger.info(f'Max iters     : {cfg.runner.max_iters}')

    # ------------------------------------------------------------------ Before
    before_metrics = {}
    if not args.no_before:
        before_metrics = run_eval_on_idd(
            cfg, args.baseline_ckpt, args.gpu_id, logger, label='BEFORE (IDD baseline)')
        before_path = os.path.join(args.work_dir, 'before_metrics.json')
        with open(before_path, 'w') as f:
            json.dump({'checkpoint': args.baseline_ckpt, 'metrics': before_metrics}, f, indent=2)
        logger.info(f'Before metrics saved → {before_path}')

    if args.eval_only:
        if args.after_ckpt:
            after_metrics = run_eval_on_idd(
                cfg, args.after_ckpt, args.gpu_id, logger, label='AFTER (CARLA fine-tuned)')
            after_path = os.path.join(args.work_dir, 'after_metrics.json')
            with open(after_path, 'w') as f:
                json.dump({'checkpoint': args.after_ckpt, 'metrics': after_metrics}, f, indent=2)
            if before_metrics:
                print_comparison(before_metrics, after_metrics, logger)
        return

    # ------------------------------------------------------------------ Prepare CARLA
    logger.info('\n--- Preparing CARLA dataset ---')
    prepare_carla_data(args.carla_dir, logger)

    # ------------------------------------------------------------------ Train
    # Resume from checkpoint if specified, otherwise load IDD baseline weights
    if args.resume_from:
        cfg.resume_from = args.resume_from
        cfg.load_from = None
    else:
        cfg.load_from = args.baseline_ckpt

    run_training(cfg, args.work_dir, args.gpu_id, args.seed, timestamp, logger)

    # ------------------------------------------------------------------ Find best ckpt
    best_ckpt = None
    for fname in sorted(os.listdir(args.work_dir)):
        if fname.startswith('best_mIoU') and fname.endswith('.pth'):
            best_ckpt = os.path.join(args.work_dir, fname)
    if best_ckpt is None:
        latest = os.path.join(args.work_dir, 'latest.pth')
        if os.path.exists(latest):
            best_ckpt = os.path.realpath(latest)

    if best_ckpt is None:
        logger.warning('No checkpoint found after training — skipping after eval.')
        return

    logger.info(f'Best checkpoint: {best_ckpt}')

    # ------------------------------------------------------------------ After
    after_metrics = run_eval_on_idd(
        cfg, best_ckpt, args.gpu_id, logger, label='AFTER (CARLA fine-tuned)')
    after_path = os.path.join(args.work_dir, 'after_metrics.json')
    with open(after_path, 'w') as f:
        json.dump({'checkpoint': best_ckpt, 'metrics': after_metrics}, f, indent=2)
    logger.info(f'After metrics saved → {after_path}')

    # ------------------------------------------------------------------ Comparison
    if before_metrics:
        print_comparison(before_metrics, after_metrics, logger)

    # Save combined results
    combined = {
        'baseline_ckpt':       args.baseline_ckpt,
        'carla_finetune_ckpt': best_ckpt,
        'before':              before_metrics,
        'after':               after_metrics,
        'delta':               {k: round(after_metrics.get(k, 0) - before_metrics.get(k, 0), 4)
                                for k in before_metrics},
    }
    comb_path = os.path.join(args.work_dir, 'comparison.json')
    with open(comb_path, 'w') as f:
        json.dump(combined, f, indent=2)
    logger.info(f'Full comparison saved → {comb_path}')


if __name__ == '__main__':
    main()
