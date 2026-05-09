"""
Train SegMAN-Small (CNN-branch decoder) on IDD-20k-II for binary road segmentation.

The SegMANDecoder runs two parallel paths in forward_winssm():
  CNN branch — conv_downsample_2/4 + pixel_unshuffle (local texture, multi-scale)
  SSM branch — VSSM selective scan over concatenated features (long-range context)
Both are fused with a short-path 1x1 and global-context path, then a final ConvModule.

Run from the SegMAN root:
    conda run -n segmanaser python train_segman_cnn_branch_idd20k.py
Flags: --config, --work-dir, --gpu-id, --seed, --resume-from, --no-test, --iters (default 20000).

Outputs: iter_*.pth, best_mIoU_iter_*.pth, latest.pth, <ts>.log[.json],
test_results.json, metrics_summary.json — all in work_dirs/segman_small_cnn_branch_idd20k/.
Metrics: aAcc, mIoU, mAcc, mDice, mFscore + per-class IoU/Acc/Dice/Fscore/Precision/Recall,
plus dataset-level precision/recall/F1/IoU for road and image-level F1@IoU>=0.5.
"""

import copy
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEG_DIR    = os.path.join(SCRIPT_DIR, 'segmentation')
sys.path.insert(0, SEG_DIR)
os.chdir(SEG_DIR)

# Pre-load PyTorch native libs so the selective_scan CUDA extension can
# resolve libc10.so, libtorch.so, etc. at import time.
import ctypes, glob as _glob
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


# ── Constants ─────────────────────────────────────────────────────────────────
ROAD_CLASS  = 1    # class index for "road" in the binary mask
IGNORE_VAL  = 255  # pixels to skip during evaluation


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description='Train SegMAN-Small (CNN branch) on IDD-20k-II road segmentation')
    p.add_argument(
        '--config',
        default='local_configs/segman/small/segman_s_cnn_branch_idd20k.py',
        help='Config file (relative to segmentation/)')
    p.add_argument(
        '--work-dir',
        default='work_dirs/segman_small_cnn_branch_idd20k',
        help='Directory for checkpoints and logs')
    p.add_argument('--gpu-id',      type=int, default=0)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--resume-from', default=None)
    p.add_argument('--no-test',     action='store_true',
                   help='Skip evaluation after training')
    p.add_argument('--iters',       type=int, default=None,
                   help='Override max_iters (default: from config = 20000)')
    return p.parse_args()


# ── Pixel-level stats for road class ─────────────────────────────────────────

def _pixel_stats(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Return (TP, FP, FN, TN) for the road class; ignore IGNORE_VAL pixels."""
    valid = (gt_mask != IGNORE_VAL)
    pred_road = pred_mask[valid] == ROAD_CLASS
    gt_road   = gt_mask[valid]   == ROAD_CLASS
    tp = int(( pred_road &  gt_road).sum())
    fp = int(( pred_road & ~gt_road).sum())
    fn = int((~pred_road &  gt_road).sum())
    tn = int((~pred_road & ~gt_road).sum())
    return tp, fp, fn, tn


# ── Comprehensive extra metrics ───────────────────────────────────────────────

def compute_extra_metrics(results, dataset):
    """
    Compute global pixel-level and image-level metrics for the road class.

    Parameters
    ----------
    results : list[np.ndarray]
        Per-pixel argmax predictions from single_gpu_test (pre_eval=False).
    dataset  : MMSeg dataset with .get_gt_seg_maps() method.

    Returns
    -------
    dict with keys:
        precision_road, recall_road, f1_road, iou_road,
        specificity_road, accuracy_overall,
        f1_at_50,
        pixel_tp, pixel_fp, pixel_fn, pixel_tn,
        img_tp50, img_fp50, img_fn50
    """
    total_tp = total_fp = total_fn = total_tn = 0
    tp50 = fp50 = fn50 = 0

    for pred, gt in zip(results, dataset.get_gt_seg_maps()):
        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred.astype(np.uint8),
                              (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        tp, fp, fn, tn = _pixel_stats(pred, gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        # Image-level F1@50 ─ an image is a TP when road-IoU ≥ 0.5
        denom = tp + fp + fn
        iou   = tp / denom if denom > 0 else 0.0
        has_gt   = bool((gt[gt != IGNORE_VAL] == ROAD_CLASS).any())
        has_pred = bool((pred == ROAD_CLASS).any())

        if has_gt and has_pred:
            if iou >= 0.50:
                tp50 += 1
            else:
                fp50 += 1
                fn50 += 1
        elif has_pred and not has_gt:
            fp50 += 1
        elif has_gt and not has_pred:
            fn50 += 1

    # ── Global pixel-level metrics ────────────────────────────────────────────
    total_pixels  = total_tp + total_fp + total_fn + total_tn

    precision    = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall       = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1           = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0)
    iou_road     = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    specificity  = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    accuracy     = (total_tp + total_tn) / total_pixels if total_pixels > 0 else 0.0

    # ── Image-level F1@50 ─────────────────────────────────────────────────────
    p50      = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0
    r50      = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
    f1_at_50 = (2 * p50 * r50 / (p50 + r50)) if (p50 + r50) > 0 else 0.0

    return dict(
        precision_road   = round(precision,   4),
        recall_road      = round(recall,      4),
        f1_road          = round(f1,          4),
        iou_road_global  = round(iou_road,    4),
        specificity_road = round(specificity, 4),
        accuracy_overall = round(accuracy,    4),
        f1_at_50         = round(f1_at_50,    4),
        # raw counts for transparency
        pixel_tp   = total_tp,
        pixel_fp   = total_fp,
        pixel_fn   = total_fn,
        pixel_tn   = total_tn,
        img_tp50   = tp50,
        img_fp50   = fp50,
        img_fn50   = fn50,
    )


# ── MMSeg pre_eval metrics ────────────────────────────────────────────────────

def compute_mmseg_metrics(results_pre_eval):
    """
    Convert pre_eval results to the full MMSeg metric set:
      aAcc, mIoU, mAcc, mDice, mFscore + per-class IoU/Acc/Dice/Fscore/Precision/Recall
    """
    ret = pre_eval_to_metrics(
        results_pre_eval,
        metrics=['mIoU', 'mDice', 'mFscore'],
        nan_to_num=0,
        beta=1,
    )

    out = {
        'aAcc':   float(ret['aAcc']),
        'mIoU':   float((ret['IoU'][0]      + ret['IoU'][1])      / 2),
        'mAcc':   float((ret['Acc'][0]      + ret['Acc'][1])      / 2),
        'mDice':  float((ret['Dice'][0]     + ret['Dice'][1])     / 2),
        'mFscore':float((ret['Fscore'][0]   + ret['Fscore'][1])   / 2),
        'mPrec':  float((ret['Precision'][0]+ ret['Precision'][1])/ 2),
        'mRecall':float((ret['Recall'][0]   + ret['Recall'][1])   / 2),
        # per-class
        'IoU.background': float(ret['IoU'][0]),
        'IoU.road':       float(ret['IoU'][1]),
        'Acc.background': float(ret['Acc'][0]),
        'Acc.road':       float(ret['Acc'][1]),
        'Dice.background':float(ret['Dice'][0]),
        'Dice.road':      float(ret['Dice'][1]),
        'Fscore.background': float(ret['Fscore'][0]),
        'Fscore.road':       float(ret['Fscore'][1]),
        'Precision.background': float(ret['Precision'][0]),
        'Precision.road':       float(ret['Precision'][1]),
        'Recall.background':    float(ret['Recall'][0]),
        'Recall.road':          float(ret['Recall'][1]),
    }
    return out


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_log_json(log_json_path):
    train_history, val_history = [], []
    if not os.path.exists(log_json_path):
        return train_history, val_history

    with open(log_json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            mode = entry.get('mode')
            if mode == 'train':
                train_history.append({
                    'iter':      entry.get('iter'),
                    'loss':      entry.get('loss'),
                    'lr':        entry.get('lr'),
                    'loss_ce':   entry.get('decode.loss_ce'),
                    'loss_dice': entry.get('decode.loss_dice'),
                })
            elif mode == 'val':
                val_history.append({
                    'iter':           entry.get('iter'),
                    'mIoU':           entry.get('mIoU'),
                    'mAcc':           entry.get('mAcc'),
                    'aAcc':           entry.get('aAcc'),
                    'IoU.background': entry.get('IoU.background'),
                    'IoU.road':       entry.get('IoU.road'),
                })

    return train_history, val_history


# ── Evaluation on a dataset split ────────────────────────────────────────────

def run_evaluation(cfg, checkpoint_path, gpu_id, logger, split='test'):
    """
    Evaluate the model checkpoint on the given split using both MMSeg's
    pre_eval pipeline (per-pixel metrics) and custom global pixel stats.
    """
    logger.info('=' * 60)
    logger.info(f'EVALUATION  ({split.upper()})')
    logger.info(f'Checkpoint : {checkpoint_path}')
    logger.info('=' * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = copy.deepcopy(cfg.data[split])
    ds_cfg.test_mode = True
    dataset = build_dataset(ds_cfg)

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        dist=False,
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = copy.deepcopy(cfg)
    model_cfg.model.pretrained = None
    model_cfg.model.train_cfg  = None
    model = build_segmentor(model_cfg.model, test_cfg=model_cfg.get('test_cfg'))

    ckpt = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = ckpt['meta'].get('CLASSES', dataset.CLASSES)
    model.PALETTE = ckpt['meta'].get('PALETTE', dataset.PALETTE)

    model = revert_sync_batchnorm(model)
    model = build_dp(model, get_device(), device_ids=[gpu_id])

    # ── Two inference passes ──────────────────────────────────────────────────
    # Pass 1: pre_eval=True  → structured per-pixel accumulation for MMSeg metrics
    results_pre = single_gpu_test(model, loader, show=False, pre_eval=True)
    mmseg_metrics = compute_mmseg_metrics(results_pre)

    # Pass 2: pre_eval=False → raw argmax masks for custom global pixel stats
    results_raw = single_gpu_test(model, loader, show=False, pre_eval=False)
    extra_metrics = compute_extra_metrics(results_raw, dataset)

    all_metrics = {**mmseg_metrics, **extra_metrics}

    logger.info(f'--- Metrics ({split}) ---')
    for k, v in all_metrics.items():
        if isinstance(v, float):
            logger.info(f'  {k:<30s}: {v:.4f}')
        else:
            logger.info(f'  {k:<30s}: {v}')

    return all_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.gpu_ids  = [args.gpu_id]
    cfg.device   = get_device()

    if args.resume_from:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = False

    if args.iters is not None:
        cfg.runner.max_iters = args.iters

    if cfg.get('checkpoint_config') is None:
        cfg.checkpoint_config = dict(by_epoch=False, interval=2000)

    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # ── Logger ────────────────────────────────────────────────────────────────
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file  = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger    = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info('=' * 60)
    logger.info('SegMAN-Small (CNN branch) | IDD-20k-II Road Segmentation')
    logger.info('=' * 60)
    logger.info(f'Config    : {args.config}')
    logger.info(f'Work dir  : {args.work_dir}')
    logger.info(f'Max iters : {cfg.runner.max_iters}')
    logger.info(f'Batch size: {cfg.data.samples_per_gpu} samples/GPU')
    logger.info(f'GPU id    : {args.gpu_id}')
    logger.info(f'Seed      : {args.seed}')
    logger.info('')
    logger.info('CNN branch in SegMANDecoder:')
    logger.info('  conv_downsample_2 (3×3 stride-2) + pixel_unshuffle')
    logger.info('  conv_downsample_4 (5×5 stride-4) + pixel_unshuffle')
    logger.info('  reduce_channels (1×1 per scale) → concat → VSSM block')

    # ── Environment ───────────────────────────────────────────────────────────
    setup_multi_processes(cfg)
    env_info = '\n'.join(f'{k}: {v}' for k, v in collect_env().items())
    logger.info('Environment:\n' + '-' * 60 + '\n' + env_info + '\n' + '-' * 60)

    # ── Seed ──────────────────────────────────────────────────────────────────
    seed = init_random_seed(args.seed, device=cfg.device)
    set_random_seed(seed)
    cfg.seed = seed
    meta = dict(seed=seed, exp_name=os.path.basename(args.config))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )
    model.init_weights()

    warnings.filterwarnings('ignore', message='SyncBN is only supported.*')
    model = revert_sync_batchnorm(model)

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f'Parameters: {total_p:.1f}M total, {train_p:.1f}M trainable')

    # ── Dataset ───────────────────────────────────────────────────────────────
    datasets = [build_dataset(cfg.data.train)]
    logger.info(f'Training samples (with repeats): {len(datasets[0])}')

    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE,
        )
    model.CLASSES = datasets[0].CLASSES
    meta.update(cfg.checkpoint_config.meta)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info('Starting training...')
    train_start = time.time()

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=(not args.no_test),
        timestamp=timestamp,
        meta=meta,
    )

    train_elapsed = time.time() - train_start
    logger.info(
        f'Training finished in {train_elapsed / 3600:.2f} h ({train_elapsed:.0f} s)')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    test_metrics = {}
    if not args.no_test:
        best_ckpt = None
        for fname in sorted(os.listdir(cfg.work_dir)):
            if fname.startswith('best_mIoU') and fname.endswith('.pth'):
                best_ckpt = os.path.join(cfg.work_dir, fname)
        if best_ckpt is None:
            latest = os.path.join(cfg.work_dir, 'latest.pth')
            if os.path.exists(latest):
                best_ckpt = os.path.realpath(latest)

        if best_ckpt and os.path.exists(best_ckpt):
            test_metrics = run_evaluation(
                cfg, best_ckpt, args.gpu_id, logger, split='test')
            results_path = os.path.join(cfg.work_dir, 'test_results.json')
            with open(results_path, 'w') as f:
                json.dump({'checkpoint': best_ckpt, 'metrics': test_metrics},
                          f, indent=2)
            logger.info(f'Test results saved to: {results_path}')
        else:
            logger.warning('No checkpoint found; skipping evaluation.')

    # ── Metrics summary ───────────────────────────────────────────────────────
    log_json_path = None
    for fname in sorted(os.listdir(cfg.work_dir)):
        if fname.endswith('.log.json'):
            log_json_path = os.path.join(cfg.work_dir, fname)

    train_history, val_history = (
        parse_log_json(log_json_path) if log_json_path else ([], []))

    best_val_miou = best_val_iter = None
    for v in val_history:
        if v.get('mIoU') is not None:
            if best_val_miou is None or v['mIoU'] > best_val_miou:
                best_val_miou = v['mIoU']
                best_val_iter = v['iter']

    summary = {
        'experiment': {
            'description':        'SegMAN-Small (CNN branch) on IDD-20k-II',
            'config':             args.config,
            'work_dir':           args.work_dir,
            'seed':               seed,
            'max_iters':          cfg.runner.max_iters,
            'train_time_seconds': round(train_elapsed, 1),
            'cnn_branch_desc':    (
                'conv_downsample_2 (3x3 stride-2) + conv_downsample_4 (5x5 stride-4)'
                ' + pixel_unshuffle + reduce_channels + VSSM fusion'
            ),
        },
        'train': {
            'logged_iterations': len(train_history),
            'final_loss': train_history[-1].get('loss') if train_history else None,
            'final_lr':   train_history[-1].get('lr')   if train_history else None,
            'history':    train_history,
        },
        'validation': {
            'best_mIoU': best_val_miou,
            'best_iter': best_val_iter,
            'history':   val_history,
        },
        'test': test_metrics,
    }

    summary_path = os.path.join(cfg.work_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Final printout ────────────────────────────────────────────────────────
    sep = '=' * 60
    logger.info(sep)
    logger.info('RESULTS SUMMARY  —  SegMAN-Small CNN Branch | IDD-20k-II')
    logger.info(sep)

    if best_val_miou:
        logger.info(
            f'Best Val  mIoU      : {best_val_miou:.4f}  (iter {best_val_iter})')

    if test_metrics:
        def _fmt(k):
            v = test_metrics.get(k)
            return f'{v:.4f}' if isinstance(v, float) else str(v)

        logger.info('')
        logger.info('Test Set Metrics (val split used as test):')
        logger.info('')
        logger.info('  ── MMSeg pre_eval metrics ──')
        logger.info(f'  Overall Pixel Accuracy (aAcc)  : {_fmt("aAcc")}')
        logger.info(f'  Mean IoU               (mIoU)  : {_fmt("mIoU")}')
        logger.info(f'  Mean Acc               (mAcc)  : {_fmt("mAcc")}')
        logger.info(f'  Mean Dice             (mDice)  : {_fmt("mDice")}')
        logger.info(f'  Mean F1              (mFscore) : {_fmt("mFscore")}')
        logger.info(f'  Mean Precision         (mPrec) : {_fmt("mPrec")}')
        logger.info(f'  Mean Recall           (mRecall): {_fmt("mRecall")}')
        logger.info('')
        logger.info('  ── Per-class metrics ──')
        logger.info(f'  IoU  : bg={_fmt("IoU.background"):<8}  road={_fmt("IoU.road")}')
        logger.info(f'  Acc  : bg={_fmt("Acc.background"):<8}  road={_fmt("Acc.road")}')
        logger.info(f'  Dice : bg={_fmt("Dice.background"):<8}  road={_fmt("Dice.road")}')
        logger.info(f'  F1   : bg={_fmt("Fscore.background"):<8}  road={_fmt("Fscore.road")}')
        logger.info(f'  Prec : bg={_fmt("Precision.background"):<8}  road={_fmt("Precision.road")}')
        logger.info(f'  Rec  : bg={_fmt("Recall.background"):<8}  road={_fmt("Recall.road")}')
        logger.info('')
        logger.info('  ── Global pixel-level road metrics ──')
        logger.info(f'  Precision (road)       : {_fmt("precision_road")}')
        logger.info(f'  Recall    (road)       : {_fmt("recall_road")}')
        logger.info(f'  F1        (road)       : {_fmt("f1_road")}')
        logger.info(f'  IoU       (road)       : {_fmt("iou_road_global")}')
        logger.info(f'  Specificity (road)     : {_fmt("specificity_road")}')
        logger.info(f'  Accuracy  (overall)    : {_fmt("accuracy_overall")}')
        logger.info('')
        logger.info('  ── Image-level metrics ──')
        logger.info(f'  F1@50 (image-level)    : {_fmt("f1_at_50")}')

    logger.info(sep)
    logger.info(f'Checkpoints : {cfg.work_dir}/')
    logger.info(f'Full log    : {log_file}')
    logger.info(f'Summary     : {summary_path}')
    logger.info(sep)


if __name__ == '__main__':
    main()
