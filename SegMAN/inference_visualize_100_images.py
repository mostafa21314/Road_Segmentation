"""
Run the SegMAN-Tiny IDD checkpoint on 100 val images and save 4-panel figures
per image: Input | Ground Truth | Prediction | Error Map. No metrics; visualisation
only. Used for quick qualitative inspection of the baseline IDD model.

Run from the SegMAN root:
    conda run -n segmanaser python run_inference_100.py
"""
import os
import sys
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'segmentation'))

import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

from mmseg.models import build_segmentor
from mmseg.utils import build_dp
from mmseg.datasets.pipelines import Compose

warnings.filterwarnings('ignore')

CONFIG     = 'segmentation/work_dirs/segman_t_idd20k/segman_t_idd20k.py'
CHECKPOINT = 'segmentation/work_dirs/segman_t_idd20k/best_mIoU_iter_16000.pth'
OUT_DIR    = 'eval_results/inference_100'
NUM_IMAGES = 100
CLASSES    = ('background', 'road')
PALETTE    = [[0, 0, 0], [128, 64, 128]]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
cfg = mmcv.Config.fromfile(CONFIG)
cfg.model.pretrained = None
cfg.model.train_cfg  = None
cfg.gpu_ids = [0]

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
ckpt  = load_checkpoint(model, CHECKPOINT, map_location='cpu')
model.CLASSES = ckpt['meta'].get('CLASSES', CLASSES)
model.PALETTE = ckpt['meta'].get('PALETTE', PALETTE)
model = revert_sync_batchnorm(model)
model = build_dp(model, 'cuda', device_ids=[0])
model.eval()
print("Model ready.")

# ── Dataset paths ────────────────────────────────────────────────────────────
img_dir  = os.path.join(cfg.data.test.data_root, cfg.data.test.img_dir)
ann_dir  = os.path.join(cfg.data.test.data_root, cfg.data.test.ann_dir)
img_suf  = cfg.data.test.img_suffix
seg_suf  = cfg.data.test.seg_map_suffix

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(img_suf)])

# Evenly spread 100 images across the dataset
indices   = np.linspace(0, len(img_files) - 1, NUM_IMAGES, dtype=int)
selected  = [img_files[i] for i in indices]

# ── Inference pipeline ───────────────────────────────────────────────────────
test_pipeline = Compose(cfg.data.test.pipeline)

def run_inference(img_path):
    data = dict(img_info=dict(filename=img_path), img_prefix='')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [0])[0]
    with torch.no_grad():
        return model(return_loss=False, **data)[0].astype(np.uint8)

# ── Colorize mask ────────────────────────────────────────────────────────────
def colorize(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [20,  20,  20]
    rgb[mask == 1] = [128, 64, 128]
    return rgb

legend_cls = [
    mpatches.Patch(color=(20/255, 20/255, 20/255),   label='Background'),
    mpatches.Patch(color=(128/255, 64/255, 128/255), label='Road'),
]
legend_err = [
    mpatches.Patch(color=(0,     200/255, 0),          label='TP — Correct Road'),
    mpatches.Patch(color=(220/255, 50/255, 50/255),    label='FP — False Road'),
    mpatches.Patch(color=(255/255, 200/255, 0),        label='FN — Missed Road'),
    mpatches.Patch(color=(30/255,  30/255, 30/255),    label='TN — Correct BG'),
]

# ── Main loop ────────────────────────────────────────────────────────────────
print(f"Running inference on {NUM_IMAGES} images...")
for i, img_name in enumerate(selected):
    ann_name = img_name.replace(img_suf, seg_suf)
    img_path = os.path.join(img_dir, img_name)
    ann_path = os.path.join(ann_dir, ann_name)

    img_bgr  = cv2.imread(img_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt_mask  = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)

    pred = run_inference(img_path)
    if pred.shape != gt_mask.shape:
        pred = cv2.resize(pred, (gt_mask.shape[1], gt_mask.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # Per-image IoU and F1 for title
    inter = np.logical_and(pred == 1, gt_mask == 1).sum()
    union = np.logical_or( pred == 1, gt_mask == 1).sum()
    iou_r = inter / union if union > 0 else 0.0
    prec  = inter / (pred == 1).sum()  if (pred == 1).sum()  > 0 else 0.0
    rec   = inter / (gt_mask == 1).sum() if (gt_mask == 1).sum() > 0 else 0.0
    f1_r  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Error map
    err = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    err[(pred == 1) & (gt_mask == 1)] = [0,   200,  0]
    err[(pred == 1) & (gt_mask == 0)] = [220, 50,   50]
    err[(pred == 0) & (gt_mask == 1)] = [255, 200,  0]
    err[(pred == 0) & (gt_mask == 0)] = [30,  30,   30]
    overlay = (img_rgb * 0.45 + err * 0.55).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'[{i+1:03d}/{NUM_IMAGES}]  {img_name}   |   Road IoU = {iou_r:.3f}   F1 = {f1_r:.3f}',
        fontsize=10, fontweight='bold', y=1.01
    )

    axes[0].imshow(img_rgb)
    axes[0].set_title('Input Image', fontsize=11)

    axes[1].imshow(colorize(gt_mask))
    axes[1].set_title('Ground Truth', fontsize=11)
    axes[1].legend(handles=legend_cls, loc='lower left', fontsize=8, framealpha=0.85)

    axes[2].imshow(colorize(pred))
    axes[2].set_title('Prediction', fontsize=11)
    axes[2].legend(handles=legend_cls, loc='lower left', fontsize=8, framealpha=0.85)

    axes[3].imshow(overlay)
    axes[3].set_title('Error Map (on Image)', fontsize=11)
    axes[3].legend(handles=legend_err, loc='lower left', fontsize=7,
                   framealpha=0.85, ncol=2)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{i+1:03d}_{Path(img_name).stem}.png')
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1:3d}/{NUM_IMAGES}] {out_path}")

print(f"\nDone. All {NUM_IMAGES} visualizations saved to:\n  {os.path.abspath(OUT_DIR)}/")
