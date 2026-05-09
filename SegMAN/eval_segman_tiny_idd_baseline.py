"""
Evaluate the SegMAN-Tiny IDD baseline checkpoint on the IDD-20k-II val split.
Loads CONFIG/CHECKPOINT, runs MMSeg single_gpu_test, computes pixel + image-level
metrics (mIoU, mAcc, F1, Precision, Recall, F1@IoU>=0.5), and saves confusion matrix,
metrics bar chart, radar summary, per-image distribution, and 4-panel overlays
(image | GT | prediction | error map) to OUT_DIR.

Run from the SegMAN root:
    conda run -n segmanaser python eval_and_visualize.py
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

# Add segmentation directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'segmentation'))

import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_dp
from mmseg.core.evaluation.metrics import pre_eval_to_metrics

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG     = 'segmentation/work_dirs/segman_t_idd20k/segman_t_idd20k.py'
CHECKPOINT = 'segmentation/work_dirs/segman_t_idd20k/best_mIoU_iter_16000.pth'
OUT_DIR    = 'eval_results'
NUM_VIS    = 10
CLASSES    = ('background', 'road')
PALETTE    = [[0, 0, 0], [128, 64, 128]]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/visualizations', exist_ok=True)

# ── Build model ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading config and model...")
cfg = mmcv.Config.fromfile(CONFIG)
cfg.model.pretrained = None
cfg.data.test.test_mode = True
cfg.gpu_ids = [0]
cfg.model.train_cfg = None

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, CHECKPOINT, map_location='cpu')
model.CLASSES = checkpoint['meta'].get('CLASSES', CLASSES)
model.PALETTE = checkpoint['meta'].get('PALETTE', PALETTE)

model = revert_sync_batchnorm(model)
model = build_dp(model, 'cuda', device_ids=[0])
model.eval()
print("Model loaded.")

# ── Build dataset / dataloader ───────────────────────────────────────────────
print("\nBuilding test dataset...")
dataset    = build_dataset(cfg.data.test)
dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                              dist=False, shuffle=False)
print(f"Test set: {len(dataset)} images")

# ── Run inference (pre_eval mode) ────────────────────────────────────────────
print("\nRunning inference on all test images...")
results = single_gpu_test(model, dataloader, show=False, out_dir=None,
                          efficient_test=False, opacity=0.5, pre_eval=True)
print("Inference complete.")

# ── Global metrics ───────────────────────────────────────────────────────────
print("\nComputing metrics...")
ret = pre_eval_to_metrics(results, metrics=['mIoU', 'mDice', 'mFscore'],
                          nan_to_num=0, beta=1)

# ret['aAcc']    → scalar  (overall pixel accuracy)
# ret['IoU']     → array[2]  per-class IoU
# ret['Acc']     → array[2]  per-class accuracy (recall)
# ret['Dice']    → array[2]  per-class Dice
# ret['Fscore']  → array[2]  per-class F1
# ret['Precision']→ array[2]  per-class precision
# ret['Recall']  → array[2]  per-class recall (same as Acc)

macc      = float(ret['aAcc'])
iou_bg,   iou_road   = float(ret['IoU'][0]),      float(ret['IoU'][1])
acc_bg,   acc_road   = float(ret['Acc'][0]),       float(ret['Acc'][1])
dice_bg,  dice_road  = float(ret['Dice'][0]),      float(ret['Dice'][1])
f1_bg,    f1_road    = float(ret['Fscore'][0]),    float(ret['Fscore'][1])
prec_bg,  prec_road  = float(ret['Precision'][0]), float(ret['Precision'][1])
rec_bg,   rec_road   = float(ret['Recall'][0]),    float(ret['Recall'][1])
miou  = (iou_bg  + iou_road)  / 2
mdice = (dice_bg + dice_road) / 2
mf1   = (f1_bg   + f1_road)   / 2

header = f"{'Metric':<28} {'Background':>12} {'Road':>12} {'Mean':>10}"
sep    = "-" * 64
rows   = [
    ("IoU",           iou_bg,  iou_road,  miou),
    ("Dice",          dice_bg, dice_road, mdice),
    ("F1-Score",      f1_bg,   f1_road,   mf1),
    ("Precision",     prec_bg, prec_road, (prec_bg+prec_road)/2),
    ("Recall",        rec_bg,  rec_road,  (rec_bg+rec_road)/2),
    ("Class Accuracy",acc_bg,  acc_road,  macc),
]

print("\n" + "=" * 64)
print("EVALUATION RESULTS")
print("=" * 64)
print(header)
print(sep)
for name, bg, road, mean in rows:
    print(f"{name:<28} {bg:>12.4f} {road:>12.4f} {mean:>10.4f}")
print(sep)
print(f"{'Overall Pixel Accuracy':<28} {macc:>12.4f}")
print(f"{'mIoU':<28} {miou:>12.4f}")
print(f"{'F1@50 Road':<28} {f1_road:>12.4f}")
print("=" * 64)

with open(f'{OUT_DIR}/metrics.txt', 'w') as fh:
    fh.write("SegMAN-T IDD20K Road Segmentation — Evaluation\n")
    fh.write(f"Checkpoint: {CHECKPOINT}\n")
    fh.write("=" * 64 + "\n")
    fh.write(header + "\n" + sep + "\n")
    for name, bg, road, mean in rows:
        fh.write(f"{name:<28} {bg:>12.4f} {road:>12.4f} {mean:>10.4f}\n")
    fh.write(sep + "\n")
    fh.write(f"{'Overall Pixel Accuracy':<28} {macc:>12.4f}\n")
    fh.write(f"{'mIoU':<28} {miou:>12.4f}\n")
    fh.write(f"{'F1@50 Road':<28} {f1_road:>12.4f}\n")

print(f"Saved → {OUT_DIR}/metrics.txt")

# ── Bar chart ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('SegMAN-T IDD20K — Evaluation Metrics', fontsize=14, fontweight='bold')

cats = ['Background', 'Road', 'Mean']
x    = np.arange(len(cats))
w    = 0.28

ax = axes[0]
for offset, vals, label, color in [
        (-w, [iou_bg,  iou_road,  miou],  'IoU',  '#4C72B0'),
        ( 0, [dice_bg, dice_road, mdice], 'Dice', '#55A868'),
        ( w, [f1_bg,   f1_road,   mf1],   'F1',   '#C44E52'),
]:
    bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylim(0, 1.1); ax.set_ylabel('Score'); ax.set_title('IoU / Dice / F1')
ax.legend(); ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
acc_vals   = [acc_bg, acc_road, macc]
acc_labels = ['Background\nAccuracy', 'Road\nAccuracy', 'Overall\nPixel Acc']
colors2    = ['#4C72B0', '#55A868', '#8172B2']
bars2 = ax2.bar(acc_labels, acc_vals, color=colors2, alpha=0.85, width=0.5)
for b in bars2:
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
             f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=10)
ax2.set_ylim(0, 1.1); ax2.set_ylabel('Accuracy'); ax2.set_title('Accuracy per Class')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/metrics_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_DIR}/metrics_bar_chart.png")

# ── Confusion-matrix pixel breakdown ────────────────────────────────────────
pre_eval_results = tuple(zip(*results))
total_intersect = sum(pre_eval_results[0]).numpy()
total_pred      = sum(pre_eval_results[2]).numpy()
total_gt        = sum(pre_eval_results[3]).numpy()

TP = total_intersect
FP = total_pred - total_intersect
FN = total_gt   - total_intersect
TN_bg = total_gt[0] - FP[0]  # approximate: background correct = gt_bg - bg_pred_as_road

conf_matrix = np.array([[TP[0], FN[0]],
                         [FP[0], TP[1]]])

fig, ax = plt.subplots(figsize=(6, 5))
norm_conf = conf_matrix / conf_matrix.sum()
im = ax.imshow(norm_conf, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred: BG', 'Pred: Road'], fontsize=12)
ax.set_yticks([0, 1]); ax.set_yticklabels(['GT: BG',   'GT: Road'],   fontsize=12)
ax.set_title('Pixel-Level Confusion Matrix\n(normalized)', fontsize=12)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{norm_conf[i,j]:.3f}\n({conf_matrix[i,j]:.1e})',
                ha='center', va='center', fontsize=10,
                color='white' if norm_conf[i,j] > 0.4 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_DIR}/confusion_matrix.png")

# ── Per-image metric distribution ────────────────────────────────────────────
per_img_iou_road = []
per_img_f1_road  = []
for res in results:
    inter = res[0].numpy()
    union = res[1].numpy()
    pred  = res[2].numpy()
    gt    = res[3].numpy()
    iou_r = inter[1] / union[1] if union[1] > 0 else np.nan
    prec  = inter[1] / pred[1]  if pred[1]  > 0 else 0.0
    rec   = inter[1] / gt[1]    if gt[1]    > 0 else 0.0
    f1_r  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    per_img_iou_road.append(iou_r)
    per_img_f1_road.append(f1_r)

per_img_iou_road = np.array(per_img_iou_road)
per_img_f1_road  = np.array(per_img_f1_road)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Per-Image Metric Distribution — Road Class', fontsize=13, fontweight='bold')

valid_iou = per_img_iou_road[~np.isnan(per_img_iou_road)]
axes[0].hist(valid_iou, bins=40, color='#4C72B0', alpha=0.8, edgecolor='white')
axes[0].axvline(np.mean(valid_iou), color='red', linestyle='--',
                label=f'Mean = {np.mean(valid_iou):.3f}')
axes[0].axvline(np.median(valid_iou), color='orange', linestyle='--',
                label=f'Median = {np.median(valid_iou):.3f}')
axes[0].set_xlabel('Road IoU'); axes[0].set_ylabel('# Images')
axes[0].set_title('Distribution of Road IoU per Image')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].hist(per_img_f1_road, bins=40, color='#C44E52', alpha=0.8, edgecolor='white')
axes[1].axvline(np.mean(per_img_f1_road), color='blue', linestyle='--',
                label=f'Mean = {np.mean(per_img_f1_road):.3f}')
axes[1].axvline(np.median(per_img_f1_road), color='orange', linestyle='--',
                label=f'Median = {np.median(per_img_f1_road):.3f}')
axes[1].set_xlabel('F1-Score (Road)'); axes[1].set_ylabel('# Images')
axes[1].set_title('Distribution of Road F1 per Image')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/per_image_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_DIR}/per_image_distribution.png")

# ── Radar chart ──────────────────────────────────────────────────────────────
radar_labels = ['mIoU', 'Road IoU', 'Road F1', 'Road Dice', 'Road Precision',
                'Road Recall', 'Pixel Acc']
radar_vals   = [miou, iou_road, f1_road, dice_road, prec_road, rec_road, macc]

angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
v_plot = radar_vals + [radar_vals[0]]
a_plot = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.fill(a_plot, v_plot, alpha=0.25, color='#4C72B0')
ax.plot(a_plot, v_plot, 'o-', linewidth=2, color='#4C72B0')
ax.set_thetagrids(np.degrees(angles), radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.set_title('SegMAN-T Metrics Overview\n(IDD20K Road Segmentation)',
             fontsize=13, fontweight='bold', pad=20)
for angle, val in zip(angles, radar_vals):
    ax.annotate(f'{val:.3f}', xy=(angle, val), xytext=(angle, min(val + 0.09, 0.97)),
                ha='center', va='center', fontsize=9, color='#222')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/radar_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_DIR}/radar_summary.png")


# ── Visualization: prediction vs ground truth ────────────────────────────────
print(f"\nGenerating {NUM_VIS} prediction visualizations...")

from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

test_pipeline = Compose(cfg.data.test.pipeline)

def colorize(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [20,  20,  20]   # background: near-black
    rgb[mask == 1] = [128, 64, 128]   # road: purple
    return rgb

def run_inference(img_path):
    data = dict(img_info=dict(filename=img_path), img_prefix='')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [0])[0]
    with torch.no_grad():
        return model(return_loss=False, **data)[0]

img_dir_path = os.path.join(cfg.data.test.data_root, cfg.data.test.img_dir)
ann_dir_path = os.path.join(cfg.data.test.data_root, cfg.data.test.ann_dir)
img_suffix   = cfg.data.test.img_suffix
seg_suffix   = cfg.data.test.seg_map_suffix

img_files = sorted([f for f in os.listdir(img_dir_path) if f.endswith(img_suffix)])

# Select: spread + hard cases (low IoU)
n = len(img_files)
spread_idx = np.linspace(0, n - 1, NUM_VIS // 2, dtype=int).tolist()
valid_sorted = np.argsort(per_img_iou_road[~np.isnan(per_img_iou_road)])
hard_idx = [i for i in np.argsort(per_img_iou_road) if not np.isnan(per_img_iou_road[i])][:NUM_VIS // 2]
vis_idx  = list(dict.fromkeys(spread_idx + hard_idx))[:NUM_VIS]

for k, idx in enumerate(vis_idx):
    img_name = img_files[idx]
    ann_name = img_name.replace(img_suffix, seg_suffix)
    img_path = os.path.join(img_dir_path, img_name)
    ann_path = os.path.join(ann_dir_path, ann_name)

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)

    pred_mask = run_inference(img_path).astype(np.uint8)
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    iou_r = float(per_img_iou_road[idx]) if not np.isnan(per_img_iou_road[idx]) else 0.0
    f1_r  = float(per_img_f1_road[idx])

    # Error map
    err = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    err[(pred_mask == 1) & (gt_mask == 1)] = [0,   200,  0]    # TP green
    err[(pred_mask == 1) & (gt_mask == 0)] = [220, 50,   50]   # FP red
    err[(pred_mask == 0) & (gt_mask == 1)] = [255, 200,  0]    # FN yellow
    err[(pred_mask == 0) & (gt_mask == 0)] = [30,  30,   30]   # TN dark
    overlay = (img_rgb * 0.4 + err * 0.6).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f'{img_name}  |  Road IoU = {iou_r:.3f}   F1 = {f1_r:.3f}',
                 fontsize=11, fontweight='bold', y=1.01)

    axes[0].imshow(img_rgb);             axes[0].set_title('Input Image',    fontsize=11)
    axes[1].imshow(colorize(gt_mask));   axes[1].set_title('Ground Truth',   fontsize=11)
    axes[2].imshow(colorize(pred_mask)); axes[2].set_title('Prediction',     fontsize=11)
    axes[3].imshow(overlay)
    legend_handles = [
        mpatches.Patch(color=(0,200/255,0),         label='TP — Correct Road'),
        mpatches.Patch(color=(220/255,50/255,50/255),label='FP — False Road'),
        mpatches.Patch(color=(255/255,200/255,0),    label='FN — Missed Road'),
        mpatches.Patch(color=(30/255,30/255,30/255), label='TN — Correct BG'),
    ]
    axes[3].legend(handles=legend_handles, loc='lower left', fontsize=8,
                   framealpha=0.85, ncol=2)
    axes[3].set_title('Error Map (on image)', fontsize=11)

    for ax in axes:
        ax.axis('off')

    # Add colour legend to GT/Pred panels
    for ax_idx in [1, 2]:
        legend_cls = [
            mpatches.Patch(color=(20/255,20/255,20/255),  label='Background'),
            mpatches.Patch(color=(128/255,64/255,128/255), label='Road'),
        ]
        axes[ax_idx].legend(handles=legend_cls, loc='lower left', fontsize=8, framealpha=0.85)

    plt.tight_layout()
    out_name = f'vis_{k+1:02d}_{Path(img_name).stem}.png'
    plt.savefig(f'{OUT_DIR}/visualizations/{out_name}', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [{k+1}/{len(vis_idx)}] Saved → {OUT_DIR}/visualizations/{out_name}")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(f"ALL OUTPUTS SAVED TO: {OUT_DIR}/")
print("=" * 64)
print(f"  metrics.txt                  — full metric table")
print(f"  metrics_bar_chart.png        — bar chart (IoU / Dice / F1 / Acc)")
print(f"  confusion_matrix.png         — pixel-level confusion matrix")
print(f"  per_image_distribution.png   — per-image IoU & F1 histograms")
print(f"  radar_summary.png            — radar/spider overview chart")
print(f"  visualizations/ ({len(vis_idx)} images)  — prediction vs GT panels")
print("=" * 64)
print("\nKEY METRICS SUMMARY:")
print(f"  mIoU            : {miou:.4f}")
print(f"  Road IoU        : {iou_road:.4f}")
print(f"  Road F1-Score   : {f1_road:.4f}")
print(f"  Road Dice       : {dice_road:.4f}")
print(f"  Road Precision  : {prec_road:.4f}")
print(f"  Road Recall     : {rec_road:.4f}")
print(f"  Pixel Accuracy  : {macc:.4f}")
print(f"  Road Accuracy   : {acc_road:.4f}")
