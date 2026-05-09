"""
Evaluate the SegMAN-Small CNN-branch checkpoint on the IDD-20k-II val split and
render 100 sample visualizations. Pixel-level metrics: IoU, F1, Precision, Recall,
Accuracy (Dice omitted — equivalent to F1 in the binary case).

Default checkpoint: work_dirs/segman_small_cnn_branch_idd20k/best_mIoU_iter_18000.pth

Run from the SegMAN root:
    conda run -n segmanaser python eval_and_visualize_cnn.py
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

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_dp
from mmseg.datasets.pipelines import Compose
from mmseg.core.evaluation.metrics import pre_eval_to_metrics

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG     = 'segmentation/work_dirs/segman_small_cnn_branch_idd20k/segman_s_cnn_branch_idd20k.py'
CHECKPOINT = 'segmentation/work_dirs/segman_small_cnn_branch_idd20k/best_mIoU_iter_18000.pth'
OUT_DIR    = 'eval_results/segman_s_cnn'
NUM_VIS    = 100
CLASSES    = ('background', 'road')
PALETTE    = [[0, 0, 0], [128, 64, 128]]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/visualizations', exist_ok=True)

# ── Build model ──────────────────────────────────────────────────────────────
print("=" * 64)
print("Loading SegMAN-S (CNN branch) model...")
cfg = mmcv.Config.fromfile(CONFIG)
cfg.model.pretrained = None
cfg.data.test.test_mode = True
cfg.gpu_ids = [0]
cfg.model.train_cfg = None

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
ckpt  = load_checkpoint(model, CHECKPOINT, map_location='cpu')
model.CLASSES = ckpt['meta'].get('CLASSES', CLASSES)
model.PALETTE = ckpt['meta'].get('PALETTE', PALETTE)
model = revert_sync_batchnorm(model)
model = build_dp(model, 'cuda', device_ids=[0])
model.eval()
print("Model loaded.")

# ── Dataset / dataloader ─────────────────────────────────────────────────────
print("\nBuilding test dataset...")
dataset    = build_dataset(cfg.data.test)
dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                              dist=False, shuffle=False)
print(f"Test set: {len(dataset)} images")

# ── Full evaluation (pre_eval) ───────────────────────────────────────────────
print("\nRunning inference on all test images for evaluation...")
results = single_gpu_test(model, dataloader, show=False, out_dir=None,
                          efficient_test=False, opacity=0.5, pre_eval=True)
print("Inference complete.")

# ── Global metrics (no Dice) ─────────────────────────────────────────────────
print("\nComputing metrics...")
ret = pre_eval_to_metrics(results, metrics=['mIoU', 'mFscore'], nan_to_num=0, beta=1)

macc      = float(ret['aAcc'])
iou_bg,   iou_road   = float(ret['IoU'][0]),       float(ret['IoU'][1])
acc_bg,   acc_road   = float(ret['Acc'][0]),        float(ret['Acc'][1])
f1_bg,    f1_road    = float(ret['Fscore'][0]),     float(ret['Fscore'][1])
prec_bg,  prec_road  = float(ret['Precision'][0]),  float(ret['Precision'][1])
rec_bg,   rec_road   = float(ret['Recall'][0]),     float(ret['Recall'][1])
miou  = (iou_bg  + iou_road)  / 2
mf1   = (f1_bg   + f1_road)   / 2
mprec = (prec_bg + prec_road) / 2
mrec  = (rec_bg  + rec_road)  / 2

header = f"{'Metric':<28} {'Background':>12} {'Road':>12} {'Mean':>10}"
sep    = "-" * 64
rows   = [
    ("IoU",            iou_bg,  iou_road,  miou),
    ("F1-Score",       f1_bg,   f1_road,   mf1),
    ("Precision",      prec_bg, prec_road, mprec),
    ("Recall",         rec_bg,  rec_road,  mrec),
    ("Class Accuracy", acc_bg,  acc_road,  macc),
]

print("\n" + "=" * 64)
print("EVALUATION RESULTS — SegMAN-S (CNN branch)")
print("=" * 64)
print(header); print(sep)
for name, bg, road, mean in rows:
    print(f"{name:<28} {bg:>12.4f} {road:>12.4f} {mean:>10.4f}")
print(sep)
print(f"{'Overall Pixel Accuracy':<28} {macc:>12.4f}")
print(f"{'mIoU':<28} {miou:>12.4f}")
print(f"{'Road F1 (pixel-wise)':<28} {f1_road:>12.4f}")
print("=" * 64)

with open(f'{OUT_DIR}/metrics.txt', 'w') as fh:
    fh.write("SegMAN-S (CNN branch) IDD20K Road Segmentation — Evaluation\n")
    fh.write(f"Checkpoint: {CHECKPOINT}\n")
    fh.write("=" * 64 + "\n")
    fh.write(header + "\n" + sep + "\n")
    for name, bg, road, mean in rows:
        fh.write(f"{name:<28} {bg:>12.4f} {road:>12.4f} {mean:>10.4f}\n")
    fh.write(sep + "\n")
    fh.write(f"{'Overall Pixel Accuracy':<28} {macc:>12.4f}\n")
    fh.write(f"{'mIoU':<28} {miou:>12.4f}\n")
    fh.write(f"{'Road F1 (pixel-wise)':<28} {f1_road:>12.4f}\n")
print(f"Saved → {OUT_DIR}/metrics.txt")

# ── Bar chart ────────────────────────────────────────────────────────────────
cats = ['Background', 'Road', 'Mean']
x    = np.arange(len(cats))
w    = 0.28

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('SegMAN-S (CNN branch) IDD20K — Evaluation Metrics',
             fontsize=14, fontweight='bold')

ax = axes[0]
for offset, vals, label, color in [
        (-w, [iou_bg,  iou_road,  miou],  'IoU',  '#4C72B0'),
        ( 0, [f1_bg,   f1_road,   mf1],   'F1',   '#C44E52'),
        ( w, [prec_bg, prec_road, mprec], 'Precision', '#55A868'),
]:
    bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
ax.set_title('IoU / F1 / Precision per Class')
ax.legend(); ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
acc_vals   = [acc_bg, acc_road, macc]
acc_labels = ['Background\nAccuracy', 'Road\nAccuracy', 'Overall\nPixel Acc']
bars2 = ax2.bar(acc_labels, acc_vals, color=['#4C72B0', '#55A868', '#8172B2'],
                alpha=0.85, width=0.5)
for b in bars2:
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
             f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=10)
ax2.set_ylim(0, 1.12); ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy per Class'); ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/metrics_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_DIR}/metrics_bar_chart.png")

# ── Confusion matrix ─────────────────────────────────────────────────────────
pre_eval_unpacked = tuple(zip(*results))
total_intersect = sum(pre_eval_unpacked[0]).numpy()
total_pred      = sum(pre_eval_unpacked[2]).numpy()
total_gt        = sum(pre_eval_unpacked[3]).numpy()

TP = total_intersect
FP = total_pred - total_intersect
FN = total_gt   - total_intersect
conf_matrix = np.array([[TP[0], FN[0]], [FP[0], TP[1]]])

fig, ax = plt.subplots(figsize=(6, 5))
norm = conf_matrix / conf_matrix.sum()
im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred: BG', 'Pred: Road'], fontsize=12)
ax.set_yticks([0, 1]); ax.set_yticklabels(['GT: BG',   'GT: Road'],   fontsize=12)
ax.set_title('Pixel-Level Confusion Matrix (normalized)', fontsize=12)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{norm[i,j]:.3f}\n({conf_matrix[i,j]:.1e})',
                ha='center', va='center', fontsize=10,
                color='white' if norm[i,j] > 0.4 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_DIR}/confusion_matrix.png")

# ── Per-image distributions ───────────────────────────────────────────────────
per_img_iou_road = []
per_img_f1_road  = []
for res in results:
    inter = res[0].numpy(); union = res[1].numpy()
    pred  = res[2].numpy(); gt    = res[3].numpy()
    iou_r = inter[1] / union[1] if union[1] > 0 else np.nan
    p     = inter[1] / pred[1]  if pred[1]  > 0 else 0.0
    r     = inter[1] / gt[1]    if gt[1]    > 0 else 0.0
    f1_r  = 2 * p * r / (p + r) if (p + r)  > 0 else 0.0
    per_img_iou_road.append(iou_r)
    per_img_f1_road.append(f1_r)

per_img_iou_road = np.array(per_img_iou_road)
per_img_f1_road  = np.array(per_img_f1_road)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Per-Image Metric Distribution — Road Class', fontsize=13, fontweight='bold')

valid_iou = per_img_iou_road[~np.isnan(per_img_iou_road)]
axes[0].hist(valid_iou, bins=40, color='#4C72B0', alpha=0.8, edgecolor='white')
axes[0].axvline(np.mean(valid_iou),   color='red',    linestyle='--', label=f'Mean={np.mean(valid_iou):.3f}')
axes[0].axvline(np.median(valid_iou), color='orange', linestyle='--', label=f'Median={np.median(valid_iou):.3f}')
axes[0].set_xlabel('Road IoU'); axes[0].set_ylabel('# Images')
axes[0].set_title('Road IoU Distribution'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].hist(per_img_f1_road, bins=40, color='#C44E52', alpha=0.8, edgecolor='white')
axes[1].axvline(np.mean(per_img_f1_road),   color='blue',   linestyle='--', label=f'Mean={np.mean(per_img_f1_road):.3f}')
axes[1].axvline(np.median(per_img_f1_road), color='orange', linestyle='--', label=f'Median={np.median(per_img_f1_road):.3f}')
axes[1].set_xlabel('F1-Score (Road)'); axes[1].set_ylabel('# Images')
axes[1].set_title('Road F1 Distribution'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/per_image_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_DIR}/per_image_distribution.png")

# ── Radar chart ───────────────────────────────────────────────────────────────
radar_labels = ['mIoU', 'Road IoU', 'Road F1', 'Road Precision', 'Road Recall', 'Pixel Acc']
radar_vals   = [miou, iou_road, f1_road, prec_road, rec_road, macc]

angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
v_plot = radar_vals + [radar_vals[0]]
a_plot = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.fill(a_plot, v_plot, alpha=0.25, color='#C44E52')
ax.plot(a_plot, v_plot, 'o-', linewidth=2, color='#C44E52')
ax.set_thetagrids(np.degrees(angles), radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.set_title('SegMAN-S (CNN branch) Metrics\n(IDD20K Road Segmentation)',
             fontsize=13, fontweight='bold', pad=20)
for angle, val in zip(angles, radar_vals):
    ax.annotate(f'{val:.3f}', xy=(angle, val), xytext=(angle, min(val + 0.09, 0.97)),
                ha='center', va='center', fontsize=9, color='#222')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/radar_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT_DIR}/radar_summary.png")


# ── 100-image visualization (same indices as SegMAN-T run) ───────────────────
print(f"\nGenerating {NUM_VIS} prediction visualizations...")

test_pipeline = Compose(cfg.data.test.pipeline)

img_dir_path = os.path.join(cfg.data.test.data_root, cfg.data.test.img_dir)
ann_dir_path = os.path.join(cfg.data.test.data_root, cfg.data.test.ann_dir)
img_suffix   = cfg.data.test.img_suffix
seg_suffix   = cfg.data.test.seg_map_suffix

img_files = sorted([f for f in os.listdir(img_dir_path) if f.endswith(img_suffix)])

# Same evenly-spread 100 indices used in the SegMAN-T run
indices  = np.linspace(0, len(img_files) - 1, NUM_VIS, dtype=int)
selected = [img_files[i] for i in indices]

def run_inference(img_path):
    data = dict(img_info=dict(filename=img_path), img_prefix='')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [0])[0]
    with torch.no_grad():
        return model(return_loss=False, **data)[0].astype(np.uint8)

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
    mpatches.Patch(color=(0,          200/255, 0),          label='TP — Correct Road'),
    mpatches.Patch(color=(220/255,    50/255,  50/255),      label='FP — False Road'),
    mpatches.Patch(color=(255/255,    200/255, 0),           label='FN — Missed Road'),
    mpatches.Patch(color=(30/255,     30/255,  30/255),      label='TN — Correct BG'),
]

for i, img_name in enumerate(selected):
    ann_name = img_name.replace(img_suffix, seg_suffix)
    img_path = os.path.join(img_dir_path, img_name)
    ann_path = os.path.join(ann_dir_path, ann_name)

    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    pred    = run_inference(img_path)
    if pred.shape != gt_mask.shape:
        pred = cv2.resize(pred, (gt_mask.shape[1], gt_mask.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    inter = np.logical_and(pred == 1, gt_mask == 1).sum()
    union = np.logical_or( pred == 1, gt_mask == 1).sum()
    iou_r = inter / union if union > 0 else 0.0
    p     = inter / (pred == 1).sum()    if (pred == 1).sum()    > 0 else 0.0
    r     = inter / (gt_mask == 1).sum() if (gt_mask == 1).sum() > 0 else 0.0
    f1_r  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    err = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    err[(pred == 1) & (gt_mask == 1)] = [0,   200, 0]
    err[(pred == 1) & (gt_mask == 0)] = [220, 50,  50]
    err[(pred == 0) & (gt_mask == 1)] = [255, 200, 0]
    err[(pred == 0) & (gt_mask == 0)] = [30,  30,  30]
    overlay = (img_rgb * 0.45 + err * 0.55).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'[{i+1:03d}/{NUM_VIS}]  {img_name}   |   Road IoU = {iou_r:.3f}   F1 = {f1_r:.3f}',
        fontsize=10, fontweight='bold', y=1.01
    )
    axes[0].imshow(img_rgb);           axes[0].set_title('Input Image',         fontsize=11)
    axes[1].imshow(colorize(gt_mask)); axes[1].set_title('Ground Truth',        fontsize=11)
    axes[1].legend(handles=legend_cls, loc='lower left', fontsize=8, framealpha=0.85)
    axes[2].imshow(colorize(pred));    axes[2].set_title('Prediction',          fontsize=11)
    axes[2].legend(handles=legend_cls, loc='lower left', fontsize=8, framealpha=0.85)
    axes[3].imshow(overlay);           axes[3].set_title('Error Map (on Image)', fontsize=11)
    axes[3].legend(handles=legend_err, loc='lower left', fontsize=7, framealpha=0.85, ncol=2)
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'visualizations', f'{i+1:03d}_{Path(img_name).stem}.png')
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1:3d}/{NUM_VIS}] {out_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(f"ALL OUTPUTS SAVED TO: {os.path.abspath(OUT_DIR)}/")
print("=" * 64)
print(f"  metrics.txt                  — full metric table")
print(f"  metrics_bar_chart.png        — bar chart")
print(f"  confusion_matrix.png         — pixel confusion matrix")
print(f"  per_image_distribution.png   — per-image IoU & F1 histograms")
print(f"  radar_summary.png            — radar chart")
print(f"  visualizations/ ({NUM_VIS} images)  — Input / GT / Pred / Error map")
print("=" * 64)
print("\nKEY METRICS:")
print(f"  mIoU            : {miou:.4f}")
print(f"  Road IoU        : {iou_road:.4f}")
print(f"  Road F1         : {f1_road:.4f}")
print(f"  Road Precision  : {prec_road:.4f}")
print(f"  Road Recall     : {rec_road:.4f}")
print(f"  Pixel Accuracy  : {macc:.4f}")
