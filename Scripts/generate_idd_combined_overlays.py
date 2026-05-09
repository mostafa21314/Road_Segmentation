"""
Stage 2 of the IDD combined overlay pipeline.

Reads FENet lane predictions cached by Scripts/export_idd_fenet_lanes.py and
overlays them together with SegMAN(IDD20k) driveable predictions on each
IDD validation image.

Overlay (BGR):
  BLUE polygonal fill — driveable area predicted by SegMAN (class 1)
  COLOURED polylines  — FENet predicted lanes (one colour per lane)

Two-stage pipeline:
    # 1) in the fenet env (python 3.8, has fenet.ops.nms_impl)
    deactivate  # if a non-conda venv is currently active
    conda activate fenet
    cd FENet && python ../Scripts/export_idd_fenet_lanes.py

    # 2) in the segman env (python 3.10, has mmseg)
    deactivate; conda activate segman
    cd SegMAN/segmentation && python /home/g6/Mostafa/Road_Segmentation/Scripts/generate_idd_combined_overlays.py

(The pipeline is split because FENet's nms_impl .so is built for py3.8 and
can't be loaded inside the py3.10 segman env.)

Output:
    /home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation/data/idd20k/
        combined_overlays/<stem>.png
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
SEGMAN_SEG_DIR = Path("/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation")
SEGMAN_CFG     = SEGMAN_SEG_DIR / "work_dirs/segman_t_idd20k/segman_t_idd20k.py"
SEGMAN_CKPT    = SEGMAN_SEG_DIR / "work_dirs/segman_t_idd20k/best_mIoU_iter_16000.pth"

IDD_IMG_DIR = SEGMAN_SEG_DIR / "data/idd20k/img_dir/val"
CACHE_DIR   = SEGMAN_SEG_DIR / "data/idd20k/fenet_lanes_cache"
OUT_DIR     = SEGMAN_SEG_DIR / "data/idd20k/combined_overlays"

ROAD_CLASS  = 1
# ─────────────────────────────────────────────────────────────────────────────

# 12 distinct BGR colours for lanes (same style as FENet's visualization.COLORS)
LANE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (255, 0, 128), (0, 128, 255), (0, 255, 128),
]

sys.path.insert(0, str(SEGMAN_SEG_DIR))
try:
    from mmseg.apis import init_segmentor, inference_segmentor
except ImportError:
    print("[ERROR] mmseg not importable. Run in the `segman` conda env.")
    sys.exit(1)


def load_segman():
    print(f"Loading SegMAN from {SEGMAN_CKPT.name} ...")
    model = init_segmentor(str(SEGMAN_CFG), str(SEGMAN_CKPT), device='cuda:0')

    # pixel_unshuffle(factor=2) at stage-3 features (input/16) requires H,W
    # divisible by 32 — pad the resized image before the model sees it.
    ms_aug = model.cfg.data.test.pipeline[1]
    transforms = ms_aug['transforms']
    if not any(t.get('type') == 'Pad' for t in transforms):
        for i, t in enumerate(transforms):
            if t.get('type') == 'Normalize':
                transforms.insert(i, dict(type='Pad', size_divisor=32,
                                          pad_val=0, seg_pad_val=255))
                break
        print("  Inserted Pad(size_divisor=32) into SegMAN test pipeline.")
    print("  SegMAN ready.\n")
    return model


def draw_lanes(img, lanes, width=6):
    polylines = []
    for pts in lanes:
        xys = [(int(x), int(y)) for x, y in pts if x > 0 and y > 0]
        polylines.append(xys)
    polylines.sort(key=lambda xys: xys[0][0] if xys else 0)

    for idx, xys in enumerate(polylines):
        color = LANE_COLORS[idx % len(LANE_COLORS)]
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], color, thickness=width)


def render_overlay(img_bgr, driveable, lanes):
    out = (img_bgr * 0.6).astype(np.uint8)

    if driveable.shape != out.shape[:2]:
        driveable = cv2.resize(
            driveable.astype(np.uint8),
            (out.shape[1], out.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    out[driveable] = (
        0.45 * out[driveable].astype(np.float32)
        + np.array([180, 60, 0], dtype=np.float32)
    ).astype(np.uint8)

    draw_lanes(out, lanes, width=6)
    return out


def main():
    if not CACHE_DIR.is_dir():
        print(f"[ERROR] Lane cache dir not found: {CACHE_DIR}")
        print("        Run Scripts/export_idd_fenet_lanes.py first "
              "(inside the fenet env).")
        return

    cache_files = sorted(CACHE_DIR.glob("*.npy"))
    if not cache_files:
        print(f"[ERROR] No cached lane .npy files in {CACHE_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(cache_files)} cached lane files → rendering overlays")
    model = load_segman()

    missing = 0
    for npy_path in tqdm(cache_files, desc="IDD overlays"):
        jpg_path = IDD_IMG_DIR / f"{npy_path.stem}.jpg"
        if not jpg_path.exists():
            missing += 1
            continue

        img_bgr = cv2.imread(str(jpg_path))
        if img_bgr is None:
            missing += 1
            continue

        result = inference_segmentor(model, str(jpg_path))
        drive  = np.asarray(result[0], dtype=np.uint8) == ROAD_CLASS

        lanes = np.load(str(npy_path), allow_pickle=True)
        lanes = [np.asarray(l, dtype=np.float32) for l in lanes]

        out = render_overlay(img_bgr, drive, lanes)
        cv2.imwrite(str(OUT_DIR / f"{npy_path.stem}.png"), out)

    if missing:
        print(f"  Skipped {missing} images (missing .jpg or unreadable).")
    print(f"\nWrote overlays to {OUT_DIR}")
    print("Colour key:")
    print("  BLUE      — SegMAN driveable area")
    print("  polylines — FENet predicted lanes (one colour each)")


if __name__ == "__main__":
    main()
