"""
Generate driveable-area overlays using the SegMAN-tiny IDD20k checkpoint.

Runs SegMAN inference on a random sample of CULane rural training images and
overlays the predicted driveable mask against the GT lane annotations, producing
visualisations similar to vehicle_masks_overlays/.

Overlay colour key (BGR):
  BLUE   — driveable area predicted by SegMAN (class 1)
  GREEN  — GT lane pixel that falls ON the driveable surface
  RED    — GT lane pixel that falls OFF the driveable surface (unusual — flags
            cases where the annotator drew a lane where SegMAN sees undriveable)
  (non-lane driveable area is shown as blue only)

Output:
    <dataset_root>/driveable_overlays/<stem>.png

Environment: run in the `segman` conda env from the SegMAN segmentation dir,
or any env that has mmseg installed (pip install -e . inside SegMAN/segmentation).

Usage:
    cd /home/g6/Mostafa/Road_Segmentation/SegMAN/segmentation
    python ../../Scripts/generate_driveable_overlays.py
"""

import os
import sys
import random
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
SEGMAN_SEG_DIR = Path("/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation")
CONFIG_PATH    = SEGMAN_SEG_DIR / "work_dirs/segman_t_idd20k/segman_t_idd20k.py"
CKPT_PATH      = SEGMAN_SEG_DIR / "work_dirs/segman_t_idd20k/best_mIoU_iter_16000.pth"

DATASET_ROOT   = Path("/home/g6/Mostafa/Road_Segmentation/CULane_Rural_Subset(1)/CULane_Rural_Subset")
OUT_DIR        = DATASET_ROOT / "driveable_overlays"

SPLIT    = "test"
SAMPLE_FRACTION = 0.5          # 50% of the split
SEED      = 42
ROAD_CLASS = 1   # class 1 = road/driveable in the IDD20k model
# ─────────────────────────────────────────────────────────────────────────────

# mmseg must be importable — add SegMAN/segmentation to sys.path if needed
sys.path.insert(0, str(SEGMAN_SEG_DIR))

try:
    from mmseg.apis import init_segmentor, inference_segmentor
except ImportError:
    print("[ERROR] Cannot import mmseg. Run this script in the `segman` conda "
          "env from inside SegMAN/segmentation/ after `pip install -e .`")
    sys.exit(1)


def load_model():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    print(f"Loading SegMAN from {CKPT_PATH.name} ...")
    model = init_segmentor(str(CONFIG_PATH), str(CKPT_PATH), device='cuda:0')

    # SegMAN decoder uses pixel_unshuffle(factor=2) on stage-3 features (input/16),
    # so input spatial dims must be multiples of 32. CULane's 1640x590 resized
    # with keep_ratio to img_scale=(1024, 576) becomes 1024x368 (368/16=23, odd
    # -> pixel_unshuffle fails). Insert a Pad step with size_divisor=32 in the
    # MultiScaleFlipAug transforms so the resized image is padded to the next
    # multiple of 32 (e.g. 1024x384) before hitting the model.
    ms_aug = model.cfg.data.test.pipeline[1]
    transforms = ms_aug['transforms']
    has_pad = any(t.get('type') == 'Pad' for t in transforms)
    if not has_pad:
        for i, t in enumerate(transforms):
            if t.get('type') == 'Normalize':
                transforms.insert(i, dict(type='Pad', size_divisor=32,
                                          pad_val=0, seg_pad_val=255))
                break
        print("  Inserted Pad(size_divisor=32) into test pipeline.")

    print("  Model ready.\n")
    return model


def make_overlay(img_bgr, driveable_mask, lane_mask):
    """
    img_bgr       : (H, W, 3) original image
    driveable_mask: (H, W)    bool  — True where SegMAN predicts road
    lane_mask     : (H, W)    uint8 — values 0 (bg) or 1-4 (lane IDs)
    """
    out = (img_bgr * 0.55).astype(np.uint8)

    lane_any     = (lane_mask >= 1) & (lane_mask <= 4)
    lane_on_road = lane_any & driveable_mask
    lane_off_road = lane_any & ~driveable_mask

    # blue driveable area (non-lane)
    only_drive = driveable_mask & ~lane_any
    out[only_drive] = (0.45 * out[only_drive].astype(np.float32) +
                       np.array([180, 60, 0], dtype=np.float32)).astype(np.uint8)

    # red — GT lane on NON-driveable (potential problem area)
    out[lane_off_road] = np.array([0, 0, 200], dtype=np.uint8)

    # green — GT lane on driveable (expected case)
    out[lane_on_road] = (0.3 * out[lane_on_road].astype(np.float32) +
                         np.array([0, 200, 0], dtype=np.float32)).astype(np.uint8)

    return out, lane_on_road.sum(), lane_off_road.sum()


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_dir  = DATASET_ROOT / SPLIT
    lane_dir = DATASET_ROOT / "laneseg_label_w16" / SPLIT

    if not img_dir.is_dir():
        print(f"[ERROR] Image dir not found: {img_dir}")
        return
    if not lane_dir.is_dir():
        print(f"[ERROR] Lane mask dir not found: {lane_dir}. "
              "Run Scripts/generate_seg_masks.py first.")
        return

    all_jpgs = sorted(img_dir.glob("*.jpg"))
    if not all_jpgs:
        print(f"[ERROR] No .jpg files in {img_dir}")
        return

    n_pick = max(1, int(round(len(all_jpgs) * SAMPLE_FRACTION)))
    picks = random.sample(all_jpgs, n_pick)
    print(f"Sampling {n_pick} / {len(all_jpgs)} images from {SPLIT}/ "
          f"({100*SAMPLE_FRACTION:.0f}%)\n")
    model = load_model()

    total_on  = 0
    total_off = 0

    skipped = 0
    for jpg_path in tqdm(picks, desc=f"{SPLIT} overlays"):
        stem = jpg_path.stem
        lane_path = lane_dir / (stem + ".png")

        lane_mask = cv2.imread(str(lane_path), cv2.IMREAD_GRAYSCALE)
        if lane_mask is None:
            skipped += 1
            continue

        img_bgr = cv2.imread(str(jpg_path))

        # SegMAN inference — returns list of (H, W) int arrays (one per image)
        result = inference_segmentor(model, str(jpg_path))
        pred = np.array(result[0], dtype=np.uint8)  # (H', W')

        # Resize prediction to original image resolution if needed
        if pred.shape != img_bgr.shape[:2]:
            pred = cv2.resize(pred, (img_bgr.shape[1], img_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        driveable = (pred == ROAD_CLASS)

        out, n_on, n_off = make_overlay(img_bgr, driveable, lane_mask)
        total_on  += n_on
        total_off += n_off

        out_name = f"{stem}__on{n_on}_off{n_off}.png"
        cv2.imwrite(str(OUT_DIR / out_name), out)

    if skipped:
        print(f"  Skipped {skipped} images with missing lane masks")

    print(f"\nWrote {len(picks)} overlays to {OUT_DIR}")
    print(f"Total lane pixels ON  driveable surface : {total_on}")
    print(f"Total lane pixels OFF driveable surface : {total_off}")
    pct = 100 * total_off / (total_on + total_off + 1e-6)
    print(f"Overall off-road lane fraction          : {pct:.1f}%")
    print()
    print("Colour key:")
    print("  BLUE  — driveable area (non-lane)")
    print("  GREEN — GT lane on driveable surface  (expected)")
    print("  RED   — GT lane on NON-driveable      (SegMAN disagrees with annotator)")


if __name__ == "__main__":
    main()
