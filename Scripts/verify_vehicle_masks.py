"""
Phase 2: Sanity-check the "GT lanes pass through vehicles" assumption.

The occluded-lane loss only makes sense if CULane's ground-truth lane polylines
actually extend through vehicles (i.e. annotators drew lanes as if the car
weren't there). If they stopped at vehicles instead, up-weighting those pixels
would teach the model the wrong thing.

This script:
  1. Picks N random training images that have a non-empty YOLO vehicle mask.
  2. For each, writes a side-by-side-style overlay showing:
        - original image (darkened)
        - GT lane pixels in GREEN    (laneseg_label_w16, classes 1-4)
        - vehicle mask   in RED      (semi-transparent)
        - overlap        in YELLOW   ← these are the pixels the loss targets
  3. Prints how many picks actually have overlap.

Output:
    <dataset_root>/vehicle_masks_overlays/*.png

Inspect the overlays visually: if yellow pixels appear INSIDE cars (i.e. the
lane polyline passes under the vehicle), the assumption holds. If lanes
clearly stop at vehicle edges, reconsider the loss.
"""

import random
import numpy as np
import cv2
from pathlib import Path

DATASET_ROOT = Path("/home/g6/Mostafa/Road_Segmentation/CULane_Rural_Subset(1)/CULane_Rural_Subset")
OUT_DIR = DATASET_ROOT / "vehicle_masks_overlays"
N_SAMPLES = 12
SPLIT = "train"
SEED = 42


def make_overlay(img, lane_mask, veh_mask):
    out = (img * 0.6).astype(np.uint8)  # darken for contrast

    lane_any = (lane_mask >= 1) & (lane_mask <= 4)
    veh_any  = veh_mask > 0
    overlap  = lane_any & veh_any

    # OpenCV uses BGR
    out[veh_any]  = (0.5 * out[veh_any].astype(np.float32) +
                     np.array([0, 0, 180], dtype=np.float32)).astype(np.uint8)
    out[lane_any] = (0.3 * out[lane_any].astype(np.float32) +
                     np.array([0, 200, 0], dtype=np.float32)).astype(np.uint8)
    out[overlap]  = np.array([0, 255, 255], dtype=np.uint8)  # yellow
    return out


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_dir  = DATASET_ROOT / SPLIT
    lane_dir = DATASET_ROOT / "laneseg_label_w16" / SPLIT
    veh_dir  = DATASET_ROOT / "vehicle_masks" / SPLIT

    if not veh_dir.is_dir():
        print(f"[ERROR] Vehicle masks not found at {veh_dir}")
        print("        Run Scripts/generate_vehicle_masks.py first.")
        return

    veh_files = sorted(veh_dir.glob("*.png"))
    if not veh_files:
        print(f"[ERROR] {veh_dir} is empty.")
        return

    print(f"Scanning {len(veh_files)} vehicle masks in {SPLIT}/...")
    candidates = []
    for vp in veh_files:
        m = cv2.imread(str(vp), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.any():
            candidates.append(vp.stem)

    print(f"  {len(candidates)} images have at least one vehicle "
          f"({100.0*len(candidates)/len(veh_files):.1f}%)")
    if not candidates:
        print("[WARN] No vehicle detections — nothing to overlay.")
        return

    picks = random.sample(candidates, min(N_SAMPLES, len(candidates)))

    n_with_overlap = 0
    total_overlap_pixels = 0
    for stem in picks:
        img       = cv2.imread(str(img_dir  / (stem + ".jpg")))
        lane_mask = cv2.imread(str(lane_dir / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
        veh_mask  = cv2.imread(str(veh_dir  / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if img is None or lane_mask is None or veh_mask is None:
            print(f"  [SKIP] Missing inputs for {stem}")
            continue

        overlap_pixels = int(((lane_mask >= 1) & (lane_mask <= 4) & (veh_mask > 0)).sum())
        if overlap_pixels > 0:
            n_with_overlap += 1
        total_overlap_pixels += overlap_pixels

        out = make_overlay(img, lane_mask, veh_mask)
        cv2.imwrite(str(OUT_DIR / f"{stem}__overlap{overlap_pixels}.png"), out)

    print(f"\nWrote {len(picks)} overlays to {OUT_DIR}")
    print(f"  {n_with_overlap}/{len(picks)} picks have GT lane pixels inside a vehicle mask")
    print(f"  total overlapping pixels across picks: {total_overlap_pixels}")
    print("\nIf yellow pixels consistently appear inside cars, the occluded-lane")
    print("loss has a valid signal. If lanes clearly stop at vehicle edges, the")
    print("assumption breaks and the loss should be reconsidered.")


if __name__ == "__main__":
    main()
