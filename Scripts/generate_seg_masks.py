"""
Step 2: Generate laneseg_label_w16 segmentation masks for FENet.

FENet's base_dataset.py loads masks as:
    label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
and uses them as a per-pixel segmentation target with num_classes=5:
    0 = background
    1 = lane 1
    2 = lane 2
    3 = lane 3
    4 = lane 4

Output structure (mirrors image paths):
    <dataset_root>/laneseg_label_w16/train/driver_xxx.png
    <dataset_root>/laneseg_label_w16/val/driver_xxx.png
    <dataset_root>/laneseg_label_w16/test/driver_xxx.png

Annotation files are expected as *.lines.txt (created by rename_annotations.py).
Falls back to *.txt if *.lines.txt is not found.

Adapted from SegMAN/convert_culane_to_masks.py, with multi-class lane IDs
instead of binary masks.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT = Path("/home/g6/Mostafa/Road_Segmentation/CULane_Rural_Subset(1)/CULane_Rural_Subset")
SPLITS = ["train", "val", "test"]
IMG_HEIGHT = 590
IMG_WIDTH = 1640
LANE_THICKNESS = 16  # matches the w16 in laneseg_label_w16


def parse_annotation(anno_path: Path):
    """Parse a CULane *.lines.txt annotation file.
    Returns a list of lanes, each lane is a list of (x, y) int tuples."""
    lanes = []
    if not anno_path.exists():
        return lanes
    with open(anno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(float, line.split()))
            points = []
            for i in range(0, len(values) - 1, 2):
                x, y = int(round(values[i])), int(round(values[i + 1]))
                points.append((x, y))
            if len(points) >= 2:
                lanes.append(points)
    return lanes


def create_mask(lanes, height=IMG_HEIGHT, width=IMG_WIDTH, thickness=LANE_THICKNESS):
    """Create a multi-class segmentation mask.
    Each lane is drawn with its 1-based index (1, 2, 3, 4).
    Later lanes overwrite earlier ones if they overlap."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for lane_idx, lane in enumerate(lanes):
        lane_id = lane_idx + 1  # 1-indexed
        if lane_id > 4:
            # CULane standard is 4 lanes; skip extras
            break
        pts = np.array(lane, dtype=np.int32)
        # Filter points outside the image
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < width) & \
                (pts[:, 1] >= 0) & (pts[:, 1] < height)
        pts = pts[valid]
        if len(pts) >= 2:
            cv2.polylines(mask, [pts], isClosed=False,
                          color=lane_id, thickness=thickness)
    return mask


def process_split(split: str):
    split_dir = DATASET_ROOT / split
    out_dir = DATASET_ROOT / "laneseg_label_w16" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(split_dir.glob("*.jpg"))
    if not jpg_files:
        print(f"  [WARN] No .jpg files found in {split_dir}")
        return

    missing_anno = 0
    for jpg_path in tqdm(jpg_files, desc=f"{split}"):
        stem = jpg_path.stem  # e.g. driver_xxx.MP4_00150

        # Prefer .lines.txt (after step 1), fall back to .txt
        anno_path = jpg_path.with_name(stem + ".lines.txt")
        if not anno_path.exists():
            anno_path = jpg_path.with_suffix(".txt")

        if not anno_path.exists():
            missing_anno += 1

        lanes = parse_annotation(anno_path)
        mask = create_mask(lanes)

        out_path = out_dir / (stem + ".png")
        cv2.imwrite(str(out_path), mask)

    if missing_anno:
        print(f"  [WARN] {missing_anno} images had no annotation file — saved as all-background masks")

    generated = len(list(out_dir.glob("*.png")))
    print(f"  Generated {generated} masks → {out_dir}")


def main():
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Lane thickness: {LANE_THICKNESS}px  |  Classes: 0=bg, 1-4=lane ID\n")

    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if not split_dir.is_dir():
            print(f"[SKIP] Directory not found: {split_dir}")
            continue
        print(f"Processing {split}/...")
        process_split(split)

    print("\nDone.")
    print("Output at:", DATASET_ROOT / "laneseg_label_w16")


if __name__ == "__main__":
    main()
