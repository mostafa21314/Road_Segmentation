"""
Phase 1: Generate binary vehicle masks for every image in the CULane rural subset.

Uses a pretrained YOLOv8-seg model (COCO) and unions the masks of COCO classes
car / motorcycle / bus / truck into a single binary mask per image. Output matches
the image paths one-to-one, mirroring the laneseg_label_w16 directory layout:

    <dataset_root>/vehicle_masks/train/driver_xxx.png
    <dataset_root>/vehicle_masks/val/driver_xxx.png
    <dataset_root>/vehicle_masks/test/driver_xxx.png

Each mask is uint8 at the original image resolution (590x1640), pixel values:
    0   = no vehicle
    255 = vehicle

Used later by the occluded-lane segmentation loss to up-weight pixels where
the ground-truth lane is hidden behind a vehicle.

Environment: requires `ultralytics` (pip install ultralytics). Safe to run in
the segman conda env or a dedicated env; not required in the fenet env.
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

DATASET_ROOT = Path("/home/g6/Mostafa/Road_Segmentation/CULane_Rural_Subset(1)/CULane_Rural_Subset")
SPLITS = ["train", "val", "test"]
IMG_HEIGHT = 590
IMG_WIDTH = 1640

# yolov8s-seg: 22 MB, better on distant / small vehicles
# yolov8n-seg:  6 MB, ~2x faster, slightly coarser masks
MODEL_WEIGHTS = "yolov8s-seg.pt"

# COCO class IDs treated as vehicles
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

CONF_THRESHOLD = 0.25  # YOLO default; permissive is fine, we only use the union


def process_split(model, split: str):
    split_dir = DATASET_ROOT / split
    out_dir = DATASET_ROOT / "vehicle_masks" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(split_dir.glob("*.jpg"))
    if not jpg_files:
        print(f"  [WARN] No .jpg files in {split_dir}")
        return

    n_with_vehicles = 0
    for jpg_path in tqdm(jpg_files, desc=f"{split}"):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        results = model(str(jpg_path), conf=CONF_THRESHOLD, verbose=False)
        r = results[0]

        if r.masks is not None and r.boxes is not None:
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            mask_data = r.masks.data.cpu().numpy()  # [N, H', W'] at inference size

            for i, cid in enumerate(cls_ids):
                if cid in VEHICLE_CLASSES:
                    m = cv2.resize(mask_data[i], (IMG_WIDTH, IMG_HEIGHT),
                                   interpolation=cv2.INTER_NEAREST)
                    mask[m > 0.5] = 255

            if mask.any():
                n_with_vehicles += 1

        out_path = out_dir / (jpg_path.stem + ".png")
        cv2.imwrite(str(out_path), mask)

    pct = 100.0 * n_with_vehicles / len(jpg_files) if jpg_files else 0
    print(f"  {len(jpg_files)} images  |  {n_with_vehicles} ({pct:.1f}%) have at least one vehicle")
    print(f"  Saved to {out_dir}")


def main():
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"YOLO weights: {MODEL_WEIGHTS}  (auto-downloads on first run)")
    print(f"Vehicle classes (COCO): car(2), motorcycle(3), bus(5), truck(7)")
    print(f"Confidence threshold  : {CONF_THRESHOLD}\n")

    model = YOLO(MODEL_WEIGHTS)

    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if not split_dir.is_dir():
            print(f"[SKIP] Directory not found: {split_dir}")
            continue
        print(f"Processing {split}/...")
        process_split(model, split)

    print("\nDone.")
    print("Output at:", DATASET_ROOT / "vehicle_masks")


if __name__ == "__main__":
    main()
