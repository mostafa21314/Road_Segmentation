"""
Convert CULane lane annotations (coordinate .txt files) to binary segmentation masks (.png).
Also reorganizes the dataset into MMSegmentation-compatible directory structure.

Output structure:
    data/culane/
    ├── img_dir/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── ann_dir/
        ├── train/
        ├── val/
        └── test/
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def parse_culane_annotation(txt_path):
    """Parse a CULane .txt annotation file into a list of lanes.
    Each lane is a list of (x, y) coordinate tuples."""
    lanes = []
    if not os.path.exists(txt_path):
        return lanes
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(float, line.split()))
            # Values are alternating x, y pairs
            points = []
            for i in range(0, len(values) - 1, 2):
                x, y = values[i], values[i + 1]
                points.append((int(round(x)), int(round(y))))
            if len(points) >= 2:
                lanes.append(points)
    return lanes


def create_segmentation_mask(lanes, height, width, line_thickness=16):
    """Create a binary segmentation mask from lane points.
    Class 0 = background, Class 1 = lane."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for lane in lanes:
        pts = np.array(lane, dtype=np.int32)
        # Filter out points outside image bounds
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < width) & \
                (pts[:, 1] >= 0) & (pts[:, 1] < height)
        pts = pts[valid]
        if len(pts) >= 2:
            cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=line_thickness)
    return mask


def process_split(src_dir, split, img_list_file, out_root, img_height=590, img_width=1640):
    """Process one dataset split (train/val/test)."""
    img_out = os.path.join(out_root, 'img_dir', split)
    ann_out = os.path.join(out_root, 'ann_dir', split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)

    # Read image list
    with open(img_list_file, 'r') as f:
        entries = [line.strip() for line in f if line.strip()]

    skipped = 0
    for entry in tqdm(entries, desc=f"Processing {split}"):
        # entry format: "train/driver_xxx.jpg" or "val/driver_xxx.jpg"
        img_name = os.path.basename(entry)
        base_name = os.path.splitext(img_name)[0]

        img_path = os.path.join(src_dir, entry)
        txt_path = os.path.join(src_dir, os.path.dirname(entry), base_name + '.txt')

        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Get actual image dimensions
        img = Image.open(img_path)
        w, h = img.size

        # Create symlink for image (saves disk space)
        dst_img = os.path.join(img_out, img_name)
        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(img_path), dst_img)

        # Create segmentation mask
        lanes = parse_culane_annotation(txt_path)
        mask = create_segmentation_mask(lanes, h, w)
        mask_path = os.path.join(ann_out, base_name + '.png')
        Image.fromarray(mask).save(mask_path)

    if skipped:
        print(f"  Warning: skipped {skipped} missing images in {split}")


def main():
    parser = argparse.ArgumentParser(description='Convert CULane to segmentation masks')
    parser.add_argument('--src', type=str,
                        default='/home/g6/Mostafa/Road_Segmentation/CULane_Rural_Subset(1)/CULane_Rural_Subset',
                        help='Path to CULane_Rural_Subset root')
    parser.add_argument('--dst', type=str,
                        default='/home/g6/Mostafa/Road_Segmentation/SegMAN/data/culane',
                        help='Output directory for MMSeg-compatible dataset')
    parser.add_argument('--thickness', type=int, default=16,
                        help='Lane line thickness in pixels for mask generation')
    args = parser.parse_args()

    print(f"Source: {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Lane thickness: {args.thickness}")

    for split in ['train', 'val', 'test']:
        list_file = os.path.join(args.src, f'{split}.txt')
        if not os.path.exists(list_file):
            print(f"Skipping {split}: {list_file} not found")
            continue
        print(f"\nProcessing {split}...")
        process_split(args.src, split, list_file, args.dst)

    print("\nDone! Dataset prepared at:", args.dst)
    print("Structure:")
    for d in ['img_dir/train', 'img_dir/val', 'img_dir/test', 'ann_dir/train', 'ann_dir/val', 'ann_dir/test']:
        full = os.path.join(args.dst, d)
        if os.path.exists(full):
            count = len(os.listdir(full))
            print(f"  {d}: {count} files")


if __name__ == '__main__':
    main()
