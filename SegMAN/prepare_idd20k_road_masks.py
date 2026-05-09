"""
prepare_idd_road_data.py  —  Prepare IDD-20k-II for binary road segmentation.

Reads polygon annotations from the IDD-20k-II dataset and creates binary PNG
masks suitable for MMSeg's CustomDataset.

Mask values:
    0  = background
    1  = road  (labels: "road" and "drivable fallback")

Output layout  (relative to the SegMAN root):
    data/idd20k/
        img_dir/
            train/  <flat .jpg files>
            val/    <flat .jpg files>
        ann_dir/
            train/  <flat .png masks>
            val/    <flat .png masks>

Images are symlinked (not copied) to save disk space.

Run from the SegMAN root:
    conda run -n segmanaser python prepare_idd_road_data.py

Optional flags:
    --idd-root   Path to idd20kII/ directory
                 (default: ../../../idd-20k-II/idd20kII)
    --out-root   Output data root
                 (default: data/idd20k)
    --splits     Which splits to process  (default: train val)
    --road-labels  Labels counted as road
                 (default: "road" "drivable fallback")
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description='Prepare IDD-20k-II road segmentation masks')
    p.add_argument('--idd-root',
                   default=str(SCRIPT_DIR.parent.parent / 'idd-20k-II' / 'idd20kII'),
                   help='Path to idd20kII/ root directory')
    p.add_argument('--out-root',
                   default=str(SCRIPT_DIR / 'data' / 'idd20k'),
                   help='Output data root (will be created)')
    p.add_argument('--splits', nargs='+', default=['train', 'val'],
                   help='Dataset splits to process')
    p.add_argument('--road-labels', nargs='+',
                   default=['road', 'drivable fallback'],
                   help='Annotation labels treated as road foreground')
    return p.parse_args()


def polygon_to_mask(polygon_points, img_h, img_w):
    """Return a binary numpy array with 1 inside the polygon."""
    mask = Image.new('L', (img_w, img_h), 0)
    flat = [(x, y) for x, y in polygon_points]
    if len(flat) >= 3:
        ImageDraw.Draw(mask).polygon(flat, fill=1)
    return np.array(mask, dtype=np.uint8)


def process_annotation(json_path, road_labels):
    """
    Read one *_gtFine_polygons.json and return a binary uint8 mask.
    Returns (mask_array, img_h, img_w).
    """
    with open(json_path) as f:
        ann = json.load(f)

    img_h = ann['imgHeight']
    img_w = ann['imgWidth']
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    road_set = set(road_labels)
    for obj in ann.get('objects', []):
        if obj.get('label') in road_set:
            pts = obj.get('polygon', [])
            if pts:
                mask |= polygon_to_mask(pts, img_h, img_w)

    return mask, img_h, img_w


def flat_name(video_id, frame_stem):
    """
    Map (video_id, frame stem) → a unique flat filename stem.
    e.g. ('379', 'frame12092') → '379__frame12092'
    """
    return f'{video_id}__{frame_stem}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_split(split, idd_root, out_root, road_labels):
    img_src_root = Path(idd_root) / 'leftImg8bit' / split
    ann_src_root = Path(idd_root) / 'gtFine' / split

    img_dst_dir = Path(out_root) / 'img_dir' / split
    ann_dst_dir = Path(out_root) / 'ann_dir' / split
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    ann_dst_dir.mkdir(parents=True, exist_ok=True)

    if not ann_src_root.exists():
        print(f'  [WARN] No annotations found for split "{split}" at {ann_src_root}')
        print(f'         Skipping annotation generation (test set has no GT).')
        # Still symlink images so they can be used for inference
        n_img = 0
        for video_dir in sorted(img_src_root.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            for img_file in sorted(video_dir.glob('*_leftImg8bit.jpg')):
                frame_stem = img_file.stem.replace('_leftImg8bit', '')
                dst_name = flat_name(video_id, frame_stem) + '.jpg'
                dst_path = img_dst_dir / dst_name
                if not dst_path.exists():
                    dst_path.symlink_to(img_file.resolve())
                n_img += 1
        print(f'  Linked {n_img} images (no masks) for split "{split}"')
        return n_img, 0

    video_dirs = sorted(ann_src_root.iterdir())
    n_ok, n_skip = 0, 0

    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name

        for json_file in sorted(video_dir.glob('*_gtFine_polygons.json')):
            # derive stem: remove '_gtFine_polygons' suffix
            frame_stem = json_file.stem.replace('_gtFine_polygons', '')

            # --- annotation mask ---
            mask_name = flat_name(video_id, frame_stem) + '.png'
            mask_path = ann_dst_dir / mask_name

            if not mask_path.exists():
                try:
                    mask, img_h, img_w = process_annotation(json_file, road_labels)
                    Image.fromarray(mask).save(mask_path)
                except Exception as e:
                    print(f'  [ERROR] {json_file}: {e}')
                    n_skip += 1
                    continue

            # --- image symlink ---
            img_src = img_src_root / video_id / (frame_stem + '_leftImg8bit.jpg')
            img_dst = img_dst_dir / (flat_name(video_id, frame_stem) + '.jpg')

            if not img_dst.exists():
                if img_src.exists():
                    img_dst.symlink_to(img_src.resolve())
                else:
                    print(f'  [WARN] Image not found: {img_src}')
                    n_skip += 1
                    continue

            n_ok += 1

    print(f'  Split "{split}": {n_ok} pairs created, {n_skip} skipped')
    return n_ok, n_skip


def main():
    args = parse_args()

    print('=' * 60)
    print('IDD-20k-II Road Segmentation Data Preparation')
    print('=' * 60)
    print(f'IDD root   : {args.idd_root}')
    print(f'Output root: {args.out_root}')
    print(f'Road labels: {args.road_labels}')
    print(f'Splits     : {args.splits}')
    print()

    if not Path(args.idd_root).exists():
        sys.exit(f'[ERROR] IDD root not found: {args.idd_root}')

    total_ok = 0
    for split in args.splits:
        print(f'Processing split: {split}')
        ok, _ = process_split(split, args.idd_root, args.out_root, args.road_labels)
        total_ok += ok

    print()
    print(f'Done.  {total_ok} total pairs written to {args.out_root}')
    print()

    # Quick stats
    for split in args.splits:
        img_dir = Path(args.out_root) / 'img_dir' / split
        ann_dir = Path(args.out_root) / 'ann_dir' / split
        n_img = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
        n_ann = len(list(ann_dir.glob('*.png'))) if ann_dir.exists() else 0
        print(f'  {split}: {n_img} images, {n_ann} masks')


if __name__ == '__main__':
    main()
