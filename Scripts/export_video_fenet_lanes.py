"""
Stage 1 of the combined-video pipeline.

Picks ~60 consecutive frames from one source clip (CULane MP4 / IDD scene),
runs FENet on each, and dumps:
  - <cache_dir>/<stem>.npy        per-frame predicted lanes (object array of
                                  (N, 2) float32 arrays in source-image coords)
  - <cache_dir>/frame_list.txt    one absolute jpg path per line, in order

Stage 2 (Scripts/render_combined_video.py) reads frame_list.txt and the .npy
files to render a video without needing the FENet package.

Run inside the `fenet` conda env:
    conda activate fenet
    cd /home/g6/Mostafa/Road_Segmentation/FENet
    python ../Scripts/export_video_fenet_lanes.py --dataset {indian,culane}
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
FENET_ROOT  = Path("/home/g6/Mostafa/Road_Segmentation/FENet")
FENET_CFG   = FENET_ROOT / "configs/fenet/FENetV2_dla34_culane_rural_finetune.py"
FENET_CKPT  = FENET_ROOT / "work_dirs/fenetv2/dla34_culane_rural_finetune/" \
                           "20260418_055835_lr_2e-04_b_8/ckpt/9.pth"

IDD_IMG_DIR    = Path("/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation/"
                      "data/idd20k/img_dir/val")
IDD_CACHE_DIR  = Path("/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation/"
                      "data/idd20k/video_lanes_cache")

CULANE_ROOT       = Path("/home/g6/Mostafa/Road_Segmentation/"
                         "CULane_Rural_Subset(1)/CULane_Rural_Subset")
CULANE_IMG_DIR    = CULANE_ROOT / "test"
CULANE_CACHE_DIR  = CULANE_ROOT / "video_lanes_cache"

NUM_FRAMES = 60
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(FENET_ROOT))

from fenet.utils.config import Config
from fenet.models.registry import build_net


def pick_culane_frames(source_dir: Path, clip: Optional[str]):
    """Group jpgs in source_dir by source MP4 stem; pick the requested clip
    (substring match) or the longest one if clip is None."""
    groups = defaultdict(list)
    for jpg in source_dir.glob("*.jpg"):
        stem = jpg.stem
        if ".MP4_" not in stem:
            continue
        prefix, _idx = stem.rsplit(".MP4_", 1)
        groups[prefix + ".MP4"].append(jpg)
    if not groups:
        raise SystemExit(f"[ERROR] No CULane jpgs in {source_dir}")

    if clip:
        matches = [k for k in groups if clip in k]
        if not matches:
            print(f"[culane] No clips in {source_dir} matched '{clip}'.")
            print(f"[culane] Available clips (top 10 by frame count):")
            for k in sorted(groups, key=lambda k: -len(groups[k]))[:10]:
                print(f"    {len(groups[k]):4d}  {k}")
            raise SystemExit(2)
        if len(matches) > 1:
            print(f"[culane] '{clip}' matched {len(matches)} clips; using the "
                  f"first by name: {sorted(matches)[0]}")
        chosen = sorted(matches)[0]
    else:
        chosen = max(groups, key=lambda k: len(groups[k]))

    frames = sorted(groups[chosen])[:NUM_FRAMES]
    print(f"[culane] source dir : {source_dir}")
    print(f"[culane] source clip: {chosen}")
    print(f"[culane] using {len(frames)} frames "
          f"(out of {len(groups[chosen])} available)")
    return frames


def pick_idd_frames(source_dir: Path, clip: Optional[str]):
    """Group jpgs by scene id (chars before '__'); pick the requested scene
    (substring match) or the longest one if clip is None."""
    groups = defaultdict(list)
    for jpg in source_dir.glob("*.jpg"):
        scene = jpg.stem.split("__", 1)[0]
        groups[scene].append(jpg)
    if not groups:
        raise SystemExit(f"[ERROR] No IDD jpgs in {source_dir}")

    if clip:
        matches = [k for k in groups if clip in k]
        if not matches:
            print(f"[indian] No scenes in {source_dir} matched '{clip}'.")
            print(f"[indian] Available scenes (top 10 by frame count):")
            for k in sorted(groups, key=lambda k: -len(groups[k]))[:10]:
                print(f"    {len(groups[k]):4d}  {k}")
            raise SystemExit(2)
        chosen = sorted(matches)[0]
    else:
        chosen = max(groups, key=lambda k: len(groups[k]))

    frames = sorted(groups[chosen])[:NUM_FRAMES]
    print(f"[indian] source dir: {source_dir}")
    print(f"[indian] scene     : {chosen}")
    print(f"[indian] using {len(frames)} frames "
          f"(out of {len(groups[chosen])} available)")
    return frames


def load_fenet():
    print(f"Loading FENet cfg: {FENET_CFG.name}")
    cfg = Config.fromfile(str(FENET_CFG))
    cfg.gpus = 1
    cfg.load_from = None
    cfg.finetune_from = None

    net = build_net(cfg).cuda().eval()
    print(f"  Loading weights: {FENET_CKPT}")

    # Checkpoint was saved while wrapped in DataParallel — strip module. prefix
    # so strict-loading actually catches mismatches.
    ckpt = torch.load(str(FENET_CKPT), map_location="cpu")
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"  Loaded weights. missing={len(missing)} unexpected={len(unexpected)}")
    print("  FENet ready.\n")
    return net, cfg


def predict_lanes(net, cfg, img_bgr):
    H0, W0 = img_bgr.shape[:2]
    img_culane = cv2.resize(img_bgr, (cfg.ori_img_w, cfg.ori_img_h),
                            interpolation=cv2.INTER_LINEAR)
    img_cut = img_culane[cfg.cut_height:, :, :]
    img_in  = cv2.resize(img_cut, (cfg.img_w, cfg.img_h),
                         interpolation=cv2.INTER_LINEAR)
    img_in  = img_in.astype(np.float32) / 255.0
    tensor  = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        output = net({'img': tensor})
    lanes = net.heads.get_lanes(output)[0]

    sx = W0 / float(cfg.ori_img_w)
    sy = H0 / float(cfg.ori_img_h)
    out = []
    for lane in lanes:
        pts = lane.to_array(cfg)
        if len(pts) < 2:
            continue
        pts = pts.astype(np.float32)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        out.append(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["indian", "culane"], required=True)
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="Override the directory of jpgs to sample from. "
                         "Default: rural test for culane, idd val for indian. "
                         "Useful for pointing at the train split which has "
                         "longer clips (e.g. temp_AML/.../culane/img_dir/train).")
    ap.add_argument("--clip", type=str, default=None,
                    help="Substring filter on the clip / scene name to use. "
                         "If omitted, the longest available clip is picked.")
    args = ap.parse_args()

    if args.dataset == "indian":
        source_dir = args.source_dir or IDD_IMG_DIR
        frames     = pick_idd_frames(source_dir, args.clip)
        cache_dir  = IDD_CACHE_DIR
    else:
        source_dir = args.source_dir or CULANE_IMG_DIR
        frames     = pick_culane_frames(source_dir, args.clip)
        cache_dir  = CULANE_CACHE_DIR

    # Wipe stale cache from prior runs (different dataset / NUM_FRAMES) so the
    # video stage doesn't pick up extras.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    (cache_dir / "frame_list.txt").write_text(
        "\n".join(str(p) for p in frames) + "\n"
    )
    print(f"Caching predictions in {cache_dir}\n")

    net, cfg = load_fenet()
    for jpg in tqdm(frames, desc=f"FENet/{args.dataset}"):
        img_bgr = cv2.imread(str(jpg))
        if img_bgr is None:
            continue
        lanes = predict_lanes(net, cfg, img_bgr)
        np.save(cache_dir / f"{jpg.stem}.npy",
                np.asarray(lanes, dtype=object), allow_pickle=True)

    print(f"\nWrote {len(frames)} .npy files + frame_list.txt to {cache_dir}")


if __name__ == "__main__":
    main()
