"""
Stage 1 of the IDD combined overlay pipeline.

Runs FENet (CULane rural finetune) on a random sample of IDD validation
images and saves each image's predicted lanes to a .npy file, so that stage 2
(generate_idd_combined_overlays.py, segman env) can load them without needing
the FENet package.

Each .npy file stores an object array of shape (num_lanes,) where every entry
is an (N, 2) float32 array of (x, y) points in the SOURCE IMAGE pixel space
(1920 x 1080 for IDD val).

Usage (run in the `fenet` conda env — first `deactivate` any active venv):
    deactivate  # if a non-conda venv is currently active
    conda activate fenet
    cd /home/g6/Mostafa/Road_Segmentation/FENet
    python ../Scripts/export_idd_fenet_lanes.py
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
FENET_ROOT  = Path("/home/g6/Mostafa/Road_Segmentation/FENet")
FENET_CFG   = FENET_ROOT / "configs/fenet/FENetV2_dla34_culane_rural_finetune.py"
FENET_CKPT  = FENET_ROOT / "work_dirs/fenetv2/dla34_culane_rural_finetune/" \
                           "20260418_055835_lr_2e-04_b_8/ckpt/9.pth"

IDD_IMG_DIR = Path("/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation/"
                   "data/idd20k/img_dir/val")
CACHE_DIR   = Path("/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation/"
                   "data/idd20k/fenet_lanes_cache")

N_SAMPLES = 30
SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(FENET_ROOT))

from fenet.utils.config import Config
from fenet.models.registry import build_net


def load_fenet():
    print(f"Loading FENet cfg: {FENET_CFG.name}")
    cfg = Config.fromfile(str(FENET_CFG))
    cfg.gpus = 1
    cfg.load_from = None
    cfg.finetune_from = None

    net = build_net(cfg).cuda().eval()
    print(f"  Loading weights: {FENET_CKPT}")

    # Checkpoint was saved while wrapped in nn.DataParallel, so every key has
    # a "module." prefix. fenet.utils.net_utils.load_network calls
    # load_state_dict(strict=False) and silently drops every mismatch — leaving
    # the model on random init. Strip the prefix before loading.
    ckpt = torch.load(str(FENET_CKPT), map_location="cpu")
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"  Loaded weights. missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"    first missing:    {missing[:3]}")
        print(f"    first unexpected: {unexpected[:3]}")
    print("  FENet ready.\n")
    return net, cfg


def predict_lanes(net, cfg, img_bgr):
    """Returns a list of (N, 2) float32 arrays in SOURCE IMAGE pixel coords."""
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
    random.seed(SEED)

    all_imgs = sorted(IDD_IMG_DIR.glob("*.jpg"))
    if not all_imgs:
        print(f"[ERROR] No .jpg files in {IDD_IMG_DIR}")
        return

    picks = random.sample(all_imgs, min(N_SAMPLES, len(all_imgs)))
    print(f"Sampling {len(picks)} / {len(all_imgs)} IDD val images")
    print(f"Caching lane predictions in {CACHE_DIR}\n")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    net, cfg = load_fenet()

    for jpg in tqdm(picks, desc="FENet lanes"):
        img_bgr = cv2.imread(str(jpg))
        if img_bgr is None:
            continue
        lanes = predict_lanes(net, cfg, img_bgr)
        np.save(CACHE_DIR / f"{jpg.stem}.npy",
                np.asarray(lanes, dtype=object), allow_pickle=True)

    print(f"\nWrote {len(picks)} .npy files to {CACHE_DIR}")


if __name__ == "__main__":
    main()
