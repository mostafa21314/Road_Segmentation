"""
Stage 2 of the combined-video pipeline.

Reads the per-frame FENet lane cache produced by export_video_fenet_lanes.py,
runs SegMAN(IDD20k) on each frame to get a driveable mask, composites both
on top of the original image, and:
  - displays the result live in a cv2 window
  - writes the same frames to an MP4 file

Run inside the `segman` conda env:
    conda activate segman
    cd /home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation
    python /home/g6/Mostafa/Road_Segmentation/Scripts/render_combined_video.py \\
        --dataset {indian,culane}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
SEGMAN_SEG_DIR = Path("/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation")
SEGMAN_CFG     = SEGMAN_SEG_DIR / "work_dirs/segman_t_idd20k/segman_t_idd20k.py"
SEGMAN_CKPT    = SEGMAN_SEG_DIR / "work_dirs/segman_t_idd20k/best_mIoU_iter_16000.pth"

IDD_CACHE_DIR    = SEGMAN_SEG_DIR / "data/idd20k/video_lanes_cache"
IDD_OUT_VIDEO    = SEGMAN_SEG_DIR / "data/idd20k/combined_video.mp4"

CULANE_ROOT      = Path("/home/g6/Mostafa/Road_Segmentation/"
                        "CULane_Rural_Subset(1)/CULane_Rural_Subset")
CULANE_CACHE_DIR = CULANE_ROOT / "video_lanes_cache"
CULANE_OUT_VIDEO = CULANE_ROOT / "combined_video.mp4"

ROAD_CLASS  = 1
FPS         = 10
DISPLAY_WIDTH = 1280   # cv2.imshow window scaled to this width; MP4 stays full-res
# ─────────────────────────────────────────────────────────────────────────────

LANE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (255, 0, 128), (0, 128, 255), (0, 255, 128),
]

sys.path.insert(0, str(SEGMAN_SEG_DIR))
try:
    from mmseg.apis import init_segmentor, inference_segmentor
except ImportError:
    print("[ERROR] mmseg not importable. Run inside the `segman` conda env.")
    sys.exit(1)


def load_segman():
    print(f"Loading SegMAN from {SEGMAN_CKPT.name} ...")
    model = init_segmentor(str(SEGMAN_CFG), str(SEGMAN_CKPT), device='cuda:0')

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


def render_frame(img_bgr, drive, lanes):
    out = (img_bgr * 0.6).astype(np.uint8)
    if drive.shape != out.shape[:2]:
        drive = cv2.resize(drive.astype(np.uint8),
                           (out.shape[1], out.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    out[drive] = (
        0.45 * out[drive].astype(np.float32)
        + np.array([180, 60, 0], dtype=np.float32)
    ).astype(np.uint8)
    draw_lanes(out, lanes, width=6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["indian", "culane"], required=True)
    ap.add_argument("--smooth", type=int, default=4,
                    help="Insert N image-space cross-fade frames between each "
                         "pair of model-inferenced frames to hide jumps in the "
                         "source data. 0 = off. Wall-clock pace is preserved "
                         "by scaling the output FPS.")
    ap.add_argument("--fps", type=float, default=FPS,
                    help="Real-frame playback rate. Inserted cross-fade frames "
                         "increase the output FPS proportionally so wall-clock "
                         "speed matches this value.")
    args = ap.parse_args()

    if args.dataset == "indian":
        cache_dir = IDD_CACHE_DIR
        out_video = IDD_OUT_VIDEO
    else:
        cache_dir = CULANE_CACHE_DIR
        out_video = CULANE_OUT_VIDEO

    list_file = cache_dir / "frame_list.txt"
    if not list_file.exists():
        print(f"[ERROR] {list_file} not found. "
              f"Run export_video_fenet_lanes.py --dataset {args.dataset} first.")
        return
    frame_paths = [Path(l) for l in list_file.read_text().splitlines() if l.strip()]
    if not frame_paths:
        print(f"[ERROR] {list_file} is empty.")
        return

    # Probe first frame to size the writer
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        print(f"[ERROR] Cannot read {frame_paths[0]}")
        return
    H, W = first.shape[:2]

    # Bump output FPS so the inserted cross-fade frames don't slow playback.
    smooth_k    = max(0, args.smooth)
    base_fps    = max(0.1, float(args.fps))
    out_fps     = base_fps * (1 + smooth_k)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), fourcc, out_fps, (W, H))
    if not writer.isOpened():
        print(f"[ERROR] Could not open VideoWriter at {out_video}")
        return
    total_frames = len(frame_paths) + smooth_k * max(0, len(frame_paths) - 1)
    print(f"Writing video to {out_video}  ({W}x{H} @ {out_fps}fps, "
          f"{len(frame_paths)} real + {smooth_k} cross-fade between each → "
          f"{total_frames} total)")

    model = load_segman()

    win = f"combined ({args.dataset})"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    disp_h = int(DISPLAY_WIDTH * H / W)
    cv2.resizeWindow(win, DISPLAY_WIDTH, disp_h)

    delay_ms = max(1, int(1000 / out_fps))
    quit_early = False
    prev_frame = None  # last rendered overlay, for cross-fade

    def show_and_write(f):
        writer.write(f)
        cv2.imshow(win, f)
        key = cv2.waitKey(delay_ms) & 0xFF
        return key in (ord('q'), 27)

    for jpg in tqdm(frame_paths, desc=f"render/{args.dataset}"):
        img_bgr = cv2.imread(str(jpg))
        if img_bgr is None:
            continue

        result = inference_segmentor(model, str(jpg))
        drive  = np.asarray(result[0], dtype=np.uint8) == ROAD_CLASS

        npy = cache_dir / f"{jpg.stem}.npy"
        lanes = []
        if npy.exists():
            lanes = [np.asarray(l, dtype=np.float32)
                     for l in np.load(str(npy), allow_pickle=True)]

        frame = render_frame(img_bgr, drive, lanes)

        # Image-space cross-fade between prev_frame and frame to hide source-
        # data gaps. The model is only run on real frames; in-between frames
        # are just lerped pixels of two already-rendered overlays.
        if prev_frame is not None and smooth_k > 0:
            prev_f = prev_frame.astype(np.float32)
            cur_f  = frame.astype(np.float32)
            for k in range(1, smooth_k + 1):
                t = k / (smooth_k + 1)
                blend = ((1.0 - t) * prev_f + t * cur_f).astype(np.uint8)
                if show_and_write(blend):
                    quit_early = True
                    break
            if quit_early:
                break

        if show_and_write(frame):
            quit_early = True
            break
        prev_frame = frame

    writer.release()
    cv2.destroyAllWindows()
    if quit_early:
        print("\n[INFO] Stopped early; partial MP4 was still written.")
    print(f"\nDone. Video: {out_video}")


if __name__ == "__main__":
    main()
