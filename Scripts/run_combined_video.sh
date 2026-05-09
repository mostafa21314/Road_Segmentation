#!/usr/bin/env bash
# Run SegMAN + FENet inference on consecutive frames from one dataset and
# play/save the combined overlay as a video.
#
# Usage:
#   ./Scripts/run_combined_video.sh <indian|culane> [smooth_k] [fps] [clip] [source_dir]
#
#   smooth_k (optional, default 4): number of cross-fade frames inserted
#     between each pair of model-inferenced frames to hide gaps in the source
#     data. Pass 0 to disable smoothing entirely.
#   fps (optional, default 10): real-frame playback rate. The output MP4's
#     actual FPS becomes fps*(1+smooth_k) so wall-clock pace matches this.
#   clip (optional): substring filter on clip / scene name. If omitted the
#     longest available clip is auto-picked. Pass '' to skip this position.
#   source_dir (optional): override directory of jpgs to sample from.
#     Examples for culane:
#       /home/g6/temp_AML/Road_Segmentation/SegMAN/data/culane/img_dir/train
#       /home/g6/temp_AML/fresh/SegMAN/culane_subset/test
#     The train split has clips with up to 109 frames vs 31 in the rural test.
#
# Pipeline:
#   1) export_video_fenet_lanes.py --dataset $1   (fenet env, py3.8)
#        Picks 60 consecutive frames from the longest source clip in the chosen
#        dataset, runs FENet on each, caches per-frame lane predictions plus a
#        frame_list.txt to <dataset>/video_lanes_cache/.
#   2) render_combined_video.py    --dataset $1 --smooth $2   (segman env, py3.10)
#        Reads the cache, runs SegMAN(IDD20k) per frame, composites blue
#        driveable + coloured lanes, displays a live cv2 window AND writes
#        <dataset>/combined_video.mp4 at 10fps.
#
# Two envs are required because FENet's nms_impl .so is built for py3.8 and
# can't be loaded inside the py3.10 segman env.

set -eo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <indian|culane> [smooth_k] [fps]"
    exit 2
fi

DATASET="$1"
case "$DATASET" in
    indian|culane) ;;
    *) echo "[ERROR] dataset must be 'indian' or 'culane' (got: $DATASET)"; exit 2 ;;
esac

SMOOTH_K="${2:-4}"
if ! [[ "$SMOOTH_K" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] smooth_k must be a non-negative integer (got: $SMOOTH_K)"
    exit 2
fi

FPS="${3:-10}"
if ! [[ "$FPS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "[ERROR] fps must be a positive number (got: $FPS)"
    exit 2
fi

CLIP="${4-}"
SOURCE_DIR="${5-}"

EXPORT_EXTRA_ARGS=()
if [[ -n "$CLIP" ]]; then
    EXPORT_EXTRA_ARGS+=("--clip" "$CLIP")
fi
if [[ -n "$SOURCE_DIR" ]]; then
    if [[ ! -d "$SOURCE_DIR" ]]; then
        echo "[ERROR] source_dir does not exist: $SOURCE_DIR"
        exit 2
    fi
    EXPORT_EXTRA_ARGS+=("--source-dir" "$SOURCE_DIR")
fi

REPO_ROOT="/home/g6/Mostafa/Road_Segmentation"
FENET_DIR="${REPO_ROOT}/FENet"
SEGMAN_DIR="/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation"
CONDA_SH="/home/g6/miniconda3/etc/profile.d/conda.sh"

# ── 0. drop any non-conda venv that may be shadowing conda's python ─────────
# `deactivate` is a shell function defined by the venv activator; running
# this script via `bash …` spawns a fresh subshell that does NOT inherit
# that function, so we strip the venv from PATH and unset its env vars
# manually instead of calling deactivate.
if [[ -n "${VIRTUAL_ENV-}" ]]; then
    echo "[INFO] Stripping non-conda venv ${VIRTUAL_ENV} from PATH"
    # Remove every occurrence of "$VIRTUAL_ENV/bin" from PATH.
    PATH=$(echo ":${PATH}:" | sed "s|:${VIRTUAL_ENV}/bin:|:|g" | sed 's|^:||;s|:$||')
    export PATH
    unset VIRTUAL_ENV PYTHONHOME
fi

# Sanity: conda init script must exist.
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] conda init not found at ${CONDA_SH}"
    exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"

# ── 1. FENet lane export (fenet env) ────────────────────────────────────────
echo
echo "==> Step 1/2: export_video_fenet_lanes.py --dataset ${DATASET}  (fenet env)"
conda activate fenet
echo "[INFO] Active python: $(which python)  ($(python --version 2>&1))"
cd "${FENET_DIR}"
python "${REPO_ROOT}/Scripts/export_video_fenet_lanes.py" --dataset "${DATASET}" "${EXPORT_EXTRA_ARGS[@]}"
conda deactivate

# ── 2. SegMAN + render video (segman env) ───────────────────────────────────
echo
echo "==> Step 2/2: render_combined_video.py --dataset ${DATASET}  (segman env)"
conda activate segman
echo "[INFO] Active python: $(which python)  ($(python --version 2>&1))"
cd "${SEGMAN_DIR}"
python "${REPO_ROOT}/Scripts/render_combined_video.py" --dataset "${DATASET}" --smooth "${SMOOTH_K}" --fps "${FPS}"
conda deactivate

echo
echo "Done."
