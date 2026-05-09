#!/usr/bin/env bash
# Reproduce the SegMAN + FENet combined overlays on the Indian Driving Dataset
# (IDD20k) validation samples.
#
# Pipeline:
#   1) Scripts/export_idd_fenet_lanes.py        — FENet (CULane rural finetune)
#      runs on 30 random IDD val jpgs, caches predicted lanes per image to
#      data/idd20k/fenet_lanes_cache/<stem>.npy  (run inside the `fenet` env).
#   2) Scripts/generate_idd_combined_overlays.py — SegMAN(IDD20k) predicts the
#      driveable area, combines it with the cached FENet lanes, writes to
#      data/idd20k/combined_overlays/<stem>.png  (run inside the `segman` env).
#   3) Copies the produced overlays into the CULane_Rural_Subset tree under
#      driveable_overlays_indian/, so all visualization assets live together.
#
# All steps are idempotent — re-running this script is safe.
#
# Notes on environment:
#   - Stage 1 needs the `fenet` env (py3.8, has fenet.ops.nms_impl).
#   - Stage 2 needs the `segman` env (py3.10, has mmseg).
#   - The two envs are required because FENet's nms_impl .so is built for
#     py3.8 and can't be loaded inside the py3.10 segman env.
#   - If a non-conda venv (e.g. federated_unlearning) is active, its python
#     will shadow conda's. We `deactivate` and unset VIRTUAL_ENV first.

# `set -u` is intentionally OFF: conda's (de)activate scripts reference
# unbound variables (e.g. _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED) and would
# trip nounset. We still want -e and pipefail.
set -eo pipefail

REPO_ROOT="/home/g6/Mostafa/Road_Segmentation"
FENET_DIR="${REPO_ROOT}/FENet"
SEGMAN_DIR="/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation"
CONDA_SH="/home/g6/miniconda3/etc/profile.d/conda.sh"

IDD_DATA_DIR="${SEGMAN_DIR}/data/idd20k"
COMBINED_OVERLAYS_SRC="${IDD_DATA_DIR}/combined_overlays"

DATASET_ROOT="${REPO_ROOT}/CULane_Rural_Subset(1)/CULane_Rural_Subset"
TARGET_OVERLAYS="${DATASET_ROOT}/driveable_overlays_indian"

# ── 0. drop any non-conda venv that may be shadowing conda's python ─────────
# `deactivate` is a shell function defined by the venv activator; running
# this script via `bash …` spawns a fresh subshell that does NOT inherit
# that function, so we strip the venv from PATH and unset its env vars
# manually instead of calling deactivate.
if [[ -n "${VIRTUAL_ENV-}" ]]; then
    echo "[INFO] Stripping non-conda venv ${VIRTUAL_ENV} from PATH"
    PATH=$(echo ":${PATH}:" | sed "s|:${VIRTUAL_ENV}/bin:|:|g" | sed 's|^:||;s|:$||')
    export PATH
    unset VIRTUAL_ENV PYTHONHOME
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] conda init not found at ${CONDA_SH}"
    exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"

# ── 1. FENet lane export (fenet env) ────────────────────────────────────────
echo
echo "==> Step 1/3: export_idd_fenet_lanes.py  (fenet env)"
conda activate fenet
echo "[INFO] Active python: $(which python)  ($(python --version 2>&1))"
cd "${FENET_DIR}"
python "${REPO_ROOT}/Scripts/export_idd_fenet_lanes.py"
conda deactivate

# ── 2. SegMAN driveable + lane render (segman env) ──────────────────────────
echo
echo "==> Step 2/3: generate_idd_combined_overlays.py  (segman env)"
conda activate segman
echo "[INFO] Active python: $(which python)  ($(python --version 2>&1))"
cd "${SEGMAN_DIR}"
python "${REPO_ROOT}/Scripts/generate_idd_combined_overlays.py"
conda deactivate

# ── 3. relocate overlays into the CULane subset tree ────────────────────────
echo
echo "==> Step 3/3: copy overlays → ${TARGET_OVERLAYS}"
if [[ ! -d "${COMBINED_OVERLAYS_SRC}" ]]; then
    echo "[ERROR] Expected overlay source dir not found: ${COMBINED_OVERLAYS_SRC}"
    exit 1
fi

shopt -s nullglob
overlays=( "${COMBINED_OVERLAYS_SRC}"/*.png )
shopt -u nullglob

if (( ${#overlays[@]} == 0 )); then
    echo "[ERROR] No overlay PNGs found in ${COMBINED_OVERLAYS_SRC}"
    exit 1
fi

mkdir -p "${TARGET_OVERLAYS}"
cp -f "${overlays[@]}" "${TARGET_OVERLAYS}/"
echo "[INFO] Copied ${#overlays[@]} overlay(s)."

echo
echo "Done. Overlays at: ${TARGET_OVERLAYS}/"
