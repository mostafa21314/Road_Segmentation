#!/usr/bin/env bash
# Reproduce the SegMAN driveable-area overlays on the CULane Rural Subset.
#
# Pipeline:
#   1) Scripts/rename_annotations.py        — <x>.txt → <x>.lines.txt
#   2) Scripts/generate_seg_masks.py        — laneseg_label_w16/<split>/*.png
#   3) Scripts/generate_driveable_overlays.py — runs SegMAN(IDD20k) and writes
#      <dataset>/driveable_overlays/<stem>__on{N}_off{M}.png
#
# All steps are idempotent — re-running this script is safe.
#
# Notes on environment:
#   - Uses the `segman` conda env (py3.10, has mmseg).
#   - The SegMAN model registrations live in
#     /home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation, so step 3 cd's
#     there before invoking python.
#   - If a non-conda venv (e.g. federated_unlearning) is active, its python
#     will shadow conda's. We `deactivate` and unset VIRTUAL_ENV first.

# `set -u` is intentionally OFF: conda's (de)activate scripts reference
# unbound variables (e.g. _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED) and would
# trip nounset. We still want -e and pipefail.
set -eo pipefail

REPO_ROOT="/home/g6/Mostafa/Road_Segmentation"
SEGMAN_DIR="/home/g6/temp_AML/Road_Segmentation/SegMAN/segmentation"
CONDA_SH="/home/g6/miniconda3/etc/profile.d/conda.sh"
DATASET_ROOT="${REPO_ROOT}/CULane_Rural_Subset(1)/CULane_Rural_Subset"

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

# ── 1. activate segman env ──────────────────────────────────────────────────
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] conda init not found at ${CONDA_SH}"
    exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate segman
echo "[INFO] Active python: $(which python)  ($(python --version 2>&1))"

# ── 2. preprocess: rename annotations ───────────────────────────────────────
echo
echo "==> Step 1/3: rename_annotations.py"
python "${REPO_ROOT}/Scripts/rename_annotations.py"

# ── 3. preprocess: build laneseg_label_w16/ ─────────────────────────────────
echo
echo "==> Step 2/3: generate_seg_masks.py"
python "${REPO_ROOT}/Scripts/generate_seg_masks.py"

# ── 4. SegMAN driveable overlays ────────────────────────────────────────────
echo
echo "==> Step 3/3: generate_driveable_overlays.py"
cd "${SEGMAN_DIR}"
python "${REPO_ROOT}/Scripts/generate_driveable_overlays.py"

echo
echo "Done. Overlays at: ${DATASET_ROOT}/driveable_overlays/"
