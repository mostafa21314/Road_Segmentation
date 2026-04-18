# Road Segmentation — Lane Detection Finetuning on CULane Rural Subset

This repo contains scripts and configs for finetuning two lane detection models on a curated rural subset of the CULane dataset:

- **[FENet](https://github.com/WangLiman/FENet)** (Focusing Enhanced Network for Lane Detection) — Wang & Zhong, IEEE ICME 2024. [[arXiv]](https://arxiv.org/abs/2312.17163) [[IEEE]](https://ieeexplore.ieee.org/document/10687857)
- **[SegMAN](https://github.com/yunxiangfu2001/SegMAN)** (Omni-scale Context Modeling with SSMs and Local Attention for Semantic Segmentation) — Fu et al., CVPR 2025. [[arXiv]](https://arxiv.org/abs/2412.11890)

---

## Repository Structure

```
Road_Segmentation/
├── Scripts/                                              ← data preparation (our work)
│   ├── rename_annotations.py                             ← Step 1 (FENet)
│   ├── generate_seg_masks.py                             ← Step 2 (FENet)
│   ├── generate_list_files.py                            ← Step 3 (FENet)
│   └── setup_fenet_data_link.py                         ← Step 4 (FENet)
├── FENet/
│   └── configs/fenet/
│       └── FENetV2_dla34_culane_rural_finetune.py        ← our finetune config (our work)
├── SegMAN/
│   ├── convert_culane_to_masks.py                        ← CULane → MMSeg mask conversion (our work)
│   ├── train.py                                          ← SegMAN training entry point (our work)
│   ├── validate.py                                       ← SegMAN validation script (our work)
│   ├── scripts/
│   │   └── train_culane.sh                               ← training launch script (our work)
│   └── segmentation/
│       └── local_configs/
│           ├── segman/small/segman_s_culane.py           ← SegMAN-Small CULane config (our work)
│           └── _base_/datasets/culane_590x590.py         ← CULane dataset config (our work)
└── CULane_Rural_Subset(1)/CULane_Rural_Subset/           ← curated dataset (not committed)
```

---

## What Each Contribution File Does

### Data Preparation Scripts (`Scripts/`)

| Script | Purpose |
|--------|---------|
| `rename_annotations.py` | Renames `*.txt` lane annotation files to `*.lines.txt` inside `train/`, `val/`, `test/`. Required because FENet's dataset loader looks for the `.lines.txt` suffix. |
| `generate_seg_masks.py` | Generates `laneseg_label_w16/` segmentation masks (grayscale PNGs where pixel value = lane ID 0–4) from the annotation files. Used by FENet's segmentation loss during training. |
| `generate_list_files.py` | Creates `list/train_gt.txt`, `list/val.txt`, `list/test.txt`, and `list/test_split/` category files in the format FENet's dataset loader expects. `train_gt.txt` includes per-image lane existence flags. |
| `setup_fenet_data_link.py` | Creates the symlink `FENet/data/CULane → CULane_Rural_Subset/` so FENet's hardcoded `dataset_path = './data/CULane'` resolves to our rural dataset. |

### FENet Config (`FENet/configs/fenet/FENetV2_dla34_culane_rural_finetune.py`)

Adapted from the full-CULane baseline config with the following changes for finetuning on a small rural dataset (2440 training images):

| Parameter | Baseline | Finetune | Reason |
|-----------|----------|----------|--------|
| `pretrained` | `True` | `False` | Weights come from `--load_from` checkpoint |
| `batch_size` | 24 | 8 | More gradient steps per epoch on small dataset |
| `lr` | `0.6e-3` | `2e-4` | Lower LR for finetuning |
| `total_iter` | full-dataset-based | 2440-based | Cosine scheduler `T_max` must match actual dataset size |
| `eval_ep` | 3 | 1 | Evaluate every epoch on a short run |
| `save_ep` | 10 | 5 | Save checkpoints more frequently |
| `workers` | 10 | 4 | Fewer workers needed for smaller dataset |
| `log_interval` | 500 | 50 | Only ~305 iters/epoch — otherwise no logs appear |

### SegMAN Scripts (`SegMAN/`)

| File | Purpose |
|------|---------|
| `convert_culane_to_masks.py` | Converts CULane coordinate annotations (`.txt` files) into binary segmentation masks (`.png`) and reorganises the dataset into MMSegmentation's `img_dir/` + `ann_dir/` structure. Creates symlinks for images to save disk space. |
| `scripts/train_culane.sh` | Bash script that launches single-GPU SegMAN-Small training with the CULane config. Run from `SegMAN/segmentation/`. |
| `segmentation/local_configs/segman/small/segman_s_culane.py` | SegMAN-Small model config for CULane. Sets 2-class output (background + lane), uses a custom `ProximityWeightedCELoss` that up-weights lane boundary pixels, and uses AdamW with a poly LR schedule. |
| `segmentation/local_configs/_base_/datasets/culane_590x590.py` | MMSegmentation dataset config for CULane at 590×1640 resolution. Defines train/val/test pipelines with augmentation (random crop, flip, colour jitter) and repeat sampling. |
| `validate.py` | Validation script for evaluating a trained SegMAN checkpoint. |

---

## Prerequisites

### FENet environment

```bash
conda create -n fenet python=3.8 -y
conda activate fenet
cd /home/g6/Mostafa/Road_Segmentation/FENet
pip install -r requirements.txt
python setup.py build develop      # compiles the NMS C extension — required
```

### SegMAN environment

```bash
conda create -n segman python=3.8 -y
conda activate segman
cd /home/g6/Mostafa/Road_Segmentation/SegMAN
pip install -r requirements.txt
cd segmentation
pip install -e .                   # installs mmseg from source
```

---

## FENet Finetuning

### Step 1 — Prepare the dataset (run once)

```bash
conda activate fenet
cd /home/g6/Mostafa/Road_Segmentation

python Scripts/rename_annotations.py     # rename *.txt → *.lines.txt
python Scripts/generate_seg_masks.py     # generate laneseg_label_w16/ masks
python Scripts/generate_list_files.py    # generate list/ index files
python Scripts/setup_fenet_data_link.py  # create FENet/data/CULane symlink
```

Expected output from the last step:
```
[OK] data/CULane/list/train_gt.txt
[OK] data/CULane/list/val.txt
[OK] data/CULane/list/test.txt
[OK] data/CULane/laneseg_label_w16
```

### Step 2 — Run finetuning

```bash
conda activate fenet
cd /home/g6/Mostafa/Road_Segmentation/FENet

python main.py configs/fenet/FENetV2_dla34_culane_rural_finetune.py \
    --load_from ./fenetv2_culane_dla34.pth --gpus 0 --view
```

- `--load_from` — starts from the pretrained FENetV2 checkpoint instead of random init
- `--gpus 0` — uses GPU 0
- `--view` — saves visualization images after each epoch

### FENet outputs

```
FENet/work_dirs/fenetv2/dla34_culane_rural_finetune/
├── ckpt/
│   ├── 0.pth, 1.pth, ...   ← one checkpoint per epoch (15 total)
├── visualization/           ← predicted lane overlays (overwritten each epoch)
└── *.log                    ← training logs
```

Metrics reported after each epoch (evaluated on the test split, ~407 rural images):

| Metric | Description |
|--------|-------------|
| F1@50 | F1 score at IoU threshold 0.50 (headline metric) |
| F1@75 | F1 score at IoU threshold 0.75 |
| mF1 | Mean F1 across IoU thresholds 0.50–0.95 |
| Precision / Recall | At each threshold |

---

## SegMAN Finetuning

### Step 1 — Prepare the dataset (run once)

```bash
conda activate segman
cd /home/g6/Mostafa/Road_Segmentation/SegMAN

python convert_culane_to_masks.py
```

This generates `SegMAN/data/culane/` with `img_dir/` and `ann_dir/` splits.

### Step 2 — Download pretrained backbone

Place `SegMAN_Encoder_s.pth.tar` in `SegMAN/segmentation/pretrained/`. The config expects it at `pretrained/SegMAN_Encoder_s.pth.tar` relative to the `segmentation/` working directory.

### Step 3 — Run training

```bash
conda activate segman
cd /home/g6/Mostafa/Road_Segmentation/SegMAN/segmentation

bash ../scripts/train_culane.sh
```

Or manually:

```bash
python tools/train.py \
    local_configs/segman/small/segman_s_culane.py \
    --work-dir work_dirs/segman_s_culane \
    --gpu-id 0 \
    --seed 15
```

### SegMAN outputs

```
SegMAN/segmentation/work_dirs/segman_s_culane/
├── best_mIoU_iter_*.pth     ← best checkpoint (by mIoU on val)
├── latest.pth               ← most recent checkpoint
├── *.log.json               ← training metrics per iteration
└── visualizations/          ← val image predictions every 2000 iters
```

Metrics reported every 2000 iterations (evaluated on the val split):

| Metric | Description |
|--------|-------------|
| mIoU | Mean IoU across background + lane classes |
| mFscore | Mean F-score across classes |

### Resuming training from a checkpoint

```bash
python tools/train.py \
    local_configs/segman/small/segman_s_culane.py \
    --work-dir work_dirs/segman_s_culane \
    --resume-from work_dirs/segman_s_culane/latest.pth \
    --gpu-id 0
```

---

## References

```bibtex
@INPROCEEDINGS{10687857,
  author={Wang, Liman and Zhong, Hanyang},
  booktitle={2024 IEEE International Conference on Multimedia and Expo (ICME)},
  title={FENet: Focusing Enhanced Network for Lane Detection},
  year={2024},
  doi={10.1109/ICME57554.2024.10687857}
}

@inproceedings{SegMAN,
  title={SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation},
  author={Yunxiang Fu and Meng Lou and Yizhou Yu},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
