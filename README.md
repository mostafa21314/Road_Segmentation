# Road Segmentation

Scripts and configs for two experiments using [FENet](https://github.com/WangLiman/FENet) and [SegMAN](https://github.com/yunxiangfu2001/SegMAN):

1. **Lane detection** — finetuning on a curated rural subset of CULane.
2. **Binary road segmentation** — training/finetuning SegMAN on IDD-20k-II with optional CARLA synthetic data.

**Papers:**
- FENet — Wang & Zhong, IEEE ICME 2024. [[arXiv]](https://arxiv.org/abs/2312.17163)
- SegMAN — Fu et al., CVPR 2025. [[arXiv]](https://arxiv.org/abs/2412.11890)

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
│   ├── prepare_idd20k_road_masks.py                      ← IDD-20k-II mask preparation (our work)
│   ├── train_segman_tiny_idd_baseline.py                 ← IDD baseline training (our work)
│   ├── train_segman_small_cnn_branch_idd.py              ← CNN-branch variant training (our work)
│   ├── finetune_segman_tiny_on_carla.py                  ← CARLA-only finetune (our work)
│   ├── finetune_segman_tiny_mixed_idd_carla.py           ← Mixed IDD+CARLA finetune (our work)
│   ├── finetune_segman_tiny_carla_frozen_backbone.py     ← Frozen-backbone CARLA finetune (our work)
│   ├── eval_segman_tiny_idd_baseline.py                  ← evaluation script (our work)
│   ├── eval_segman_small_cnn_branch.py                   ← CNN-branch evaluation (our work)
│   ├── inference_visualize_100_images.py                 ← qualitative visualization (our work)
│   ├── train.py                                          ← SegMAN CULane training entry point (our work)
│   ├── validate.py                                       ← SegMAN CULane validation script (our work)
│   ├── scripts/
│   │   └── train_culane.sh                               ← training launch script (our work)
│   ├── simulator/
│   │   └── carla_collect_town07_rgb_seg.py               ← CARLA data collection (our work)
│   └── segmentation/
│       └── local_configs/
│           ├── segman/
│           │   ├── tiny/segman_t_idd20k.py               ← IDD baseline config (our work)
│           │   ├── tiny/segman_t_carla_finetune.py       ← CARLA-only finetune config (our work)
│           │   ├── tiny/segman_t_mixed_finetune.py       ← Mixed IDD+CARLA config (our work)
│           │   ├── tiny/segman_t_carla_freeze_finetune.py← Frozen-backbone config (our work)
│           │   ├── small/segman_s_cnn_branch_idd20k.py   ← CNN-branch config (our work)
│           │   └── small/segman_s_culane.py              ← SegMAN-Small CULane config (our work)
│           └── _base_/datasets/
│               ├── idd20k.py                             ← IDD-20k-II dataset config (our work)
│               └── culane_590x590.py                     ← CULane dataset config (our work)
└── CULane_Rural_Subset(1)/CULane_Rural_Subset/           ← curated dataset (not committed)
```

---

## Setup

**FENet** (Python 3.8, CUDA 12.1):
```bash
conda create -n fenet python=3.8 -y
conda activate fenet
cd FENet
pip install -r requirements-fenet.txt
python setup.py build develop   # compiles NMS C extension — required
```

**SegMAN** (Python 3.10, CUDA 12.1):
```bash
conda create -n segman python=3.8 -y
conda activate segman
cd SegMAN
pip install -r requirements-segmanaser.txt
cd segmentation && pip install -e .
```

---

## Binary Road Segmentation — IDD-20k-II + CARLA

### Experiments

Binary segmentation (background=0, road=1) using three training strategies. All use `CrossEntropy(class_weight=[1.0, 3.0]) + DiceLoss(loss_weight=3.0)` and pretrained encoder weights from `segmentation/pretrained/SegMAN_Encoder_{t,s}.pth.tar`.

| Experiment | Script | Description |
|---|---|---|
| IDD baseline | `train_segman_tiny_idd_baseline.py` | SegMAN-Tiny trained on IDD-20k-II for 40k iterations. |
| CNN-branch variant | `train_segman_small_cnn_branch_idd.py` | SegMAN-Small with a parallel CNN branch (`conv_downsample_2/4` + `pixel_unshuffle`) in the decoder, trained for 20k iterations. |
| CARLA-only finetune | `finetune_segman_tiny_on_carla.py` | Adapts the IDD baseline using only CARLA Town07 synthetic data. |
| Mixed IDD+CARLA | `finetune_segman_tiny_mixed_idd_carla.py` | Jointly finetunes on IDD and CARLA with the backbone frozen in early stages. |
| Frozen-backbone CARLA | `finetune_segman_tiny_carla_frozen_backbone.py` | CARLA 80/20 split with frozen backbone stages; evaluates before/after on both domains. |

### Configs (`SegMAN/segmentation/local_configs/`)

| Config | Description |
|---|---|
| `_base_/datasets/idd20k.py` | MMSeg dataset config for IDD-20k-II. |
| `_base_/models/segman.py` | Base SegMAN model definition. |
| `segman/tiny/segman_t_idd20k.py` | Baseline IDD training config. |
| `segman/tiny/segman_t_carla_finetune.py` | CARLA-only finetune config. |
| `segman/tiny/segman_t_mixed_finetune.py` | Mixed IDD+CARLA finetune config. |
| `segman/tiny/segman_t_carla_freeze_finetune.py` | Frozen-backbone CARLA finetune config. |
| `segman/small/segman_s_cnn_branch_idd20k.py` | CNN-branch decoder config. |

### Dataset Preparation

```bash
# IDD-20k-II: convert polygon JSON annotations to binary PNGs
# Outputs flat binary masks to data/idd20k/{img_dir,ann_dir}/{train,val}/
cd SegMAN && python prepare_idd20k_road_masks.py

# CARLA: collect paired RGB + drivable-area masks (255=drivable, 0=background)
# Requires CARLA server running on 127.0.0.1:2000 with Town07 loaded
python simulator/carla_collect_town07_rgb_seg.py --frames 5000 --out ./Carladataset
```

### Training

All scripts run from `SegMAN/`:

```bash
python train_segman_tiny_idd_baseline.py
python train_segman_small_cnn_branch_idd.py
python finetune_segman_tiny_on_carla.py
python finetune_segman_tiny_mixed_idd_carla.py
python finetune_segman_tiny_carla_frozen_backbone.py
```

### Evaluation

```bash
python eval_segman_tiny_idd_baseline.py   # mIoU, mAcc, pixel P/R/F1, image-level F1@IoU≥0.5
python eval_segman_small_cnn_branch.py
python inference_visualize_100_images.py  # qualitative 4-panel overlay figures
```

Outputs (confusion matrix, bar chart, radar, per-image overlays) written to `eval_results/`.

---

## Lane Detection — CULane Rural Subset

### Experiments

Finetuning FENet and training SegMAN on a curated subset of 2440 rural CULane images.

| Model | Config / Script | Description |
|---|---|---|
| FENet | `FENet/configs/fenet/FENetV2_dla34_culane_rural_finetune.py` | DLA-34 backbone finetuned from a CULane pretrained checkpoint. `batch_size=8`, `lr=2e-4`, cosine scheduler. Outputs per-epoch F1@50/75, mF1, P/R. |
| SegMAN | `segmentation/local_configs/segman/small/segman_s_culane.py` | SegMAN-Small, 2-class output, `ProximityWeightedCELoss`, AdamW + poly LR, 590×1640 input. Outputs mIoU and mFscore every 2000 iters. |

### Data Preparation Scripts (`Scripts/`)

| Script | Purpose |
|--------|---------|
| `rename_annotations.py` | Renames `*.txt` lane annotation files to `*.lines.txt` inside `train/`, `val/`, `test/`. Required because FENet's dataset loader looks for the `.lines.txt` suffix. |
| `generate_seg_masks.py` | Generates `laneseg_label_w16/` segmentation masks (grayscale PNGs where pixel value = lane ID 0–4) from the annotation files. Used by FENet's segmentation loss during training. |
| `generate_list_files.py` | Creates `list/train_gt.txt`, `list/val.txt`, `list/test.txt`, and `list/test_split/` category files in the format FENet's dataset loader expects. `train_gt.txt` includes per-image lane existence flags. |
| `setup_fenet_data_link.py` | Creates the symlink `FENet/data/CULane → CULane_Rural_Subset/` so FENet's hardcoded `dataset_path = './data/CULane'` resolves to our rural dataset. |

### SegMAN CULane Scripts (`SegMAN/`)

| File | Purpose |
|------|---------|
| `convert_culane_to_masks.py` | Converts CULane coordinate annotations (`.txt` files) into binary segmentation masks (`.png`) and reorganises the dataset into MMSegmentation's `img_dir/` + `ann_dir/` structure. Creates symlinks for images to save disk space. |
| `scripts/train_culane.sh` | Bash script that launches single-GPU SegMAN-Small training with the CULane config. Run from `SegMAN/segmentation/`. |
| `segmentation/local_configs/segman/small/segman_s_culane.py` | SegMAN-Small model config for CULane. Sets 2-class output (background + lane), uses a custom `ProximityWeightedCELoss` that up-weights lane boundary pixels, and uses AdamW with a poly LR schedule. |
| `segmentation/local_configs/_base_/datasets/culane_590x590.py` | MMSegmentation dataset config for CULane at 590×1640 resolution. Defines train/val/test pipelines with augmentation (random crop, flip, colour jitter) and repeat sampling. |
| `validate.py` | Validation script for evaluating a trained SegMAN checkpoint. |

### FENet Config Changes

The finetune config (`FENetV2_dla34_culane_rural_finetune.py`) is adapted from the full-CULane baseline with these changes for a small rural dataset (2440 training images):

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

### Running FENet

#### Step 1 — Prepare the dataset (run once)

```bash
conda activate fenet
cd Road_Segmentation

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

#### Step 2 — Run finetuning

```bash
conda activate fenet
cd FENet

python main.py configs/fenet/FENetV2_dla34_culane_rural_finetune.py \
    --load_from ./fenetv2_culane_dla34.pth --gpus 0 --view
```

- `--load_from` — starts from the pretrained FENetV2 checkpoint instead of random init
- `--gpus 0` — uses GPU 0
- `--view` — saves visualization images after each epoch

#### FENet outputs

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

### Running SegMAN CULane

#### Step 1 — Prepare the dataset (run once)

```bash
conda activate segman
cd SegMAN

python convert_culane_to_masks.py
```

This generates `SegMAN/data/culane/` with `img_dir/` and `ann_dir/` splits.

#### Step 2 — Download pretrained backbone

Place `SegMAN_Encoder_s.pth.tar` in `SegMAN/segmentation/pretrained/`. The config expects it at `pretrained/SegMAN_Encoder_s.pth.tar` relative to the `segmentation/` working directory.

#### Step 3 — Run training

```bash
conda activate segman
cd SegMAN/segmentation

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

#### SegMAN outputs

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

#### Resuming training from a checkpoint

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
