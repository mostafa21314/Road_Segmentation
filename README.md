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
├── Scripts/                                                        ← CULane data prep
│   ├── rename_annotations.py
│   ├── generate_seg_masks.py
│   ├── generate_list_files.py
│   └── setup_fenet_data_link.py
├── FENet/
│   └── configs/fenet/
│       └── FENetV2_dla34_culane_rural_finetune.py                  ← finetune config
├── SegMAN/
│   ├── convert_culane_to_masks.py                                  ← CULane → MMSeg masks
│   ├── validate.py                                                 ← CULane validation
│   ├── prepare_idd20k_road_masks.py                                ← IDD-20k-II → binary masks
│   ├── train_segman_tiny_idd_baseline.py                           ← IDD baseline training
│   ├── train_segman_small_cnn_branch_idd.py                        ← CNN-branch variant training
│   ├── finetune_segman_tiny_on_carla.py                            ← CARLA-only finetune
│   ├── finetune_segman_tiny_mixed_idd_carla.py                     ← mixed IDD+CARLA finetune
│   ├── finetune_segman_tiny_carla_frozen_backbone.py               ← CARLA finetune, frozen backbone
│   ├── eval_segman_tiny_idd_baseline.py                            ← evaluate IDD baseline
│   ├── eval_segman_small_cnn_branch.py                             ← evaluate CNN-branch model
│   ├── inference_visualize_100_images.py                           ← qualitative inference
│   ├── simulator/
│   │   └── carla_collect_town07_rgb_seg.py                         ← CARLA dataset collection
│   ├── scripts/
│   │   └── train_culane.sh
│   └── segmentation/local_configs/
│       ├── _base_/
│       │   ├── datasets/culane_590x590.py
│       │   ├── datasets/idd20k.py                                  ← IDD-20k-II dataset config
│       │   └── models/segman.py                                    ← base SegMAN model config
│       └── segman/
│           ├── small/segman_s_culane.py
│           ├── small/segman_s_cnn_branch_idd20k.py                 ← CNN-branch decoder config
│           └── tiny/
│               ├── segman_t_idd20k.py                              ← baseline IDD config
│               ├── segman_t_carla_finetune.py                      ← CARLA-only finetune config
│               ├── segman_t_mixed_finetune.py                      ← mixed IDD+CARLA config
│               └── segman_t_carla_freeze_finetune.py               ← frozen-backbone CARLA config
└── CULane_Rural_Subset(1)/CULane_Rural_Subset/                     ← dataset (not committed)
```
---

## Setup

**FENet** (`conda env: fenet`, Python 3.8):
```bash
cd Road_Segmentation/FENet && pip install -r requirements.txt
python setup.py build develop   # compiles NMS C extension
```

**SegMAN — CULane** (`conda env: segman`, Python 3.8):
```bash
cd Road_Segmentation/SegMAN && pip install -r requirements.txt
cd segmentation && pip install -e .
```

**SegMAN — IDD/CARLA** (`conda env: segmanaser`): same as above; scripts are invoked via `conda run -n segmanaser`.

---

## Binary Road Segmentation — IDD-20k-II + CARLA (`SegMAN/`)

Three experiments on binary road segmentation (background=0, road=1):

**1. IDD-20k-II baseline** — SegMAN-Tiny trained from pretrained encoder weights on IDD-20k-II for 40k iterations.

**2. CNN-branch variant** — SegMAN-Small with a parallel CNN branch (`conv_downsample_2/4` + `pixel_unshuffle`) added to `forward_winssm` in the decoder, trained on IDD-20k-II for 20k iterations.

**3. CARLA synthetic finetuning** — three strategies for adapting the IDD baseline using CARLA Town07 data:
- CARLA-only finetune (`finetune_segman_tiny_on_carla.py`)
- Mixed IDD+CARLA with early backbone frozen (`finetune_segman_tiny_mixed_idd_carla.py`)
- CARLA 80/20 split with frozen backbone stages, 4-way before/after eval on both domains (`finetune_segman_tiny_carla_frozen_backbone.py`)

All configs use `CrossEntropy(class_weight=[1.0, 3.0]) + DiceLoss(loss_weight=3.0)`.

### Dataset preparation

```bash
# Generate IDD-20k-II binary masks
conda run -n segmanaser python prepare_idd20k_road_masks.py

# Collect CARLA paired RGB + mask frames (requires CARLA running on 127.0.0.1:2000, Town07)
python simulator/carla_collect_town07_rgb_seg.py --frames 5000 --out ./Carladataset
```

`prepare_idd20k_road_masks.py` converts the IDD-20k-II polygon JSON annotations into flat binary PNGs under `data/idd20k/{img_dir,ann_dir}/{train,val}/`. `carla_collect_town07_rgb_seg.py` drives an autopilot vehicle, collecting pixel-aligned RGB + drivable-area masks (255 = drivable, 0 = background).

### Training

```bash
conda run -n segmanaser python train_segman_tiny_idd_baseline.py       # IDD baseline
conda run -n segmanaser python train_segman_small_cnn_branch_idd.py    # CNN-branch variant
conda run -n segmanaser python finetune_segman_tiny_on_carla.py        # CARLA-only
conda run -n segmanaser python finetune_segman_tiny_mixed_idd_carla.py # mixed
conda run -n segmanaser python finetune_segman_tiny_carla_frozen_backbone.py  # frozen backbone
```

All scripts run from `SegMAN/` and expect pretrained encoder weights at `segmentation/pretrained/SegMAN_Encoder_{t,s}.pth.tar`.

### Evaluation

```bash
conda run -n segmanaser python eval_segman_tiny_idd_baseline.py   # IDD baseline metrics + plots
conda run -n segmanaser python eval_segman_small_cnn_branch.py    # CNN-branch metrics + plots
conda run -n segmanaser python inference_visualize_100_images.py  # qualitative 4-panel figures
```

Eval scripts compute mIoU, mAcc, pixel P/R/F1, and image-level F1@IoU≥0.5, and write confusion matrix, bar chart, radar, and per-image overlay figures to `eval_results/`.

---

## Lane Detection — CULane Rural Subset

### Data Preparation Scripts (`Scripts/`)

| Script | Purpose |
|--------|---------|
| `rename_annotations.py` | Renames `*.txt` annotation files to `*.lines.txt` (required by FENet's loader). |
| `generate_seg_masks.py` | Generates `laneseg_label_w16/` grayscale masks (pixel value = lane ID 0–4) used by FENet's segmentation loss. |
| `generate_list_files.py` | Creates `list/train_gt.txt`, `list/val.txt`, `list/test.txt`, and `list/test_split/` in FENet's expected format. |
| `setup_fenet_data_link.py` | Creates the symlink `FENet/data/CULane → CULane_Rural_Subset/` to satisfy FENet's hardcoded dataset path. |

### FENet Config (`FENet/configs/fenet/FENetV2_dla34_culane_rural_finetune.py`)

Adapted from the full-CULane baseline for finetuning on 2440 rural images: `pretrained=False` (weights from `--load_from`), `batch_size=8`, `lr=2e-4`, cosine scheduler `T_max` matched to dataset size, `eval_ep=1`, `log_interval=50`.

### SegMAN CULane Scripts (`SegMAN/`)

| File | Purpose |
|------|---------|
| `convert_culane_to_masks.py` | Converts CULane `.txt` annotations to binary masks and lays out `img_dir/` + `ann_dir/` for MMSeg. |
| `scripts/train_culane.sh` | Launches single-GPU SegMAN-Small training from `SegMAN/segmentation/`. |
| `segmentation/local_configs/segman/small/segman_s_culane.py` | SegMAN-Small config for CULane: 2-class output, `ProximityWeightedCELoss`, AdamW + poly LR. |
| `segmentation/local_configs/_base_/datasets/culane_590x590.py` | MMSeg dataset config for CULane at 590×1640 with augmentation and repeat sampling. |
| `validate.py` | Evaluates a trained SegMAN checkpoint on CULane. |

### Running FENet

```bash
# Prepare dataset (once)
python Scripts/rename_annotations.py && python Scripts/generate_seg_masks.py
python Scripts/generate_list_files.py && python Scripts/setup_fenet_data_link.py

# Finetune
cd FENet
python main.py configs/fenet/FENetV2_dla34_culane_rural_finetune.py \
    --load_from ./fenetv2_culane_dla34.pth --gpus 0 --view
```

Outputs (per-epoch F1@50/75, mF1, P/R) in `FENet/work_dirs/fenetv2/dla34_culane_rural_finetune/`.

### Running SegMAN CULane

```bash
cd SegMAN && python convert_culane_to_masks.py   # prepare dataset (once)
# place SegMAN_Encoder_s.pth.tar in segmentation/pretrained/ first
cd segmentation && bash ../scripts/train_culane.sh
```

Outputs (mIoU, mFscore every 2000 iters) in `SegMAN/segmentation/work_dirs/segman_s_culane/`.

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
