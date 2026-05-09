# Road Segmentation

Scripts and configs for two experiments using [FENet](https://github.com/WangLiman/FENet) and [SegMAN](https://github.com/yunxiangfu2001/SegMAN):

1. **Lane detection** — finetuning on a curated rural subset of CULane.
2. **Binary road segmentation** — training/finetuning SegMAN on IDD-20k-II with optional CARLA synthetic data.

**Papers:**
- FENet — Wang & Zhong, IEEE ICME 2024. [[arXiv]](https://arxiv.org/abs/2312.17163)
- SegMAN — Fu et al., CVPR 2025. [[arXiv]](https://arxiv.org/abs/2412.11890)

---

## Setup

**FENet** (Python 3.8, CUDA 12.1):
```bash
cd FENet
pip install -r requirements-fenet.txt
python setup.py build develop   # compiles NMS C extension
```

**SegMAN** (Python 3.10, CUDA 12.1):
```bash
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

| Script | Description |
|---|---|
| `rename_annotations.py` | Renames `*.txt` annotation files to `*.lines.txt` (required by FENet's data loader). |
| `generate_seg_masks.py` | Generates `laneseg_label_w16/` grayscale masks (pixel value = lane ID 0–4) for FENet's segmentation loss. |
| `generate_list_files.py` | Creates `list/train_gt.txt`, `list/val.txt`, `list/test.txt`, and `list/test_split/` in FENet's expected format. |
| `setup_fenet_data_link.py` | Creates the symlink `FENet/data/CULane → CULane_Rural_Subset/` to satisfy FENet's hardcoded dataset path. |

### SegMAN CULane Scripts (`SegMAN/`)

| File | Description |
|---|---|
| `convert_culane_to_masks.py` | Converts CULane `.txt` annotations to binary masks and lays out `img_dir/` + `ann_dir/` for MMSeg. |
| `scripts/train_culane.sh` | Launches single-GPU SegMAN-Small training from `SegMAN/segmentation/`. |
| `validate.py` | Evaluates a trained SegMAN checkpoint on the CULane rural subset. |
| `segmentation/local_configs/_base_/datasets/culane_590x590.py` | MMSeg dataset config for CULane at 590×1640 with augmentation and repeat sampling. |

### Running FENet

```bash
# Prepare dataset (once, from repo root)
python Scripts/rename_annotations.py
python Scripts/generate_seg_masks.py
python Scripts/generate_list_files.py
python Scripts/setup_fenet_data_link.py

# Finetune
cd FENet
python main.py configs/fenet/FENetV2_dla34_culane_rural_finetune.py \
    --load_from ./fenetv2_culane_dla34.pth --gpus 0
```

Outputs in `FENet/work_dirs/`.

### Running SegMAN CULane

```bash
cd SegMAN
python convert_culane_to_masks.py   # prepare dataset (once)
# place SegMAN_Encoder_s.pth.tar in segmentation/pretrained/
cd segmentation && bash ../scripts/train_culane.sh
```

Outputs in `SegMAN/segmentation/work_dirs/`.

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
