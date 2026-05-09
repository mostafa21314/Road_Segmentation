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
├── Scripts/                                              ← data prep + visualisation (our work)
│   ├── rename_annotations.py                             ← FENet prep — Step 1
│   ├── generate_seg_masks.py                             ← FENet prep — Step 2
│   ├── generate_list_files.py                            ← FENet prep — Step 3
│   ├── setup_fenet_data_link.py                          ← FENet prep — Step 4
│   ├── generate_vehicle_masks.py                         ← YOLOv8-seg vehicle masks
│   ├── verify_vehicle_masks.py                           ← sanity-check vehicle/lane overlap
│   ├── generate_driveable_overlays.py                    ← SegMAN driveable overlay (CULane)
│   ├── export_idd_fenet_lanes.py                         ← FENet inference cache (IDD)
│   ├── generate_idd_combined_overlays.py                 ← SegMAN+FENet still overlay (IDD)
│   ├── export_video_fenet_lanes.py                       ← FENet inference cache (video pipeline)
│   ├── render_combined_video.py                          ← SegMAN+FENet video render
│   ├── run_driveable_overlays.sh                         ← wrapper: CULane driveable pipeline
│   ├── run_driveable_overlays_indian.sh                  ← wrapper: IDD combined-still pipeline
│   └── run_combined_video.sh                             ← wrapper: combined-video pipeline
├── FENet/
│   └── configs/fenet/
│       ├── FENetV2_dla34_culane_rural_finetune.py        ← rural finetune config (our work)
│       └── FENetV2_dla34_culane_rural_finetune_vehicle.py← vehicle-aware variant (our work)
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

### Vehicle-aware FENet finetune (YOLOv8 cars as positive anchors)

Hypothesis: when a vehicle covers a lane in CULane, the annotators still drew the lane polyline through it — so the GT lane "passes under" the car. The base seg loss treats those pixels like any other lane pixel; the vehicle-aware variant up-weights them so the model is pushed to *infer* the lane through the occlusion (the car becomes a positive anchor for where a lane should be).

Pipeline:

1. **Generate vehicle masks** (one-time, from any env that has `ultralytics`):

   ```bash
   pip install ultralytics                 # only needed once
   python Scripts/generate_vehicle_masks.py
   ```

   Runs `yolov8s-seg.pt` (auto-downloads on first use, ~22 MB) on every image in `train/`, `val/`, `test/`. Unions COCO classes `car (2)`, `motorcycle (3)`, `bus (5)`, `truck (7)` into one binary mask per image at the original `1640×590` resolution. Output:
   ```
   <dataset>/vehicle_masks/{train,val,test}/<stem>.png   (uint8, 0 or 255)
   ```
   The bigger `yolov8s` is preferred over `yolov8n` because rural CULane has many distant / small vehicles. Confidence threshold is permissive (0.25) since we only use the union, not per-detection scores.

2. **Verify the assumption** — before touching the loss, confirm that GT lanes really do pass through vehicles:

   ```bash
   python Scripts/verify_vehicle_masks.py
   ```

   Picks 30 random training images that have at least one vehicle and writes overlays to `<dataset>/vehicle_masks_overlays/<stem>.png`:
   - Red = vehicle mask, green = GT lane, **yellow = the overlap pixels the loss will target**.
   If lanes clearly stop at vehicle edges (no yellow), the loss would teach the wrong thing — bail out before training.

3. **Run finetuning with the vehicle-aware config**:

   ```bash
   conda activate fenet
   cd FENet

   python main.py configs/fenet/FENetV2_dla34_culane_rural_finetune_vehicle.py \
       --load_from ./fenetv2_culane_dla34.pth --gpus 0 --view
   ```

   How the loss change works (no extra branch, no extra parameters):

   | Step | What happens |
   |---|---|
   | `BaseDataset` | If `cfg.use_vehicle_masks` is `True`, loads `vehicle_masks/<split>/<stem>.png` next to each lane mask and **bit-packs it into the seg label** (`label += 16 * (vehicle > 0)`). |
   | `GenerateLaneLine` | Imgaug runs on the bit-packed seg mask, so geometric transforms stay consistent between the lane channel and the vehicle channel. |
   | `SplitVehicleFromSeg` | New transform inserted between augmentation and `ToTensor`. Splits the mask back into `seg` (lane IDs 0-4) and `vehicle_mask` (0/1). |
   | `ToTensor` | Includes `'vehicle_mask'` in the keys so the head sees it on every batch. |
   | `proximity_seg_loss` | When `vehicle_mask` is provided, GT lane pixels that fall inside a vehicle get `occluded_lane_weight` (default 8.0) instead of 1.0. Border pixels and bg pixels keep their normal weights. |

   The default config uses `occluded_lane_weight = 8.0`. Lower it (e.g. 4.0) if the model starts hallucinating lanes through every car.

4. **Output layout** is identical to the baseline finetune, just under a separate `work_dir`:
   ```
   FENet/work_dirs/fenetv2/dla34_culane_rural_finetune_vehicle/
   ```
   so you can do an A/B comparison against the non-vehicle finetune by checkpoint and visualisation diff.

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

## Visualisation & Inference Pipelines

Three end-to-end pipelines that combine SegMAN driveable predictions with FENet lane predictions. All three are wrapped in self-contained bash scripts in `Scripts/` — you can run them directly without prepping the shell. Each wrapper:

- Strips any active non-conda venv from `PATH` (the `federated_unlearning` venv on this machine shadows conda otherwise).
- Sources `/home/g6/miniconda3/etc/profile.d/conda.sh` and switches between the `fenet` (py3.8) and `segman` (py3.10) envs as needed. Two envs are required because FENet's compiled `nms_impl.so` is built for py3.8 and can't be loaded inside the py3.10 segman env where `mmseg` lives. Each pipeline that touches both models runs FENet first to dump predictions to disk, then runs SegMAN against the cache.

### 1. SegMAN driveable overlay on CULane

```bash
./Scripts/run_driveable_overlays.sh
```

Runs SegMAN(IDD20k) on CULane test images and overlays the predicted driveable area against the **CULane GT lane masks** (no FENet involved). Useful for spotting cases where the annotator drew a lane on what SegMAN considers undriveable.

| Stage | Script | Env | Effect |
|---|---|---|---|
| 1 | `rename_annotations.py` | segman | `<x>.txt` → `<x>.lines.txt` (idempotent). |
| 2 | `generate_seg_masks.py` | segman | Builds `<dataset>/laneseg_label_w16/<split>/*.png`. |
| 3 | `generate_driveable_overlays.py` | segman | Runs SegMAN on a 50% sample of the test split and writes `<dataset>/driveable_overlays/<stem>__on{N}_off{M}.png`. |

Colour key in the overlays: **blue** = driveable area, **green** = GT lane on driveable, **red** = GT lane on non-driveable.

### 2. SegMAN + FENet still overlay on IDD (reverse direction)

```bash
./Scripts/run_driveable_overlays_indian.sh
```

The reverse: runs SegMAN(IDD20k) **and** the rural-finetuned FENet on Indian Driving Dataset val frames, then overlays both predictions on each image. Two-stage because we cross conda envs.

| Stage | Script | Env | Effect |
|---|---|---|---|
| 1 | `export_idd_fenet_lanes.py` | fenet | Picks 30 random IDD val jpgs, runs FENet, caches per-frame lanes to `idd20k/fenet_lanes_cache/<stem>.npy`. |
| 2 | `generate_idd_combined_overlays.py` | segman | Loads each cached prediction, runs SegMAN, writes `idd20k/combined_overlays/<stem>.png`. |
| 3 | (bash) | — | Copies the PNGs into `<dataset>/driveable_overlays_indian/` so all visualisations live together. |

Colour key: **blue** = SegMAN driveable area, **coloured polylines** = FENet lanes (one colour per lane).

### 3. Combined-inference video on either dataset

```bash
./Scripts/run_combined_video.sh <indian|culane> [smooth_k] [fps] [clip] [source_dir]
```

Runs both models on **consecutive frames** from one source clip and produces a video showing the fused output frame-by-frame. Live `cv2.imshow` window plus a saved MP4. Press `q` or `Esc` in the window to stop early — the partial MP4 still flushes.

| Stage | Script | Env | Effect |
|---|---|---|---|
| 1 | `export_video_fenet_lanes.py --dataset $1` | fenet | Picks up to 60 frames from the longest clip in the chosen dataset (or the clip matched by `clip`), runs FENet, caches lanes + a `frame_list.txt` to `<dataset>/video_lanes_cache/`. |
| 2 | `render_combined_video.py --dataset $1 --smooth $2 --fps $3` | segman | Reads `frame_list.txt`, runs SegMAN per frame, composites blue driveable + coloured lanes, writes `<dataset>/combined_video.mp4` while showing a live window. |

Positional arguments (all optional except the dataset):

| Arg | Default | Meaning |
|---|---|---|
| `dataset` | — | `indian` (IDD val, scene `420` is the default longest) or `culane` (rural test). |
| `smooth_k` | 4 | Cross-fade frames inserted between each pair of model-inferenced frames to hide gaps in the source data. `0` disables smoothing entirely. The MP4's stored FPS is scaled to `fps*(1+smooth_k)` so wall-clock pace matches `fps`. |
| `fps` | 10 | Real-frame playback rate. Use 5 for very sparse data, 30 for densely consecutive clips. |
| `clip` | longest | Substring matched against the clip / scene name (e.g. `0325` for the CULane source MP4 ending in `0325`). On no match, the script lists the top-10 candidates and exits. Pass `""` to skip this slot. |
| `source_dir` | dataset default | Override the directory of jpgs to sample from. The CULane train split at `temp_AML/.../culane/img_dir/train` has clips with up to 109 frames vs 31 in the rural test, so it's better for longer videos. |

**Common invocations:**

```bash
# defaults: rural test + auto-pick clip + smooth=4, fps=10
./Scripts/run_combined_video.sh culane

# IDD scene 497 instead of the auto-picked 420
./Scripts/run_combined_video.sh indian 4 10 497

# CULane train split, clip with substring "0325", no smoothing, 5 fps
./Scripts/run_combined_video.sh culane 0 5 0325 \
    /home/g6/temp_AML/Road_Segmentation/SegMAN/data/culane/img_dir/train

# more smoothing, slower playback
./Scripts/run_combined_video.sh indian 8 5
```

A note on "skipped" frames: CULane and IDD both sample frames non-uniformly from their source videos (gaps of 1 sec to 6+ sec are common). The script never drops anything — every available frame in the chosen clip is rendered. Cross-fading hides the visible jumps without faking new model predictions.

To change the frame cap, edit `NUM_FRAMES` at the top of [Scripts/export_video_fenet_lanes.py](Scripts/export_video_fenet_lanes.py).

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
