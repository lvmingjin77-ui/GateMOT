# Checkpoints

Download the checkpoints below to `pretrain/` using the listed filenames. Full benchmark reproduction of these downloads is still being verified.

| Dataset | Filename | Download |
|---|---|---|
| BEE24 | `gatemot_bee24.pth` | [Google Drive](https://drive.google.com/file/d/1JoFAW258G3SiDVU27-f2W3zKdSonm916/view?usp=drive_link) |
| MOT17 | `gatemot_mot17.pth` | [Google Drive](https://drive.google.com/file/d/1BkWeOFmDDr5b6sP5bY_k2elt8OSqb1LH/view?usp=drive_link) |
| SportsMOT | `gatemot_sportsmot.pth` | [Google Drive](https://drive.google.com/file/d/1I6cLWAVBhptTKjjuuGDeR_OvFdrCZt8E/view?usp=drive_link) |
| MOT20 | `gatemot_mot20.pth` | [Google Drive](https://drive.google.com/file/d/1HufLPtgFWm9_QShCzFtSZ0LukkQReSxY/view?usp=drive_link) |

## Run the checkpoints

Complete [installation and data preparation](../README.md#installation), then run the block for your dataset from the repository root. Separate ImageNet weights are not needed for inference. These checkpoints use DLA-34 and passed strict loading and a forward check with the settings below.

### BEE24

```bash
export DATASET=bee24 DATA_ROOT=/path/to/BEE24
export LOAD_MODEL=pretrain/gatemot_bee24.pth
export TRACKERS_FOLDER=results/bee24_test
SPLIT=test bash scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER" \
  --K 20 --track_thresh 0.2 --pre_thresh 0.5 --new_thresh 0.4 --max_age 60
```

### MOT17

```bash
export DATASET=mot17 DATA_ROOT=/path/to/MOT17
export LOAD_MODEL=pretrain/gatemot_mot17.pth
export TRACKERS_FOLDER=results/mot17_test
SPLIT=test bash scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER" \
  --wh_weight 0
```

### SportsMOT

The main SportsMOT training and inference scripts default to DLA-60; the DLA-34 override below applies only to this older downloaded checkpoint.

```bash
export DATASET=sportsmot DATA_ROOT=/path/to/SportsMOT/dataset
export LOAD_MODEL=pretrain/gatemot_sportsmot.pth
export TRACKERS_FOLDER=results/sportsmot_test
SPLIT=test bash scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER" \
  --arch dla34
```

For official validation, set `SPLIT=val` and choose a fresh output directory.

### MOT20

This checkpoint was trained for one epoch on 256 MOT20 training frames for pipeline testing. It passed inference on eight real test frames, but produced no tracks at the default thresholds. It is not a converged benchmark model.

```bash
export DATASET=mot20 DATA_ROOT=/path/to/MOT20
export LOAD_MODEL=pretrain/gatemot_mot20.pth
export TRACKERS_FOLDER=results/mot20_test
SPLIT=test bash scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER"
```

### Evaluation

When GT is available, keep `DATASET`, `DATA_ROOT`, and `TRACKERS_FOLDER` from the corresponding block and run:

```bash
SPLIT=test bash scripts/eval_hota.sh
```

Use `SPLIT=val` if you tracked the SportsMOT validation split. Hidden test data supports inference without GT, but local metrics require its annotations. Use fresh result and evaluation directories for each run.

## Backbone initialization

For training, download the ImageNet pretrained weights from the [DLA Model Zoo](http://dl.yf.io/dla/models/imagenet/) into `pretrain/`, keeping the original filenames.
