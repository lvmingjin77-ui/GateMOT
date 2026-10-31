# Checkpoints

Checkpoints for the four main experiments. Each model is trained on the full training split and evaluated on the test split. Download a checkpoint to `pretrain/` using the listed filename.

| Dataset | Backbone | HOTA | MOTA | IDF1 | Filename | Download |
|---|---|---:|---:|---:|---|---|
| BEE24 | DLA-34 | 48.4 | 67.8 | 64.5 | `gatemot_bee24.pth` | [Google Drive](https://drive.google.com/file/d/1JoFAW258G3SiDVU27-f2W3zKdSonm916/view?usp=drive_link) |
| MOT17 | DLA-34 | 63.3 | 78.0 | 77.9 | `gatemot_mot17.pth` | [Google Drive](https://drive.google.com/file/d/1BkWeOFmDDr5b6sP5bY_k2elt8OSqb1LH/view?usp=drive_link) |
| SportsMOT | DLA-60 | 76.3 | 96.5 | 79.0 | `gatemot_sportsmot.pth` | [Google Drive](https://drive.google.com/file/d/1I6cLWAVBhptTKjjuuGDeR_OvFdrCZt8E/view?usp=drive_link) |
| MOT20 | DLA-34 | 62.8 | 77.6 | 77.3 | `gatemot_mot20.pth` | [Google Drive](https://drive.google.com/file/d/1HufLPtgFWm9_QShCzFtSZ0LukkQReSxY/view?usp=drive_link) |

Scores are the test-set results reported in the paper. MOT17 and MOT20 use private detections. All files are also in the [Google Drive folder](https://drive.google.com/drive/folders/1Pmd2Y_tkax5D5gQQpRpCh0YTwBl27qiD).

## Run the checkpoints

Complete [installation and data preparation](../README.md#installation), then run the block for your dataset from the repository root. ImageNet weights are not needed for inference. The checkpoints use Q-Attention heads and the default settings of `scripts/test.sh`, which selects the backbone for each dataset.

### BEE24

```bash
export DATASET=bee24 DATA_ROOT=/path/to/BEE24
export LOAD_MODEL=pretrain/gatemot_bee24.pth
export TRACKERS_FOLDER=results/bee24_test
SPLIT=test bash scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER"
```

### MOT17

```bash
export DATASET=mot17 DATA_ROOT=/path/to/MOT17
export LOAD_MODEL=pretrain/gatemot_mot17.pth
export TRACKERS_FOLDER=results/mot17_test
SPLIT=test bash scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER"
```

### SportsMOT

```bash
export DATASET=sportsmot DATA_ROOT=/path/to/SportsMOT/dataset
export LOAD_MODEL=pretrain/gatemot_sportsmot.pth
export TRACKERS_FOLDER=results/sportsmot_test
SPLIT=test bash scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER"
```

For official validation, set `SPLIT=val` and choose a fresh output directory. The official test-server output of this model is in [SportsMOTEvaluationOutput](../SportsMOTEvaluationOutput/scoring_result/scores.txt).

### MOT20

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
