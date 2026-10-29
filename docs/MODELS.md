# Models

The GateMOT URLs below reserve filenames for the `v1.0` GitHub release. Uploads are pending, so they will not download a model yet. Checkpoint epoch, training settings and evaluation output will be supplied with each model.

## Official test models

Save downloaded models in `pretrain/`.

| Dataset argument | Checkpoint |
|---|---|
| bee24 | [gatemot_bee24.pth](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_bee24.pth) |
| mot17 | [gatemot_mot17.pth](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_mot17.pth) |
| mot20 | [gatemot_mot20.pth](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_mot20.pth) |
| sportsmot | [gatemot_sportsmot.pth](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_sportsmot.pth) |
| dancetrack | [gatemot_dancetrack.pth](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_dancetrack.pth) |

```bash
DATA_ROOT=/path/to/MOT17 scripts/prepare_data.sh mot17 --splits test
DATA_ROOT=/path/to/MOT17 SPLIT=test LOAD_MODEL=pretrain/gatemot_mot17.pth \
  scripts/test.sh mot17 --output_dir results/mot17_test
```

Change the dataset, root and filename together for another model. `test.sh` selects DLA-60 for SportsMOT and DLA-34 for the other datasets.

## Half-training models

These filenames are reserved for models trained on the first half of each training sequence and evaluated on the second half. They are separate from full-training models.

- [gatemot_bee24_halftrain.pth](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_bee24_halftrain.pth)
- [gatemot_mot17_halftrain.pth](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_mot17_halftrain.pth)
- [gatemot_mot20_halftrain.pth](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_mot20_halftrain.pth)

```bash
DATA_ROOT=/path/to/MOT17 scripts/prepare_data.sh mot17 --splits halftrain halfval
DATA_ROOT=/path/to/MOT17 SPLIT=halfval LOAD_MODEL=pretrain/gatemot_mot17_halftrain.pth \
  scripts/test.sh mot17 --output_dir results/mot17_halfval
```

For your own training run, set `LOAD_MODEL` to its generated `model_last.pth` or a selected numbered checkpoint instead.

## Backbone initialization

Official ImageNet weights are available from the DLA project:

- [dla34-ba72cf86.pth](http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth)
- [dla60-24839fc4.pth](http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth)

Place these files in `pretrain/` before training. Inference with a complete GateMOT checkpoint does not need separate ImageNet weights.
