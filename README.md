# GateMOT

**Q-gated attention for dense multi-object tracking**

GateMOT addresses feature selection in dense tracking heads, where high-resolution features meet a limited computation budget. Q-Attention uses Query as a gate on Key, locally pools the selected features, and fuses them with an ungated Value path. It avoids global pairwise affinity computation while giving each prediction task its own input-dependent selection.

This repository contains the DLA-based training, online association, data conversion, and evaluation pipeline. The manuscript is under review. See [implementation details](docs/IMPLEMENTATION.md) and [ablation settings](docs/CONTROLS.md).

## Results and models

Results below are reported in the manuscript. Checkpoint downloads will be added here. The script column links to the current training entry point. Checkpoint-specific settings will accompany each weight release.

| Dataset | Backbone | HOTA | MOTA | IDF1 | Training script | Weights |
|---|---|---:|---:|---:|---|---|
| BEE24-test | DLA-34 | 48.4 | 67.8 | 64.5 | [train.sh bee24](scripts/train.sh) | — |
| SportsMOT | DLA-60 | 76.3 | 96.5 | 79.0 | [train.sh sportsmot](scripts/train.sh) | — |
| MOT17 | DLA-34 | 63.3 | 78.0 | 77.9 | [train.sh mot17](scripts/train.sh) | — |
| MOT20 | DLA-34 | 62.8 | 77.6 | 77.3 | [train.sh mot20](scripts/train.sh) | — |
| DanceTrack | DLA-34 | 61.1 | 89.8 | 64.3 | [train.sh dancetrack](scripts/train.sh) | — |

BEE24-test is the best result among the methods compared in the manuscript. MOT17/MOT20 use private detections. The corrected SportsMOT AssA is 65.8 ([evaluation output](https://anonymous.4open.science/r/mt7/scoring_result/scores.txt)).

## Installation

The validation environment uses Linux, Python 3.8, PyTorch 2.4.1, torchvision 0.19.1 and CUDA 12.1 wheels on an NVIDIA V100S. Package versions are constrained in `requirements.txt`. A C/C++ compiler is needed for `cython-bbox`. The default DCN implementation uses the packaged torchvision operator and does **not** require compiling the legacy CUDA extension.

```bash
conda create -n gatemot python=3.8 -y
conda activate gatemot
cd GateMOT-release
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.24.4 Cython==3.1.3 setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

For CPU installations, use the matching PyTorch CPU wheels. See [software checks](docs/VALIDATION.md) for a small train/resume/inference test that needs no dataset or pretrained weights.

Normal training initializes the backbone from ImageNet weights in `pretrain/`. Obtain DLA-34 (`dla34-ba72cf86.pth`) or DLA-60 (`dla60-24839fc4.pth`) from the [DLA project](https://github.com/ucbdrive/dla). Use `--no_pretrained` for random initialization. This changes the training setting. Inference with a complete tracking checkpoint does not require ImageNet weights.

## Data preparation

Use MOTChallenge-style directories with `train/SEQ/img1/`, `train/SEQ/gt/gt.txt`, and corresponding `val/` or `test/` directories. Image filenames must be numeric, such as `000001.jpg` or `00000001.jpg`. Obtain datasets under their own access and licensing terms. No dataset is redistributed here.

```bash
python src/tools/bee_to_coco.py --data-root /path/to/bee24
python src/tools/mot17_to_coco.py --data-root /path/to/mot17
python src/tools/mot20_to_coco.py --data-root /path/to/mot20
python src/tools/sportsmot_to_coco.py --data-root /path/to/sportsmot
python src/tools/dancetrack_to_coco.py --data-root /path/to/dancetrack
```

Converters write `annotations/*.json`. Track identities are stable across frames and distinct across sequences. Test annotations contain image metadata without GT labels. Existing old JSON files should be regenerated after updating to this release.

BEE24 uses the first `floor(N/2)` frames of each training sequence for `halftrain` and the remainder for `halfval`. MOT17/MOT20 retain the inherited split boundary at `floor(N/2)+1`, with the remaining frames in `halfval`. No boundary frame is discarded. SportsMOT/DanceTrack use their official `train`, `val`, and `test` directories. `--splits train` can convert only one split. Half-split conversion reads the training directory only.

## Training

```bash
# Fit on the training half for validation-based selection.
ANN_NAME=halftrain DATA_ROOT=/path/to/bee24 NUM_GPUS=4 scripts/train.sh bee24

# Full training split after fixing the selected settings.
DATA_ROOT=/path/to/bee24 NUM_GPUS=4 scripts/train.sh bee24

# Resume the same experiment explicitly.
DATA_ROOT=/path/to/bee24 NUM_GPUS=4 scripts/train.sh bee24 \
  --resume --load_model exp/tracking.ctdet/bee24_train_qattn_s317/model_last.pth
```

Experiment IDs default to `<dataset>_<annotation>_<head_type>_s<seed>`. Set `EXP_ID` for distinct trials. Existing checkpoints cannot be overwritten accidentally. `GLOBAL_BATCH` is the total batch across `NUM_GPUS` processes and must be divisible by that number. Changing batch size changes the optimization setting.

| Dataset | Backbone | Epochs | Global batch | LR | Decay after epoch | K | ReID weight |
|---|---|---:|---:|---:|---:|---:|---:|
| bee24 | DLA-34 | 60 | 24 | 2e-4 | 40 | 200 | 1.0 |
| mot17 | DLA-34 | 70 | 16 | 1.25e-4 | 60 | 256 | 1.0 |
| mot20 | DLA-34 | 70 | 16 | 1.15e-4 | 60 | 256 | 1.0 |
| dancetrack | DLA-34 | 90 | 16 | 1.3e-4 | 50 | 100 | 0.5 |
| sportsmot | DLA-60 | 80 | 20 | 1.2e-4 | 45 | 100 | 1.0 |

These are the defaults in [scripts/train.sh](scripts/train.sh). All scripts use input 608×1088, Adam, one Q-Attention transform per head, and an auxiliary `wh` head. Default weight decay and warmup are zero. Both can be set with `--weight_decay` and `--warmup_epochs`.

Training saves `model_last.pth` every epoch, numbered checkpoints periodically and at the final epoch, and settings in `run_config.json`. Select checkpoints on validation data. `--resume` restores model and optimizer state.

## Tracking and evaluation

```bash
SPLIT=halfval DATA_ROOT=/path/to/bee24 \
LOAD_MODEL=exp/tracking.ctdet/bee24_halftrain_qattn_s317/model_last.pth \
scripts/test.sh bee24 --output_dir results/bee24_halfval_qattn
```

Use `SPLIT=test` for the official test split and `SPLIT=val` for official validation splits. Use the checkpoint's architecture, head type, and overrides. Loading is strict by default. Use `--allow_partial_load` only for transfer initialization. `FUSE=1` folds Q-Attention's BatchNorm for inference.

Tracking uses exactly the manifest's sequences and frames, preserving original frame numbers in MOT txt files. The output directory must be fresh. `evaluation_config.json` records settings, checkpoint/annotation hashes, and evaluated frame IDs. Internal CLEAR output is diagnostic. Use TrackEval for canonical metrics.

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval
git -C third_party/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
DATASET=bee24 SPLIT=halfval DATA_ROOT=/path/to/bee24 \
TRACKERS_FOLDER=results/bee24_halfval_qattn \
scripts/eval_hota.sh
```

The wrapper builds an evaluation-only copy of GT and predictions for the exact annotation-defined frames, consistently renumbers both, and runs HOTA/CLEAR/Identity. It never modifies source GT. Set `TRACK_EVAL` for another installation and `EVAL_DIR` for a fresh bundle directory. Local evaluation requires GT. For hidden benchmark test sets, submit the original tracking txt files to the official evaluator instead. The wrapper includes NumPy compatibility for this TrackEval revision.

## Ablations and demo

See [docs/CONTROLS.md](docs/CONTROLS.md) for Conv, Conv+BN, GLU-style gating, gated convolution, and per-task replacements that retain the prediction outputs and losses. 

```bash
DEMO=/path/to/SEQ/img1 LOAD_MODEL=/path/to/model.pth scripts/demo.sh
```

For SportsMOT, set `ARCH=dla60`. Demo visualization requires a graphical display. Use `scripts/test.sh` for headless batch inference.

## Attribution and citation

The implementation builds on [CenterTrack](https://github.com/xingyizhou/CenterTrack), [DLA](https://github.com/ucbdrive/dla), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/ifzhang/ByteTrack). The optional legacy operator comes from [DCNv2](https://github.com/CharlesShang/DCNv2). Evaluation uses [TrackEval](https://github.com/JonathonLuiten/TrackEval). See [LICENSE](LICENSE), [NOTICE](NOTICE), and retained third-party notices. A final bibliographic citation will be added when publication information is available.
