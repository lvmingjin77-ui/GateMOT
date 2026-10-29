# GateMOT

**Q-Gated Attention for Dense Object Tracking**

[arXiv](https://arxiv.org/abs/2604.26353) · [PDF](https://arxiv.org/pdf/2604.26353)

GateMOT brings input- and task-dependent feature selection to dense tracking heads under a tight computation budget. Q-Attention uses Query as a gate on Key and preserves an ungated Value path, separating feature selection from costly global pairwise interactions. The same operator serves detection, motion estimation, and identity embedding.

## Results and model zoo

Test-set results reported in the manuscript. MOT17 and MOT20 use private detections. GateMOT checkpoints are coming soon.

| Dataset | Backbone | HOTA | MOTA | IDF1 | Model |
|---|---|---:|---:|---:|---|
| BEE24 | DLA-34 | 48.4 | 67.8 | 64.5 | [Coming soon](docs/MODELS.md#coming-soon) |
| SportsMOT | DLA-60 | 76.3 | 96.5 | 79.0 | [Coming soon](docs/MODELS.md#coming-soon) |
| MOT17 | DLA-34 | 63.3 | 78.0 | 77.9 | [Coming soon](docs/MODELS.md#coming-soon) |
| MOT20 | DLA-34 | 62.8 | 77.6 | 77.3 | [Coming soon](docs/MODELS.md#coming-soon) |

## Installation

Use Linux with an NVIDIA GPU and a C/C++ compiler. The reference environment is Python 3.8, PyTorch 2.4.1, torchvision 0.19.1, and CUDA 12.1.

```bash
git clone https://github.com/lvmingjin77-ui/GateMOT.git
cd GateMOT
conda create -n gatemot python=3.8 -y
conda activate gatemot
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24.4 Cython==3.1.3 setuptools wheel
pip install -r requirements.txt

git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval
git -C third_party/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

DCN uses torchvision's implementation. Before training, download the ImageNet initialization weights into `pretrain/`, keeping the filenames below. SportsMOT uses DLA-60; the other three datasets use DLA-34.

- [dla34-ba72cf86.pth](http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth)
- [dla60-24839fc4.pth](http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth)

## Data preparation

Download the datasets from their official pages. Set `DATA_ROOT` to the absolute path of the directory containing `train/` and `test/` (and `val/` for SportsMOT).

| Dataset | Download | Training split for validation | Validation split |
|---|---|---|---|
| BEE24 | [BEE24](https://holmescao.github.io/datasets/BEE24) | `halftrain` | `halfval` |
| MOT17 | [MOTChallenge](https://motchallenge.net/data/MOT17/) | `halftrain` | `halfval` |
| MOT20 | [MOTChallenge](https://motchallenge.net/data/MOT20/) | `halftrain` | `halfval` |
| SportsMOT | [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | `train` | `val` |

```text
DATA_ROOT/
  train/SEQ/img1/000001.jpg
  train/SEQ/gt/gt.txt
  val/SEQ/...                 # SportsMOT only
  test/SEQ/img1/000001.jpg
  annotations/                # generated below
```

For the SportsMOT archive, `DATA_ROOT` points to its `dataset/` subdirectory. Prepare the training and validation annotations with:

```bash
DATA_ROOT=/path/to/BEE24 scripts/prepare_data.sh bee24 --splits halftrain halfval
DATA_ROOT=/path/to/MOT17 scripts/prepare_data.sh mot17 --splits halftrain halfval
DATA_ROOT=/path/to/MOT20 scripts/prepare_data.sh mot20 --splits halftrain halfval
DATA_ROOT=/path/to/SportsMOT/dataset scripts/prepare_data.sh sportsmot --splits train val
```

For BEE24, MOT17, and MOT20, the first `floor(N/2)` frames of each training sequence form `halftrain`; the rest form `halfval`. Images stay in `train/`, and the JSON files select the frames. SportsMOT uses its official train/val split. These commands do not require test data. See [data details](docs/DATA_AND_TRAINING.md) for annotation formats and earlier split conventions.

## Training

Run the command for your dataset from the repository root:

```bash
DATA_ROOT=/path/to/BEE24 ANN_NAME=halftrain NUM_GPUS=2 scripts/train.sh bee24
DATA_ROOT=/path/to/MOT17 ANN_NAME=halftrain NUM_GPUS=2 scripts/train.sh mot17
DATA_ROOT=/path/to/MOT20 ANN_NAME=halftrain NUM_GPUS=2 scripts/train.sh mot20
DATA_ROOT=/path/to/SportsMOT/dataset ANN_NAME=train NUM_GPUS=2 scripts/train.sh sportsmot
```

Dataset settings are in [train.sh](scripts/train.sh). `NUM_GPUS` is the number of GPUs. `GLOBAL_BATCH` overrides the total batch size, which must be divisible by `NUM_GPUS`. Choose these according to available GPU memory; see [training settings and resume](docs/DATA_AND_TRAINING.md#training-settings).

Checkpoints and `run_config.json` are saved in `exp/tracking.ctdet/<dataset>_<split>_qattn_s317/`. Each run saves `model_last.pth` and numbered checkpoints. Use validation results to select the checkpoint; training does not select a best model automatically.

## Validation

Choose the block matching your training run:

```bash
# BEE24
export DATASET=bee24 DATA_ROOT=/path/to/BEE24 SPLIT=halfval
export LOAD_MODEL=exp/tracking.ctdet/bee24_halftrain_qattn_s317/model_last.pth
```

```bash
# MOT17
export DATASET=mot17 DATA_ROOT=/path/to/MOT17 SPLIT=halfval
export LOAD_MODEL=exp/tracking.ctdet/mot17_halftrain_qattn_s317/model_last.pth
```

```bash
# MOT20
export DATASET=mot20 DATA_ROOT=/path/to/MOT20 SPLIT=halfval
export LOAD_MODEL=exp/tracking.ctdet/mot20_halftrain_qattn_s317/model_last.pth
```

```bash
# SportsMOT
export DATASET=sportsmot DATA_ROOT=/path/to/SportsMOT/dataset SPLIT=val
export LOAD_MODEL=exp/tracking.ctdet/sportsmot_train_qattn_s317/model_last.pth
```

Then run tracking and compute HOTA, MOTA, and IDF1 with TrackEval:

```bash
export TRACKERS_FOLDER=results/${DATASET}_${SPLIT}
scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER"
scripts/eval_hota.sh
```

Replace `LOAD_MODEL` with a numbered checkpoint to evaluate another epoch. Use a new results directory for each run. Evaluation follows the selected JSON, including frames with no predictions, and leaves the source GT unchanged.

## Test-set inference

After choosing settings on validation data, train on the complete training split for test-set inference. For example, on MOT17:

```bash
export DATASET=mot17 DATA_ROOT=/path/to/MOT17
scripts/prepare_data.sh "$DATASET" --splits train test
ANN_NAME=train NUM_GPUS=2 scripts/train.sh "$DATASET"
SPLIT=test LOAD_MODEL=exp/tracking.ctdet/mot17_train_qattn_s317/model_last.pth \
  scripts/test.sh "$DATASET" --output_dir results/mot17_test
```

The same workflow supports BEE24 and MOT20 by changing the dataset, paths, and checkpoint. SportsMOT already trains on its official `train` split; convert `--splits test` and run its trained model with `SPLIT=test`. See [all four test commands](docs/DATA_AND_TRAINING.md#test-set-inference).

Tracking writes one MOT-format `.txt` file per sequence. Local metric evaluation requires GT; hidden test splits produce result files only. The Model Zoo will provide the checkpoints and corresponding settings for the reported results.

## More information

- [Data, training, and test commands](docs/DATA_AND_TRAINING.md)
- [Q-Attention and implementation details](docs/IMPLEMENTATION.md)
- [Decoder and task-head ablations](docs/CONTROLS.md)
- [Additional DanceTrack experiment](docs/DATA_AND_TRAINING.md#dancetrack)
- [Software checks](docs/VALIDATION.md)

## Acknowledgements

Built on [CenterTrack](https://github.com/xingyizhou/CenterTrack), [DLA](https://github.com/ucbdrive/dla), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/FoundationVision/ByteTrack). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
