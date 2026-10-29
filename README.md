# GateMOT

**Q-Gated Attention for Dense Object Tracking**

GateMOT brings task-specific feature selection to dense tracking heads. Query gates Key, while a separate Value path preserves detail, avoiding the cost of global pairwise attention.

## Results

Manuscript results. Dataset links below are reserved for model downloads; checkpoint uploads are pending. MOT17 and MOT20 use private detections.

| Dataset | Backbone | HOTA | MOTA | IDF1 |
|---|---|---:|---:|---:|
| [BEE24](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_bee24.pth) | DLA-34 | 48.4 | 67.8 | 64.5 |
| [SportsMOT](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_sportsmot.pth) | DLA-60 | 76.3 | 96.5 | 79.0 |
| [MOT17](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_mot17.pth) | DLA-34 | 63.3 | 78.0 | 77.9 |
| [MOT20](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_mot20.pth) | DLA-34 | 62.8 | 77.6 | 77.3 |
| [DanceTrack](https://github.com/lvmingjin77-ui/GateMOT/releases/download/v1.0/gatemot_dancetrack.pth) | DLA-34 | 61.1 | 89.8 | 64.3 |

## Installation

```bash
git clone https://github.com/lvmingjin77-ui/GateMOT.git
cd GateMOT
conda create -n gatemot python=3.8 -y
conda activate gatemot
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24.4 Cython==3.1.3 setuptools wheel
pip install -r requirements.txt
```

Download the ImageNet [DLA-34](http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth) or [DLA-60](http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth) weights into `pretrain/`. DCN uses torchvision by default.

## Data preparation

```bash
DATA_ROOT=/path/to/MOT17 scripts/prepare_data.sh mot17 --splits halftrain halfval
DATA_ROOT=/path/to/MOT20 scripts/prepare_data.sh mot20 --splits halftrain halfval
DATA_ROOT=/path/to/BEE24 scripts/prepare_data.sh bee24 --splits halftrain halfval
```

Each sequence's first half is used for training and its second half for validation. Images stay in `train/`; the generated JSON files select the frames. See [data layout and the complete workflow](docs/DATA_AND_TRAINING.md) for full-split training, SportsMOT and DanceTrack.

## Training and evaluation

```bash
export DATA_ROOT=/path/to/MOT17
ANN_NAME=halftrain NUM_GPUS=2 scripts/train.sh mot17
SPLIT=halfval LOAD_MODEL=exp/tracking.ctdet/mot17_halftrain_qattn_s317/model_last.pth \
  scripts/test.sh mot17 --output_dir results/mot17_halfval
DATASET=mot17 SPLIT=halfval TRACKERS_FOLDER=results/mot17_halfval scripts/eval_hota.sh
```

Install [TrackEval](docs/DATA_AND_TRAINING.md#evaluation) before evaluation. Dataset settings are in [train.sh](scripts/train.sh) and [test.sh](scripts/test.sh). For downloaded models, use the matching commands in [Models](docs/MODELS.md).

[Implementation](docs/IMPLEMENTATION.md) · [Ablations](docs/CONTROLS.md) · [Tests](docs/VALIDATION.md)

## Acknowledgements

Built on [CenterTrack](https://github.com/xingyizhou/CenterTrack), [DLA](https://github.com/ucbdrive/dla), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/FoundationVision/ByteTrack). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
