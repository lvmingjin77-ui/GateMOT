<h1 align="center">GateMOT: Q-Gated Attention<br>for Dense Object Tracking</h1>

<p align="center">
  Mingjin Lv, Zelin Liu, Feifei Shao,<br>
  Yi-Ping Phoebe Chen, Junqing Yu, Wei Yang, Zikai Song
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.26353"><img src="https://img.shields.io/badge/arXiv-2604.26353-b31b1b?style=flat-square" alt="arXiv 2604.26353"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#citation">Citation</a>
</p>

<a id="overview"></a>

## 🔍 Overview

**GateMOT turns Query into a gate, bringing task-specific feature selection to dense tracking heads.**

Detection, motion estimation, and re-identification need different cues from the same dense feature map. Selecting those cues matters most in the decoder, where the computation budget is tight. Q-Attention makes this selection affordable: Query gates Key features directly, removing global pairwise interactions, while an ungated Value path preserves details for each task.

<p align="center">
  <img src="assets/attention-comparison.png" width="900" alt="Q-Attention uses Gating-Q to select Key features and fuse them with Value, while vanilla attention uses global pairwise matrix multiplications.">
</p>

- **A gate for each task.** Detection, motion, and ReID heads learn their own input-dependent feature selection.
- **A path for detail.** An ungated Value branch complements the selected Key features.
- **Built for dense maps.** The decoder avoids the quadratic spatial affinity matrix of vanilla attention.

<details>
<summary><b>A closer look at Q-Attention</b></summary>

<p align="center">
  <img src="assets/q-attention.png" width="850" alt="Q-Attention module with a sigmoid Query gate, element-wise Key selection, local pooling, and concatenation with ungated Value before fusion.">
</p>

The selected Key features are pooled locally, concatenated with Value, and fused. See the [implementation](src/lib/model/networks/base_model.py) and [module details](docs/IMPLEMENTATION.md#q-attention).

</details>

<a id="results"></a>

## 📊 Results

Test-set results reported in the paper. MOT17 and MOT20 use private detections.

| Dataset | Backbone | HOTA ↑ | MOTA ↑ | IDF1 ↑ | Model |
|---|---|---:|---:|---:|---|
| BEE24 | DLA-34 | **48.4** | 67.8 | 64.5 | [Coming soon](docs/MODELS.md#coming-soon) |
| SportsMOT | DLA-60 | **76.3** | 96.5 | 79.0 | [Coming soon](docs/MODELS.md#coming-soon) |
| MOT17 | DLA-34 | **63.3** | 78.0 | 77.9 | [Coming soon](docs/MODELS.md#coming-soon) |
| MOT20 | DLA-34 | **62.8** | 77.6 | 77.3 | [Coming soon](docs/MODELS.md#coming-soon) |

GateMOT checkpoints and their matching configurations are coming soon. This release provides the complete main-method pipeline for all four datasets.

<a id="getting-started"></a>

## 🚀 Getting started

<a id="installation"></a>

### 1. Install

Use Linux with an NVIDIA GPU and a C/C++ compiler. The reference environment uses Python 3.8, PyTorch 2.4.1, torchvision 0.19.1, and CUDA 12.1.

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

DCN uses torchvision's implementation. Download the ImageNet initialization weights into `pretrain/`, keeping their filenames:

- **BEE24, MOT17, MOT20** — [dla34-ba72cf86.pth](http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth)
- **SportsMOT** — [dla60-24839fc4.pth](http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth)

### 2. Choose a dataset

Download a dataset using the links below. **The main experiments train on the complete training split and evaluate on the test split.**

| Dataset | `DATASET` | Training | Evaluation |
|---|---|---|---|
| [BEE24](https://holmescao.github.io/datasets/BEE24) | `bee24` | Full train | Test |
| [MOT17](https://motchallenge.net/data/MOT17/) | `mot17` | Full train | Test |
| [MOT20](https://motchallenge.net/data/MOT20/) | `mot20` | Full train | Test |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | `sportsmot` | Official train | Official test |

Set `DATA_ROOT` to the directory containing `train/` and `test/`. For SportsMOT, use the downloaded archive's `dataset/` subdirectory, which also contains the official `val/`. See [directory layout](docs/DATA_AND_TRAINING.md#directory-layout).

<a id="training"></a>

### 3. Train and track

Run from the repository root. This MOT17 example uses the complete training split. For another dataset, change `DATASET` and `DATA_ROOT` using the table above:

```bash
export DATASET=mot17 DATA_ROOT=/path/to/MOT17

# Prepare annotations and train on full train.
scripts/prepare_data.sh "$DATASET"
ANN_NAME=train NUM_GPUS=2 scripts/train.sh "$DATASET"

# Run the trained model on test images.
export LOAD_MODEL=exp/tracking.ctdet/${DATASET}_train_qattn_s317/model_last.pth
export TRACKERS_FOLDER=results/${DATASET}_test
SPLIT=test scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER"
```

Training saves checkpoints and `run_config.json` in the experiment directory. `NUM_GPUS` sets the GPU count; `GLOBAL_BATCH` sets the total batch size and must be divisible by it. The example uses the last checkpoint. The SportsMOT result in Table 2 uses `model_70.pth` from its 80-epoch schedule.

### 4. Evaluate

Tracking writes one MOT-format `.txt` file per sequence. When the evaluated split's ground truth is available locally, compute HOTA, MOTA, and IDF1 with:

```bash
SPLIT=test scripts/eval_hota.sh
```

Hidden test splits support inference only without their GT. For development, the converter also supports **optional** `halftrain`/`halfval` splits for BEE24, MOT17, and MOT20. SportsMOT has an official validation split. These validation protocols are separate from the full-training main experiments; see [validation instructions](docs/DATA_AND_TRAINING.md#optional-validation).

**More:** [training settings and resume](docs/DATA_AND_TRAINING.md#training-settings) · [evaluation details](docs/DATA_AND_TRAINING.md#evaluation).

## Documentation

| Topic | Guide |
|---|---|
| Data preparation, training, and evaluation | [Data and training](docs/DATA_AND_TRAINING.md) |
| Q-Attention, losses, and association | [Implementation](docs/IMPLEMENTATION.md) |
| Train/resume/inference checks | [Software validation](docs/VALIDATION.md) |

## Acknowledgements

Built on [CenterTrack](https://github.com/xingyizhou/CenterTrack), [DLA](https://github.com/ucbdrive/dla), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/FoundationVision/ByteTrack). We thank their authors for sharing their work. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

If you find GateMOT useful in your research, please cite:

```bibtex
@misc{lv2026gatemot,
  title         = {{GateMOT}: {Q}-Gated Attention for Dense Object Tracking},
  author        = {Lv, Mingjin and Liu, Zelin and Shao, Feifei and Chen, Yi-Ping Phoebe and Yu, Junqing and Yang, Wei and Song, Zikai},
  year          = {2026},
  eprint        = {2604.26353},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2604.26353}
}
```
