<div align="center">

# 🚪 GateMOT

### Q-Gated Attention for Dense Object Tracking

Mingjin Lv<sup>1</sup>, Zelin Liu<sup>1</sup>, Feifei Shao<sup>2</sup>, Yi-Ping Phoebe Chen<sup>3</sup>, Junqing Yu<sup>1</sup>, Wei Yang<sup>1</sup>, Zikai Song<sup>1,✉</sup>

<sup>1</sup>Huazhong University of Science and Technology &nbsp; <sup>2</sup>Zhejiang University &nbsp; <sup>3</sup>La Trobe University

<a href="https://arxiv.org/abs/2604.26353"><img src="https://img.shields.io/badge/arXiv-2604.26353-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2f80ed?style=flat-square" alt="License"></a>
<img src="https://img.shields.io/badge/PyTorch-2.4-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">

<a href="#-tldr">TL;DR</a> ·
<a href="#-results">Results</a> ·
<a href="#-quick-start">Quick start</a> ·
<a href="#-faq">FAQ</a> ·
<a href="#-citation">Citation</a>

</div>

---

## 💡 TL;DR

> 🐝 *Hundreds of tiny, look-alike bees in one frame. Which cues should each head look at?*

In vanilla attention, Query asks **"who looks like me?"** and pays a quadratic price to compare every pixel with every other pixel. In GateMOT, Query asks a cheaper question: **"what should I let through?"**

**Q-Attention** turns Query into a **gate** on Key features. Each task head gets its own gate, so detection, motion, and ReID each pick their own cues from one shared map. An ungated **Value** path keeps the details the gate might drop. No pairwise affinity matrix, linear spatial cost, and the result fits in every dense head.

<p align="center">
  <img src="assets/attention-comparison.png" width="900" alt="Q-Attention uses Gating-Q to select Key features and fuse them with Value, while vanilla attention uses global pairwise matrix multiplications.">
</p>

| | |
|---|---|
| 🎯 **A gate for each task** | Detection, motion, and ReID heads learn their own input-dependent feature selection. |
| 🧵 **A path for detail** | An ungated Value branch complements the selected Key features. |
| 🗺️ **Built for dense maps** | No quadratic spatial affinity matrix, so every high-resolution head can afford it. |

<details>
<summary><b>🔬 A closer look at Q-Attention</b></summary>

<p align="center">
  <img src="assets/q-attention.png" width="850" alt="Q-Attention module with a sigmoid Query gate, element-wise Key selection, local pooling, and concatenation with ungated Value before fusion.">
</p>

```text
V = Wv(X)      K = Wk(X)      G = sigmoid(Wq(X))
Y = BN( Conv1x1( concat( V, MaxPool3x3(G ⊙ K) ) ) )
```

The gate selects, local max pooling keeps the strongest nearby response, and Value joins back in before fusion. See the [code](src/lib/model/networks/base_model.py) and [implementation notes](docs/IMPLEMENTATION.md#q-attention).

</details>

## 📰 News

- **2026.10** 🎉 Code and checkpoints are released.
- **2026.04** 📄 Paper is available on [arXiv](https://arxiv.org/abs/2604.26353).

## 📊 Results

Test-set results reported in the paper. MOT17 and MOT20 use private detections.

| Dataset | Backbone | HOTA ↑ | DetA ↑ | AssA ↑ | MOTA ↑ | IDF1 ↑ | Evaluation output |
|---|---|---:|---:|---:|---:|---:|:---:|
| 🐝 BEE24 | DLA-34 | **48.4** | – | 44.7 | 67.8 | 64.5 | |
| ⛹️ SportsMOT | DLA-60 | **76.3** | 88.5 | 65.8 | 96.5 | 79.0 | [scores](SportsMOTEvaluationOutput/scoring_result/scores.txt) · [tracks](SportsMOTEvaluationOutput/SportsMOTResults) |
| 🚶 MOT17 | DLA-34 | **63.3** | 63.4 | 63.5 | 78.0 | 77.9 | |
| 👥 MOT20 | DLA-34 | **62.8** | 61.9 | 64.1 | 77.6 | 77.3 | |

On BEE24-test, GateMOT has the best HOTA, IDF1, and fewest ID switches (1,058) among the methods compared in the paper.

### ⚖️ Swap the head, keep everything else

Same backbone, losses, and association; only the prediction-head block changes (one-layer heads, DLA-34).

| Head | MOT17-val HOTA ↑ | BEE24-test HOTA ↑ | FPS ↑ | Params (M) |
|---|---:|---:|---:|---:|
| Conv | 60.2 | 44.3 | 14.8 | 21.85 |
| Self-attention | 63.0 | 46.3 | 4.6 | 22.30 |
| **Q-Attention** | **64.4** | **48.4** | 13.3 | 22.05 |

Q-Attention costs about 0.2M parameters more than Conv and stays close to it in speed, while self-attention slows down sharply. The Conv baseline is included in this release: set `USE_QATTN=0` when training and tracking.

### 📦 Checkpoints

Checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1Pmd2Y_tkax5D5gQQpRpCh0YTwBl27qiD). See the [checkpoint guide](docs/MODELS.md) for filenames and inference commands.

| Dataset | Download |
|---|---|
| BEE24 | [Google Drive](https://drive.google.com/file/d/1JoFAW258G3SiDVU27-f2W3zKdSonm916/view?usp=drive_link) |
| MOT17 | [Google Drive](https://drive.google.com/file/d/1BkWeOFmDDr5b6sP5bY_k2elt8OSqb1LH/view?usp=drive_link) |
| SportsMOT | [Google Drive](https://drive.google.com/file/d/1I6cLWAVBhptTKjjuuGDeR_OvFdrCZt8E/view?usp=drive_link) |
| MOT20 | [Google Drive](https://drive.google.com/file/d/1HufLPtgFWm9_QShCzFtSZ0LukkQReSxY/view?usp=drive_link) |

The checkpoints passed strict loading and a forward check. **MOT20 is a short-training checkpoint for pipeline testing.** Full benchmark reproduction of these downloads is still being verified.

## 🚀 Quick start

Five steps from a clean machine to HOTA. The full guide, including half-split validation, SportsMOT val, and resuming, lives in [Data and training](docs/DATA_AND_TRAINING.md).

<a id="installation"></a>

**1️⃣ Install.** Linux, an NVIDIA GPU, and a C/C++ compiler. Reference environment: Python 3.8, PyTorch 2.4.1, torchvision 0.19.1, CUDA 12.1.

```bash
git clone https://github.com/lvmingjin77-ui/GateMOT.git && cd GateMOT
conda create -n gatemot python=3.8 -y && conda activate gatemot
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24.4 Cython==3.1.3 setuptools wheel
pip install -r requirements.txt

git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval
git -C third_party/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

DCN uses torchvision, so nothing extra to compile. For training, put the ImageNet weights from the [DLA Model Zoo](http://dl.yf.io/dla/models/imagenet/) into `pretrain/` with their original filenames.

**2️⃣ Prepare data.** Download a dataset in MOTChallenge layout, then build the annotation files:

| Dataset | `DATASET` | `DATA_ROOT` points to |
|---|---|---|
| [BEE24](https://holmescao.github.io/datasets/BEE24) | `bee24` | the BEE24 folder |
| [MOT17](https://motchallenge.net/data/MOT17/) | `mot17` | the MOT17 folder |
| [MOT20](https://motchallenge.net/data/MOT20/) | `mot20` | the MOT20 folder |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | `sportsmot` | the archive's `dataset/` folder |

```bash
export DATASET=mot17 DATA_ROOT=/path/to/MOT17
scripts/prepare_data.sh "$DATASET"
```

This writes `annotations/train.json` and `annotations/test.json` (plus `val.json` for SportsMOT). Images stay in place. MOT17 uses only the `*-FRCNN` copy of each sequence. See the [directory layout](docs/DATA_AND_TRAINING.md#directory-layout).

<a id="training"></a>

**3️⃣ Train.** The script picks DLA-34 for BEE24/MOT17/MOT20 and DLA-60 for SportsMOT.

```bash
ANN_NAME=train NUM_GPUS=2 scripts/train.sh "$DATASET"
```

Checkpoints go to `exp/tracking.ctdet/${DATASET}_train_qattn_s317/`. `GLOBAL_BATCH` sets the total batch size and must be divisible by `NUM_GPUS`. Schedules are in [training settings](docs/DATA_AND_TRAINING.md#training-settings).

**4️⃣ Track.** One MOT-format `.txt` per sequence is written to `TRACKERS_FOLDER`.

```bash
export LOAD_MODEL=exp/tracking.ctdet/${DATASET}_train_qattn_s317/model_last.pth
export TRACKERS_FOLDER=results/${DATASET}_test
SPLIT=test scripts/test.sh "$DATASET" --output_dir "$TRACKERS_FOLDER"
```

For SportsMOT, the paper uses `model_70.pth` from the 80-epoch schedule. For downloaded weights, use the [matching commands](docs/MODELS.md#run-the-checkpoints).

**5️⃣ Evaluate.** Needs GT for the split you tracked:

```bash
SPLIT=test scripts/eval_hota.sh
```

Metrics land in `${TRACKERS_FOLDER}_eval_test/trackers/`. Hidden test splits need the benchmark's evaluation server. See [evaluation details](docs/DATA_AND_TRAINING.md#evaluation).

<details>
<summary><b>🎬 Bonus: render your own tracking video</b></summary>

```bash
DEMO=/path/to/SEQ/img1 LOAD_MODEL=/path/to/model.pth ARCH=dla34 \
  scripts/demo.sh --exp_id demo_mot17
```

The video is saved to `results/demo_mot17_img1.avi` and frames to `results/demo*.jpg`. Match `ARCH` and other options to the [checkpoint guide](docs/MODELS.md#run-the-checkpoints).

</details>

<a id="validation"></a>

## 🙋 FAQ

<details>
<summary><b>How do I run the Conv baseline?</b></summary>

Add `USE_QATTN=0` to both training and tracking, for example `USE_QATTN=0 ANN_NAME=train scripts/train.sh mot17`. Conv runs are saved under `${DATASET}_train_conv_s317/`. Use the same setting for training, resume, and inference.

</details>

<details>
<summary><b>Where is the validation set?</b></summary>

SportsMOT has an official val split. BEE24, MOT17, and MOT20 do not, so development uses a temporal half-split: the first `ceil(N/2)` frames of each training sequence form `halftrain`, the rest form `halfval`. After settings are fixed, the main model is retrained on full train. Commands are in [optional validation](docs/DATA_AND_TRAINING.md#optional-validation).

</details>

<details>
<summary><b>Why only FRCNN for MOT17?</b></summary>

MOT17 ships each video three times (DPM, FRCNN, SDP) with identical images and GT. GateMOT predicts its own detections, so one copy is enough and avoids duplicated training data and identities.

</details>

<details>
<summary><b>Training was interrupted. How do I resume?</b></summary>

```bash
ANN_NAME=train NUM_GPUS=2 scripts/train.sh "$DATASET" \
  --resume --load_model "exp/tracking.ctdet/${DATASET}_train_qattn_s317/model_last.pth"
```

</details>

## 📚 Documentation

| Topic | Guide |
|---|---|
| Checkpoint downloads and matching inference commands | [Checkpoints](docs/MODELS.md) |
| Data preparation, training, validation, and evaluation | [Data and training](docs/DATA_AND_TRAINING.md) |
| Q-Attention, losses, augmentation, and association | [Implementation](docs/IMPLEMENTATION.md) |

## 📜 License

The code is released under the [Apache-2.0 License](LICENSE). Third-party notices are in [NOTICE](NOTICE). Datasets keep their own licenses; please follow the terms of BEE24, MOTChallenge, and SportsMOT.

## 🙏 Acknowledgements

Built on [CenterTrack](https://github.com/xingyizhou/CenterTrack), [DLA](https://github.com/ucbdrive/dla), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/FoundationVision/ByteTrack). Evaluation uses [TrackEval](https://github.com/JonathonLuiten/TrackEval). We thank their authors for sharing their work.

## 📝 Citation

If GateMOT helps your research, a ⭐ and a citation would make our day:

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
