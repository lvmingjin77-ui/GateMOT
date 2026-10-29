# Ablation settings

The head transform can be changed independently of the encoder, prediction outputs, losses, and tracker. The commands below use the BEE24 training half and a common validation split.

## Whole-decoder replacements

| `HEAD_TYPE` | Transform before the final prediction layer |
|---|---|
| `qattn` | Q-gated Key pooling plus ungated Value fusion and BN |
| `conv` | 3×3 convolution, without BN |
| `conv_bn` | 3×3 convolution followed by BN |
| `glu` | BN of `feature1x1(X) * sigmoid(gate1x1(X))` |
| `gated_conv` | BN of `feature3x3(X) * sigmoid(gate3x3(X))` |

A ReLU and final prediction convolution follow each transform. `glu` uses two 1×1 projections and sigmoid gating. `gated_conv` uses 3×3 projections. Both include BN and use the same intermediate width as Q-Attention, with different parameter counts and FLOPs. Results for these configurable controls are not included here.

```bash
ANN_NAME=halftrain HEAD_TYPE=conv DATA_ROOT=/path/to/bee24 scripts/train.sh bee24
ANN_NAME=halftrain HEAD_TYPE=conv_bn DATA_ROOT=/path/to/bee24 scripts/train.sh bee24
ANN_NAME=halftrain HEAD_TYPE=glu DATA_ROOT=/path/to/bee24 scripts/train.sh bee24
ANN_NAME=halftrain HEAD_TYPE=gated_conv DATA_ROOT=/path/to/bee24 scripts/train.sh bee24
```

Evaluate each with the matching `HEAD_TYPE` and its own checkpoint. Use the same `SPLIT=halfval` and exact-subset TrackEval wrapper.

## Per-task replacement

`--head_overrides` replaces the transform in one task group while retaining its prediction outputs and losses. The examples use Conv+BN as the replacement.

```bash
# Detection group, including auxiliary width/height supervision.
EXP_ID=bee24_halftrain_detection_conv_bn ANN_NAME=halftrain DATA_ROOT=/path/to/bee24 \
scripts/train.sh bee24 --head_overrides hm:conv_bn,reg:conv_bn,ltrb_amodal:conv_bn,wh:conv_bn

# Motion group.
EXP_ID=bee24_halftrain_motion_conv_bn ANN_NAME=halftrain DATA_ROOT=/path/to/bee24 \
scripts/train.sh bee24 --head_overrides tracking:conv_bn

# Appearance group.
EXP_ID=bee24_halftrain_reid_conv_bn ANN_NAME=halftrain DATA_ROOT=/path/to/bee24 \
scripts/train.sh bee24 --head_overrides reid:conv_bn
```

Pass the same `--head_overrides` during tracking. Boxes are still decoded from that experiment's `hm`, `reg`, and `ltrb_amodal` predictions. ReID still supplies the appearance features, and the tracking head still supplies offsets.

Training settings are saved in `run_config.json`. Evaluation settings and checkpoint hashes are saved in `evaluation_config.json`.
