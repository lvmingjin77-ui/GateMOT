#!/usr/bin/env bash
# Train GateMOT on the complete training split by default.
# Usage:
#   DATA_ROOT=/path/to/bee24 NUM_GPUS=4 scripts/train.sh bee24
#   ANN_NAME=halftrain DATA_ROOT=/path/to/bee24 scripts/train.sh bee24
#   USE_BFL=0 DATA_ROOT=/path/to/bee24 scripts/train.sh bee24
# Datasets: bee24 mot17 mot20 sportsmot
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
DATASET=${1:?Usage: scripts/train.sh bee24|mot17|mot20|sportsmot}
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
NUM_GPUS=${NUM_GPUS:-1}
SEED=${SEED:-317}
shift
source "$ROOT/scripts/decoder_args.sh"

ARCH=dla34
FLIP=0.5
AUG_ROT=0
ROTATE=0
REID_WEIGHT=1.0
case "$DATASET" in
  bee24)
    EPOCHS=60; BATCH=24; LR=2e-4; STEP=40; K=200; CLS=1
    AUG_ROT=1; ROTATE=15
    IMG="${DATA_ROOT}/train"
    ANN="${DATA_ROOT}/annotations/${ANN_NAME:-train}.json"

    ;;
  mot17)
    EPOCHS=70; BATCH=16; LR=1.25e-4; STEP=60; K=256; CLS=1
    IMG="${DATA_ROOT}/train"
    ANN="${DATA_ROOT}/annotations/${ANN_NAME:-train}.json"

    ;;
  mot20)
    # The supplementary table does not record the MOT20 batch size.
    # 16 matches the MOT17 setting.
    EPOCHS=70; BATCH=16; LR=1.15e-4; STEP=60; K=256; CLS=1
    IMG="${DATA_ROOT}/train"
    ANN="${DATA_ROOT}/annotations/${ANN_NAME:-train}.json"

    ;;
  sportsmot)
    ARCH=dla60
    EPOCHS=80; BATCH=20; LR=1.2e-4; STEP=45; K=100; CLS=1
    FLIP=0
    IMG="${DATA_ROOT}/train"
    ANN="${DATA_ROOT}/annotations/${ANN_NAME:-train}.json"

    ;;
  *)
    echo "Unknown dataset: $DATASET" >&2
    exit 1
    ;;
esac

EXP_ID=${EXP_ID:-${DATASET}_${ANN_NAME:-train}_${DECODER}_s${SEED}}
BATCH=${GLOBAL_BATCH:-$BATCH}
EPOCHS=${NUM_EPOCHS:-$EPOCHS}
python -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" src/main.py \
  "${DECODER_ARGS[@]}" \
  --exp_id "$EXP_ID" \
  --same_aug_pre \
  --hm_disturb 0.05 \
  --lost_disturb 0.4 \
  --fp_disturb 0.1 \
  --num_epochs "$EPOCHS" \
  --num_workers 4 \
  --batch_size "$BATCH" \
  --input_h 608 \
  --input_w 1088 \
  --num_classes "$CLS" \
  --K "$K" \
  --lr "$LR" \
  --pre_hm \
  --optim adam \
  --num_head_conv 1 \
  --seed "$SEED" \
  --wh \
  --lr_step "$STEP" \
  --arch "$ARCH" \
  --flip "$FLIP" \
  --aug_rot "$AUG_ROT" \
  --rotate "$ROTATE" \
  --reid_weight "$REID_WEIGHT" \
  --hm_weight 1 \
  --off_weight 1 \
  --wh_weight 0.1 \
  --tracking_weight 1 \
  --ltrb_amodal_weight 0.1 \
  --custom_dataset_img_path "$IMG" \
  --custom_dataset_ann_path "$ANN" \
  "$@"
