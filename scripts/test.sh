#!/usr/bin/env bash
# Track a split with the reported Q-Attention model.
# Usage:
#   DATA_ROOT=/path/to/bee24 LOAD_MODEL=/path/to/model.pth scripts/test.sh bee24
#   SPLIT=halfval DATA_ROOT=/path/to/bee24 LOAD_MODEL=... scripts/test.sh bee24
#
# SPLIT=test reads images from test/ and annotations/test.json.
# SPLIT=halftrain or halfval reads images from train/ and the matching json.
# SportsMOT also accepts SPLIT=val, which reads val/ and annotations/val.json.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
DATASET=${1:?Usage: scripts/test.sh bee24|mot17|mot20|dancetrack|sportsmot}
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
: "${LOAD_MODEL:?Set LOAD_MODEL to a .pth checkpoint}"

shift
ARCH=dla34
NEW_THRESH=0.4
MAX_AGE=30
SPLIT=${SPLIT:-test}
case "$SPLIT" in
  halftrain|halfval)
    IMG_SPLIT=train
    ;;
  *)
    IMG_SPLIT="$SPLIT"
    ;;
esac

case "$DATASET" in
  bee24)
    K=200; CLS=1; MAX_AGE=20
    ;;
  mot17|mot20)
    K=256; CLS=1
    ;;
  dancetrack)
    K=100; CLS=1; NEW_THRESH=0.5
    ;;
  sportsmot)
    ARCH=dla60
    K=100; CLS=1
    ;;
  *)
    echo "Unknown dataset: $DATASET" >&2
    exit 1
    ;;
esac

if [[ "${FUSE:-0}" == "1" ]]; then
  set -- --fuse "$@"
fi

python src/test.py \
  --load_model "$LOAD_MODEL" \
  --track_thresh 0.4 \
  --new_thresh "$NEW_THRESH" \
  --max_age "$MAX_AGE" \
  --num_classes "$CLS" \
  --input_h 608 \
  --input_w 1088 \
  --K "$K" \
  --num_head_conv 1 \
  --pre_hm \
  --hungarian \
  --head_type "${HEAD_TYPE:-qattn}" \
  --wh \
  --arch "$ARCH" \
  --custom_dataset_img_path "${DATA_ROOT}/${IMG_SPLIT}" \
  --custom_dataset_ann_path "${DATA_ROOT}/annotations/${SPLIT}.json" \
  "$@"
