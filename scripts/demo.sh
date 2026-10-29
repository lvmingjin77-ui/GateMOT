#!/usr/bin/env bash
# Run GateMOT on an image sequence and save a visualization.
# Usage:
#   DEMO=/path/to/img1 LOAD_MODEL=/path/to/model.pth scripts/demo.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
: "${DEMO:?Set DEMO to an image directory}"
: "${LOAD_MODEL:?Set LOAD_MODEL to a .pth checkpoint}"
ARCH=${ARCH:-dla34}

python src/demo.py \
  --demo "$DEMO" \
  --load_model "$LOAD_MODEL" \
  --arch "$ARCH" \
  --num_classes 1 \
  --input_h 608 \
  --input_w 1088 \
  --video_h 608 \
  --video_w 1088 \
  --num_head_conv 1 \
  --pre_hm \
  --track_thresh 0.4 \
  --use_bfl \
  --wh \
  --save_video \
  "$@"
