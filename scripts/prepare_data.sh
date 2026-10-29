#!/usr/bin/env bash
# Convert MOT-layout data, including chronological training/validation halves.
# DATA_ROOT=/path/to/mot17 scripts/prepare_data.sh mot17 --splits halftrain halfval train
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DATASET=${1:?Usage: scripts/prepare_data.sh bee24|mot17|mot20|sportsmot|dancetrack [--splits ...]}
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
shift
case "$DATASET" in
  bee24) CONVERTER=bee_to_coco.py ;;
  mot17|mot20|sportsmot|dancetrack) CONVERTER=${DATASET}_to_coco.py ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 1 ;;
esac
python "src/tools/$CONVERTER" --data-root "$DATA_ROOT" "$@"
