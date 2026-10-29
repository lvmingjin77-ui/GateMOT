#!/usr/bin/env bash
# Convert the full train/test splits (and official SportsMOT val).
# DATA_ROOT=/path/to/mot17 scripts/prepare_data.sh mot17 --splits train test
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DATASET=${1:?Usage: scripts/prepare_data.sh bee24|mot17|mot20|sportsmot [--splits ...]}
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
shift
case "$DATASET" in
  bee24) CONVERTER=bee_to_coco.py ;;
  mot17|mot20|sportsmot) CONVERTER=${DATASET}_to_coco.py ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 1 ;;
esac
python "src/tools/$CONVERTER" --data-root "$DATA_ROOT" "$@"
