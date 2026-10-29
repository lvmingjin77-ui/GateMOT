#!/usr/bin/env bash
# Evaluate an exact annotation-defined subset. Never infer the evaluation frames from predictions.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
: "${DATASET:?Set DATASET to bee24, mot17, mot20 or sportsmot}"
: "${TRACKERS_FOLDER:?Set TRACKERS_FOLDER to the tracking txt directory}"
SPLIT=${SPLIT:-test}
IMG_SPLIT=$SPLIT
case "$SPLIT" in halftrain|halfval) IMG_SPLIT=train ;; esac
TRACK_EVAL=${TRACK_EVAL:-third_party/TrackEval}
EVAL_DIR=${EVAL_DIR:-${TRACKERS_FOLDER}_eval_${SPLIT}}
[[ -f "$TRACK_EVAL/scripts/run_mot_challenge.py" ]] || { echo "Install TrackEval at $TRACK_EVAL. See README." >&2; exit 1; }
python src/tools/prepare_trackeval.py --dataset "$DATASET" \
  --images-root "$DATA_ROOT/$IMG_SPLIT" --annotations "$DATA_ROOT/annotations/$SPLIT.json" \
  --tracks "$TRACKERS_FOLDER" --output "$EVAL_DIR"
PREPROC=False
case "$DATASET" in mot17|mot20) PREPROC=True ;; esac
python src/tools/run_trackeval.py --trackeval "$TRACK_EVAL" \
  --bundle "$EVAL_DIR" --preproc "$PREPROC"
