#!/usr/bin/env bash
# Sourced by train/test/demo. Public scripts default to GateMOT.
# USE_BFL=0 selects the original Conv baseline. An explicit --use_bfl wins.
USE_BFL=${USE_BFL:-1}
for arg in "$@"; do
  if [[ "$arg" == --use_bfl ]]; then
    USE_BFL=1
  fi
done
DECODER_ARGS=(--head_kernel 3)
case "$USE_BFL" in
  1) DECODER=qattn; DECODER_ARGS+=(--use_bfl) ;;
  0) DECODER=conv ;;
  *) echo "USE_BFL must be 1 (GateMOT) or 0 (Conv)" >&2; exit 2 ;;
esac
