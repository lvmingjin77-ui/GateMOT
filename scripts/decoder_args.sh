#!/usr/bin/env bash
# Sourced by train/test/demo. Public scripts default to GateMOT.
# USE_QATTN=0 selects the original Conv baseline. An explicit --use_qattn wins.
USE_QATTN=${USE_QATTN:-1}
for arg in "$@"; do
  if [[ "$arg" == --use_qattn ]]; then
    USE_QATTN=1
  fi
done
DECODER_ARGS=(--head_kernel 3)
case "$USE_QATTN" in
  1) DECODER=qattn; DECODER_ARGS+=(--use_qattn) ;;
  0) DECODER=conv ;;
  *) echo "USE_QATTN must be 1 (GateMOT) or 0 (Conv)" >&2; exit 2 ;;
esac
