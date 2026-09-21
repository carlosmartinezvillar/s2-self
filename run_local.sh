#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 --run-name NAME --workers N --data-dir DIR --net-dir DIR --log-dir DIR --ids \"0 1 2\" [--masks DIR] [--gpu N]"
  exit 1
}

MASK_DIR=""
GPU=0

while [ $# -gt 0 ]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --workers)  WORKERS="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --net-dir)  NET_DIR="$2"; shift 2 ;;
    --log-dir)  LOG_DIR="$2"; shift 2 ;;
    --masks)    MASK_DIR="$2"; shift 2 ;;
    --ids)      IDS="$2"; shift 2 ;;
    --gpu)      GPU="$2"; shift 2 ;;
    *) usage ;;
  esac
done

: "${RUN_NAME:?Missing --run-name}"
: "${WORKERS:?Missing --workers}"
: "${DATA_DIR:?Missing --data-dir}"
: "${NET_DIR:?Missing --net-dir}"
: "${LOG_DIR:?Missing --log-dir}"
: "${IDS:?Missing --ids, e.g. --ids \"0 1 2 3\"}"

PARAMS_FILE="./hparams/${RUN_NAME}.json"

cd "$(dirname "$0")"

MASK_ARGS=()
if [ -n "$MASK_DIR" ]; then
  MASK_ARGS=(--masks "$MASK_DIR")
fi

for MODEL_ID in $IDS; do
  MODEL_ID_PADDED=$(printf "%03d" "$MODEL_ID")
  DONE_MARKER="${LOG_DIR}/model_${MODEL_ID_PADDED}.done"
  FAIL_MARKER="${LOG_DIR}/model_${MODEL_ID_PADDED}.failed"
  STDOUT_LOG="${LOG_DIR}/model_${MODEL_ID_PADDED}_stdout.log"

  echo "Training model $MODEL_ID ($RUN_NAME) on GPU $GPU..."

  if python3 -u train.py \
    --data-dir "$DATA_DIR" \
    --net-dir "$NET_DIR" \
    --log-dir "$LOG_DIR" \
    --workers "$WORKERS" \
    --params "$PARAMS_FILE" \
    --gpu "$GPU" \
    "${MASK_ARGS[@]}" \
    --id "$MODEL_ID" 2>&1 | tee "$STDOUT_LOG"; then
    touch "$DONE_MARKER"
  else
    touch "$FAIL_MARKER"
  fi
done
