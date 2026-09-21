#!/bin/bash

set -euo pipefail

: "${RUN_NAME:?Set RUN_NAME env var, e.g. hpo_0}"
: "${WORKERS:?Set WORKERS env var}"
: "${MODEL_ID:?Set MODEL_ID env var}"

MODEL_ID_PADDED=$(printf "%03d" "$MODEL_ID")

REPO_DIR="s2-self"

CHIPS_PVC="/chips_pvc"
MODEL_PVC="/model_pvc"
LOG_PVC="/log_pvc"
CACHE_DIR="/cache"

DATA_ZIP_NAME="chips_256_sorted.zip"
DATA_DIR_NAME="chips_256_sorted"
MASK_ZIP_NAME="extra_masks.zip"
MASK_DIR_NAME="extra_masks"

ZIP_SRC="${CHIPS_PVC}/${DATA_ZIP_NAME}"
ZIP_MSK="${CHIPS_PVC}/${MASK_ZIP_NAME}"
DATA_DIR="${CACHE_DIR}/${DATA_DIR_NAME}"
MASK_DIR="${CACHE_DIR}/${MASK_DIR_NAME}"
NET_DIR="${CACHE_DIR}/model"
LOG_DIR="${CACHE_DIR}/logs"
PARAMS_FILE="./hparams/${RUN_NAME}.json"

MODEL_OUT_DIR="${MODEL_PVC}/${RUN_NAME}"
LOG_OUT_DIR="${LOG_PVC}/${RUN_NAME}"

DONE_MARKER="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}.done"
FAIL_MARKER="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}.failed"
STDOUT_LOG_LOCAL="${LOG_DIR}/model_${MODEL_ID_PADDED}_stdout.log"
STDOUT_LOG_PVC="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}_stdout.log"

mkdir -p "$MODEL_OUT_DIR" "$LOG_OUT_DIR"

echo "Copying files..."
cp "$ZIP_SRC" "$CACHE_DIR/"
cp "$ZIP_MSK" "$CACHE_DIR/"

echo "Unzipping..."
unzip -q "${CACHE_DIR}/${DATA_ZIP_NAME}" -d "$CACHE_DIR"
unzip -q "${CACHE_DIR}/${MASK_ZIP_NAME}" -d "$CACHE_DIR"

mkdir -p "$NET_DIR" "$LOG_DIR"
cd "$REPO_DIR"

# MIRROR STDOUT TO THE PVC IN THE BACKGROUND -- tee writes to local disk only,
# so the training loop never blocks on network I/O; this process copies new
# lines over to the PVC as they appear, keeping crash logs visible without
# the per-print latency hit.
touch "$STDOUT_LOG_LOCAL"
tail -F "$STDOUT_LOG_LOCAL" >> "$STDOUT_LOG_PVC" &
TAIL_PID=$!

on_exit() {
  status=$?
  kill "$TAIL_PID" 2>/dev/null || true
  cp "$STDOUT_LOG_LOCAL" "$STDOUT_LOG_PVC" 2>/dev/null || true
  if [ "$status" -eq 0 ]; then
    cp -v "${NET_DIR}"/*.pth.tar "$MODEL_OUT_DIR/" 2>/dev/null || true
    cp -v "${LOG_DIR}"/*.tsv "$LOG_OUT_DIR/" 2>/dev/null || true
    touch "$DONE_MARKER"
  else
    touch "$FAIL_MARKER"
  fi
  exit "$status"
}
trap on_exit EXIT

python3 -u train.py \
  --data-dir "$DATA_DIR" \
  --net-dir "$NET_DIR" \
  --log-dir "$LOG_DIR" \
  --workers "$WORKERS" \
  --params "$PARAMS_FILE" \
  --masks "$MASK_DIR" \
  --id "$MODEL_ID" 2>&1 | tee "$STDOUT_LOG_LOCAL"