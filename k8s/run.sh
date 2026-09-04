#!/bin/bash

set -euo pipefail

: "${RUN_NAME:?Set RUN_NAME env var, e.g. hpo_0}"
: "${WORKERS:?Set WORKERS env var}"
: "${MODEL_ID:?Set MODEL_ID env var}"

REPO_DIR="s2-self"

CHIPS_PVC="/chips_pvc"
MODEL_PVC="/model_pvc"
LOG_PVC="/log_pvc"
CACHE_DIR="/cache"

DATA_ZIP_NAME="chips_256_sorted.zip"
DATA_DIR_NAME="chips_256_sorted"

ZIP_SRC="${CHIPS_PVC}/${DATA_ZIP_NAME}"
DATA_DIR="${CACHE_DIR}/${DATA_DIR_NAME}"
NET_DIR="${CACHE_DIR}/model"
LOG_DIR="${CACHE_DIR}/logs"
PARAMS_FILE="./hparams/${RUN_NAME}.json"

MODEL_OUT_DIR="${MODEL_PVC}/${RUN_NAME}"
LOG_OUT_DIR="${LOG_PVC}/${RUN_NAME}"

DONE_MARKER="${LOG_OUT_DIR}/model_${MODEL_ID}.done"
STDOUT_LOG="${LOG_DIR}/model_${MODEL_ID}_stdout.log"

mkdir -p "$MODEL_OUT_DIR" "$LOG_OUT_DIR"

echo "Copying files..."
cp "$ZIP_SRC" "$CACHE_DIR/"

echo "Unzipping..."
unzip -q "${CACHE_DIR}/${DATA_ZIP_NAME}" -d "$CACHE_DIR"

mkdir -p "$NET_DIR" "$LOG_DIR"
cd "$REPO_DIR"

python3 -u train.py \
  --data-dir "$DATA_DIR" \
  --net-dir "$NET_DIR" \
  --log-dir "$LOG_DIR" \
  --workers "$WORKERS" \
  --params "$PARAMS_FILE" \
  --id "$MODEL_ID" 2>&1 | tee "$STDOUT_LOG"

cp -v "${NET_DIR}"/*.pth.tar "$MODEL_OUT_DIR/"
cp -v "${LOG_DIR}"/*.tsv "$LOG_OUT_DIR/"
cp -v "$STDOUT_LOG" "$LOG_OUT_DIR/"
touch "$DONE_MARKER"