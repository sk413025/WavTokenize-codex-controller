#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/sbplab/ruizi/WavTokenize-feature-analysis"
PY_BIN="/home/sbplab/miniconda3/envs/test/bin/python"

# 監控目前正在跑的 enc+mrd-fm+hubert 實驗（cuda:0）
WAIT_PATTERN="families/official/hubert_then_distalign/train_enc_hubert_fm.py --mode epoch --epochs 300 --device cuda:0 --batch_size 8 --grad_accum 2"

WAIT_LOG="${WORKDIR}/families/official/hubert_then_distalign/nohup_wait_start_distalign.log"
TRAIN_LOG="${WORKDIR}/families/official/hubert_then_distalign/nohup_train_distalign.log"
LOCK_FILE="${WORKDIR}/families/official/hubert_then_distalign/.wait_then_start_distalign.lock"

if [[ -f "${LOCK_FILE}" ]]; then
  echo "[$(date '+%F %T')] lock exists: ${LOCK_FILE}" | tee -a "${WAIT_LOG}"
  echo "[$(date '+%F %T')] another monitor may already be running. exit." | tee -a "${WAIT_LOG}"
  exit 1
fi

touch "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

echo "[$(date '+%F %T')] monitor started (env=test)." | tee -a "${WAIT_LOG}"
echo "[$(date '+%F %T')] waiting pattern: ${WAIT_PATTERN}" | tee -a "${WAIT_LOG}"

while pgrep -f "${WAIT_PATTERN}" >/dev/null; do
  count="$(pgrep -f "${WAIT_PATTERN}" | wc -l | tr -d ' ')"
  echo "[$(date '+%F %T')] enc+mrd-fm+hubert still running. matched=${count}" | tee -a "${WAIT_LOG}"
  sleep 120
done

echo "[$(date '+%F %T')] source run finished, starting distalign..." | tee -a "${WAIT_LOG}"

cd "${WORKDIR}"
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-self-supervised:${PYTHONPATH:-}"

nohup "${PY_BIN}" families/official/hubert_then_distalign/train_enc_hubert_fm_distalign.py \
  --mode epoch \
  --epochs 300 \
  --device cuda:0 \
  --batch_size 8 \
  --grad_accum 2 \
  --align_type coral \
  --lambda_dist 0.2 \
  >> "${TRAIN_LOG}" 2>&1 &

new_pid="$!"
echo "[$(date '+%F %T')] started distalign pid=${new_pid}" | tee -a "${WAIT_LOG}"
echo "[$(date '+%F %T')] train log: ${TRAIN_LOG}" | tee -a "${WAIT_LOG}"

