#!/bin/bash
# Setup and run script for Parameter Golf - Hybrid Ternary QAT + Depth Recurrence
# Usage: Copy to RunPod pod, then: bash setup_and_run.sh [smoke|full]
#   smoke = 30s quick test (default)
#   full  = 10 min training, single val at end

set -e
MODE="${1:-smoke}"
BRANCH="${BRANCH:-experimental/diagonal-masking}"

cd /workspace

# --- SETUP (skip if already done) ---
if [ ! -d "parameter-golf" ]; then
    echo "=== Cloning repo ==="
    git clone https://github.com/HenriqueTecas/parameter-golf.git
fi

cd parameter-golf

echo "=== Fetching latest code from $BRANCH ==="
git fetch origin "$BRANCH"
git checkout FETCH_HEAD -- records/track_10min_16mb/2026-03-19_HybridDeltaNet_TernaryQAT_32x512/train_gpt.py
cp records/track_10min_16mb/2026-03-19_HybridDeltaNet_TernaryQAT_32x512/train_gpt.py t.py

echo "=== Installing dependencies ==="
pip install -q sentencepiece huggingface_hub
pip install -q --upgrade torch

echo "=== Downloading FineWeb dataset ==="
python3 data/cached_challenge_fineweb.py --variant sp1024

if [ "$MODE" = "full" ]; then
    echo "=== 10-min run (val only at end) ==="
    RUN_ID=full10m \
      GRAD_ACCUM_STEPS=1 \
      TRAIN_BATCH_TOKENS=32768 \
      MAX_WALLCLOCK_SECONDS=600 \
      VAL_LOSS_EVERY=0 \
      VAL_STRIDE=1024 \
      DIAG_MASK_LAST_N=3 \
      torchrun --standalone --nproc_per_node=1 t.py
else
    echo "=== 30s smoke test ==="
    RUN_ID=smoke \
      GRAD_ACCUM_STEPS=1 \
      TRAIN_BATCH_TOKENS=32768 \
      MAX_WALLCLOCK_SECONDS=30 \
      VAL_LOSS_EVERY=0 \
      VAL_STRIDE=1024 \
      DIAG_MASK_LAST_N=3 \
      torchrun --standalone --nproc_per_node=1 t.py
fi
