#!/bin/bash
# RunPod launch script — handles PyTorch 2.4.1 compat + tmux for SSH drops
# Usage: bash run_runpod.sh [smoke|full|full8]
#   smoke = 60s test, 1 GPU
#   full  = 10-min run, 1 GPU (for single H100/5090)
#   full8 = 10-min run, 8 GPUs (for 8xH100 pod)
set -e

MODE="${1:-smoke}"
NGPU="${2:-1}"

# Auto-detect GPU count for full8
if [ "$MODE" = "full8" ]; then
    NGPU=8
    MODE="full"
fi

echo "=== RunPod Parameter Golf Runner ==="
echo "Mode: $MODE | GPUs: $NGPU"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

cd /workspace

# Setup if not already done
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/HenriqueTecas/parameter-golf.git
fi
cd parameter-golf
git fetch origin experimental/base3-fixed-recurrence
git checkout experimental/base3-fixed-recurrence
git pull origin experimental/base3-fixed-recurrence

# Dataset — only download if missing
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "=== Downloading dataset ==="
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

SEED="${SEED:-1337}"
echo "=== Starting training (SEED=$SEED) ==="

if [ "$MODE" = "full" ]; then
    RUN_ID="runpod_${NGPU}gpu_seed${SEED}_$(date +%H%M)" \
      SEED=$SEED \
      MAX_WALLCLOCK_SECONDS=600 \
      VAL_LOSS_EVERY=200 \
      TRAIN_LOG_EVERY=50 \
      torchrun --standalone --nproc_per_node=$NGPU train_gpt.py 2>&1 | tee "train_full_seed${SEED}.log"
else
    RUN_ID="runpod_smoke_$(date +%H%M)" \
      MAX_WALLCLOCK_SECONDS=60 \
      ITERATIONS=50 \
      VAL_LOSS_EVERY=0 \
      TRAIN_LOG_EVERY=10 \
      torchrun --standalone --nproc_per_node=$NGPU train_gpt.py 2>&1 | tee train_smoke.log
fi

echo ""
echo "=== DONE. Check log above for val_bpb ==="
