#!/bin/bash
set -e

echo "========================================="
echo "  Parameter Golf — Prep VM Setup"
echo "========================================="

# Step 1: Python environment
echo ""
echo "[1/5] Setting up Python venv..."
sudo apt update -qq && sudo apt install -y -qq python3-venv git
python3 -m venv ~/venv
source ~/venv/bin/activate
echo "  Python: $(python3 --version)"
echo "  Pip: $(pip --version)"

# Step 2: Install deps
echo ""
echo "[2/5] Installing Python packages..."
pip install -q numpy tqdm torch huggingface-hub sentencepiece datasets

# Step 3: Clone and checkout
echo ""
echo "[3/5] Setting up repo..."
cd /home/golf
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/HenriqueTecas/parameter-golf.git
fi
cd parameter-golf
git fetch origin experimental/base3-fixed-recurrence
git checkout experimental/base3-fixed-recurrence
git pull origin experimental/base3-fixed-recurrence
echo "  Branch: $(git branch --show-current)"
echo "  Commit: $(git log --oneline -1)"

# Step 4: Download dataset
echo ""
echo "[4/5] Downloading FineWeb dataset (this takes a while)..."
python3 data/cached_challenge_fineweb.py --variant sp1024

# Step 5: Verify everything
echo ""
echo "[5/5] Verifying setup..."
echo ""

FAIL=0

# Check Python imports
python3 -c "import torch; import sentencepiece; import numpy; import datasets; print('  Imports: OK')" || { echo "  Imports: FAILED"; FAIL=1; }

# Check dataset files
TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "  Train shards: $TRAIN_COUNT"
echo "  Val shards: $VAL_COUNT"
if [ "$TRAIN_COUNT" -lt 1 ]; then echo "  ERROR: No train shards found!"; FAIL=1; fi
if [ "$VAL_COUNT" -lt 1 ]; then echo "  ERROR: No val shards found!"; FAIL=1; fi

# Check tokenizer
if [ -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo "  Tokenizer: OK"
else
    echo "  Tokenizer: MISSING"; FAIL=1
fi

# Check train script
if [ -f "train_gpt.py" ]; then
    python3 -c "import py_compile; py_compile.compile('train_gpt.py', doraise=True)" && echo "  train_gpt.py: OK (compiles)" || { echo "  train_gpt.py: SYNTAX ERROR"; FAIL=1; }
else
    echo "  train_gpt.py: MISSING"; FAIL=1
fi

# Dataset size
DATASET_SIZE=$(du -sh data/datasets/fineweb10B_sp1024/ 2>/dev/null | cut -f1)
echo "  Dataset size: $DATASET_SIZE"

echo ""
echo "========================================="
if [ "$FAIL" -eq 0 ]; then
    echo "  ALL CHECKS PASSED"
    echo ""
    echo "  Next steps:"
    echo "  1. Stop this VM (Azure portal -> Stop)"
    echo "  2. Go to Disks -> click OS disk"
    echo "  3. Create snapshot: pgolf-ready"
    echo "  4. Launch H100 VM from snapshot"
    echo "  5. SSH in, then:"
    echo ""
    echo "     source ~/venv/bin/activate"
    echo "     cd ~/parameter-golf"
    echo "     torchrun --standalone --nproc_per_node=8 train_gpt.py"
else
    echo "  SOME CHECKS FAILED — fix errors above"
fi
echo "========================================="
