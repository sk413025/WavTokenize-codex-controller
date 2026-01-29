#!/bin/bash
#
# exp_0128: Component Check Script
#
# 檢查所有組件是否正常運作
#

set -e

# 環境變數設定
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

echo "========================================================"
echo "Exp 0128: Component Check"
echo "========================================================"

# Test 1: CUDA
echo ""
echo "Test 1: CUDA & GPU Setup"
echo "------------------------"
python -c "
import torch
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ Device count: {torch.cuda.device_count()}')
"

# Test 2: Noise Type Extraction
echo ""
echo "Test 2: Noise Type Extraction"
echo "------------------------------"
python -c "
import sys
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0128.noise_balanced_sampling.sampler import extract_noise_type

test_paths = [
    'nor_boy10_box_LDV_132.wav',
    'nor_girl3_papercup_LDV_115.wav',
    'nor_boy4_plastic_LDV_281.wav',
]

for path in test_paths:
    noise_type = extract_noise_type(path)
    assert noise_type in ['box', 'papercup', 'plastic']
    print(f'✅ {path} -> {noise_type}')
"

# Test 3: Import all required modules
echo ""
echo "Test 3: Module Imports"
echo "----------------------"
python -c "
import sys
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

# Test imports
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
print('✅ config imports')

from exp_0112_intermediate.models import TeacherStudentIntermediate
print('✅ model imports')

from exp_0112_intermediate.train_v6 import IntermediateSupervisionLossV6
print('✅ loss imports')

from exp_0128.noise_balanced_sampling.sampler import NoiseBalancedSampler
print('✅ sampler imports')

from exp_0128.noise_balanced_sampling.data_balanced import create_noise_balanced_dataloaders
print('✅ dataloader imports')
"

# Test 4: Check data cache exists
echo ""
echo "Test 4: Data Cache"
echo "------------------"
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import TRAIN_CACHE, VAL_CACHE

train_exists = Path(TRAIN_CACHE).exists()
val_exists = Path(VAL_CACHE).exists()

print(f'✅ Train cache: {TRAIN_CACHE}' if train_exists else f'❌ Train cache not found')
print(f'✅ Val cache: {VAL_CACHE}' if val_exists else f'❌ Val cache not found')

assert train_exists and val_exists, 'Data cache not found!'
"

# Test 5: Training script exists
echo ""
echo "Test 5: Training Script"
echo "-----------------------"
if [ -f "exp_0128/noise_balanced_sampling/train_short_run.py" ]; then
    echo "✅ train_short_run.py exists"
else
    echo "❌ train_short_run.py not found"
    exit 1
fi

if [ -f "exp_0128/noise_balanced_sampling/run_exp2.sh" ]; then
    echo "✅ run_exp2.sh exists"
else
    echo "❌ run_exp2.sh not found"
    exit 1
fi

# Summary
echo ""
echo "========================================================"
echo "✅ All Component Checks PASSED!"
echo "========================================================"
echo ""
echo "Ready to run experiment:"
echo "  bash exp_0128/noise_balanced_sampling/run_exp2.sh"
echo ""
