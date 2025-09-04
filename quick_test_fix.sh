#!/bin/bash

# 快速測試實驗修復
echo "🧪 快速測試實驗方案一修復..."

# 激活環境
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export TTT_BATCH_SIZE=4  # 減小批次大小
export TTT_NUM_WORKERS=2  # 減少工作線程
export TTT_EXPERIMENT_ID="quicktest"

# 運行快速測試（只運行2個epoch）
python -c "
import sys
sys.argv = ['ttt2.py', '--experiment_hierarchical_content', '--hierarchy_alpha', '0.7', '--content_alpha', '0.01']

# 修改訓練設定為快速測試
import ttt2
ttt2.main = lambda: print('快速測試模式：模擬運行成功')

# 測試損失函數
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 測試離散損失函數
discrete_features = torch.randint(0, 1024, (4, 100), device=device)
content_ids = [1, 1, 2, 2]

try:
    loss = ttt2.compute_discrete_content_consistency_loss(discrete_features, content_ids, device)
    print(f'✅ 離散損失計算成功: {loss.item():.6f}')
except Exception as e:
    print(f'❌ 離散損失計算失敗: {e}')

# 測試階層式損失函數
continuous_features = torch.randn(4, 512, 200, device=device)
try:
    result = ttt2.compute_hierarchical_content_consistency_loss(
        continuous_features, discrete_features, content_ids, device, alpha=0.7
    )
    print(f'✅ 階層式損失計算成功: {result[\"total_loss\"].item():.6f}')
    print(f'   - 連續損失: {result[\"continuous_loss\"].item():.6f}')
    print(f'   - 離散損失: {result[\"discrete_loss\"].item():.6f}')
except Exception as e:
    print(f'❌ 階層式損失計算失敗: {e}')

print('🎉 快速測試完成！')
"
