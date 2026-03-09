# exp_0128 Phase 2 快速啟動指南

## 立即執行所有實驗

```bash
# 並行執行所有 5 個實驗（推薦）
bash exp_0128/phase2/start_all_experiments.sh
```

這會在背景啟動所有實驗，預計 2-3 小時完成。

---

## 單獨執行實驗

### Entropy Regularization 實驗

```bash
# 實驗 3a: λ=0.01 (保守)
bash exp_0128/phase2/entropy_regularization/run_exp3a_lambda_0.01.sh

# 實驗 3b: λ=0.05 (中等)
bash exp_0128/phase2/entropy_regularization/run_exp3b_lambda_0.05.sh

# 實驗 3c: λ=0.1 (激進)
bash exp_0128/phase2/entropy_regularization/run_exp3c_lambda_0.1.sh
```

### Codebook Refresh 實驗

```bash
# 實驗 4a: interval=100, threshold=10 (保守)
bash exp_0128/phase2/codebook_refresh/run_exp4a_interval_100_thresh_10.sh

# 實驗 4b: interval=50, threshold=5 (激進)
bash exp_0128/phase2/codebook_refresh/run_exp4b_interval_50_thresh_5.sh
```

---

## 監控進度

### 實時查看訓練 log

```bash
# Entropy Regularization
tail -f exp_0128/phase2/logs/exp3a_lambda_0.01.log
tail -f exp_0128/phase2/logs/exp3b_lambda_0.05.log
tail -f exp_0128/phase2/logs/exp3c_lambda_0.1.log

# Codebook Refresh
tail -f exp_0128/phase2/logs/exp4a_interval_100_thresh_10.log
tail -f exp_0128/phase2/logs/exp4b_interval_50_thresh_5.log
```

### 檢查 GPU 使用狀況

```bash
watch -n 1 nvidia-smi
```

### 檢查背景進程

```bash
ps aux | grep train_entropy_reg
ps aux | grep train_codebook_refresh
```

---

## 查看結果

### 最終結果摘要

每個實驗的 `summary.json` 包含：

```bash
# Entropy Regularization
cat exp_0128/phase2/entropy_regularization/exp3a_lambda_0.01/run_*/summary.json
cat exp_0128/phase2/entropy_regularization/exp3b_lambda_0.05/run_*/summary.json
cat exp_0128/phase2/entropy_regularization/exp3c_lambda_0.1/run_*/summary.json

# Codebook Refresh
cat exp_0128/phase2/codebook_refresh/exp4a_interval_100_thresh_10/run_*/summary.json
cat exp_0128/phase2/codebook_refresh/exp4b_interval_50_thresh_5/run_*/summary.json
```

### 訓練曲線

每個實驗生成 `training_curves.png`：

```bash
# 使用 VSCode 打開圖片
code exp_0128/phase2/entropy_regularization/exp3a_lambda_0.01/run_*/training_curves.png
code exp_0128/phase2/codebook_refresh/exp4a_interval_100_thresh_10/run_*/training_curves.png
```

### Metrics 歷史

```bash
# JSON 格式
cat exp_0128/phase2/entropy_regularization/exp3a_lambda_0.01/run_*/metrics_history.json

# 使用 jq 美化
cat exp_0128/phase2/entropy_regularization/exp3a_lambda_0.01/run_*/metrics_history.json | jq '.'
```

---

## 快速判斷成功/失敗

### 使用 Python 快速檢查

```python
import json
from pathlib import Path

# 檢查所有實驗結果
experiments = {
    'Exp 3a (λ=0.01)': 'exp_0128/phase2/entropy_regularization/exp3a_lambda_0.01',
    'Exp 3b (λ=0.05)': 'exp_0128/phase2/entropy_regularization/exp3b_lambda_0.05',
    'Exp 3c (λ=0.1)': 'exp_0128/phase2/entropy_regularization/exp3c_lambda_0.1',
    'Exp 4a (i=100, t=10)': 'exp_0128/phase2/codebook_refresh/exp4a_interval_100_thresh_10',
    'Exp 4b (i=50, t=5)': 'exp_0128/phase2/codebook_refresh/exp4b_interval_50_thresh_5',
}

baseline = {'entropy': 6.07, 'top_10_mass': 0.197, 'strict_acc': 0.0091}

for name, path in experiments.items():
    run_dirs = list(Path(path).glob('run_*'))
    if run_dirs:
        summary_path = run_dirs[0] / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
                final = data['final']
                success = data['success']

                print(f"\n{name}:")
                print(f"  Success: {'✅' if success else '❌'}")
                print(f"  Entropy: {final['entropy']:.2f} (Δ{final['entropy']-baseline['entropy']:+.2f})")
                print(f"  Top-10: {final['top_10_mass']*100:.1f}% (Δ{(final['top_10_mass']-baseline['top_10_mass'])*100:+.1f}%)")
                print(f"  Acc: {final['strict_acc']*100:.2f}% (Δ{(final['strict_acc']-baseline['strict_acc'])*100:+.2f}%)")
```

### 使用 Bash 一鍵檢查

```bash
# 檢查所有實驗的成功狀態
for exp in exp_0128/phase2/entropy_regularization/exp3*/run_*/summary.json \
           exp_0128/phase2/codebook_refresh/exp4*/run_*/summary.json; do
    if [ -f "$exp" ]; then
        echo "=== $(dirname $exp) ==="
        cat "$exp" | jq '{success, final: {entropy, top_10_mass, strict_acc}}'
        echo ""
    fi
done
```

---

## 成功判準提醒

實驗成功需同時滿足：

- ✅ Entropy > 6.07 (baseline)
- ✅ Top-10 Mass < 19.7% (baseline)
- ✅ Strict Accuracy ≥ 0.82% (90% of baseline)

**任一實驗成功即為 Phase 2 成功！**

---

## 預期結果

### 如果成功

1. 進行 full training (300 epochs)
2. 測試組合方法 (Entropy Reg + Codebook Refresh)
3. 精調超參數

### 如果失敗

測試 Phase 2 中優先級方案：
- 方案 C: 降低學習率 (1e-4 → 5e-5)
- 方案 D: 減小 LoRA Rank (256 → 128)

---

## 故障排除

### 實驗卡住或 OOM

```bash
# 檢查 GPU 內存
nvidia-smi

# 終止實驗
pkill -f train_entropy_reg
pkill -f train_codebook_refresh

# 清理 GPU 內存
python -c "import torch; torch.cuda.empty_cache()"
```

### 重新啟動失敗的實驗

單獨運行失敗的實驗腳本即可（會自動創建新的 timestamped 目錄）。

---

**創建日期**: 2026-01-29
**用途**: Phase 2 實驗快速啟動與結果檢查
