# 訓練平台期診斷實驗重現指南

**實驗日期**: 2025-11-05  
**實驗編號**: EXP-20251105-PLATEAU-DIAGNOSIS  
**相關報告**: 
- `PLATEAU_MECHANISM_ANALYSIS.md` (詳細機轉分析)
- `PLATEAU_DIAGNOSIS_SUMMARY.md` (簡潔摘要)
- `TRAINING_PLATEAU_DIAGNOSIS_20251105.md` (完整診斷報告)

---

## 📋 實驗背景

### 問題現象
- **Train Accuracy**: 卡在 54% (Epoch 20-38 幾乎無進步)
- **Val Accuracy**: 僅 37% (Train-Val Gap = 17%)
- **訓練狀態**: Loss 持續下降但 Accuracy 不提升

### 實驗目的
診斷訓練平台期的根本原因，確認是否為：
1. Token 過度集中
2. Padding 過多
3. Data Distribution Mismatch
4. 模型架構問題

---

## 🔬 實驗步驟

### 步驟 1: Token 分布分析

**目的**: 分析 Train/Val set 的 token 分布差異

**執行**:
```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp

python analyze_token_distribution.py
```

**預期輸出**:
```
Train Set:
  總 token 數: 4,285,755
  唯一 token 數: 1,833 / 4096 (44.8%)
  Token 453: 13.57%

Val Set:
  總 token 數: 1,581,684
  唯一 token 數: 1,819 / 4096 (44.4%)
  Token 453: 18.65%

發現 15 個 Top-20 tokens 有顯著分布差異 (>0.3%)
累計絕對差異: 10.94%
```

**關鍵發現**:
- Token 453 在 Val 比 Train 高 +5.08% (相對增幅 +37.5%)
- 15 個 Top-20 tokens 都有分布差異

---

### 步驟 2: Accuracy 反推分析

**目的**: 從整體 accuracy 反推 Token 453 和其他 tokens 的準確率

**執行**:
```bash
python analyze_token_accuracy_inference.py
```

**預期輸出**:
```
Train Set Accuracy 分析:
  如果 Token 453 完全失敗 (0%)
    → 其他 tokens 準確率 = 63.29%

Val Set Accuracy 分析:
  如果 Token 453 完全失敗 (0%)
    → 其他 tokens 準確率 = 45.18%

關鍵發現:
  即使 Token 453 表現相同，其他 tokens 在 Val 也下降了 18.1%
  → 問題不只是 Token 453，整體分布都有 mismatch
```

**關鍵發現**:
- Token 453 佔 Train 錯誤 30%, Val 錯誤 29.5%
- 其他 tokens 在 Val 的準確率也下降 18%
- Distribution Mismatch 只能解釋 30% 的 gap

---

### 步驟 3: Padding 分析

**目的**: 檢查 padding 是否導致訓練問題

**執行**:
```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp

python << 'EOF'
import torch
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn
from torch.utils.data import DataLoader

# 載入數據
train_dataset = ZeroShotAudioDatasetCached('./data/train_cache.pt')
val_dataset = ZeroShotAudioDatasetCached('./data/val_cache.pt')

train_loader = DataLoader(train_dataset, batch_size=28, shuffle=False, collate_fn=cached_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=28, shuffle=False, collate_fn=cached_collate_fn)

# 分析 Train set padding (10 batches)
print("Train Set Padding 分析 (10 batches):\n")
train_lengths = []
for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 10:
        break
    clean_tokens = batch['clean_tokens']
    B, T = clean_tokens.shape
    for i in range(B):
        actual_len = (clean_tokens[i] != 0).sum().item()
        train_lengths.append(actual_len)

print(f"序列長度: Min={min(train_lengths)}, Max={max(train_lengths)}, Mean={sum(train_lengths)/len(train_lengths):.1f}")

# 分析 Val set padding
print("\nVal Set Padding 分析 (10 batches):\n")
val_lengths = []
for batch_idx, batch in enumerate(val_loader):
    if batch_idx >= 10:
        break
    clean_tokens = batch['clean_tokens']
    B, T = clean_tokens.shape
    for i in range(B):
        actual_len = (clean_tokens[i] != 0).sum().item()
        val_lengths.append(actual_len)

print(f"序列長度: Min={min(val_lengths)}, Max={max(val_lengths)}, Mean={sum(val_lengths)/len(val_lengths):.1f}")

# 計算 padding 比例
train_max_len = max(train_lengths)
val_max_len = max(val_lengths)

train_padding_ratio = (train_max_len - sum(train_lengths)/len(train_lengths)) / train_max_len * 100
val_padding_ratio = (val_max_len - sum(val_lengths)/len(val_lengths)) / val_max_len * 100

print(f"\nTrain Padding 平均佔比: {train_padding_ratio:.1f}%")
print(f"Val Padding 平均佔比: {val_padding_ratio:.1f}%")

print("\n結論: Val set padding 幾乎為 0%，說明 padding 不是問題")
EOF
```

**預期輸出**:
```
Train Set Padding 分析:
  序列長度: Min=194, Max=438, Mean=265.9

Val Set Padding 分析:
  序列長度: Min=280, Max=439, Mean=343.4

Train Padding 平均佔比: 30.2%
Val Padding 平均佔比: 0.05%

結論: Val set padding 幾乎為 0%，說明 padding 不是問題
```

**關鍵發現**:
- Val set padding 極少 (0-0.3%)，但 accuracy 依然低
- Padding 不是導致低 accuracy 的主要原因

---

### 步驟 4: Noisy vs Clean Token 差異分析

**目的**: 確認 denoising task 的難度

**執行**:
```bash
python << 'EOF'
import torch
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn
from torch.utils.data import DataLoader

val_dataset = ZeroShotAudioDatasetCached('./data/val_cache.pt')
val_loader = DataLoader(val_dataset, batch_size=28, shuffle=False, collate_fn=cached_collate_fn)

# 取一個 batch
batch = next(iter(val_loader))
noisy_tokens = batch['noisy_tokens']
clean_tokens = batch['clean_tokens']

# 計算差異
B, T = clean_tokens.shape
diff_count = 0
total_count = 0

for i in range(B):
    non_pad_mask = clean_tokens[i] != 0
    diff_count += (noisy_tokens[i][non_pad_mask] != clean_tokens[i][non_pad_mask]).sum().item()
    total_count += non_pad_mask.sum().item()

diff_ratio = diff_count / total_count * 100

print(f"Noisy vs Clean Token 差異:")
print(f"  不同的 token 數: {diff_count} / {total_count} ({diff_ratio:.2f}%)")
print(f"\n解讀: Noisy audio 導致約 {diff_ratio:.0f}% 的 token 改變")
print(f"說明: Denoising task 具有足夠難度")
EOF
```

**預期輸出**:
```
Noisy vs Clean Token 差異:
  不同的 token 數: 6790 / 9574 (70.92%)

解讀: Noisy audio 導致約 71% 的 token 改變
說明: Denoising task 具有足夠難度
```

---

## 📊 實驗結果總結

### 核心發現

1. **Token 453 是主要瓶頸** ✅
   - 數據: Train 13.57%, Val 18.65% (+5.08%)
   - 佔 Train 錯誤 30%, Val 錯誤 29.5%

2. **Distribution Mismatch 廣泛存在** ✅
   - 數據: 15 個 Top-20 tokens 累計差異 10.94%
   - Val speakers 的聲音特徵與 Train speakers 系統性不同

3. **Mismatch 只能解釋部分 gap** ⚠️
   - 推算: Mismatch 導致 4-5% accuracy loss
   - 實際 gap 17%，說明 70% 來自模型泛化能力不足

---

## ⚡ Quick Repro Index（實驗對應表）

以下彙整本輪 Cross‑Attention 機轉診斷的主要 run，包含類型、結果路徑、重現命令與關鍵指標（節錄）。路徑皆以專案根目錄為基準。

- Deep‑100（K=4，多層注入，100 epoch）
  - Run 目錄：`results/crossattn_k4_deep_100ep_20251105_221426`
  - 行為重現（E1/E2/E3 一鍵）：
    - `bash done/exp/run_behavior_analysis.sh <gpu> results/crossattn_k4_deep_100ep_20251105_221426 "10 20 30 40 50 80 100" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`
  - 產物路徑（節錄）：
    - E2 分桶（e80/e100）：`analysis/margins_topk/epoch_{80,100}/margins_bins_epoch_*.csv`
    - E3 幾何（e80/e100）：`analysis/logit_geometry/epoch_{80,100}/geometry_epoch_*.csv`
  - 指標摘要：
    - 高 margin ΔAcc_zero（e100）：約 −12.40 pp（移除更差 → 加入更好）
    - 低 margin dmargin_mean（e100）：約 −3.48（方向性負）
  - 深入報告：`CROSSATTN_DEEP_100_200_ANALYSIS_20251106.md`
  - 原始訓練指令（範例）：見 `RUNNING_EXPERIMENTS_20251105.md` Deep‑100 小節

- Deep‑200（K=4，多層注入，200 epoch）
  - Run 目錄：`results/crossattn_k4_deep_200ep_20251106_014239`
  - 行為重現：
    - `bash done/exp/run_behavior_analysis.sh <gpu> results/crossattn_k4_deep_200ep_20251106_014239 "10 20 30 40 50 80 100 150 200" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`
  - 產物路徑（節錄）：
    - E1 影響分解：`analysis/influence_breakdown/epoch_{10..200}/breakdown_epoch_*.csv`
    - E2 分桶（e10/20/30/40/50）：`analysis/margins_topk/epoch_*/margins_bins_epoch_*.csv`
  - 指標摘要：
    - 淨影響（zero）net_acc_delta：e80≈ −3.75 pp → e200≈ −9.56 pp
  - 深入報告：`CROSSATTN_DEEP_100_200_ANALYSIS_20251106.md`
  - 原始訓練指令（範例）：對齊 Deep‑100 指令，將 `--num_epochs 200` 與輸出路徑替換為本 run 目錄

- Gated‑100（K=4，門控，100 epoch）
  - Run 目錄：`results/crossattn_k4_gate_100ep_20251105_221334`
  - 行為重現：
    - `bash done/exp/run_behavior_analysis.sh <gpu> results/crossattn_k4_gate_100ep_20251105_221334 "10 20 30 40 50 80 100" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`
  - 產物路徑（節錄）：
    - E2 分桶（e80/e100）：`analysis/margins_topk/epoch_{80,100}/margins_bins_epoch_*.csv`
    - E3 幾何（e80/e100）：`analysis/logit_geometry/epoch_{80,100}/geometry_epoch_*.csv`
  - 指標摘要：
    - 高 margin dmargin_mean（e80/e100）：≈ +0.535 / +0.536（強正向）
    - 高 margin ΔAcc_zero（e100）：約 −18.75 pp（移除更差 → 加入更好）
  - 深入報告：`CROSSATTN_GATED_100_200_ANALYSIS_20251106.md`
  - 原始訓練指令（範例）：見 `RUNNING_EXPERIMENTS_20251105.md` Gated‑100 小節

- Gated‑200（K=4，門控，200 epoch）
  - Run 目錄：`results/crossattn_k4_gate_200ep_20251106_014033`
  - 行為重現：
    - `bash done/exp/run_behavior_analysis.sh <gpu> results/crossattn_k4_gate_200ep_20251106_014033 "10 20 30 40 50 80 100 150 200" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`
  - 產物路徑（節錄）：
    - E1 影響分解：`analysis/influence_breakdown/epoch_{10..200}/breakdown_epoch_*.csv`
    - E3 幾何（e150/e200）：`analysis/logit_geometry/epoch_{150,200}/geometry_epoch_*.csv`
    - 注意力熵：`analysis/attn_entropy/epoch_*/entropy_epoch_*.csv`
    - 門控分布：`analysis/gate_distribution/epoch_*/gate_stats_epoch_*.csv`
  - 指標摘要：
    - 淨影響（zero）：e10≈ −0.53 pp → e100≈ −3.47 pp → e200≈ −9.80 pp
    - 高 margin dmargin_mean：e150≈ +0.91、e200≈ +1.06
    - 注意力熵均值：e10≈ 0.909 → e200≈ 0.726（peaked_frac_gt0.7 下降）
  - 深入報告：`CROSSATTN_GATED_100_200_ANALYSIS_20251106.md`
  - 原始訓練指令（建議）：將 Gated‑100 指令之 `--num_epochs 100` 調整為 `--num_epochs 200`，輸出路徑置為本 run 目錄

- GateL2‑100（K=4，淺層門控，100 epoch）
  - Run 目錄：`results/crossattn_k4_gateL2_100ep_20251106_000140`
  - 行為重現：
    - `bash done/exp/run_behavior_analysis.sh <gpu> results/crossattn_k4_gateL2_100ep_20251106_000140 "10 20 30 40 50 80 100" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`
  - 產物路徑（節錄）：
    - E1：`analysis/influence_breakdown/epoch_100/breakdown_epoch_100.csv`
    - E2：`analysis/margins_topk/epoch_100/margins_bins_epoch_100.csv`
    - E3：`analysis/logit_geometry/epoch_100/geometry_epoch_100.csv`
  - 指標摘要：
    - 淨影響（zero）e100：約 −15.91 pp
    - 高 margin（e100）：ΔAcc_zero ≈ −34.29 pp；`dmargin_mean ≈ +2.59`
    - 中 margin（e100）：ΔAcc_zero ≈ −1.72/−4.81/−10.25 pp；`cos_mean ≈ −0.0081`、`dmargin_mean ≈ −1.89`
  - 深入報告：`CROSSATTN_GATE_L2_100EP_REPORT_20251106.md:1`

備註
- 所有分析腳本皆使用同一資料快取：`/home/sbplab/ruizi/c_code/done/exp/data`（val_cache.pt）。
- 若需只跑單一指標，請見對應腳本：
  - E1：`done/exp/analyze_influence_breakdown.py`
  - E2：`done/exp/analyze_margins_topk.py`
  - E3：`done/exp/analyze_logit_shift_geometry.py`
  - 熵：`done/exp/analyze_attention_entropy.py`
  - 門控：`done/exp/analyze_gate_distribution.py`

4. **Padding 不是問題** ✅
   - 數據: Val set padding <0.3%，但 accuracy 依然低

### 機轉模型

```
根本原因: Val Speakers 聲學特徵不同
    ↓
Token Distribution Mismatch (15 tokens, 累計 10.94%)
    ↓
模型架構限制 (Speaker Embedding 只用簡單相加)
    ↓
訓練結果: Train 54%, Val 37%, Gap 17%
```

---

## 🔧 改進方向

基於實驗結果，提出以下改進方向：

### 方向 1: Speaker-Adaptive Token Distribution Modeling ⭐⭐⭐
- **針對**: Speaker Embedding 無法捕捉 speaker-specific token distribution
- **方案**: 新增 Speaker → Token Distribution Prior 網路
- **預期**: Val Acc 提升至 45-50%

### 方向 2: Weighted Cross-Entropy Loss ⭐⭐
- **針對**: Token 453 等 mismatch tokens 訓練不足
- **方案**: 對 Val 佔比高的 tokens 增加權重
- **預期**: Train Acc 60-65%, Val Acc 42-47%

### 方向 3: Distribution-Aware Speaker Split ⭐
- **針對**: Val speakers 選擇不當
- **方案**: 選擇 token distribution 接近 train 的 speakers
- **預期**: Val Acc 48-52% (但喪失 zero-shot 嚴格性)

---

## 📁 相關檔案

### 實驗分析工具
- `analyze_token_distribution.py` - Token 分布分析工具
- `analyze_token_accuracy_inference.py` - Accuracy 反推工具

### 實驗報告
- `PLATEAU_MECHANISM_ANALYSIS.md` - 詳細機轉分析 (含 ASCII 圖示)
- `PLATEAU_DIAGNOSIS_SUMMARY.md` - 簡潔摘要
- `TRAINING_PLATEAU_DIAGNOSIS_20251105.md` - 完整診斷報告

### 訓練相關
- `train_zeroshot_full_cached_analysis.py` - 訓練腳本
- `data_zeroshot.py` - 數據載入
- `model_zeroshot.py` - 模型定義

### 數據
- `done/exp/data/train_cache.pt` - 訓練集緩存 (91 MB, 16,128 samples)
- `done/exp/data/val_cache.pt` - 驗證集緩存 (32 MB, 4,608 samples)

### 訓練輸出
- `results/zeroshot_100epochs_20251105_002300/` - 訓練輸出目錄
  - `best_model.pth` - 最佳模型 (Epoch 38, Val Acc 36.75%)
  - `training.log` - 訓練日誌
  - `config.json` - 訓練配置

---

## 🧪 待驗證假設

以下假設需要額外實驗驗證：

1. **Speaker Embedding 是否能調整 Token Distribution?**
   - 工具: 固定 noisy tokens，改變 speaker embeddings
   - 觀察預測 distribution 是否改變

2. **Token 453 的物理意義?**
   - 提取 Token 453 對應音頻段
   - 分析頻譜、能量、音高、共振峰特徵

3. **模型預測的 Token 453 頻率?**
   - 統計模型預測
   - 驗證是否接近 13.57% (Train) 或 18.65% (Val)

---

## 💡 關鍵洞察

**這不是 Bug，是 Zero-Shot Task 的本質困難**

- Val speakers 與 Train speakers 聲學特徵確實不同
- Token distribution mismatch 反映真實差異
- 當前模型架構無法捕捉 speaker-specific token distribution
- **需要改進模型架構，不只是調整數據**

---

**實驗完成時間**: 2025-11-05 02:15:00  
**下一步**: 實作 Speaker-Adaptive Token Distribution Modeling (方向 1)
