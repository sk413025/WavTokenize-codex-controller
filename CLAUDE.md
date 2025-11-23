# WavTokenize 項目指南

這個文件會在每次 Claude Code 會話啟動時自動加載，確保 Claude 遵循項目的最佳實踐。

---

## 🔧 調試工作流程 - 核心原則

**CRITICAL: 優先使用 PDB + stdin 重定向，而非創建臨時 Python 腳本**

### 為什麼這很重要

- ✅ **可重現性**：調試命令存儲在文本文件中，任何人都能重新執行
- ✅ **版本控制友好**：可以 commit 和追蹤調試流程
- ✅ **無侵入性**：不修改源代碼，不污染 git history
- ✅ **自動化**：一次設置，多次執行
- ✅ **完整記錄**：調試過程和結果都有記錄

### 自動化多輪調試（推薦）

使用 **pdb-debugger subagent** 進行智能多輪調試：

```
> 使用 pdb-debugger 調試 done/exp/train_with_distances.py，找出為什麼訓練停滯
> 用 pdb-debugger 檢查梯度流動問題
```

**pdb-debugger** 會自動：
1. 執行第一輪 PDB，檢查關鍵位置
2. 分析輸出，識別問題（NaN、缺失梯度、模型 collapse 等）
3. 生成新的 PDB 命令，在新位置設置斷點
4. 迭代最多 5 輪，直到找到問題根源
5. 生成完整的調試報告和修復建議

### 手動調試流程

如果需要手動控制，使用標準 PDB + stdin 方法：

```bash
# 1. 創建 PDB 命令文件（例如 pdb_commands.txt）
# 包含：斷點位置、執行參數、檢查語句

# 2. 執行自動化調試
python -m pdb script.py < pdb_commands.txt

# 3. 可選：保存輸出
python -m pdb script.py < pdb_commands.txt 2>&1 | tee debug_output.log
```

### 完整指南

詳細的調試指南、實際案例、常用命令速查、進階技巧和範本，請參考：

@done/exp/DEBUG_GUIDE.md

---

## 📊 項目背景

### 當前研究方向
- **HDF5 訓練系統**：記憶體高效的大規模資料載入
- **VQ Distance 軟目標實驗**：使用 VQ-VAE codebook distances 進行 knowledge distillation
- **Zero-Shot Denoising Transformer**：Speaker-conditioned 音訊 token 去噪

### 重要檔案位置
- 訓練腳本：`done/exp/train_with_distances.py`
- Loss functions：`done/exp/losses_with_distances.py`
- 實驗啟動：`done/exp/launch_distance_experiments.sh`
- 調試指南：`done/exp/DEBUG_GUIDE.md`
- PDB 範例：`done/exp/pdb_commands.txt`

### 當前分支
- `feat/hdf5-training-implementation`

---

## 🐛 常見調試場景

### 梯度流動驗證（PyTorch）

```python
# 在 PDB 中檢查梯度
p sum(1 for p in model.parameters() if p.grad is not None)
p sum(1 for p in model.parameters() if p.requires_grad)
p model.token_embedding.weight.grad.norm().item()
```

### 訓練停滯診斷

```python
# 檢查 loss 和預測分布
p loss.item()
p loss.requires_grad
p torch.argmax(output, dim=-1).unique(return_counts=True)
```

---

## 🎯 快速命令

- `/debug` - 查看完整調試指南
- `/agents` - 查看所有可用的 subagent（包含 pdb-debugger）
- `/context` - 查看當前載入的記憶和 context

## 🤖 可用的 Subagents

- **pdb-debugger** - 專門用於 Python PDB 多輪調試
  - 自動執行 PDB → 分析 → 迭代循環
  - 最多 5 輪智能調試
  - 生成完整的調試報告
  - 使用方式：`使用 pdb-debugger 調試 <script.py>`

---

**Note**: 這個文件會在每次 Claude Code 啟動時自動載入。保持簡潔，詳細內容使用 `@import` 引入。
