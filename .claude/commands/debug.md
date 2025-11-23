---
description: Show Python debugging guide (PDB + stdin redirection method)
---

# Python Debugging Guide

快速查看項目的標準調試方法和完整指南。

---

## 🎯 核心原則

**優先使用 PDB + stdin 重定向，而非創建臨時 Python 腳本**

### 為什麼？

- ✅ **可重現**：調試命令存儲在文本文件中
- ✅ **可追蹤**：可以 commit 到 git，完整記錄調試過程
- ✅ **無侵入**：不修改源代碼
- ✅ **自動化**：一次設置，多次執行

---

## 📝 基本用法

### 1. 創建 PDB 命令文件

創建 `pdb_commands.txt`：

```bash
# 設置斷點
b script.py:100

# 執行程式（帶參數）
run --arg1 value1

# 檢查變數
c
p variable_name

# 退出
q
```

### 2. 執行調試

```bash
python -m pdb script.py < pdb_commands.txt
```

### 3. 保存輸出（可選）

```bash
python -m pdb script.py < pdb_commands.txt 2>&1 | tee debug_output.log
```

---

## 🔍 常用 PDB 命令速查

### 斷點控制
- `b file.py:line` - 設置斷點
- `c` (continue) - 繼續到下一個斷點
- `n` (next) - 執行下一行
- `s` (step) - 進入函數
- `q` (quit) - 退出

### 變數檢查
- `p variable` - 打印變數
- `pp variable` - 美化打印
- `p dir(object)` - 查看物件屬性

### PyTorch 梯度檢查
```python
# 檢查有多少參數有梯度
p sum(1 for p in model.parameters() if p.grad is not None)

# 檢查梯度範數
p model.layer.weight.grad.norm().item()

# 檢查 loss
p loss.item()
p loss.requires_grad
```

---

## 📊 實際案例

### 梯度流動驗證

在 VQ Distance 訓練實驗中，成功使用此方法診斷梯度流動：

```bash
# 使用的命令文件
cat done/exp/pdb_commands.txt

# 執行
python -m pdb done/exp/train_with_distances.py < done/exp/pdb_commands.txt
```

**結果**：
- ✅ 發現梯度正常（52/52 參數有梯度）
- ✅ 排除梯度消失假設
- ✅ 指向真正問題：模型 collapse

---

## 📋 範本

### 基礎範本

```bash
# === 斷點 ===
b script.py:LINE

# === 執行 ===
run ARGS

# === 檢查 ===
c
p VARIABLE

# === 退出 ===
q
```

### 訓練調試範本

```bash
# === 斷點 ===
b train.py:FORWARD_LINE
b train.py:LOSS_LINE
b train.py:BACKWARD_LINE
b train.py:OPTIMIZER_LINE

# === 執行（最小配置加快調試）===
run --batch_size 2 --num_epochs 1

# === Forward 檢查 ===
c
p output.shape
p output.requires_grad

# === Loss 檢查 ===
c
p loss.item()
p loss.requires_grad

# === Backward 檢查 ===
c
p sum(1 for p in model.parameters() if p.grad is not None)

# === Optimizer 檢查 ===
c
p "Step completed"

q
```

---

## 📚 完整指南

詳細的調試指南（包含進階技巧、故障排除、更多範例）：

**done/exp/DEBUG_GUIDE.md**

此文件包含：
- 進階技巧（條件斷點、循環檢查、動態修改）
- 故障排除（斷點行數改變、互動輸入、輸出過多）
- 與其他調試方法的詳細比較
- 完整的 PyTorch 調試命令集
- 更多實際案例

---

## 🚀 快速開始

如果你現在需要調試：

1. **創建命令文件**：`vim pdb_commands.txt`
2. **設置斷點和檢查語句**（參考上面的範本）
3. **執行**：`python -m pdb your_script.py < pdb_commands.txt`
4. **分析輸出**
5. **Commit 命令文件**：記錄調試過程

---

**記住：這是項目的標準調試方法。優先使用此方法，而非創建臨時 Python 腳本！**
