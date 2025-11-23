# Python 調試指南：PDB + stdin 重定向方法

## 核心原則

**優先使用 PDB + stdin 重定向進行自動化調試，而非編寫新的 Python 腳本。**

這個方法的優勢：
- ✅ 可重現：調試命令寫在文本檔案中，隨時可以重新執行
- ✅ 可版本控制：pdb 命令檔案可以 commit 到 git
- ✅ 無侵入性：不需修改原始程式碼
- ✅ 自動化：一次設定，多次執行
- ✅ 完整記錄：調試過程和結果都有記錄

---

## 基本用法

### 1. 創建 PDB 命令檔案

創建一個 `.txt` 檔案（例如 `pdb_commands.txt`），包含所有 pdb 命令：

```bash
# 範例：pdb_commands.txt

# 設置斷點
b script.py:100
b script.py:200

# 執行程式（帶參數）
run --arg1 value1 --arg2 value2

# 第一個斷點：檢查變數
c
p variable_name
p variable.shape
p variable.requires_grad

# 第二個斷點：檢查梯度
c
p model.layer.weight.grad.norm()
p sum(1 for p in model.parameters() if p.grad is not None)

# 退出
q
```

### 2. 執行調試

```bash
python -m pdb script.py < pdb_commands.txt
```

### 3. 重定向輸出（可選）

如果輸出很多，可以同時重定向到檔案：

```bash
python -m pdb script.py < pdb_commands.txt 2>&1 | tee debug_output.log
```

---

## 實際案例：梯度流動驗證

### 背景

訓練停滯 34 epochs 無改善，懷疑梯度消失問題。

### 命令檔案：`pdb_commands.txt`

```bash
# === 設置斷點 ===
b train_with_distances.py:280  # Loss 計算後
b train_with_distances.py:289  # Backward 後
b train_with_distances.py:295  # Optimizer step 後

# === 執行訓練 ===
run --exp_name debug_test --batch_size 2 --num_workers 0 --num_epochs 1 ...

# === 斷點 1: Loss 計算後 ===
c
p loss
p loss.requires_grad
p loss.item()

# === 斷點 2: Backward 後 ===
c
p sum(1 for p in model.parameters() if p.grad is not None)
p sum(1 for p in model.parameters() if p.requires_grad)
p model.token_embedding.weight.grad.norm().item()

# === 斷點 3: Optimizer step 後 ===
c
p "Optimizer step completed"

q
```

### 執行

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp
python -m pdb train_with_distances.py < pdb_commands.txt
```

### 結果

✅ 發現梯度流動正常 (52/52 參數有梯度)
✅ 排除梯度消失假設
✅ 指向真正問題：模型 collapse

---

## 常用 PDB 命令速查

### 斷點控制
- `b file.py:line` - 在指定行設置斷點
- `b function_name` - 在函數入口設置斷點
- `c` (continue) - 繼續執行到下一個斷點
- `n` (next) - 執行下一行
- `s` (step) - 進入函數
- `r` (return) - 執行到當前函數返回

### 變數檢查
- `p variable` - 打印變數
- `pp variable` - 美化打印
- `p dir(object)` - 查看物件屬性
- `p type(variable)` - 查看類型
- `p variable.shape` - 查看 tensor shape

### PyTorch 專用檢查
```python
# 檢查梯度
p parameter.grad
p parameter.grad.norm()
p parameter.requires_grad

# 統計有梯度的參數
p sum(1 for p in model.parameters() if p.grad is not None)
p sum(1 for p in model.parameters() if p.requires_grad)

# 檢查 tensor 屬性
p tensor.shape
p tensor.dtype
p tensor.device
p tensor.requires_grad
```

### 條件斷點
```python
# 當條件滿足時才中斷
b script.py:100, epoch > 10
b script.py:200, loss.item() > 10.0
```

### 執行表達式
```python
# 在斷點處執行任意 Python 代碼
p [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
```

---

## 進階技巧

### 1. 檢查多個時間點

```bash
# 在訓練循環的多個關鍵點設置斷點
b train.py:100  # 加載數據後
b train.py:150  # Forward 後
b train.py:180  # Loss 計算後
b train.py:200  # Backward 後
b train.py:220  # Optimizer step 後
```

### 2. 循環檢查

如果需要檢查多個 batch：

```bash
# 第一個 batch
c
p batch_idx
p loss.item()

# 第二個 batch
c
p batch_idx
p loss.item()

# ... 重複
```

### 3. 動態修改變數

```bash
# 在斷點處修改變數進行實驗
c
p learning_rate
!learning_rate = 0.001
c
```

### 4. 搭配條件表達式

```python
# 只檢查異常情況
p loss.item() if not torch.isfinite(loss) else "Loss is fine"
p "NaN detected!" if torch.isnan(tensor).any() else tensor.mean().item()
```

---

## 最佳實踐

### ✅ DO

1. **命令檔案命名清晰**：`pdb_梯度檢查.txt`, `pdb_loss_debug.txt`
2. **添加註解**：在命令檔案中說明每個斷點的目的
3. **版本控制**：commit 調試命令檔案和結果
4. **記錄發現**：在 commit message 中說明調試結果
5. **參數化執行**：使用最小配置（小 batch_size, 少 epochs）加快調試

### ❌ DON'T

1. **不要創建臨時 Python 腳本**：優先使用 pdb 命令檔案
2. **不要修改源代碼添加 print**：會污染 git history
3. **不要手動輸入命令**：不可重現
4. **不要在生產環境使用**：pdb 會暫停執行

---

## 與其他調試方法的比較

| 方法 | 可重現性 | 版本控制 | 侵入性 | 靈活性 |
|------|----------|----------|--------|--------|
| **PDB + stdin** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅ |
| 手動 PDB | ❌ | ❌ | ✅✅✅ | ✅✅✅ |
| print() | ✅✅ | ✅ | ❌ | ❌ |
| logging | ✅✅✅ | ✅✅ | ✅✅ | ✅ |
| 專用腳本 | ✅✅ | ✅✅ | ✅ | ✅✅ |

---

## 故障排除

### 問題：斷點行數改變

**症狀**：程式碼修改後，斷點位置不對

**解決**：
```bash
# 搜尋關鍵行
grep -n "keyword" script.py

# 更新 pdb_commands.txt 中的行號
```

### 問題：程式需要互動輸入

**症狀**：程式期待用戶輸入，但 stdin 被 pdb 命令佔用

**解決**：
```python
# 方法 1: 使用環境變數
export INPUT_VALUE="test"
python -m pdb script.py < pdb_commands.txt

# 方法 2: 修改程式支援非互動模式
parser.add_argument('--non-interactive', action='store_true')
```

### 問題：輸出太多

**解決**：
```bash
# 只保留關鍵輸出
python -m pdb script.py < pdb_commands.txt 2>&1 | grep -E "(Pdb)|variable_name"

# 或重定向到檔案後分析
python -m pdb script.py < pdb_commands.txt &> debug.log
grep "variable_name" debug.log
```

---

## 範本

### 基礎範本

```bash
# === 設置斷點 ===
b script.py:LINE_NUMBER

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
b train.py:DATA_LOAD_LINE
b train.py:FORWARD_LINE
b train.py:LOSS_LINE
b train.py:BACKWARD_LINE
b train.py:OPTIMIZER_LINE

# === 執行 ===
run --batch_size 2 --num_epochs 1 --debug

# === 數據檢查 ===
c
p batch['input'].shape
p batch['target'].shape

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

## 參考資源

- Python PDB 官方文檔: https://docs.python.org/3/library/pdb.html
- PDB 命令速查: https://docs.python.org/3/library/pdb.html#debugger-commands

---

**記住：優先使用 PDB + stdin 重定向，而非創建臨時 Python 腳本！**
