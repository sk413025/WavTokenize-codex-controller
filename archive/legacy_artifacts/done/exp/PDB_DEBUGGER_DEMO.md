# pdb-debugger Subagent 使用示範

這是一個完整的示範，展示如何使用 `pdb-debugger` subagent 進行自動多輪調試。

---

## 示範場景

**問題腳本**：[example_buggy_script.py](example_buggy_script.py)

這個腳本計算多組數據的平均值，但會崩潰並顯示：
```
ZeroDivisionError: division by zero
```

---

## 使用方式 1：在 Claude Code 中調用 pdb-debugger

在 Claude Code 會話中直接說：

```
> 使用 pdb-debugger 調試 done/exp/example_buggy_script.py，
  找出為什麼會出現 ZeroDivisionError
```

### pdb-debugger 會自動執行：

#### Round 1: 初始探索
1. 讀取源碼，識別可能出錯的位置
2. 創建 `pdb_commands_round1.txt`：
   ```bash
   b example_buggy_script.py:18  # 除法操作處
   run
   c
   p numbers
   p total
   p count
   ```
3. 執行 PDB，保存到 `pdb_output_round1.log`
4. **分析發現**：空列表導致 `count = 0`

#### Round 2: 驗證修復（如果需要）
pdb-debugger 會建議修復方案並驗證。

---

## 使用方式 2：手動模擬（本次示範）

### 已創建的文件

1. **[example_buggy_script.py](example_buggy_script.py)** - 有 bug 的腳本
2. **[pdb_demo_round1.txt](pdb_demo_round1.txt)** - Round 1 的 PDB 命令
3. **[pdb_demo_output_round1.log](pdb_demo_output_round1.log)** - Round 1 的輸出
4. **[pdb_demo_analysis_round1.md](pdb_demo_analysis_round1.md)** - Round 1 的分析報告

### 執行過程

```bash
# Round 1: 初始探索
python -m pdb example_buggy_script.py < pdb_demo_round1.txt 2>&1 | tee pdb_demo_output_round1.log
```

### 分析結果

通過 Round 1 的調試，我們發現：

| Dataset | numbers | total | count | count==0 | 結果 |
|---------|---------|-------|-------|----------|------|
| 0 | `[1,2,3,4,5]` | 15 | 5 | False | ✅ 正常 |
| 1 | `[10,20,30]` | 60 | 3 | False | ✅ 正常 |
| 2 | `[]` | 0 | **0** | **True** | ⚠️ 問題！|

**問題根源**：`example_buggy_script.py:18` - 空列表導致除零錯誤

### 修復建議

```python
def calculate_average(numbers):
    """計算數字列表的平均值"""
    if not numbers:  # ← 添加檢查
        return 0

    return sum(numbers) / len(numbers)
```

---

## pdb-debugger 的優勢

與手動調試相比，pdb-debugger subagent 會：

### ✅ 自動化
- 自動識別關鍵斷點位置
- 自動生成 PDB 命令
- 自動執行多輪迭代

### ✅ 智能分析
- 分析變數狀態，識別異常值
- 對比多輪結果，縮小問題範圍
- 提供根因分析（行號 + 原因 + 證據）

### ✅ 完整記錄
- 保存所有 PDB 命令文件（可復現）
- 保存所有輸出日誌（可追溯）
- 生成分析報告（可分享）

### ✅ 多輪迭代
- 最多 5 輪調試
- 每輪基於前一輪的發現調整策略
- 自動終止（找到根因或達到最大輪數）

---

## 完整的 pdb-debugger 工作流程

```
┌─────────────────────────────────────────────────────────┐
│ 用戶：使用 pdb-debugger 調試 example_buggy_script.py   │
└────────────────────┬────────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │ pdb-debugger 啟動     │
         └───────────┬───────────┘
                     ↓
    ╔════════════════════════════════════╗
    ║  Round 1: 初始探索                 ║
    ╠════════════════════════════════════╣
    ║ 1. 讀取源碼                        ║
    ║ 2. 識別關鍵位置（第 18 行除法）   ║
    ║ 3. 創建 pdb_commands_round1.txt   ║
    ║ 4. 執行 PDB                        ║
    ║ 5. 保存 pdb_output_round1.log     ║
    ║ 6. 分析輸出                        ║
    ╚════════════════╤═══════════════════╝
                     ↓
         ┌───────────────────────┐
         │ 分析 Round 1 結果     │
         │                       │
         │ ✅ 發現：空列表時     │
         │    count = 0          │
         │                       │
         │ ⚠️ 異常：除零錯誤     │
         │                       │
         │ 🔍 根因：第 18 行     │
         │    沒有檢查空列表     │
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │ 生成修復建議          │
         │                       │
         │ if not numbers:       │
         │     return 0          │
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │ 生成最終報告          │
         │                       │
         │ - 問題：除零錯誤      │
         │ - 位置：第 18 行      │
         │ - 原因：空列表        │
         │ - 修復：添加檢查      │
         └───────────────────────┘
```

---

## 實際輸出範例

### PDB 命令文件（Round 1）
```bash
# === PDB 示範 Round 1: 初始探索 ===
# 目標：定位錯誤發生的位置

b example_buggy_script.py:18
run
c
p numbers
p total
p count
p count == 0
```

### PDB 輸出（Round 1，部分）
```
> example_buggy_script.py(18)calculate_average()
-> average = total / count
(Pdb) numbers
[]
(Pdb) total
0
(Pdb) count
0
(Pdb) count == 0
True
```

### 分析報告（Round 1）
見 [pdb_demo_analysis_round1.md](pdb_demo_analysis_round1.md)

---

## 對比：手動 vs pdb-debugger

| 方面 | 手動調試 | pdb-debugger Subagent |
|------|----------|----------------------|
| **設置斷點** | 需要手動猜測位置 | 自動識別關鍵位置 |
| **PDB 命令** | 手動編寫 | 自動生成 |
| **執行** | 手動運行 | 自動執行 |
| **分析** | 人工閱讀輸出 | 自動分析異常 |
| **迭代** | 手動調整斷點 | 自動多輪迭代（最多5輪）|
| **記錄** | 可能忘記保存 | 自動保存所有文件 |
| **報告** | 需要手動整理 | 自動生成報告 |
| **時間** | 30-60 分鐘 | 5-10 分鐘 |

---

## 何時使用 pdb-debugger

### ✅ 適合場景
- 訓練停滯、loss 不下降
- NaN/Inf 錯誤
- 梯度消失/爆炸
- 模型 collapse
- 複雜的多步驟問題
- 需要多輪迭代追蹤

### ⚠️ 不適合場景
- 非常簡單的一行錯誤（直接看 traceback 更快）
- 語法錯誤（Python 會直接報錯）
- Import 錯誤（不需要 PDB）

---

## 總結

pdb-debugger subagent 通過自動化多輪 PDB 調試，可以：

1. **快速定位**：自動識別關鍵位置
2. **智能分析**：自動識別異常和問題
3. **可重現**：所有命令和輸出都保存
4. **可追溯**：完整的調試記錄
5. **可分享**：生成完整的報告

**使用方式**：
```
> 使用 pdb-debugger 調試 <你的腳本.py>
```

pdb-debugger 會自動處理剩下的工作！
