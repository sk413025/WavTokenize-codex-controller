# PDB 示範分析 - Round 1

## 執行時間
2025-11-22

## 檢查目標
- 定位 ZeroDivisionError 的根本原因
- 檢查 `calculate_average` 函數的輸入和狀態

## 斷點位置
- `example_buggy_script.py:18` - 除法操作前

## 關鍵發現

### 第一次斷點（Dataset 0）
```
numbers = [1, 2, 3, 4, 5]
total = 15
count = 5
count == 0 → False ✅
```
**結論**：正常運行

### 第二次斷點（Dataset 1）
```
numbers = [10, 20, 30]
total = 60
count = 3
count == 0 → False ✅
```
**結論**：正常運行

### 第三次斷點（Dataset 2）
```
numbers = []          ← 空列表！
total = 0
count = 0             ← 問題！
count == 0 → True ⚠️
```
**結論**：**找到問題根源！**

## 問題定位

**文件**：`example_buggy_script.py`
**行號**：18
**函數**：`calculate_average()`
**問題**：當輸入為空列表時，`count = 0`，導致除零錯誤

**錯誤代碼**：
```python
average = total / count  # count 為 0 時會出錯
```

## 根本原因

函數 `calculate_average()` 沒有處理空列表的情況。

## 建議修復

### 選項 1：添加檢查（推薦）
```python
def calculate_average(numbers):
    """計算數字列表的平均值"""
    if not numbers:  # 檢查空列表
        return 0  # 或者 raise ValueError("Cannot calculate average of empty list")

    total = 0
    count = 0

    for num in numbers:
        total += num
        count += 1

    average = total / count
    return average
```

### 選項 2：使用 try-except
```python
def calculate_average(numbers):
    """計算數字列表的平均值"""
    total = 0
    count = 0

    for num in numbers:
        total += num
        count += 1

    try:
        average = total / count
    except ZeroDivisionError:
        return 0  # 或者拋出更明確的錯誤

    return average
```

### 選項 3：使用 Python 內建函數
```python
def calculate_average(numbers):
    """計算數字列表的平均值"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
```

## 是否需要繼續調試？

- [x] **否，已找到根因**
- [ ] 是，需要進一步調查

**原因**：通過 Round 1 的 PDB 調試，我們明確定位了問題：
1. 空列表導致 count = 0
2. 除以 0 觸發 ZeroDivisionError
3. 位置：第 18 行

## 下一步

實施修復（建議選項 3），然後重新測試。
