# Token 序列分析與專案清理 - 20250904

## 實驗背景
為了深入理解 WavTokenizer 如何編碼不同音檔的特徵，建立了 token 序列分析工具，並清理了過時的實驗檔案。

## 實驗動機與目的
- **動機**: 需要驗證 WavTokenizer 對不同說話者、內容、背景噪音的 token 編碼差異
- **目的**: 建立自動化工具來分析音檔的離散 token 序列，以支援後續的解離性 (disentanglement) 研究

## 主要變更

### 1. 新增 Token 序列分析工具
- **新增檔案**: `compare_token_sequences.py`
- **功能**: 批次處理 ./test 資料夾下所有 .wav 檔，輸出每個音檔的 token 序列
- **輸出格式**: 
  - `.npy` 檔案：供程式進一步分析使用
  - `.txt` 檔案：供人工檢視，以逗號分隔
- **技術特點**:
  - 使用正確的 WavTokenizer.from_pretrained0802() 初始化方式
  - 自動轉換音檔格式（24kHz, 單聲道）
  - 提供詳細的 token shape、獨特種類、分布統計

### 2. 實驗音檔與結果
- **新增**: test/ 資料夾包含 9 個測試音檔（不同說話者、材質組合）
- **新增**: test/out/ 包含對應的 token 序列分析結果
- **涵蓋條件**:
  - 說話者: nor_boy1, nor_girl1
  - 材質: box, clean, mac
  - 背景: LDV (噪音) vs clean (乾淨)

### 3. 專案清理
- **刪除**: backup_202508140339/ 整個備份目錄
- **刪除**: 過時的實驗報告和分析文件
- **保留**: 核心程式碼和當前實驗結果

### 4. 輔助腳本
- **新增**: `quick_test_fix.sh` - 快速測試修復腳本
- **新增**: `run_content_comparison_experiments.sh` - 內容比較實驗腳本

## 實際執行結果
透過 `compare_token_sequences.py` 成功分析了 9 個測試音檔，每個音檔都產生了對應的 token 序列檔案，為後續的解離性分析奠定基礎。

## 解讀實驗結果
- Token 序列成功提取，格式正確
- 不同音檔的 token 分布顯示出預期的差異性
- 工具運行穩定，支援批次處理

## 根據實驗結果的下一步計劃
1. 分析不同條件下的 token 分布差異
2. 量化說話者、內容、噪音對 token 選擇的影響
3. 建立 token 序列的統計分析方法
4. 整合到主要的解離性實驗流程中

## 實驗重現步驟
1. 準備測試音檔放置於 ./test/ 目錄
2. 執行 `python compare_token_sequences.py`
3. 檢查 ./test/out/ 目錄下的 token 序列結果
4. 使用 .txt 檔案進行人工檢視，.npy 檔案進行程式分析

## 檔案結構變更
```
新增:
├── compare_token_sequences.py
├── quick_test_fix.sh  
├── run_content_comparison_experiments.sh
├── test/
│   ├── 9個測試音檔.wav
│   └── out/
│       ├── 18個.npy token檔案
│       └── 18個.txt token檔案

修改:
├── REPORT.md (更新實驗記錄)
├── ttt2.py (程式碼優化)
└── ttt2_out1/ (實驗結果更新)

刪除:
├── backup_202508140339/ (整個備份目錄)
└── 多個過時的實驗文件
```

此次提交為 WavTokenizer 的 token 序列分析建立了完整的工具鏈，並清理了專案結構，為後續深入的解離性研究做好準備。
