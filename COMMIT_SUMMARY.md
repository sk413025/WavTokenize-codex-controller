# Git Commit 總結

## 📝 Commit 信息
- **Commit Hash**: f8fa445f306441d0352c127aab20751d8fd60f80
- **分支**: c_code
- **作者**: sbplab
- **日期**: 2025-09-02 04:35:25 -0400
- **標題**: 實驗設計：添加內容一致性損失的連續與離散特徵比較實驗

## 📊 文件變更統計
- **新增文件**: 4 個
- **修改文件**: 1 個
- **總行數變更**: +2121 -20
- **影響文件總數**: 5 個

## 📁 詳細文件變更

### 新增文件 (4個)
1. **EXPERIMENT_GUIDE.md** (+162 行)
   - 完整的實驗運行指南
   - 包含兩種方案的詳細說明
   - 評估指標和故障排除指南

2. **run_experiment_discrete_content.sh** (+121 行)
   - 方案二：純離散內容一致性損失的自動化腳本
   - 包含完整的環境設定和日誌記錄
   - 自動更新實驗報告功能

3. **run_experiment_hierarchical_content.sh** (+117 行)
   - 方案一：階層式內容一致性損失的自動化腳本
   - 包含完整的環境設定和日誌記錄
   - 自動更新實驗報告功能

4. **ttdata.py** (+1396 行)
   - 音頻數據集處理類別
   - 新增 `max_sentences_per_speaker` 參數
   - 支持限制每位與者的句子數量

### 修改文件 (1個)
1. **ttt2.py** (+325 -20 行)
   - 新增實驗損失函數：
     - `compute_discrete_content_consistency_loss()`
     - `compute_hierarchical_content_consistency_loss()`
     - `compute_hybrid_loss_with_discrete_content()`
     - `compute_hybrid_loss_with_hierarchical_content()`
   - 新增實驗參數支持
   - 修改輸出目錄配置以支持實驗模式
   - 更新訓練和驗證循環以支持實驗模式

## 🧪 實驗功能總結

### 核心實驗功能
1. **方案一：階層式內容一致性損失**
   - 結合連續特徵（embedding）和離散特徵（codebook index）
   - 可調節權重比例（預設 0.7:0.3）
   - 目標：平衡韻律保留與語意一致性

2. **方案二：純離散內容一致性損失**
   - 僅使用離散特徵進行內容約束
   - 支援 KL 散度和序列相似度兩種計算方式
   - 目標：強化語意一致性

### 實驗環境配置
- 數據限制：每位與者前100句話
- 材質限制：僅使用 box 材質
- 批次大小：8
- 自動化腳本包含完整環境設定

### 評估與分析
- 自動生成實驗報告
- 包含 t-SNE 可視化
- 支援損失曲線對比
- 提供詳細的評估指標指南

## 🔧 技術實現細節

### 損失函數實現
- **離散特徵損失**：支援機率分布（KL散度）和離散索引（序列相似度）
- **階層式損失**：加權結合連續和離散特徵
- **混合損失**：整合特徵損失和內容一致性損失

### 參數設定
- `--experiment_hierarchical_content`: 啟用方案一
- `--experiment_discrete_content`: 啟用方案二
- `--hierarchy_alpha`: 階層式損失中連續特徵權重
- `--content_alpha`: 內容一致性損失權重

### 輸出管理
- 自動創建實驗專用輸出目錄
- 格式：`exp1-hierarchical-{timestamp}` 和 `exp2-discrete-{timestamp}`
- 包含模型、特徵、可視化和音頻樣本

## 🚀 使用方法

### 快速開始
```bash
# 方案一：階層式內容一致性損失
./run_experiment_hierarchical_content.sh

# 方案二：純離散內容一致性損失
./run_experiment_discrete_content.sh
```

### 參數調整
```bash
# 調整階層權重
python ttt2.py --experiment_hierarchical_content --hierarchy_alpha 0.8

# 調整內容權重
python ttt2.py --experiment_discrete_content --content_alpha 0.05
```

## ✅ 測試驗證
- 所有損失函數已通過測試
- 實驗環境配置完成
- 腳本具備執行權限
- 支援 CUDA 和 CPU 運行

## 📈 預期影響
1. **研究價值**：為內容一致性損失設計提供實證依據
2. **實用性**：兩種方案可直接用於生產環境
3. **可擴展性**：實驗框架可適用於其他損失函數比較
4. **可重現性**：完整的自動化腳本和詳細文檔

---

**此 commit 已準備好進行實驗運行和結果分析。**
