# TTT2 修復分支檔案依賴分析報告

**實驗編號:** FILE_DEP_20250814  
**分析時間:** 2025-08-14  
**分支:** fix-ttt2-residual-block-and-manifold  
**分析函式:** analyze_ttt2_file_dependencies  

## 執行 run_fixed_ttt2_branch.sh 所需檔案

### 🔧 核心執行檔案（必需）

#### 主要 Python 檔案
1. **`ttt2.py`** - 主要訓練腳本 ✅ 存在
2. **`test_ttt2_fixes.py`** - 修復驗證測試 ✅ 已創建
3. **`ttdata.py`** - 數據載入模組 ✅ 用戶確認存在

#### 模型相關目錄
4. **`decoder/`** 目錄及其內容 ✅ 存在
   - `__init__.py`
   - `models.py`
   - `modules.py`
   - `loss.py`
   - `dataset.py`
   - 等其他模組

5. **`encoder/`** 目錄及其內容 ✅ 存在
   - `__init__.py`
   - `model.py`
   - `quantization/` 子目錄
   - `modules/` 子目錄

#### 配置檔案
6. **`config/`** 目錄 ✅ 存在
   - `wavtokenizer_*.yaml` 配置檔案

#### 支援模組
7. **`utils/`** 目錄 ✅ 存在
8. **`fairseq/`** 目錄 ✅ 存在
9. **`metrics/`** 目錄 ✅ 存在

### 📁 輸出目錄（執行時創建）
- `logs/` - 日誌輸出目錄
- `results/tsne_outputs/b-output4/` - 訓練結果輸出

### 📝 報告檔案（自動更新）
- `REPORT.md` - 實驗報告文件

## 🗑️ 可能不需要的檔案（可刪除）

### 實驗相關但非此分支必需
1. **`analyze_audio_quality.py`** - 音頻品質分析（非訓練必需）
2. **`analyze_semantic_layer.py`** - 語義層分析（非訓練必需）
3. **`check_feature_shapes.py`** - 特徵形狀檢查（非訓練必需）
4. **`discrete_loss.py`** - 離散損失實驗（可能重複）
5. **`infer.py`** - 推理腳本（非訓練必需）
6. **`inspect_model.py`** - 模型檢查工具（非訓練必需）
7. **`loss_visualization.py`** - 損失視覺化（非訓練必需）
8. **`tsne.py`** - t-SNE 分析（非此分支訓練必需）
9. **`train.py`** - 可能是舊的訓練腳本
10. **`try3.py`** - 實驗性腳本
11. **`time_alignment_fix.py`** - 時間對齊修復（可能已整合）
12. **`update_min_content_samples.py`** - 內容樣本更新工具
13. **`verify_wavtokenizer.py`** - WavTokenizer 驗證工具

### 測試和外部實驗檔案
14. **`test_ttt2_outside.py`** - 外部測試（非此分支）
15. **`run_ttt2_outside_test.sh`** - 外部測試腳本
16. **`run_layer_visualization.sh`** - 層視覺化腳本
17. **`ttt2_outside_test_results/`** - 外部測試結果目錄

### 文檔檔案（保留但非執行必需）
18. **各種 `.md` 文檔檔案** - 保留作為參考，但非執行必需
19. **`docs/`** 目錄內容 - 文檔資料
20. **`b-0813/`** 目錄 - 舊實驗結果

### 圖片和結果檔案
21. **`result.png`** - 舊結果圖片
22. **`wavtokenizer.txt`** - 可能是配置或日誌

## 🔍 檔案狀態檢查結果

### ✅ 已確認存在且必需
- `ttt2.py` - 主訓練腳本
- `ttdata.py` - 數據載入（用戶確認）
- `decoder/` - 解碼器模組
- `encoder/` - 編碼器模組
- `config/` - 配置檔案
- `utils/`, `fairseq/`, `metrics/` - 支援模組

### ✅ 已創建
- `test_ttt2_fixes.py` - 修復驗證測試

### ⚠️ 需要確認的依賴
- PyTorch 和相關套件
- CUDA 支援
- conda 環境 'test'

## 💡 建議操作

### 可以安全刪除（建議備份後刪除）
```bash
# 實驗分析工具（非訓練必需）
rm analyze_audio_quality.py analyze_semantic_layer.py check_feature_shapes.py
rm discrete_loss.py infer.py inspect_model.py loss_visualization.py
rm tsne.py train.py try3.py time_alignment_fix.py
rm update_min_content_samples.py verify_wavtokenizer.py

# 外部測試相關
rm test_ttt2_outside.py run_ttt2_outside_test.sh run_layer_visualization.sh
rm -rf ttt2_outside_test_results/

# 舊結果檔案
rm result.png wavtokenizer.txt
rm -rf b-0813/  # 如果不需要舊實驗結果
```

### 保留但非執行必需
```bash
# 文檔檔案（建議保留）
# *.md 檔案
# docs/ 目錄
```

## 🎯 最小執行環境

最小執行環境只需要：
1. `run_fixed_ttt2_branch.sh`
2. `ttt2.py`
3. `test_ttt2_fixes.py`
4. `ttdata.py`
5. `decoder/` 目錄
6. `encoder/` 目錄
7. `config/` 目錄
8. `utils/`, `fairseq/`, `metrics/` 目錄
9. `requirements.txt` 或 `environment.yml`
10. `REPORT.md`

總計約 10-15 個核心檔案/目錄即可運行此分支的訓練。

---

**註記:** 建議在刪除檔案前先備份，確保不會意外刪除重要檔案。可以先將不需要的檔案移動到 `backup/` 目錄中。
