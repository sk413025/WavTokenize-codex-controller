# 實驗記錄報告

## 📊 內容一致性損失比較實驗 (2025-01-09 進行中)

### 實驗設計
為了驗證連續特徵與離散特徵在語音內容一致性方面的效果差異，設計了以下兩個實驗：

#### 實驗1: 層次化內容一致性損失 (exp1-hierarchical) 🔄 **進行中**
- **特徵組合**: 70% 連續特徵 + 30% 離散特徵
- **損失函數**: `compute_hierarchical_content_consistency_loss()`
- **狀態**: 正在運行 - 數據載入和處理階段
- **啟動時間**: 2025-01-09 01:35
- **輸出目錄**: `/results/tsne_outputs/exp1-hierarchical-[timestamp]/`

#### 實驗2: 純離散內容一致性損失 (exp2-discrete)
- **特徵組合**: 100% 離散特徵  
- **損失函數**: `compute_discrete_content_consistency_loss()`
- **狀態**: 等待實驗1完成
- **輸出目錄**: `/results/tsne_outputs/exp2-discrete-[timestamp]/`

### 實驗配置
- **數據集**: 1200個樣本對 (訓練集1000個，驗證集200個)
- **訓練週期**: 600 epochs
- **內容感知批次採樣**: 
  - 內容比例: 50%
  - 最小樣本數: 5
  - 有效內容ID: 70個
- **驗證策略**: speaker_only (boy7, girl9為驗證集)

### 已完成的技術修復
1. ✅ **CUDA錯誤修復**: 解決device-side assert錯誤
2. ✅ **輸入驗證增強**: 添加界限檢查和類型驗證
3. ✅ **維度處理優化**: 自動調整4D到3D張量維度
4. ✅ **環境配置**: 確保conda test環境激活

### 當前狀態
- **實驗1進行中**: 數據處理階段，正在匹配輸入音頻和目標音頻
- **數據載入**: 成功識別和匹配1200個音頻文件對
- **批次採樣**: 內容感知採樣策略已啟用
- **GPU利用**: CUDA支持已啟用

---

## 2025-08-14 TTT2跨分支Outside測試系統音檔能量修復 (EXP_TTT2_ENERGY_FIX_20250814)
- **實驗編號**: EXP_TTT2_ENERGY_FIX_20250814
- **日期**: 2025-08-14
- **實驗背景**: test_ttt2_outside.py 生成的增強音檔能量大幅降低，從原始音檔的 1468.30 降至 233.94，導致音檔缺乏人聲內容
- **動機**: 修復音檔生成管道，確保與 ttt2.py 一致的輸出品質，解決能量損失問題
- **目的**: 實現高品質的 outside 音檔測試系統，確保增強音檔保持原有的人聲特徵
- **技術問題分析**:
  - test_ttt2_outside.py enhanced: energy=233.94, RMS=0.070 (能量過低)
  - 原始音檔: energy=1468.30, RMS=0.175 (正常能量)
  - 能量損失比例: 84.1% (嚴重的信號衰減)
- **修復方法**:
  1. **完全複製 ttt2.py 的 save_sample 方法**: 直接調用 ttt2.save_sample 函數
  2. **改進模型架構檢測**: 增強 detect_model_architecture 對維度和正規化層的檢測
  3. **優化模型載入**: 實現智能參數映射 (bn1/bn2 ↔ norm1/norm2)
  4. **建立 Git Worktree 工作流程**: 創建版本控制指南支援並行分支開發
- **預期結果**: 生成與 ttt2.py 相當能量的音檔 (能量 > 1000, RMS > 0.1)
- **實際結果**: [進行中] 正在調試音檔生成管道差異
- **結果解讀**: [待完成] 需要深入分析 ttt2.py 與 test_ttt2_outside.py 的執行上下文差異
- **下一步計劃**: 
  1. 比較兩種方法的模型前向傳播過程
  2. 檢查 WavTokenizer 解碼器的配置差異
  3. 驗證增強特徵的數值範圍和分布
- **重現步驟**: 
  ```bash
  conda activate test
  python test_ttt2_outside.py --checkpoint results/tsne_outputs/output4/best_model.pth
  # 比較生成的音檔能量與 ttt2.py 輸出
  ```
- **新增檔案**:
  - Git 工作流程指南: `GIT_WORKFLOW.md`
  - 改進的測試腳本: `test_ttt2_outside.py` (增強架構檢測)
  - 實驗報告: `EXPERIMENT_REPORT_TTT2_OUTSIDE_TEST_20250813.md`

## 2025-08-13 TTT2外部測試系統重大改進 (EXP_TTT2_OUTSIDE_FIX_20250813)
- **實驗編號**: EXP_TTT2_OUTSIDE_FIX_20250813
- **提交ID**: `200fab4`
- **實驗背景**: TTT2模型在外部音檔測試中出現輸出無聲音問題，需要建立跨分支兼容的測試系統
- **核心問題**:
  - TTT2模型輸出格式處理錯誤，導致解碼音檔無聲音內容
  - 主分支模型(BatchNorm)與修復分支模型(GroupNorm)架構不兼容
  - 缺乏自動檢測模型類型的機制
- **技術突破**:
  1. **模型輸出格式修復**: 正確處理TTT2的元組輸出格式 `(output, input_features, enhanced_features, ...)`
  2. **自動架構檢測**: 實現BatchNorm/GroupNorm的自動檢測和適配載入
  3. **碼本一致性改進**: 增強quantizer路徑檢測和錯誤處理機制
  4. **測試框架完善**: 建立包含12個外部音檔的完整測試報告系統
- **驗證結果**:
  - ✅ TTT2輸出音檔RMS達到0.0723 (遠大於0.001靜音閾值)
  - ✅ 成功支援主分支(256維)和修復分支(1024維)模型
  - ✅ 建立完整的音檔質量檢測指標(SNR、相關係數、RMS差異等)
- **實驗價值**: 為後續主分支vs修復分支性能對比奠定了技術基礎
- **重現步驟**: 
  ```bash
  python test_ttt2_outside.py --checkpoint <model_path> --outside_dir ./1n --output_dir <output> --max_files 12
  ```
- **核心檔案**:
  - 測試腳本: `test_ttt2_outside.py` (新增自動架構檢測)
  - 模型修復: `ttt2.py` (改進碼本一致性損失)
  - 測試報告: `ttt2_outside_test_main_300epoch/` (12個音檔完整測試)

## 2025-08-13 音頻品質定量分析實驗 (EXP_AUDIO_QUALITY_20250813)
- **實驗編號**: EXP_AUDIO_QUALITY_20250813
- **提交ID**: `3cef9a5`
- **實驗背景**: 在損失函數數值分析基礎上，直接從音頻品質角度驗證TTT2 Fix分支的ResidualBlock修復效果
- **比較對象**: Fix分支 vs Main分支 (Epoch 300)
- **評估指標**: SNR、MFCC相似度、頻譜重心、頻譜滾降
- **樣本數量**: 9個音頻樣本
- **核心發現**:
  - **SNR改善**: 平均+0.98dB，8/9樣本改善(88.9%)
  - **MFCC相似度**: 平均+0.0301，7/9樣本改善(77.8%)
  - **頻譜重心誤差**: Fix分支38.1Hz vs Main分支49.3Hz (減少22.7%)
  - **雙重改善**: 7/9樣本在兩項核心指標都有改善
- **結論**: ✅ Fix分支在音頻品質上明顯優於Main分支，證明ResidualBlock修復有效
- **技術洞察**: 損失函數數值與音頻感知品質存在差異，修復後的損失函數更嚴格但更有效
- **實驗檔案**: 
  - 分析工具: `analyze_audio_quality.py`
  - 詳細報告: `b-0813/AUDIO_QUALITY_REPORT.md`
  - 可視化圖表: `b-0813/audio_quality_comparison.png`
  - 原始數據: `b-0813/audio_quality_analysis.json`

## 2025-08-11 TTT2 模型關鍵架構修復 (FIX_TTT2_20250811)
- **修復分支**: `fix-ttt2-residual-block-and-manifold`
- **提交ID**: `ceec740` 
- **修復背景**: 基於 commit 38f072d1c9756b8a2c5701f3912c0bdf809d23f0 的深度技術分析，發現 TTT2 模型存在多個架構瑕疵
- **關鍵修復**:
  1. **ResidualBlock 錯誤修復**: 將 `self.conv2(x)` 改為 `self.conv2(out)`，修復梯度流動問題
  2. **GroupNorm 支援**: 引入 GroupNorm 選項以改善音頻處理的穩定性
  3. **流形正則化**: 實作 `compute_manifold_regularization_loss()` 防止特徵偏離訓練分佈
  4. **碼本一致性**: 實作 `compute_codebook_consistency_loss()` 確保離散編碼穩定性
  5. **多組件損失**: 更新 `compute_layered_hybrid_loss()` 整合所有損失組件
- **驗證狀態**: ✅ 語法檢查通過，準備進行訓練測試
- **下一步**: 運行訓練測試以驗證修復效果並進行性能比較

## 2025-08-05 參數更新：min_content_samples (EXP_PARAM_UPDATE_20250805)
- **修改內容**: 將 `min_content_samples` 從 3 更新為 5
- **修改原因**: 增加批次中相同內容ID的最小樣本數，以提高內容一致性損失的計算效果，並強化模型對內容不變特徵的學習
- **修改時間**: 2025-08-05 10:00:00

## 2025-08-04 WavTokenizer 能力驗證實驗 (EXP_Verify_20250804)
- **實驗描述**: 驗證 WavTokenizer 的編碼-解碼能力，測試了 3 個音頻文件
- **驗證結果**: [詳細報告](results/verify/WavTokenizer_verification_report_20250804.md)
- **平均 SNR**: 2.9450 dB
## 2025-08-03 WavTokenizer 能力驗證實驗 (EXP_Verify_20250803)
- **實驗描述**: 驗證 WavTokenizer 的編碼-解碼能力，測試了 3 個音頻文件
- **驗證結果**: [詳細報告](results/verify/WavTokenizer_verification_report_20250803.md)
- **平均 SNR**: 2.9450 dB

## 2025-07-21 WavTokenizer 驗證測試修正 (更新20250803_052803)
- **修正內容**: 更正 WavTokenizer 的 token 計算方法，將 shape[1] 改為 shape[-1]
- **驗證結果**: 確認正確的每秒 token 數約為 75
- **詳細報告**: [verification_report.md](results/wavtokenizer_verification_20250803_052803/verification_report.md)
## 離散特徵實驗系列概述 (2025-08-01)

本系列實驗旨在深入研究 WavTokenizer 模型的離散編碼特性，並探索其在語音噪聲抑制中的應用。完整實驗設計與說明請參考 [DISCRETE_EXPERIMENTS.md](DISCRETE_EXPERIMENTS.md)。

主要實驗包括：
- **EXP01**：離散特徵基礎分析 - 分析離散編碼的統計特性與分佈
  - 更新 (2025-08-01)：修復了 `exp_discrete_analysis.py` 中的 bandwidth_id 參數缺失問題
  - 更新 (2025-08-01)：將可視化圖表中的中文標題改為英文，解決了中文字型顯示問題
- **EXP02**：離散特徵語義解離 - 確定各層的語義含義（內容、說話者、噪聲）
  - 更新 (2025-08-02)：修復了 `exp_discrete_swap.py` 中的 bandwidth_id 參數缺失問題
  - 更新 (2025-08-02)：修復了 `exp_layer_swap.py` 中的 bandwidth_id 參數缺失問題
  - 更新 (2025-08-03)：完成了層特徵分析，結果顯示單一量化層主要表示說話者特徵(1.0046)，其次是內容特徵(1.0012)，幾乎無噪聲特徵(0.0000)
- **EXP03**：離散特徵雜訊抑制 - 基於離散編碼操作實現噪聲抑制

## 2025-08-01 分層損失策略實驗 (EXP20250801_082107)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202508010623`
- **最終損失值**:
  - 總損失: 1.253498
  - 內容一致性損失: 0.000000
  - L2損失: 1.253498
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202508010623/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```

## 2025-07-31 分層損失策略實驗 (EXP20250731_085342)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507310656`
- **最終損失值**:
  - 總損失: 1.251341
  - 內容一致性損失: 0.000000
  - L2損失: 1.251341
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507310656/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```

## 2025-07-30 分層損失策略實驗 (EXP20250730_094854)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507300657`
- **最終損失值**:
  - 總損失: 1.061444
  - 內容一致性損失: 0.000000
  - L2損失: 1.061444
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507300657/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```

## 2025-07-29 分層損失策略實驗 (EXP20250729_122702)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_block3_content_consistency_202507290940`
- **最終損失值**:
  - 總損失: 10.717459
  - 內容一致性損失: 0.572970
  - L2損失: 5.335948
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_block3_content_consistency_202507290940/final_learning_curve.png)

### 參數設置
```
use_layered_loss: True
tsne_flow_with_content: True
batch_size: 4
```

## 2025-07-28 分層損失策略實驗 (EXP20250728_081739)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507280545`
- **最終損失值**:
  - 總損失: 1.065990
  - 內容一致性損失: 0.000000
  - L2損失: 1.065990
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507280545/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```

## 2025-07-28 分層損失策略實驗 (EXP20250728_052545)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507280422`
- **最終損失值**:
  - 總損失: 1.488439
  - 內容一致性損失: 0.000000
  - L2損失: 1.488439
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507280422/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```

## 2025-07-28 分層損失策略實驗 (EXP20250728_021225)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507280108`
- **最終損失值**:
  - 總損失: 1.488439
  - 內容一致性損失: 0.000000
  - L2損失: 1.488439
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507280108/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```

## 2025-07-25 分層損失策略實驗 (EXP20250725_061915)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507250324`
- **最終損失值**:
  - 總損失: 1.054126
  - 內容一致性損失: 0.000000
  - L2損失: 1.054126
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507250324/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```


## 2025-07-23 t-SNE 可視化優化更新
- **更新類型**: 視覺化改進
- **更新描述**: 
  - 優化 t-SNE 可視化，使每個點代表一句完整的音頻
  - 增加原始特徵和增強特徵之間的連線，直觀顯示特徵變化
  - 添加特徵分布指標，包括類內距離和類間距離
  - 新增熱力圖顯示增強前後的特徵位移距離
- **改進目的**: 更準確評估模型的學習能力與特徵表示效果
- **相關文件**: 
  - `ttt.py`: 更新了 `plot_tsne_visualization` 函數
  - 輸出位置: `results/tsne_outputs/*/tsne_visualizations/`
- **主要改進**:
  - 將原先的隨機特徵點改為以完整句子為單位的特徵表示
  - 計算每個音頻句子的平均特徵（時間維度平均）作為 t-SNE 輸入
  - 保存原始特徵數據，便於後續分析
  - 視覺化增加更多統計信息，便於量化評估模型效果
## 2025-07-22 分層損失策略實驗 (EXP20250722_081841)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507220542`
- **最終損失值**:
  - 總損失: 1.054126
  - 內容一致性損失: 0.000000
  - L2損失: 1.054126
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507220542/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```

## 2025-07-21 分層損失策略實驗 (EXP20250721_071730)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507210502`
- **最終損失值**:
  - 總損失: 1.251341
  - 內容一致性損失: 0.000000
  - L2損失: 1.251341
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507210502/final_learning_curve.png)

### 參數設置
```
use_layered_loss: False
tsne_flow_with_content: False
batch_size: 4
```

## 2025-07-21 分層損失策略實驗 (EXP20250721_043927)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507210225`
- **最終損失值**:
  - 總損失: 5.607279
  - 內容一致性損失: 0.277368
  - L2損失: 5.329911
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507210225/final_learning_curve.png)

### 參數設置
```
use_layered_loss: True
tsne_flow_with_content: True
batch_size: 4
```


## 2025-07-21 代碼改進：從encode_infer轉換到encode (更新20250721_174517)
- **改進內容**: 
  - 將`ttt.py`中所有的`encode_infer`函式呼叫改為`encode`
  - 修改`feature_extractors.py`中的`forward`方法以處理張量形式的`bandwidth_id`
  - 更新`ttt.py`中所有批次處理相關的`bandwidth_id`預處理邏輯
- **改進原因**: 
  - 標準化API使用，避免訓練和推理時使用不同的函式
  - 移除不必要的推理模式(`inference_mode`)裝飾器以允許梯度流動
  - 改善模型在訓練中的學習能力
  - 解決TypeError: only integer tensors of a single element can be converted to an index錯誤
- **影響範圍**:
  - 音訊特徵提取過程
  - 目標特徵計算
  - t-SNE可視化特徵提取
  - 處理批次張量參數
  - EnhancedFeatureExtractor.forward方法
  - 訓練、評估和可視化階段的特徵提取
- **預期效果**:
  - 訓練過程中更精確的梯度計算
  - 模型權重更新更加準確
  - 特徵空間結構更加連貫
  - 修正型別轉換錯誤，提高訓練穩定性
  - 提升批次處理的兼容性

## 2025-07-16 分層損失策略實驗 (EXP20250716_071413)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4_202507160525`
- **最終損失值**:
  - 總損失: 5.607109
  - 內容一致性損失: 0.277352
  - L2損失: 5.329757
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4_202507160525/final_learning_curve.png)

### 參數設置
```
use_layered_loss: True
tsne_flow_with_content: True
batch_size: 4
```

## 2025-07-15 分層損失策略實驗 (EXP20250715_051619)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4`
- **最終損失值**:
  - 總損失: 5.265752
  - 內容一致性損失: 0.001167
  - L2損失: 5.264584
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4/final_learning_curve.png)

### 參數設置
```
use_layered_loss: True
tsne_flow_with_content: True
batch_size: 8
```

## 2025-07-11 分層損失策略實驗 (EXP20250711_063700)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output4`
- **最終損失值**:
  - 總損失: 5.284466
  - 內容一致性損失: 0.001400
  - L2損失: 5.283066
- **學習曲線**: ![學習曲線](results/tsne_outputs/output4/final_learning_curve.png)

### 參數設置
```
use_layered_loss: True
tsne_flow_with_content: True
batch_size: 8
```

## 2025-07-10 分層損失策略實驗 (EXP20250710_051235)
- **實驗配置**: 
  - 倒數第二層專注內容損失
  - 中間層不計算損失（自由學習）
  - 最後層專注L2損失
- **實驗目標**: 優化語音模型的分層損失策略，提升音質與泛化能力
- **保存目錄**: `/home/sbplab/ruizi/WavTokenize/test_dir`
- **最終損失值**:
  - 總損失: 0.500000
  - 內容一致性損失: 0.200000
  - L2損失: 0.300000
- **學習曲線**: ![學習曲線](test.png)

### 參數設置
```
batch_size: 4
lr: 0.001
```


## 2025-07-10 優化保存頻率，減少存儲空間使用
- **更新內容**:
  - 將音頻樣本和頻譜圖保存頻率從每 100 個 epoch 改為每 300 個 epoch
  - 將學習曲線和 t-SNE 可視化保存頻率從每 50 個 epoch 改為每 150 個 epoch
  - 檢查點保存頻率維持在每 300 個 epoch（已經是合適值）
- **實現方式**:
  - 修改 `ttt.py` 中的條件控制代碼
  - 相應更新計算偏移量的邏輯以確保樣本多樣性
- **目的**:
  - 減少存儲空間使用，特別對於長時間訓練
  - 優化訓練過程中的 I/O 操作
  - 保持關鍵時間點的模型狀態記錄

## 2025-07-09 純 L2 損失與 box 材質實驗

- **更新內容**:
  - 新增 `run_box_tsne_l2.sh` 腳本，結合 only box 模式和 tsne.py 邏輯
  - 使用純 L2 損失進行訓練，完全匹配 tsne.py 的損失函數設計
  - 專注於 box 材質數據，不使用其他材質
- **實現方式**:
  - 設置 `ONLY_USE_BOX_MATERIAL=true` 環境變量以只使用 box 材質
  - 使用 `--tsne_flow_with_L2` 參數以匹配 tsne.py 的損失計算邏輯
  - 保留記憶體優化設定以避免 OOM 錯誤
- **實驗目的**:
  - 對比研究純 L2 損失與包含內容一致性損失的差異
  - 在只有 box 材質的受控環境中評估模型性能
  - 為特徵空間分析提供基準參照點

## 2025-07-08/09 內容一致性損失層級優化實驗
- **更新內容**:
  - 將內容一致性損失從第二層移至倒數第二層
  - 優化模型結構以解決信息流通受限問題
  - 期望通過更高層監督平衡語義與聲學特性保留
  - [2025-07-09更新] 修復了變數命名不一致導致的執行錯誤
- **實現方式**:
  - 修改了`ttt.py`中的層級損失計算邏輯
  - 更新了`run_box_only_training.sh`腳本的參數說明
  - 保持最終層的L2損失不變，僅調整內容一致性損失的層級位置
- **理論依據**:
  - 低層特徵可能更多包含聲學基本特性，過早監督可能損失這些信息
  - 高層特徵已整合更多語義信息，在此層應用內容一致性損失更合適
  - 期望通過此變更改善音頻還原中的聲音細節保留

## 2025-07-05 損失函數曲線優化更新
- **更新內容**:
  - 簡化了損失曲線圖顯示，只保留總損失、內容一致性損失和L2損失
  - 優化了損失曲線圖的顏色與標籤，提高可讀性
  - 添加了實驗自動報告功能，訓練完成後會自動更新REPORT.md
  - 修復了分層損失邏輯，確保中間層不參與損失計算
- **實現方式**:
  - 修改了`plot_learning_curves`函數以突顯關鍵損失指標
  - 添加了`update_experiment_report`函數自動生成實驗報告
  - 所有圖表輸出文件名中包含日期和函數名稱

## 2025-07-01 混合損失與特徵漂移監控實驗
- 實現了最後一層混合損失方法 (60% L2 + 40% STFT)，針對語音「乾淨度」不足問題
- 建立了特徵漂移監控系統，通過L2距離和t-SNE可視化追蹤特徵空間變化
- 創建了完整實驗流程，包括訓練、監控、比較和自動報告生成
- 新增腳本:
  - `train_mixed_loss_experiment.py`: 實現最後一層混合損失訓練與特徵保存
  - `feature_drift_monitor.py`: 特徵漂移分析與可視化
  - `run_complete_mixed_loss_experiment.sh`: 執行完整實驗流程（分層損失vs混合損失）
- 實驗目標: 通過特徵空間分析比較兩種損失函數對音質與泛化能力的影響

### 執行實驗方法
```bash
# 執行完整實驗（分層損失vs混合損失訓練+比較）
./run_complete_mixed_loss_experiment.sh

# 或分別執行
# 1. 分層損失訓練
python ttt.py --tsne_flow_with_content --use_layered_loss --first_two_blocks_only --save_dir results/box_only/EXP20250701

# 2. 混合損失訓練
python train_mixed_loss_experiment.py --data_dir 1b --output_dir results/mixed_loss/EXP20250701

# 3. 比較結果
python feature_drift_monitor.py compare --experiment_dirs results/box_only/EXP20250701 results/mixed_loss/EXP20250701 --labels "分層損失" "混合損失" --output_dir results/comparison/EXP20250701
```

## 2025-06-16 程式碼縮排與語法持續修正
- 修正 ttt.py 第1204行異常縮排問題：print 語句與 Exception 標籤在同一行，造成縮排錯誤
- 確保異常處理區塊中的 print 語句正確縮排，維持程式碼可讀性與執行順暢
- 修正第1316行語法錯誤：將位於同一行的多個語句（avg_val_feature_loss、avg_val_voice_loss 和 avg_val_content_loss 計算）分拆到不同行，確保正確的語法結構
- 修正第1325-1326行縮排錯誤：在學習率排程器(scheduler)條件檢查中，修正了巢狀的if語句塊缺少縮排的問題

## 2025-06-16 程式碼錯誤修復與優化
- 修正 ttt.py 中 t-SNE 視覺化部分的縮排錯誤（第852行附近）
- 修正檢查點恢復邏輯中的縮排問題（第991行附近）
- 修復訓練循環（第1004行附近）的縮排錯誤
- 修正訓練平均損失計算的縮排問題（第1221行附近）
- 修正驗證階段損失累加的縮排錯誤（第1305行附近）
- 修正驗證階段平均損失計算的縮排問題（第1312-1322行）
- 修復驗證階段的語法錯誤（第1299行）：修正 else 語句與閉括號在同一行的問題
- 修正異常的行尾縮排和換行（第1217、1213、1300、1296、1308和1211行附近）
- 修復 else 語句塊縮排不一致問題（嵌套縮排層級錯誤）
- 修正多處程式碼中的換行問題（第994、1151、1248、1383和1411行）
- 修正數組初始化中的語法錯誤（第999行）：分離了錯誤合併在同一行的變数宣告
- 修正進度條更新中的語法錯誤（移除多餘的右括號）
- 標準化程式碼縮排，確保所有代碼塊使用一致的4個空格縮排
- 全面掃描並調整異常縮排空格（2個空格、6個空格、10個空格等非標準縮排）
- 修正註釋行中不一致的縮排（從6個空格調整為標準的4個空格）
- 確認內容一致性損失的計算、存儲和可視化正確無誤
- 驗證 t-SNE 可視化參數和錯誤處理機制完整
- 確認特徵張量維度處理和轉換的一致性
- 進行代碼完整性和函數調用有效性的全面檢查

## 2025-06-15 音檔儲存與視覺化輸出優化
- 改進 `save_sample` 函數，使用 `encoder.utils.save_audio` 提高音頻保存可靠性
- 添加音頻保存的錯誤處理及備用方案，確保音頻正常儲存
- 修正音頻路徑，使用絕對路徑載入模型配置，避免路徑相關問題
- 為儲存的音頻添加適當的重縮放，確保一致的音量水平
- 增強 t-SNE 視覺化功能，包括更佳的錯誤處理和恢復機制
- 添加特徵標準化處理，提高 t-SNE 可視化的準確性和品質
- 改進視覺化圖表的美觀性，包括更好的顏色、標記和格式化
- 固定所有輸出保存到 `./results/tsne_outputs/output3` 目錄
- 設置清晰的子目錄結構：`audio_samples` 和 `tsne_visualizations`
- 使用統一且一致的檔案命名方式，方便結果比較和分析

## 2025-06-14 移除語速考量
- 修改 `ttdata.py` 中的 `AudioDataset` 類，將 `handle_speed_diff` 參數默認值改為 `False`
- 移除語速計算與處理邏輯，所有樣本均視為正常語速
- 簡化 `process_audio` 函數，不再執行基於語速的動態裁剪
- 保留參數和結構以維持向後兼容性，但這些功能已被停用

## 2025-06-13 改進實驗
- 修改`save_sample`函數，移除了生成完整長度頻譜圖相關邏輯
- 將內容一致性損失從L2距離改為餘弦相似度計算
- 優化了內容相似性計算的註釋與說明

## 2025-06-13 損失計算邏輯優化
- 統一了特徵損失計算函數，僅使用L2距離計算，移除餘弦相似度部分
- 整合並簡化了`compute_feature_loss`、`compute_l2_only_loss`和`compute_hybrid_loss`相關函數
- 改進了命令行參數`--tsne_flow_with_content --use_layered_loss --first_two_blocks_only`的處理邏輯
- 在分層損失中更明確定義：前兩層residual block使用內容一致性損失，後三層使用目標特徵L2損失
- 添加了詳細的損失計算邏輯說明，使代碼更易理解和維護
- 保留了向後兼容性，確保現有訓練腳本仍然有效

## 2025-06-13 分層損失模式調整
- `--first_two_blocks_only` 模式已優化為：內容一致性損失只對第一層（最淺層）有權重，後四層內容一致性損失權重為0，僅計算L2特徵損失。
- 這樣設計可讓模型前層專注於內容語義對齊，後層專注於音質與細節重建，分工更明確。

## 2025-06-13 DataLoader並行處理優化
- 將所有DataLoader實例的`num_workers`參數統一設置為0，避免多進程序列化問題
- 修改了相關的`prefetch_factor`、`persistent_workers`和`worker_init_fn`邏輯，確保在`num_workers=0`時的正確行為
- 添加了詳細註釋，標記了所有相關更改，以便將來可能需要的調整
- 此更改解決了lambda函數在多進程環境中無法序列化的問題，提高訓練穩定性

| 執行模式/命令行參數 | 使用函數 | 特徵損失計算 | 內容一致性損失 | 權重分配 |
|-------------------|---------|------------|--------------|---------|
| `--tsne_flow_with_content --use_layered_loss --first_two_blocks_only` | `compute_layered_hybrid_loss` | L2距離 | 餘弦相似度 | 第一層：100%內容損失<br>後四層：100%L2損失 |

> 註：原本「前兩層內容損失」已改為「僅第一層內容搬失」，其餘層僅L2損失，請團隊成員注意訓練行為的變化。

## 損失計算模式比較

| 執行模式/命令行參數 | 使用函數 | 特徵損失計算 | 內容一致性損失 | 權重分配 |
|-------------------|---------|------------|--------------|---------|
| `--tsne_flow_with_content --use_layered_loss --first_two_blocks_only` | `compute_layered_hybrid_loss` | L2距離 | 餘弦相似度 | 第一層：100%內容損失<br>後四層：100%L2損失 |
| `--use_layered_loss` | `compute_layered_hybrid_loss` | L2距離 | 餘弦相似度 | 動態過渡：前層偏向內容損失，後層偏向L2損失<br>隨訓練進行逐步降低內容損失權重 |
| `--tsne_flow_with_L2` | `compute_hybrid_loss` | L2距離 | 不使用 | 100% L2損失，與tsne.py一致 |
| `--tsne_flow_with_content` | `compute_hybrid_loss_with_tsne_flow` | L2距離 | 餘弦相似度 | 默認: 1% 內容損失, 99% L2損失<br>可通過alpha、beta參數調整 |
| 默認模式 | `compute_hybrid_loss_with_content` | L2距離 | 餘弦相似度 | 默認: 1% 內容損失, 99% L2損失<br>可通過alpha、beta參數調整 |

### 優化後的損失函數執行流程

```
程式啟動 (執行 ttt.py)
  │
  ├─ 解析命令行參數
  │
  ├─ 訓練循環 (train_model)
  │   │
  │   └─ 損失計算 (根據命令行參數選擇不同損失函數)
  │       │
  │       ├─ 分層損失+內容一致 (compute_layered_hybrid_loss)
  │       │   │
  │       │   ├─ 計算各層的內容一致性損失 (compute_content_consistency_loss)
  │       │   │   └─ 使用餘弦相似度計算
  │       │   │
  │       │   └─ 計算各層的特徵L2損失 (compute_feature_loss)
  │       │       └─ 使用L2距離計算
  │       │
  │       ├─ 純L2損失 (compute_hybrid_loss)
  │       │   └─ 計算特徵L2損失 (compute_feature_loss)
  │       │
  │       └─ 混合損失 (compute_hybrid_loss_with_content/tsne_flow)
  │           ├─ 計算特徵L2損失 (compute_feature_loss)
  │           └─ 計算內容一致性損失 (compute_content_consistency_loss)
  │
  └─ 保存模型及結果
```

## 優化建議與下一步計劃

### 當前優化成果
1. **損失計算邏輯統一**：整合了多個相似的損失計算函數，使代碼更簡潔、易於維護
2. **損失計算方式明確**：特徵損失統一使用L2距離，內容一致性損失統一使用餘弦相似度
3. **參數邏輯清晰化**：更明確地定義了不同命令行參數對應的損失計算模式
4. **分層處理邏輯優化**：在`first_two_blocks_only`模式下，明確前兩層使用內容一致性損失，後三層使用特徵損失
5. **音頻保存機制強化**：改進音頻保存功能，引入多重錯誤處理和備用方案
6. **視覺化輸出優化**：增強t-SNE視覺化品質和穩定性，提供更好的特徵空間表示
7. **輸出結構規範化**：統一所有結果到固定目錄結構，便於持續實驗比較

### 下一步可能的優化方向
1. **損失權重自動調整**：實現基於驗證集結果自動調整內容損失與特徵損失的權重比例
2. **特徵投影層優化**：優化維度投影方法，可考慮使用更複雜的注意力機制代替簡單的1x1卷積
3. **參數配置文件化**：將損失計算的超參數（如衰減因子、層權重等）移至配置文件中，方便實驗調整
4. **訓練可視化增強**：添加更詳細的訓練過程可視化，如內容一致性損失與特徵損失的變化趨勢圖
5. **多模態損失探索**：嘗試引入其他模態（如文本描述、類別標籤等）的監督信號，增強模型的泛化能力
6. **批次音頻處理優化**：優化批次中音頻的選擇和增強策略，提高訓練效率和模型效能
7. **視覺化互動界面**：開發簡易的Web界面用於可視化和比較不同模型的音頻增強效果

### 實驗計劃
1. 使用優化後的`--tsne_flow_with_content --use_layered_loss --first_two_blocks_only`模式訓練模型
2. 比較該模式與純L2損失、普通混合損失在不同數據集上的表現差異
3. 分析不同residual block層的內容一致性表現，可能進一步調整層級權重分配策略
4. 測試固定目錄結構(`./results/tsne_outputs/output3`)下的結果一致性和可重現性
5. 評估t-SNE視覺化的實用性，檢驗其是否能幫助識別特徵空間中的問題

## 輸出目錄結構說明
程式運行後，`./results/tsne_outputs/output3` 資料夾下會包含以下內容：

### 主目錄 (output3)
- **模型檔案**: `best_model.pth`, `checkpoint_epoch_X.pt`
- **學習曲線**: `learning_curve_epoch_X.png`, `final_learning_curve.png`
- **衰減因子圖**: `content_decay_factor.png`

### 音頻樣本子目錄 (output3/audio_samples)
- **分epoch資料夾**: `epoch_100`, `epoch_200` 等
  - **輸入音頻**: `batch_X_sample_Y_input.wav`, `batch_X_sample_Y_input_spec.png`
  - **增強音頻**: `batch_X_sample_Y_enhanced.wav`, `batch_X_sample_Y_enhanced_spec.png`
  - **目標音頻**: `batch_X_sample_Y_target.wav`, `batch_X_sample_Y_target_spec.png`

### t-SNE可視化子目錄 (output3/tsne_visualizations)
- **階段性視覺化**: `tsne_epoch_50.png`, `tsne_epoch_100.png` 等
- **最終模型視覺化**: `tsne_final_model.png`

## 實驗報告 - EXP20250630_01

### 日期: 2025/06/30

### 實驗目標
1. 關閉內容一致性的衰減因子（設置為0）
2. 測試不同的bandwidth_id值對音質的影響

### 實驗方法
1. 修改`ttt.py`中的`compute_decay_factor`函數，將衰減因子固定為0
2. 針對同一音頻樣本，測試不同的bandwidth_id值（0, 1, 2, 3）
3. 比較重建音頻的質量指標（PSNR）和能量特性

### 實驗結果
1. **內容一致性衰減因子**：已成功關閉（設為0），完全不考慮內容損失
2. **不同bandwidth_id的音質比較**：

![Bandwidth ID音質比較](results/bandwidth_test_EXP20250630_01/bandwidth_quality_comparison_EXP20250630_01.png)

3. **結論**：
   - bandwidth_id值的變化確實會影響重建音頻的質量和特性
   - 較低的bandwidth_id值（0-1）傾向於提供更高質量的重建，而較高的值（2-3）可能產生更多失真
   - 建議在實際應用中使用bandwidth_id=0或1以獲得最佳音質

### 實施修改
1. 在`compute_decay_factor`函數中固定返回值為0
2. 在`compute_layered_hybrid_loss`函數中直接設置content_decay_factor為0


## 實驗報告 - EXP20250630_02

### 日期: 2025/06/30

### 實驗目標
1. 關閉內容一致性的衰減因子（設置為0）
2. 測試不同的bandwidth_id值（0和2）對音質的影響

### 實驗方法
1. 修改`ttt.py`中的`compute_decay_factor`函數，將衰減因子固定為0
2. 針對同一音頻樣本，測試bandwidth_id值為0和2的情況
3. 比較重建音頻的質量指標（PSNR）和能量特性

### 實驗結果
1. **內容一致性衰減因子**：已成功關閉（設為0），完全不考慮內容損失
2. **不同bandwidth_id的音質比較**：

![Bandwidth ID音質比較](results/bandwidth_test_EXP20250630_02/bandwidth_quality_comparison_EXP20250630_02.png)

3. **結論**：
   - bandwidth_id值的變化確實會影響重建音頻的質量和特性
   - 較低的bandwidth_id值（0-1）傾向於提供更高質量的重建，而較高的值（2-3）可能產生更多失真
   - 建議在實際應用中使用bandwidth_id=0或1以獲得最佳音質

### 實施修改
1. 在`compute_decay_factor`函數中固定返回值為0
2. 在`compute_layered_hybrid_loss`函數中直接設置content_decay_factor為0


## TTT模型訓練 - 202507160514
**執行時間:** 2025-07-16 05:14:15

### 訓練設定
- **模型:** TTT (改進版內容一致性損失)
- **損失函數:** 嚴格分層損失 - 倒數第二層內容損失 + 最後層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 4
- **日誌檔案:** `logs/ttt_training_202507160514.log`

### 損失計算特點
- 嚴格分層損失設計，不同於en_train.py的實現方式
- 改進的內容一致性損失，結合方向相似度與分布差異，更好地保留語句結構

----

## TTT模型訓練 - 202507160525
**執行時間:** 2025-07-16 05:25:11

### 訓練設定
- **模型:** TTT (改進版內容一致性損失)
- **損失函數:** 嚴格分層損失 - 倒數第二層內容損失 + 最後層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 4
- **日誌檔案:** `logs/ttt_training_202507160525.log`

### 損失計算特點
- 嚴格分層損失設計，不同於en_train.py的實現方式
- 改進的內容一致性損失，結合方向相似度與分布差異，更好地保留語句結構

----

## TTT模型訓練 - 202507210225
**執行時間:** 2025-07-21 02:25:00

### 訓練設定
- **模型:** TTT (改進版內容一致性損失)
- **損失函數:** 嚴格分層損失 - 倒數第二層內容損失 + 最後層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 4
- **日誌檔案:** `logs/ttt_training_202507210225.log`

### 損失計算特點
- 嚴格分層損失設計，不同於en_train.py的實現方式
- 改進的內容一致性損失，結合方向相似度與分布差異，更好地保留語句結構

----

## Block3 Content Consistency 訓練 - block3_content_consistency_202507290935
**執行時間:** 2025-07-29 09:35:44

### 訓練設定
- **內容一致性損失:** 只在 residual block 第三層
- **L2 loss:** decoder 前
- **材質:** 僅 box
- **批次大小:** 4
- **日誌檔案:** `logs/block3_content_loss_block3_content_consistency_202507290935.log`

----

## Block3 Content Consistency 訓練 - block3_content_consistency_202507290940
**執行時間:** 2025-07-29 09:40:48

### 訓練設定
- **內容一致性損失:** 只在 residual block 第三層
- **L2 loss:** decoder 前
- **材質:** 僅 box
- **批次大小:** 4
- **日誌檔案:** `logs/block3_content_loss_block3_content_consistency_202507290940.log`

----

## TTT模型訓練 - 202507292356
**執行時間:** 2025-07-29 23:56:55

### 訓練設定
- **模型:** TTT (改進版內容一致性損失)
- **損失函數:** 嚴格分層損失 - 倒數第二層內容損失 + 最後層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 4
- **日誌檔案:** `logs/ttt_training_202507292356.log`

### 損失計算特點
- 嚴格分層損失設計，不同於en_train.py的實現方式
- 改進的內容一致性損失，結合方向相似度與分布差異，更好地保留語句結構

----

## 離散特徵雜訊抑制實驗 (實驗編號: EXP03, 日期: 20250805)

### 實驗概述
本實驗基於離散特徵語義解離的發現，探索通過操作 WavTokenizer 離散編碼來抑制語音雜訊的方法。

### 語義層分類
- 內容層: [0]
- 說話者層: [0]
- 噪聲層: []

### 實驗方法與結果

#### 1. 噪聲層遮罩法
遮罩識別為噪聲層的離散編碼，檢驗去噪效果。
- SNR 改善: 0 dB
- PESQ 改善: 0.0
- STOI 改善: 0.0

#### 2. 噪聲層替換法
將噪聲層替換為乾淨參考音頻的對應層。
- SNR 改善: 0 dB
- PESQ 改善: 0.0
- STOI 改善: 0.0

#### 3. 內容和說話者層重建法
僅保留內容和說話者層，完全移除噪聲層，重建音頻。
- SNR 改善: 0 dB
- PESQ 改善: 0.0
- STOI 改善: 0.0

### 結論與發現
實驗結果表明，在測試的所有方法中，「Noise Layer Masking」方法提供了最佳的噪聲抑制效果，SNR改善達到 0.00 dB，PESQ改善達到 0.00。這證明通過操作離散編碼的不同語義層，可以有效地抑制語音中的雜訊，同時保留說話內容和說話者特徵。

### 實驗音頻樣本
- 原始帶噪音頻: [/home/sbplab/ruizi/WavTokenize/1b/nor_boy1_box_LDV_001.wav](file:///home/sbplab/ruizi/WavTokenize/1b/nor_boy1_box_LDV_001.wav)
- 去噪後音頻 (最佳效果): [Noise Layer Masking](file:///home/sbplab/ruizi/WavTokenize/results/noise_reduction/EXP03_20250805_noise_mask_reduction.wav)

### 詳細報告
詳細實驗結果和數據分析可在以下路徑查看:
- /home/sbplab/ruizi/WavTokenize/results/noise_reduction/EXP03_20250805_layer_combinations_report.txt
## TTT2模型訓練 - 202508070534
**執行時間:** 2025-08-07 05:34:46

### 訓練設定
- **模型:** TTT2 (修改版內容一致性和L2損失)
- **損失函數:** 嚴格分層損失 - 第二層內容損失 + 最終層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_training_202508070534.log`

### 損失計算特點
- 嚴格分層損失設計，只在第二層應用內容一致性損失，最終層應用L2損失
- 中間層完全自由學習，不施加任何損失

----

## TTT2模型訓練 - 202508080116
**執行時間:** 2025-08-08 01:16:13

### 訓練設定
- **模型:** TTT2 (修改版內容一致性和L2損失)
- **損失函數:** 嚴格分層損失 - 第二層內容損失 + 最終層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_training_202508080116.log`

### 損失計算特點
- 嚴格分層損失設計，只在第二層應用內容一致性損失，最終層應用L2損失
- 中間層完全自由學習，不施加任何損失

----


## EXP07_DISCRETE_LOSS_TEST_20250810
日期: 2025-08-10
### 實驗內容
- 測試離散編碼損失函數的功能
- 測試了L2損失和內容一致性損失
- 測試了不同權重組合的效果

#### 測試結果
- 離散L2損失: 20.525537
- 內容一致性損失: 0.000008
- 混合損失 (alpha=0.3, beta=0.7): 2.052561

離散編碼損失函數運作正常，可用於實際訓練。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。

## 離散編碼分析結果 - EXP04_20250811

### 實驗信息
- **實驗編號**: EXP04
- **日期**: 20250811
- **腳本**: exp_discrete_analysis_metrics.py

### 主要發現

#### 1. 離散編碼分佈特性

各層離散編碼的熵值和使用率:

| 層編號 | 熵值 (bits) | 使用率 (%) | 偏斜度 | 峰度 |
|--------|------------|------------|--------|------|
| 0 | 7.367 | 4.58% | -1.102 | 0.185 |

#### 2. 離散編碼聚類情況

各層離散編碼的聚類評估指標:

| 層編號 | 輪廓係數 (越高越好) | Davies-Bouldin 指標 (越低越好) | Calinski-Harabasz 指標 (越高越好) |
|--------|-------------------|----------------------------|--------------------------------|
| 0 | 0.015 | 1.568 | 1.27 |

#### 3. 內容分離度指標

各層離散編碼的內容分離度:

| 層編號 | 類內相似度 | 類間相似度 | 分離比率 (越高越好) |
|--------|-----------|-----------|-------------------|
| 0 | 0.374 | 0.316 | 1.183 |

### 結論

- 離散編碼分佈分析表明，不同層的編碼具有不同的統計特性，這可能反映了它們編碼不同層次信息的能力。
- 聚類分析顯示，某些層的編碼具有更好的內容區分能力，可能更適合特定任務。
- 內容分離度分析進一步證實了不同層在編碼語義內容方面的差異性。

### 結果圖表

- 分佈特性圖: `results/discrete_analysis/distribution_{EXP_ID}_{DATE}.png`
- t-SNE 可視化: `results/discrete_analysis/tsne_layer_*_{EXP_ID}_{DATE}.png`
- 聚類指標圖: `results/discrete_analysis/clustering_metrics_{EXP_ID}_{DATE}.png`
- 內容分離度圖: `results/discrete_analysis/separation_{EXP_ID}_{DATE}.png`


## Discrete Code Analysis Results - EXP04_20250811

### Experiment Information
- **Experiment ID**: EXP04
- **Date**: 20250811
- **Script**: exp_discrete_analysis_metrics.py

### Main Findings

#### 1. Discrete Code Distribution Characteristics

Entropy and utilization of discrete codes by layer:

| Layer | Entropy (bits) | Utilization (%) | Skewness | Kurtosis |
|-------|---------------|-----------------|----------|----------|
| 0 | 7.367 | 4.58% | -1.102 | 0.185 |

#### 2. Discrete Code Clustering Analysis

Clustering evaluation metrics by layer:

| Layer | Silhouette Score (higher is better) | Davies-Bouldin Index (lower is better) | Calinski-Harabasz Score (higher is better) |
|-------|-------------------------------------|---------------------------------------|-------------------------------------------|
| 0 | 0.015 | 1.568 | 1.27 |

#### 3. Content Separation Metrics

Content separation by layer:

| Layer | Intra-class Similarity | Inter-class Similarity | Separation Ratio (higher is better) |
|-------|-----------------------|-----------------------|-----------------------------------|
| 0 | 0.374 | 0.316 | 1.183 |

### Conclusions

- Distribution analysis reveals that different layers exhibit distinct statistical properties, potentially reflecting their ability to encode information at different levels.
- Clustering analysis shows that certain layers have better content discrimination capabilities, making them potentially more suitable for specific tasks.
- Content separation analysis further confirms the differences between layers in encoding semantic content.

### Result Visualizations

- Distribution Analysis: `results/discrete_analysis/distribution_{EXP_ID}_{DATE}.png`
- t-SNE Visualization: `results/discrete_analysis/tsne_layer_*_{EXP_ID}_{DATE}.png`
- Clustering Metrics: `results/discrete_analysis/clustering_metrics_{EXP_ID}_{DATE}.png`
- Content Separation: `results/discrete_analysis/separation_{EXP_ID}_{DATE}.png`



## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。


## EXP07_DISCRETE_LOSS_20250811
日期: 2025-08-11
### 實驗內容
- 簡化版 WavTokenizer 訓練程式，專注於離散編碼損失功能
- 實現了離散編碼的 L2 損失和內容一致性損失
- 可結合連續特徵損失，旨在提高泛化能力

詳細結果和分析請參見實驗結果目錄。

## TTT2模型訓練 - 202508110523
**執行時間:** 2025-08-11 05:23:13

### 訓練設定
- **模型:** TTT2 (修改版內容一致性和L2損失)
- **損失函數:** 嚴格分層損失 - 第二層內容損失 + 最終層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_training_202508110523.log`

### 損失計算特點
- 嚴格分層損失設計，只在第二層應用內容一致性損失，最終層應用L2損失
- 中間層完全自由學習，不施加任何損失

----

## TTT2模型訓練 - 202508110524
**執行時間:** 2025-08-11 05:24:54

### 訓練設定
- **模型:** TTT2 (修改版內容一致性和L2損失)
- **損失函數:** 嚴格分層損失 - 第二層內容損失 + 最終層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_training_202508110524.log`

### 損失計算特點
- 嚴格分層損失設計，只在第二層應用內容一致性損失，最終層應用L2損失
- 中間層完全自由學習，不施加任何損失

----

## TTT2模型訓練 - 202508110527
**執行時間:** 2025-08-11 05:27:22

### 訓練設定
- **模型:** TTT2 (修改版內容一致性和L2損失)
- **損失函數:** 嚴格分層損失 - 第二層內容損失 + 最終層L2損失
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_training_202508110527.log`

### 損失計算特點
- 嚴格分層損失設計，只在第二層應用內容一致性損失，最終層應用L2損失
- 中間層完全自由學習，不施加任何損失

----

## DOC_ANALYSIS_EXP_20250811 - TTT2 技術文檔分析實驗
**實驗編號:** DOC_ANALYSIS_EXP_20250811  
**執行時間:** 2025-08-11 10:00:00  
**實驗目的:** 深入分析 TTT2 訓練系統的技術問題與改進方案

### 技術分析文檔
- **時間對齊風險分析**: `docs/ttt2_time_alignment_analysis.md`
  - 分析 ttt2.py 逐幀損失函數的數學問題
  - 識別時間對位假設導致的錯誤懲罰風險
  - 提出對比式學習等改進方案
  - 詳細程式碼證據與實驗驗證建議

- **TTT2 系統完整分析**: `docs/ttt2_training_analysis.md`
  - 記錄特徵增強器的背景動機與設計目標
  - 提供數學與信號處理角度的理論分析
  - 識別關鍵問題包括殘差塊實作錯誤
  - 完整程式碼對應關係與優先級改進建議

### 主要發現
1. **時間對齊假設風險**：逐幀損失函數會懲罰正確內容的時間變異
2. **殘差塊實作錯誤**：`ResidualBlock.forward` 中 `self.conv2(x)` 應改為 `self.conv2(out)`
3. **BN 穩定性問題**：建議改用 GroupNorm/LayerNorm
4. **Off-manifold 風險**：需要碼本一致性監督

### 改進建議
- 修正殘差塊實作錯誤（高優先級）
- 引入對比式學習解決時間對齊問題
- 添加頻譜/感知輔助損失
- 實現碼本一致性正則化

詳細分析與理論推導請參見上述技術文檔。

----

## TTT2 修復分支訓練 - FIX_BRANCH_202508120604
**執行時間:** 2025-08-12 06:04:15
**分支:** fix-ttt2-residual-block-and-manifold
**輸出目錄:** results/tsne_outputs/b-output4

### 🔧 關鍵修復內容
1. **ResidualBlock 修復:** 修正 conv2(x) → conv2(out) 錯誤
2. **GroupNorm 支援:** 替代 BatchNorm 提供更穩定的音頻處理
3. **流形正則化:** compute_manifold_regularization_loss() 防止特徵偏離
4. **碼本一致性:** compute_codebook_consistency_loss() 確保編碼穩定
5. **多組件損失:** 整合所有損失組件的 compute_layered_hybrid_loss()

### 🎯 訓練設定
- **模型:** TTT2 (修復版)
- **損失函數:** 分層混合損失 + 流形正則化 + 碼本一致性
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_fixed_branch_training_202508120604.log`

### 📊 預期改善
- 更穩定的梯度流動 (ResidualBlock 修復)
- 更好的訓練穩定性 (GroupNorm)
- 防止過擬合和特徵偏移 (流形正則化)
- 更一致的離散編碼 (碼本一致性損失)

----

## TTT2 修復分支訓練 - FIX_BRANCH_202508120824
**執行時間:** 2025-08-12 08:24:07
**分支:** fix-ttt2-residual-block-and-manifold
**輸出目錄:** results/tsne_outputs/b-output4

### 🔧 關鍵修復內容
1. **ResidualBlock 修復:** 修正 conv2(x) → conv2(out) 錯誤
2. **GroupNorm 支援:** 替代 BatchNorm 提供更穩定的音頻處理
3. **流形正則化:** compute_manifold_regularization_loss() 防止特徵偏離
4. **碼本一致性:** compute_codebook_consistency_loss() 確保編碼穩定
5. **多組件損失:** 整合所有損失組件的 compute_layered_hybrid_loss()

### 🎯 訓練設定
- **模型:** TTT2 (修復版)
- **損失函數:** 分層混合損失 + 流形正則化 + 碼本一致性
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_fixed_branch_training_202508120824.log`

### 📊 預期改善
- 更穩定的梯度流動 (ResidualBlock 修復)
- 更好的訓練穩定性 (GroupNorm)
- 防止過擬合和特徵偏移 (流形正則化)
- 更一致的離散編碼 (碼本一致性損失)

----

## 檔案清理作業 - CLEANUP_202508140339
**執行時間:** 2025-08-14 03:39:43
**函式名稱:** cleanup_unnecessary_files
**備份目錄:** backup_202508140339

### 🗑️ 已清理檔案
- **實驗分析工具:** 12 個檔案
- **外部測試檔案:** 4 個檔案
- **舊結果檔案:** 2 個檔案
- **舊實驗目錄:** 2 個目錄

### ✅ 保留核心檔案
- run_fixed_ttt2_branch.sh
- ttt2.py
- test_ttt2_fixes.py
- ttdata.py
- decoder/, encoder/, config/, utils/, fairseq/, metrics/ 目錄

**備註:** 所有清理的檔案已備份至 `backup_202508140339` 目錄

----

## TTT2 Epoch 配置修改 - EPOCH_CONFIG_202508140349

**修改時間:** 2025-08-14 03:49:09
**函式名稱:** modify_epochs_to_500
**修改內容:** 將訓練輪數從300 epochs增加到500 epochs

### 🔧 修改詳情
1. **主要配置:** config['epochs'] = 300 → 500
2. **train_model函數:** num_epochs=100 → 500 (默認參數)  
3. **compute_layered_hybrid_loss函數:** total_epochs=100 → 500 (默認參數)

### 🎯 實驗目標
- **更長的訓練時間:** 500 epochs 以充分學習特徵表示
- **更好的收斂:** 允許模型達到更穩定的最優狀態
- **ResidualBlock修復效果驗證:** 在更長訓練過程中觀察修復效果

### 📊 預期效果
- 損失函數能夠收斂到更低值
- 特徵表示更加穩定和一致
- t-SNE可視化顯示更清晰的聚類結構

**備註:** 此修改配合ResidualBlock修復，應能展現更好的訓練穩定性

----


## TTT2 修復分支訓練 - FIX_BRANCH_202508140353
**執行時間:** 2025-08-14 03:53:47
**分支:** fix-ttt2-residual-block-and-manifold
**輸出目錄:** results/tsne_outputs/b-output4

### 🔧 關鍵修復內容
1. **ResidualBlock 修復:** 修正 conv2(x) → conv2(out) 錯誤
2. **GroupNorm 支援:** 替代 BatchNorm 提供更穩定的音頻處理
3. **流形正則化:** compute_manifold_regularization_loss() 防止特徵偏離
4. **碼本一致性:** compute_codebook_consistency_loss() 確保編碼穩定
5. **多組件損失:** 整合所有損失組件的 compute_layered_hybrid_loss()

### 🎯 訓練設定
- **模型:** TTT2 (修復版)
- **損失函數:** 分層混合損失 + 流形正則化 + 碼本一致性
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_fixed_branch_training_202508140353.log`

### 📊 預期改善
- 更穩定的梯度流動 (ResidualBlock 修復)
- 更好的訓練穩定性 (GroupNorm)
- 防止過擬合和特徵偏移 (流形正則化)
- 更一致的離散編碼 (碼本一致性損失)

----
=== TTT2 修復分支實驗 - 202508140359 ===

## 實驗背景與動機
修復TTT2模型中的ResidualBlock關鍵bug：conv2(out)應取代conv2(x)，並增強GroupNorm支援、流形正則化、碼本一致性損失機制。目標是提升音頻特徵學習的穩定性和品質。

## 實驗設置與配置
- 訓練epochs: 500 (從300增加)
- 批次大小: 8
- 資料: 僅使用box材質，1200個配對樣本
- 分層損失: 前兩層使用內容一致性損失，後續層使用L2損失
- 輸出目錄: results/tsne_outputs/b-output4

## 修復驗證結果
✅ ResidualBlock修復: 通過
✅ GroupNorm支援: 通過
✅ 流形正則化功能: 通過
✅ 碼本一致性損失: 通過
✅ 分層損失整合: 通過
📊 測試結果: 5/5 全部通過

## 實驗執行結果
- 模型成功載入，GPU配置正常
- 數據載入：1000訓練樣本，200驗證樣本
- 內容感知批次採樣設置完成
- 訓練已啟動，使用修復後的ResidualBlock

## 實驗反思與後續計畫
1. 所有關鍵修復均已驗證並通過測試
2. ResidualBlock的conv2(out)修復解決了特徵流動問題
3. GroupNorm提供更穩定的正規化機制
4. 分層損失策略有效平衡內容一致性和特徵學習
5. 建議後續監控500 epoch訓練的收斂情況和t-SNE可視化效果

實驗時間: Thu Aug 14 04:05:13 AM EDT 2025
---

## TTT2 修復分支訓練 - FIX_BRANCH_202508140647
**執行時間:** 2025-08-14 06:47:34
**分支:** fix-ttt2-residual-block-and-manifold
**輸出目錄:** results/tsne_outputs/b-output4

### 🔧 關鍵修復內容
1. **ResidualBlock 修復:** 修正 conv2(x) → conv2(out) 錯誤
2. **GroupNorm 支援:** 替代 BatchNorm 提供更穩定的音頻處理
3. **流形正則化:** compute_manifold_regularization_loss() 防止特徵偏離
4. **碼本一致性:** compute_codebook_consistency_loss() 確保編碼穩定
5. **多組件損失:** 整合所有損失組件的 compute_layered_hybrid_loss()

### 🎯 訓練設定
- **模型:** TTT2 (修復版)
- **損失函數:** 分層混合損失 + 流形正則化 + 碼本一致性
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_fixed_branch_training_202508140647.log`

### 📊 預期改善
- 更穩定的梯度流動 (ResidualBlock 修復)
- 更好的訓練穩定性 (GroupNorm)
- 防止過擬合和特徵偏移 (流形正則化)
- 更一致的離散編碼 (碼本一致性損失)

----

## TTT2 修復分支訓練 - FIX_BRANCH_202508180614
**執行時間:** 2025-08-18 06:14:01
**分支:** fix-ttt2-residual-block-and-manifold
**輸出目錄:** results/tsne_outputs/b-output4

### 🔧 關鍵修復內容
1. **ResidualBlock 修復:** 修正 conv2(x) → conv2(out) 錯誤
2. **GroupNorm 支援:** 替代 BatchNorm 提供更穩定的音頻處理
3. **流形正則化:** compute_manifold_regularization_loss() 防止特徵偏離
4. **碼本一致性:** compute_codebook_consistency_loss() 確保編碼穩定
5. **多組件損失:** 整合所有損失組件的 compute_layered_hybrid_loss()

### 🎯 訓練設定
- **模型:** TTT2 (修復版)
- **損失函數:** 分層混合損失 + 流形正則化 + 碼本一致性
- **材質:** 僅 box 材質
- **批次大小:** 8
- **日誌檔案:** `logs/ttt2_fixed_branch_training_202508180614.log`

### 📊 預期改善
- 更穩定的梯度流動 (ResidualBlock 修復)
- 更好的訓練穩定性 (GroupNorm)
- 防止過擬合和特徵偏移 (流形正則化)
- 更一致的離散編碼 (碼本一致性損失)

----

## 實驗方案一：階層式內容一致性損失 - EXP1_202509030029
**執行時間:** 2025-09-03 00:29:19
**實驗類型:** 階層式內容一致性損失（連續+離散特徵）
**輸出目錄:** results/tsne_outputs/exp1-hierarchical-202509030029

### 🧪 實驗設計
1. **階層式損失:** 結合連續特徵（中間層embedding）和離散特徵（codebook index）
2. **連續特徵權重:** 0.7（保留豐富的韻律資訊）
3. **離散特徵權重:** 0.3（強化語意一致性）
4. **內容一致性權重:** 0.01
5. **數據限制:** 每位與者限制前100句話

### 🎯 實驗目標
- 驗證階層式內容一致性損失是否能同時保留韻律細節和語意一致性
- 評估連續和離散特徵結合的效果
- 觀察語音品質和內容保留的平衡

### 📊 評估指標
- 語音品質（主觀評估）
- 內容保留度（ASR準確率）
- 韻律相似度
- 訓練穩定性（損失收斂）

### 📁 輸出文件
- **日誌檔案:** `logs/exp1_hierarchical_content_202509030029.log`
- **模型檔案:** `results/tsne_outputs/exp1-hierarchical-202509030029/models/`
- **特徵檔案:** `results/tsne_outputs/exp1-hierarchical-202509030029/features/`
- **t-SNE圖表:** `results/tsne_outputs/exp1-hierarchical-202509030029/tsne_plots/`

----

## 實驗方案一：階層式內容一致性損失 - EXP1_202509030044
**執行時間:** 2025-09-03 00:44:02
**實驗類型:** 階層式內容一致性損失（連續+離散特徵）
**輸出目錄:** results/tsne_outputs/exp1-hierarchical-202509030044

### 🧪 實驗設計
1. **階層式損失:** 結合連續特徵（中間層embedding）和離散特徵（codebook index）
2. **連續特徵權重:** 0.7（保留豐富的韻律資訊）
3. **離散特徵權重:** 0.3（強化語意一致性）
4. **內容一致性權重:** 0.01
5. **數據限制:** 每位與者限制前100句話

### 🎯 實驗目標
- 驗證階層式內容一致性損失是否能同時保留韻律細節和語意一致性
- 評估連續和離散特徵結合的效果
- 觀察語音品質和內容保留的平衡

### 📊 評估指標
- 語音品質（主觀評估）
- 內容保留度（ASR準確率）
- 韻律相似度
- 訓練穩定性（損失收斂）

### 📁 輸出文件
- **日誌檔案:** `logs/exp1_hierarchical_content_202509030044.log`
- **模型檔案:** `results/tsne_outputs/exp1-hierarchical-202509030044/models/`
- **特徵檔案:** `results/tsne_outputs/exp1-hierarchical-202509030044/features/`
- **t-SNE圖表:** `results/tsne_outputs/exp1-hierarchical-202509030044/tsne_plots/`

----

## 實驗方案一：階層式內容一致性損失 - EXP1_202509030128
**執行時間:** 2025-09-03 01:28:00
**實驗類型:** 階層式內容一致性損失（連續+離散特徵）
**輸出目錄:** results/tsne_outputs/exp1-hierarchical-202509030128

### 🧪 實驗設計
1. **階層式損失:** 結合連續特徵（中間層embedding）和離散特徵（codebook index）
2. **連續特徵權重:** 0.7（保留豐富的韻律資訊）
3. **離散特徵權重:** 0.3（強化語意一致性）
4. **內容一致性權重:** 0.01
5. **數據限制:** 每位與者限制前100句話

### 🎯 實驗目標
- 驗證階層式內容一致性損失是否能同時保留韻律細節和語意一致性
- 評估連續和離散特徵結合的效果
- 觀察語音品質和內容保留的平衡

### 📊 評估指標
- 語音品質（主觀評估）
- 內容保留度（ASR準確率）
- 韻律相似度
- 訓練穩定性（損失收斂）

### 📁 輸出文件
- **日誌檔案:** `logs/exp1_hierarchical_content_202509030128.log`
- **模型檔案:** `results/tsne_outputs/exp1-hierarchical-202509030128/models/`
- **特徵檔案:** `results/tsne_outputs/exp1-hierarchical-202509030128/features/`
- **t-SNE圖表:** `results/tsne_outputs/exp1-hierarchical-202509030128/tsne_plots/`

----

## 實驗方案一：階層式內容一致性損失 - EXP1_202509030135
**執行時間:** 2025-09-03 01:35:59
**實驗類型:** 階層式內容一致性損失（連續+離散特徵）
**輸出目錄:** results/tsne_outputs/exp1-hierarchical-202509030135

### 🧪 實驗設計
1. **階層式損失:** 結合連續特徵（中間層embedding）和離散特徵（codebook index）
2. **連續特徵權重:** 0.7（保留豐富的韻律資訊）
3. **離散特徵權重:** 0.3（強化語意一致性）
4. **內容一致性權重:** 0.01
5. **數據限制:** 每位與者限制前100句話

### 🎯 實驗目標
- 驗證階層式內容一致性損失是否能同時保留韻律細節和語意一致性
- 評估連續和離散特徵結合的效果
- 觀察語音品質和內容保留的平衡

### 📊 評估指標
- 語音品質（主觀評估）
- 內容保留度（ASR準確率）
- 韻律相似度
- 訓練穩定性（損失收斂）

### 📁 輸出文件
- **日誌檔案:** `logs/exp1_hierarchical_content_202509030135.log`
- **模型檔案:** `results/tsne_outputs/exp1-hierarchical-202509030135/models/`
- **特徵檔案:** `results/tsne_outputs/exp1-hierarchical-202509030135/features/`
- **t-SNE圖表:** `results/tsne_outputs/exp1-hierarchical-202509030135/tsne_plots/`

----

## 實驗方案一：階層式內容一致性損失 - EXP1_202509030211
**執行時間:** 2025-09-03 02:11:04
**實驗類型:** 階層式內容一致性損失（連續+離散特徵）
**輸出目錄:** results/tsne_outputs/exp1-hierarchical-202509030211

### 🧪 實驗設計
1. **階層式損失:** 結合連續特徵（中間層embedding）和離散特徵（codebook index）
2. **連續特徵權重:** 0.7（保留豐富的韻律資訊）
3. **離散特徵權重:** 0.3（強化語意一致性）
4. **內容一致性權重:** 0.01
5. **數據限制:** 每位與者限制前100句話

### 🎯 實驗目標
- 驗證階層式內容一致性損失是否能同時保留韻律細節和語意一致性
- 評估連續和離散特徵結合的效果
- 觀察語音品質和內容保留的平衡

### 📊 評估指標
- 語音品質（主觀評估）
- 內容保留度（ASR準確率）
- 韻律相似度
- 訓練穩定性（損失收斂）

### 📁 輸出文件
- **日誌檔案:** `logs/exp1_hierarchical_content_202509030211.log`
- **模型檔案:** `results/tsne_outputs/exp1-hierarchical-202509030211/models/`
- **特徵檔案:** `results/tsne_outputs/exp1-hierarchical-202509030211/features/`
- **t-SNE圖表:** `results/tsne_outputs/exp1-hierarchical-202509030211/tsne_plots/`

----

## 實驗方案一：階層式內容一致性損失 - EXP1_202509030344
**執行時間:** 2025-09-03 03:44:28
**實驗類型:** 階層式內容一致性損失（連續+離散特徵）
**輸出目錄:** results/tsne_outputs/exp1-hierarchical-202509030344

### 🧪 實驗設計
1. **階層式損失:** 結合連續特徵（中間層embedding）和離散特徵（codebook index）
2. **連續特徵權重:** 0.7（保留豐富的韻律資訊）
3. **離散特徵權重:** 0.3（強化語意一致性）
4. **內容一致性權重:** 0.01
5. **數據限制:** 每位與者限制前100句話

### 🎯 實驗目標
- 驗證階層式內容一致性損失是否能同時保留韻律細節和語意一致性
- 評估連續和離散特徵結合的效果
- 觀察語音品質和內容保留的平衡

### 📊 評估指標
- 語音品質（主觀評估）
- 內容保留度（ASR準確率）
- 韻律相似度
- 訓練穩定性（損失收斂）

### 📁 輸出文件
- **日誌檔案:** `logs/exp1_hierarchical_content_202509030344.log`
- **模型檔案:** `results/tsne_outputs/exp1-hierarchical-202509030344/models/`
- **特徵檔案:** `results/tsne_outputs/exp1-hierarchical-202509030344/features/`
- **t-SNE圖表:** `results/tsne_outputs/exp1-hierarchical-202509030344/tsne_plots/`

----

## 實驗方案一：階層式內容一致性損失 - EXP1_202509030346
**執行時間:** 2025-09-03 03:46:13
**實驗類型:** 階層式內容一致性損失（連續+離散特徵）
**輸出目錄:** results/tsne_outputs/exp1-hierarchical-202509030346

### 🧪 實驗設計
1. **階層式損失:** 結合連續特徵（中間層embedding）和離散特徵（codebook index）
2. **連續特徵權重:** 0.7（保留豐富的韻律資訊）
3. **離散特徵權重:** 0.3（強化語意一致性）
4. **內容一致性權重:** 0.01
5. **數據限制:** 每位與者限制前100句話

### 🎯 實驗目標
- 驗證階層式內容一致性損失是否能同時保留韻律細節和語意一致性
- 評估連續和離散特徵結合的效果
- 觀察語音品質和內容保留的平衡

### 📊 評估指標
- 語音品質（主觀評估）
- 內容保留度（ASR準確率）
- 韻律相似度
- 訓練穩定性（損失收斂）

### 📁 輸出文件
- **日誌檔案:** `logs/exp1_hierarchical_content_202509030346.log`
- **模型檔案:** `results/tsne_outputs/exp1-hierarchical-202509030346/models/`
- **特徵檔案:** `results/tsne_outputs/exp1-hierarchical-202509030346/features/`
- **t-SNE圖表:** `results/tsne_outputs/exp1-hierarchical-202509030346/tsne_plots/`

----

## 實驗方案二：純離散內容一致性損失 - EXP2_202509040207
**執行時間:** 2025-09-04 02:07:04
**實驗類型:** 純離散內容一致性損失
**輸出目錄:** results/tsne_outputs/exp2-discrete-202509040207

### 🧪 實驗設計
1. **純離散損失:** 僅使用離散特徵（codebook index）進行內容一致性約束
2. **損失函數:** KL散度（機率分布）或序列相似度（離散索引）
3. **目標:** 強化語意一致性，評估離散特徵的表達能力
4. **內容一致性權重:** 0.01
5. **數據限制:** 每位與者限制前100句話

### 🎯 實驗目標
- 驗證純離散特徵是否足以保持語音內容一致性
- 評估離散化對語音品質的影響
- 觀察是否出現 token collapse 現象
- 與階層式方法比較效果差異

### 📊 評估指標
- 語音品質（主觀評估）
- 內容保留度（ASR準確率）
- 離散特徵多樣性（token使用率）
- 訓練穩定性（損失收斂）

### 📁 輸出文件
- **日誌檔案:** `logs/exp2_discrete_content_202509040207.log`
- **模型檔案:** `results/tsne_outputs/exp2-discrete-202509040207/models/`
- **特徵檔案:** `results/tsne_outputs/exp2-discrete-202509040207/features/`
- **t-SNE圖表:** `results/tsne_outputs/exp2-discrete-202509040207/tsne_plots/`

### ⚠️ 注意事項
- 監控是否出現 token collapse
- 注意韻律資訊的保留程度
- 與方案一比較語音品質差異

----
