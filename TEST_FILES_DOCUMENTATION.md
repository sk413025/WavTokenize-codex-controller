# 測試檔案文檔

## 📋 **目前保留的測試檔案**

### ✅ **核心功能測試**

#### 1. `simple_wavtokenizer_test.py`
- **用途**: WavTokenizer 基本功能測試
- **保留原因**: 作為系統基準測試，驗證 WavTokenizer 載入和基本運作
- **使用場景**: 環境變更或模型更新後的基本功能驗證

#### 2. `test_discrete_flow.py`
- **用途**: 離散 token 處理流程測試
- **保留原因**: 驗證核心的 encode → discrete tokens → decode 流程
- **使用場景**: 離散化邏輯出現問題時的診斷工具

#### 3. `test_bandwidth_params.py`
- **用途**: 測試不同 bandwidth_id 參數對重建品質的影響
- **保留原因**: 後續實驗可能需要優化 bandwidth 設定
- **使用場景**: 音質優化實驗、參數調優

#### 4. `test_different_configs.py`
- **用途**: 測試不同 WavTokenizer 配置檔案
- **保留原因**: 支援不同模型配置的對比實驗
- **使用場景**: 模型配置對比、性能評估

### 🔧 **修復和擴展測試**

#### 5. `test_token_loss_fix.py`
- **用途**: Token Loss 功能修復和驗證測試
- **保留原因**: 如果重新啟用 token loss，此檔案是必要的
- **使用場景**: Token Loss 和 CrossEntropy Loss 對比實驗

#### 6. `t2_outside_test.py`
- **用途**: 外部環境或不同設定下的測試
- **保留原因**: 用於不同環境下的功能驗證
- **使用場景**: 部署到不同環境時的功能確認

## 🗑️ **已刪除的測試檔案**

以下檔案因為已完成驗證或功能已整合到主程式中而被刪除：

### **音檔保存相關**
- `test_audio_save.py` - 基本音檔保存測試
- `test_audio_save_fix.sh` - 音檔保存修復腳本
- `quick_audio_save_test.py` - 快速音檔保存測試
- `verify_audio_fix.py` - 音檔修復驗證
- `verify_audio_reconstruction_fix.py` - 音檔重建驗證

### **WavTokenizer 重建相關**
- `test_wavtokenizer_direct_reconstruction.py` - WavTokenizer 直接重建測試
- `test_wavtokenizer_reconstruction.py` - WavTokenizer 重建品質測試
- `test_real_speech_wavtokenizer.py` - 真實語音 WavTokenizer 測試

### **問題修復相關**
- `test_mask_fix.py` - Mask 維度問題修復測試
- `test_real_speech_denoising.py` - 真實語音降噪測試
- `quick_test_fix.sh` - 快速修復測試腳本

## 📊 **驗證總結**

### ✅ **已確認的功能**
1. **音檔儲存功能**: 正常運作，可儲存 input/target/enhanced 三種音檔
2. **WavTokenizer 重建**: 相關係數約 0.3-0.4，符合預期
3. **Mask 維度問題**: 已修復，Transformer 正常運作
4. **音檔重建邏輯**: 已統一與 ttt2.py 一致的處理方式

### 🎯 **當前實驗狀態**
- **短期修改**: ✅ 完成，訓練時間延長至 200 epochs
- **音檔儲存**: ✅ 正常，與 ttt2.py 邏輯一致
- **系統架構**: ✅ 穩定，準備執行延長訓練實驗

### 🚀 **下一步行動**
執行 200 epochs 延長訓練實驗：
```bash
bash run_crossentropy_experiment.sh
```

## 📝 **測試檔案使用建議**

1. **日常開發**: 主要使用 `simple_wavtokenizer_test.py` 進行基本功能驗證
2. **參數調優**: 使用 `test_bandwidth_params.py` 和 `test_different_configs.py`
3. **問題診斷**: 使用 `test_discrete_flow.py` 檢查離散化流程
4. **功能擴展**: 如重新啟用 token loss，參考 `test_token_loss_fix.py`

---
**最後更新**: 2025-09-26  
**實驗編號**: TRAINING_EXT_20250926