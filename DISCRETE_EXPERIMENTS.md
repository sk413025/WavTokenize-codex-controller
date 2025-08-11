# WavTokenizer 離散特徵分析與噪聲抑制實驗

## 實驗目的

本系列實驗旨在深入分析 WavTokenizer 模型中離散編碼的特性，並探索如何利用離散特徵進行語音噪聲抑制。基於以下假設：
1. WavTokenizer 的離散編碼中，不同層可能編碼了不同類型的語義信息（內容、說話者特徵、環境噪聲等）
2. 通過有選擇地操作這些離散編碼，可以實現語音增強和噪聲抑制

## 檔案結構與功能

### 1. 離散特徵基礎分析 (`exp_discrete_analysis.py`)

**實驗編號**：EXP01  
**執行日期**：2025-08-01  
**主要功能**：
- 分析 WavTokenizer 離散編碼的基本統計特性
- 確定離散編碼的層結構與形狀
- 計算每一層的統計指標（熵值、唯一值數量、數值分佈等）
- 可視化離散編碼的分佈特徵

**輸出目錄**：`/home/sbplab/ruizi/WavTokenize/results/discrete_analysis/`  
**輸出檔案**：
- `EXP01_YYYYMMDD_analysis_report.txt`：詳細統計報告
- `EXP01_YYYYMMDD_discrete_dist_*.png`：各音頻檔案的離散編碼分佈可視化
- `EXP01_YYYYMMDD_discrete_code_*.pt`：保存的離散編碼數據

### 2. 離散特徵語義解離實驗 (`exp_discrete_swap.py`)

**實驗編號**：EXP02  
**執行日期**：2025-08-01  
**主要功能**：
- 通過特徵交換實驗確定不同層的語義含義
- 內容交換實驗：分析哪些層主要包含語音內容信息
- 說話者交換實驗：分析哪些層主要包含說話者身份信息
- 噪聲交換實驗：分析哪些層主要包含環境和噪聲信息
- 綜合分析各層的語義特性，生成層分類報告

**輸出目錄**：`/home/sbplab/ruizi/WavTokenize/results/discrete_semantic/`  
**輸出檔案**：
- `EXP02_YYYYMMDD_content_swap_layer*.wav`：內容交換實驗音頻
- `EXP02_YYYYMMDD_speaker_swap_layer*.wav`：說話者交換實驗音頻
- `EXP02_YYYYMMDD_noise_swap_layer*.wav`：噪聲交換實驗音頻
- `EXP02_YYYYMMDD_*_layer_impact.png`：各層影響程度可視化
- `EXP02_YYYYMMDD_comprehensive_layer_analysis.txt`：綜合層分析報告
- `EXP02_YYYYMMDD_summary_report.txt`：實驗總結報告

### 3. 離散特徵雜訊抑制實驗 (`exp_noise_reduction.py`)

**實驗編號**：EXP03  
**執行日期**：2025-08-01  
**主要功能**：
- 基於離散編碼的語義解離結果，實現噪聲抑制
- 噪聲層遮罩法：將識別為噪聲層的離散編碼遮罩為零
- 噪聲層替換法：將噪聲層替換為乾淨音頻的對應層
- 內容和說話者層重建法：僅保留內容和說話者層，重建音頻
- 多層組合實驗：測試不同層組合的去噪效果，找出最佳組合

**輸出目錄**：`/home/sbplab/ruizi/WavTokenize/results/noise_reduction/`  
**輸出檔案**：
- `EXP03_YYYYMMDD_noise_mask_reduction.wav`：噪聲層遮罩處理結果
- `EXP03_YYYYMMDD_noise_replacement_reduction.wav`：噪聲層替換處理結果
- `EXP03_YYYYMMDD_content_speaker_reconstruction.wav`：內容和說話者層重建結果
- `EXP03_YYYYMMDD_combo_*.wav`：各種層組合的處理結果
- `EXP03_YYYYMMDD_layer_combinations_report.txt`：層組合實驗報告
- `EXP03_YYYYMMDD_layer_combinations_comparison.png`：層組合效果對比可視化

### 4. 實驗運行腳本 (`run_discrete_experiment.sh`)

**功能**：
- 自動執行完整實驗流程
- 按順序執行三個實驗腳本
- 收集並記錄實驗日誌
- 實驗結果自動更新到中央報告 (`REPORT.md`)

**輸出目錄**：`/home/sbplab/ruizi/WavTokenize/logs/`  
**輸出檔案**：
- `EXP_DISCRETE_YYYYMMDD.log`：實驗執行日誌

## 如何運行實驗

1. 確保安裝了所有必要依賴項：
   ```bash
   pip install -r requirements.txt
   ```

2. 運行完整實驗流程：
   ```bash
   ./run_discrete_experiment.sh
   ```

3. 單獨運行特定實驗：
   ```bash
   python exp_discrete_analysis.py  # 基礎分析
   python exp_discrete_swap.py      # 語義解離實驗
   python exp_noise_reduction.py    # 噪聲抑制實驗
   ```

## 實驗結果查看

1. 查看綜合實驗報告：
   ```
   /home/sbplab/ruizi/WavTokenize/REPORT.md
   ```

2. 查看各實驗詳細報告：
   ```
   /home/sbplab/ruizi/WavTokenize/results/discrete_analysis/EXP01_*_analysis_report.txt
   /home/sbplab/ruizi/WavTokenize/results/discrete_semantic/EXP02_*_comprehensive_layer_analysis.txt
   /home/sbplab/ruizi/WavTokenize/results/noise_reduction/EXP03_*_layer_combinations_report.txt
   ```

3. 聆聽處理後的音頻樣本：
   ```
   /home/sbplab/ruizi/WavTokenize/results/discrete_semantic/EXP02_*_*_swap_layer*.wav
   /home/sbplab/ruizi/WavTokenize/results/noise_reduction/EXP03_*_*.wav
   ```

## 預期實驗成果

1. **離散層語義映射**：確定 WavTokenizer 中哪些層主要編碼內容、說話者特徵和噪聲信息
2. **噪聲抑制效果評估**：比較不同離散編碼操作方法的噪聲抑制效果
3. **最佳層組合方案**：發現最有效的層組合方案，實現最佳的語音增強效果
4. **噪聲抑制機制理解**：深入理解 WavTokenizer 離散編碼與語音噪聲的關係
