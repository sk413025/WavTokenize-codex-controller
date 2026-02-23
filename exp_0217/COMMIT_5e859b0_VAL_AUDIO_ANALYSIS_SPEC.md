# Commit 5e859b0 實驗：Val 音質不足根因分析規格（SPEC）

## 1) 分析範圍（Scope）

### In Scope
- 對 `exp_0216/runs/augmented_long_20260216` 的結果做完整診斷。
- 建立「MSE / Token 指標 / 感知音質（PESQ、STOI）」的對照關係。
- 釐清 val 音質不佳的主要影響因子（資料分布、目標函數、容量瓶頸）。

### Out of Scope
- 不在本階段直接改模型訓練邏輯。
- 不在本階段重做 300 epoch 長訓練。

---

## 2) 輸入資料與來源

- Commit: `5e859b0b8ae0bf54d7cf77449f239ae2aeaa0edb`
- 實驗主目錄：`exp_0216/runs/augmented_long_20260216`
- 必要檔案：
  - `config.json`
  - `summary.json`
  - `metrics_history.json`
  - `train.log`
  - `best_model.pt`
  - `final_model.pt`
  - `checkpoints/checkpoint_epoch*.pt`
- 音檔樣本目錄：
  - `audio_samples/train/epoch_*`
  - `audio_samples/val/epoch_*`

---

## 3) 輸出規格

建議輸出目錄：`exp_0217/analysis_commit_5e859b0/`

### 必要輸出
1. `baseline_metrics_table.csv`
2. `audio_quality_by_epoch.json`
3. `audio_quality_by_epoch.md`
4. `stratified_quality_report.md`
5. `hypothesis_scoring.json`
6. `next_experiment_recommendation.md`

### 圖表輸出
1. `mse_vs_pesq_stoi.png`
2. `quality_by_t453_bin.png`
3. `quality_by_snr_bin.png`
4. `quality_by_length_bin.png`

---

## 4) 分析模組規格

## M0. 基線固化（Metadata Freeze）
- 目的：避免後續分析混到其他 run。
- 動作：複製 commit hash、run path、config/summary 指紋（hash）。

## M1. Epoch 指標時間線
- 從 `metrics_history.json` 提取：
  - `train_total_loss`, `val_total_loss`
  - `feature_mse`
  - `entropy`, `top10_mass`, `used_codes`
- 必含 epoch 節點：`1, 50, 71, 78, 100, 150, 200, 222, 250, 300`
- 輸出：表格 + 趨勢圖。

## M2. 感知音質評估（PESQ/STOI）

### 2.1 評估對象
- Epoch: `050, 100, 150, 200, 220, 222(best), 250, 300(final)`
- Split: `train`, `val`

### 2.2 樣本數規格
- 主要評估：每個 split 每個 epoch 至少 `N=200`（若資源不足最少 `N=100`）
- 快速檢查：可先用既有 audio_samples（每 epoch 僅少量）做 smoke check

### 2.3 指標定義
- `PESQ_noisy`, `PESQ_recon`, `ΔPESQ = recon - noisy`
- `STOI_noisy`, `STOI_recon`, `ΔSTOI = recon - noisy`
- 需同時輸出 mean/std 與 95% CI

### 2.4 實作注意
- 既有腳本可參考：`exp_0125/tracin_token_collapse_589e6d/stepS3_audio_pesq_stoi.py`
- 需調整 recon 檔名支援：`*_vq_recon.wav`（exp_0216 命名）
- 取樣率統一到 16k（PESQ wideband）

## M3. 分層誤差分析（Stratified Analysis）

### 3.1 分桶維度
1. `T453 ratio`：`[0,0.1)`, `[0.1,0.2)`, `[0.2,0.3)`, `[0.3,0.5]`
2. `SNR`：`<0dB`, `0~10dB`, `10~20dB`, `>20dB`
3. `duration`：`<2s`, `2~5s`, `>5s`

### 3.2 分析輸出
- 每桶輸出：`feature_mse`, `ΔPESQ`, `ΔSTOI`, `top10_mass proxy`
- 找出 val 最差 10% 樣本，做錯誤案例清單（id + 指標 + 音檔路徑）

## M4. 假設評分與決策
- 對 H1~H4 各給 0~2 分：
  - 0 = 證據不足/不支持
  - 1 = 部分支持
  - 2 = 強支持
- 產出主要根因（Primary）+ 次要根因（Secondary）
- 對應下一輪可執行改動（最多 3 項）

---

## 5) 統計與判定規格

- 相關分析：
  - `corr(feature_mse, ΔPESQ)`
  - `corr(feature_mse, ΔSTOI)`
  - `corr(T453_ratio, ΔPESQ/ΔSTOI)`
- 檢定方式：
  - 連續變數：Spearman
  - 分桶差異：Mann-Whitney U（或 bootstrap CI）
- 顯著性門檻：`p < 0.05`（若多重比較需做 FDR 修正）

---

## 6) 風險與對策

1. **風險：既有每 epoch 音檔樣本太少**
   - 對策：新增 checkpoint 批次推論腳本，固定抽樣 `N>=100`。
2. **風險：PESQ/STOI 套件環境不一致**
   - 對策：在輸出中記錄 package version 與執行環境。
3. **風險：只看平均值掩蓋失敗案例**
   - 對策：強制輸出最差 10% slice 報告。

---

## 7) 與研究目標對齊（Why this spec matters）

本規格把「訓練表徵品質」與「驗證音質」串成同一條證據鏈：
- 若 MSE 好但 PESQ/STOI 差，表示現行研究目標函數需要補齊感知向量。
- 若 val 特定分布（如高 T453）退化，表示需在資料策略上提升泛化。

這能直接支持論文主張：
> 不只要在訓練集收斂，而是要在驗證集穩定產生高品質離散去噪表示。
