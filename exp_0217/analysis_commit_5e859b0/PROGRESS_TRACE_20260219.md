# Progress Trace (Auditable)

日期：2026-02-19  
分析目錄：`exp_0217/analysis_commit_5e859b0`

## M1 文件對齊（100%）
- 已讀文件：
  - `exp_0217/COMMIT_5e859b0_VAL_AUDIO_ANALYSIS_PLAN.md`
  - `exp_0217/COMMIT_5e859b0_VAL_AUDIO_ANALYSIS_SPEC.md`
  - `exp_0217/PRE_MODIFICATION_EVALUATION_20260217.md`
  - `exp_0217/README.md`
  - `exp_0216/README.md`
  - `THESIS_ARCHITECTURE.md`

## M2 現況盤點（100%）
- 盤點輸出：
  - `spec_coverage_status_20260219.json`
- 初始覆蓋：
  - `3/10`
- 最終覆蓋：
  - `10/10`

## M3 根因分析（100%）
- 基線指標：
  - `baseline_metrics_table.csv`
- 音質評估（N=100，available epochs）：
  - `audio_quality_by_epoch.json`
  - `audio_quality_by_epoch.md`
- 分層分析：
  - `stratified_quality_report.md`
- 圖表：
  - `mse_vs_pesq_stoi.png`
  - `quality_by_t453_bin.png`
  - `quality_by_snr_bin.png`
  - `quality_by_length_bin.png`

## M4 決策輸出（100%）
- 假設評分：
  - `hypothesis_scoring.json`
- 決策建議：
  - `next_experiment_recommendation.md`

## M5 若 Go 方案（100%）
- 最小改動方案與驗收門檻：
  - `next_experiment_recommendation.md`
  - `FINAL_DECISION_REPORT_20260219.md`

## M5 執行狀態（M1 實驗進行中）
- 執行 run：
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug`
- 單因子改動：
  - `t453_min_weight: 0.2 -> 0.3`（其餘條件固定）
- 設定快照：
  - `epochs=100`, `seed=42`, `batch_size=8`, `grad_accum=2`
  - `lora_rank=64`, `lora_alpha=128`
  - `save_checkpoint_every=10`, `save_audio_interval=50`

### 已觀測結果（epoch 1~12）
- 來源：`exp_0217/runs/t453_m1_minw03_epoch100_debug/metrics_history.json`
- 指標（epoch1）：
  - `train_total_loss=0.071076`
  - `val_total_loss=0.109126`
  - `feature_mse=0.074210`
  - `entropy=9.705683`
  - `top10_mass=0.073969`
  - `used_codes=1369`
  - `P2=pass`, `P3=fail`
- 指標（epoch2）：
  - `train_total_loss=0.059618`
  - `val_total_loss=0.082932`
  - `feature_mse=0.048616`
  - `entropy=8.902934`
  - `top10_mass=0.129524`
  - `used_codes=1019`
  - `P2=pass`, `P3=fail`
- 指標（epoch3）：
  - `train_total_loss=0.055951`
  - `val_total_loss=0.080923`
  - `feature_mse=0.047366`
  - `entropy=9.198084`
  - `top10_mass=0.104634`
  - `used_codes=1204`
  - `P2=pass`, `P3=fail`
- 指標（epoch4）：
  - `train_total_loss=0.053657`
  - `val_total_loss=0.078875`
  - `feature_mse=0.045769`
  - `entropy=8.913223`
  - `top10_mass=0.127326`
  - `used_codes=1046`
  - `P2=pass`, `P3=fail`
- 指標（epoch5）：
  - `train_total_loss=0.051848`
  - `val_total_loss=0.077573`
  - `feature_mse=0.044673`
  - `entropy=9.153075`
  - `top10_mass=0.099704`
  - `used_codes=1125`
  - `P2=pass`, `P3=fail`
- 指標（epoch6）：
  - `train_total_loss=0.051096`
  - `val_total_loss=0.075968`
  - `feature_mse=0.043267`
  - `entropy=9.075493`
  - `top10_mass=0.122270`
  - `used_codes=1168`
  - `P2=pass`, `P3=fail`
- 指標（epoch7）：
  - `train_total_loss=0.049861`
  - `val_total_loss=0.075780`
  - `feature_mse=0.043054`
  - `entropy=9.034194`
  - `top10_mass=0.104802`
  - `used_codes=1128`
  - `P2=pass`, `P3=fail`
- 指標（epoch8）：
  - `train_total_loss=0.049379`
  - `val_total_loss=0.075361`
  - `feature_mse=0.042783`
  - `entropy=8.782626`
  - `top10_mass=0.158329`
  - `used_codes=1027`
  - `P2=pass`, `P3=fail`
- 指標（epoch9）：
  - `train_total_loss=0.049152`
  - `val_total_loss=0.075826`
  - `feature_mse=0.043354`
  - `entropy=8.991943`
  - `top10_mass=0.123076`
  - `used_codes=1101`
  - `P2=pass`, `P3=fail`
- 指標（epoch10）：
  - `train_total_loss=0.048262`
  - `val_total_loss=0.077853`
  - `feature_mse=0.045319`
  - `entropy=8.795537`
  - `top10_mass=0.167140`
  - `used_codes=1149`
  - `P2=pass`, `P3=fail`
- 指標（epoch11）：
  - `train_total_loss=0.047791`
  - `val_total_loss=0.074827`
  - `feature_mse=0.042413`
  - `entropy=8.672192`
  - `top10_mass=0.168185`
  - `used_codes=1037`
  - `P2=pass`, `P3=fail`
- 指標（epoch12）：
  - `train_total_loss=0.046887`
  - `val_total_loss=0.077772`
  - `feature_mse=0.045284`
  - `entropy=8.911071`
  - `top10_mass=0.107032`
  - `used_codes=1137`
  - `P2=pass`, `P3=fail`
- 變化（epoch1 -> epoch12）：
  - `val_mse: 0.074210 -> 0.045284`（-0.028926）
- 日誌定位：
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:89`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:93`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:215`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:224`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:233`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:242`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:251`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:260`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:269`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:278`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:286`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:294`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:303`

### 目前進度（執行中）
- 最新進度條（log 解析）：
  - `epoch 2, 464/1296 (~36%)`
- 已完成 epoch summary：
  - `1/100`

### 監控工具與快照（本輪新增）
- 監控腳本（只讀）：
  - `exp_0217/analysis_commit_5e859b0/monitor_m1_progress.py`
- 快照輸出：
  - `exp_0217/analysis_commit_5e859b0/m1_progress_snapshot.json`
- 快照結果（本輪）：
  - `epochs_logged=100`
  - `latest_progress=training_completed`
  - `latest_logged_metrics: mse=0.039526, P2=True, P3=False`

### checkpoint 里程碑（本輪達成）
- 已生成：
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/checkpoints/checkpoint_epoch010.pt`
- 意義：
  - 達成 `save_checkpoint_every=10` 的首個 checkpoint 驗收節點

### M1 完整執行完成（本輪更新）
- 訓練完成：
  - `epochs_logged=100`
  - checkpoints: `epoch010~epoch100` 皆存在
- 來源：
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/metrics_history.json`
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/checkpoints/`
- 終點指標（epoch100）：
  - `val_total_loss=0.071558`
  - `feature_mse=0.039526`
  - `P2=pass`, `P3=fail`

### M1 音質驗收補件（本輪新增）
- 輸出：
  - `audio_quality_m1_epoch050.json/.md`
  - `audio_quality_m1_epoch060.json/.md`
  - `audio_quality_m1_epoch100.json/.md`
- 驗收核對報告：
  - `M1_ACCEPTANCE_EVALUATION_20260222.md`
- 結論：
  - 中期進度（checkpoint 里程碑）已達成
  - 完整驗收（ΔPESQ/ΔSTOI/MSE門檻）**未達成（No-Go）**

### 風險與補件
- 風險：
  - `Audio save failed: Could not load libtorchcodec`
- 來源：
  - `exp_0217/runs/t453_m1_minw03_epoch100_debug/train.log:95`
- 影響：
  - `audio_samples` 尚無可用輸出，可能影響後續音質驗收
- 補件策略（不改訓練設定）：
  - 保持訓練繼續，於 `epoch50/100` checkpoint 完成後以獨立推論/評估腳本補做 `N>=100` 音質評估

## 統計檢定補件（已完成）
- 輸出：
  - `statistical_tests_summary.json`
  - `statistical_tests_summary.md`
- 方法：
  - Spearman
  - Mann-Whitney U
  - Bootstrap 95% CI
  - Benjamini-Hochberg FDR

## epoch222 補件（本輪新增）
- 問題：原 run 無 `checkpoint_epoch222.pt`
- 補件執行：
  - `replay_epoch222_from220.py`（從 epoch220 checkpoint 同設定續跑到 222）
  - 產物：`replay_from220_to222/checkpoints/checkpoint_epoch222_replay.pt`
- 補件評估：
  - `audio_quality_epoch222_replay.json`
  - `audio_quality_epoch222_replay.md`

## 唯一剩餘缺口
- `epoch_222` checkpoint 不存在，該節點音質不可評估。
- 已在下列文件明確標記：
  - `audio_quality_by_epoch.md`
  - `hypothesis_scoring.json`
  - `next_experiment_recommendation.md`

## 2026-02-23 補件：M2 定義與執行 Gate（本輪新增）
- 新增文件：
  - `M2_DEFINITION_20260223.md`
  - `M2_EXECUTION_GATE_20260223.md`

### 補件目的
- 解決「M2」語意混淆：
  1. 里程碑 M2（現況盤點）
  2. SPEC 模組 M2（PESQ/STOI 分析）
- 定義 M1 No-Go 後是否可進入下一輪最小改動的客觀 gate。

### 關鍵結論
- 里程碑 M2：已完成（`M2_current_inventory=100`）。
- SPEC 模組 M2：已完成（含 `epoch222` 缺口註記與 replay 補件）。
- 跨實驗比較覆蓋：`0206_plan_ori / 0206_v2 / 0216 / 0217`，且 `all_epoch300=true`。
- M2 執行 Gate：**Go（可進入執行規劃）**，但新一輪實驗尚未啟動。

### 證據來源
- `spec_coverage_status_20260219.json`
- `cross_experiment_300epoch_check_20260219.json`
- `M1_ACCEPTANCE_EVALUATION_20260222.md`
- `hypothesis_scoring.json`

## 2026-02-23 補件：M2 執行啟動（本輪新增）
- Run：
  - `exp_0217/runs/t453_m2_interw002_epoch100_20260223`
- 目的：
  - 依 `M2_EXECUTION_GATE_20260223.md` 啟動下一輪單因子改動驗證
- 單因子改動（相對 exp_0216 baseline）：
  - `intermediate_weight: 0.03 -> 0.02`
- 固定條件：
  - `t453_min_weight=0.2`, `t453_ramp_epochs=150`
  - `lambda_quant=1.0`, `beta_commit=1.0`
  - `epochs=100`

### 啟動快照
- `m2_progress_snapshot_20260223.json`
- 快照值：
  - `status=running`
  - `epochs_target=100`
  - `epochs_logged=0`（等待 epoch1 summary 寫入）
  - `latest_progress=Epoch 1 ... 242/1296 (~19%)`

### 當前阻塞
- `metrics_history.json` 尚未生成（屬正常，需待 epoch summary 落盤）
- 下一步：待 epoch1 結束後補 `mse/P2/P3` 首次量化快照
