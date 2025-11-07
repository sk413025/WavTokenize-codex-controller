# CrossAttn Deep（100/200 epoch）補充分析與連結

目的與方法
- 目的：檢驗「speaker 通道的淨效益」與「方向性是否朝向目標」；聚焦於停滯（plateau）原因。
- 指標：
  - E1 Influence：`analysis/influence_breakdown/epoch_E/breakdown_epoch_E.csv`（ΔAcc、C→W/W→C）
  - E2 Margins/Top‑k：`analysis/margins_topk/epoch_E/margins_bins_epoch_E.csv`（依 margin 分桶的 ΔAcc/flip/top‑k）
  - E3 Logit Geometry：`analysis/logit_geometry/epoch_E/geometry_epoch_E.csv`（cos_mean、Δmargin）

重現步驟（任一 Deep run）
- 指令（自動跑 E1/E2/E3）：
  `bash done/exp/run_behavior_analysis.sh <gpu> <results_dir> "10 20 30 40 50 80 100" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`
- 或個別腳本：
  - Influence：`python -u done/exp/analyze_influence_breakdown.py --results_dir <dir> --cache_dir <cache> --epochs 10 20 ...`
  - Margins/Top‑k：`python -u done/exp/analyze_margins_topk.py --results_dir <dir> --cache_dir <cache> --epochs 10 20 ... --k 5`
  - Geometry：`python -u done/exp/analyze_logit_shift_geometry.py --results_dir <dir> --cache_dir <cache> --epochs 10 20 ...`

Quick Links
- Repro Index：`EXPERIMENT_REPRODUCTION_GUIDE.md` 的「⚡ Quick Repro Index」
- 原始訓練指令：見 `RUNNING_EXPERIMENTS_20251105.md`（Deep‑50/Deep‑100 範例與輸出路徑）

Deep‑100（K=4，run=results/crossattn_k4_deep_100ep_20251105_221426）
- 關鍵檔案
  - 幾何（e80）：`results/crossattn_k4_deep_100ep_20251105_221426/analysis/logit_geometry/epoch_80/geometry_epoch_80.csv`
  - 幾何（e100）：`results/crossattn_k4_deep_100ep_20251105_221426/analysis/logit_geometry/epoch_100/geometry_epoch_100.csv`
  - 分桶（e80）：`results/crossattn_k4_deep_100ep_20251105_221426/analysis/margins_topk/epoch_80/margins_bins_epoch_80.csv`
  - 分桶（e100）：`results/crossattn_k4_deep_100ep_20251105_221426/analysis/margins_topk/epoch_100/margins_bins_epoch_100.csv`
- 觀察（E3 幾何方向性）
  - e80：低 margin(0–0.02) `dmargin_mean≈-2.96`、高 margin(0.4–1.01) `≈-0.42`；`cos_mean` 在各桶多為負。
  - e100：低 margin `≈-3.48`、高 margin `≈-0.62`；方向性仍偏負。
- 觀察（E2 分桶 ΔAcc/flip）
  - e80：低 margin 移除 speaker `ΔAcc_zero≈+0.62pp`（移除更好）；高 margin `≈-8.50pp`（加入更好）。
  - e100：低 margin `≈+0.43pp`；高 margin `≈-12.40pp`（隨訓練高 margin 收益增加）。
- 結論：Deep‑100 整體已使用 speaker，但低/中 margin 的方向性與決策穩定度不足（害大於利），導致整體提升受限；高 margin token 顯示實質收益。

Deep‑200（K=4，run=results/crossattn_k4_deep_200ep_20251106_014239）
- 關鍵檔案
  - 影響分解：`results/crossattn_k4_deep_200ep_20251106_014239/analysis/influence_breakdown/epoch_100/breakdown_epoch_100.csv`（與 e80/e150/e200 對照）
  - 分桶（早期）：`results/crossattn_k4_deep_200ep_20251106_014239/analysis/margins_topk/epoch_40/margins_bins_epoch_40.csv`
- 觀察（E1 淨影響曲線，以 zero 為例）
  - e80 `net_acc_delta≈-3.75pp`、e100 `≈-4.18pp`、e150 `≈-6.75pp`、e200 `≈-9.56pp`（移除 speaker 更差，顯示依賴度上升）。
- 觀察（E2 分桶）
  - e40：高 margin `ΔAcc_zero≈-2.11pp` 已見正效益；低 margin `≈+0.06pp`、中 margin 小幅正負交替（不穩定）。
- 結論：長訓練強化了「高 margin 區的正向使用」，但低/中 margin 仍缺乏穩健方向性，成為停滯主因之一。

小結與建議
- Deep 系列在高 margin 具備實質增益；停滯主要來自低/中 margin 長尾方向性不足。
- 後續建議：對低 margin token 引入 margin‑aware gate 或方向性輔助損失；評估針對長尾的訓練采樣或對比式正則。
