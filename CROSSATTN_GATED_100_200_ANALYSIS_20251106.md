# CrossAttn Gated（100/200 epoch）補充分析與連結

目的與方法
- 目的：觀察門控是否改善「方向性」並抑制低 margin 的負面影響。
- 指標：與 Deep 相同（E1/E2/E3），另加入：
  - 注意力熵：`analysis/attn_entropy/epoch_E/entropy_epoch_E.csv`
  - 門控分佈：`analysis/gate_distribution/epoch_E/gate_stats_epoch_E.csv`

重現步驟（任一 Gated run）
- 指令（自動跑 E1/E2/E3 + 熵 + 門控）：
  `bash done/exp/run_behavior_analysis.sh <gpu> <results_dir> "10 20 30 40 50 80 100 150 200" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`

Quick Links
- Repro Index：`EXPERIMENT_REPRODUCTION_GUIDE.md` 的「⚡ Quick Repro Index」
- 原始訓練指令：見 `RUNNING_EXPERIMENTS_20251105.md`（Gated‑100 範例與輸出路徑）

Gated‑100（K=4，run=results/crossattn_k4_gate_100ep_20251105_221334）
- 關鍵檔案
  - 幾何（e80）：`results/crossattn_k4_gate_100ep_20251105_221334/analysis/logit_geometry/epoch_80/geometry_epoch_80.csv`
  - 幾何（e100）：`results/crossattn_k4_gate_100ep_20251105_221334/analysis/logit_geometry/epoch_100/geometry_epoch_100.csv`
  - 分桶（e80/e100）：`results/crossattn_k4_gate_100ep_20251105_221334/analysis/margins_topk/epoch_XX/margins_bins_epoch_XX.csv`
- 觀察（E3 幾何方向性）
  - e80：高 margin(0.4–1.01) `dmargin_mean≈+0.535`，`cos_c2w_mean>0`（正向）；低 margin 仍為小幅負值。
  - e100：高 margin `≈+0.536`；延續正向趨勢。
- 觀察（E2 分桶 ΔAcc）
  - e80：低 margin `ΔAcc_zero≈-0.20pp`、高 margin `≈-15.66pp`（移除更差）。
  - e100：高 margin `≈-18.75pp`，顯示門控強化高 margin 的實質貢獻。
- 結論：門控明顯修正高 margin 的方向性與收益，低 margin 仍稍弱。

Gated‑200（K=4，run=results/crossattn_k4_gate_200ep_20251106_014033）
- 關鍵檔案
  - 影響分解：`results/crossattn_k4_gate_200ep_20251106_014033/analysis/influence_breakdown/epoch_200/breakdown_epoch_200.csv`
  - 幾何：`results/crossattn_k4_gate_200ep_20251106_014033/analysis/logit_geometry/epoch_150/geometry_epoch_150.csv`，`epoch_200/geometry_epoch_200.csv`
  - 幾何（新增）：`epoch_80/geometry_epoch_80.csv`，`epoch_100/geometry_epoch_100.csv`
  - 注意力熵：`results/crossattn_k4_gate_200ep_20251106_014033/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
  - 門控分佈：`results/crossattn_k4_gate_200ep_20251106_014033/analysis/gate_distribution/epoch_XX/gate_stats_epoch_XX.csv`
  - 補充彙整（200ep）：`results/crossattn_k4_gate_200ep_20251106_014033/analysis/supplemental_summary_200ep.csv`
- 觀察（E1 淨影響）
  - zero：e10 `≈-0.53pp` → e100 `≈-3.47pp` → e200 `≈-9.80pp`（移除更差，依賴度持續上升）。
- 觀察（E3 幾何方向性）
  - 高 margin：e80 `≈+0.12`、e100 `≈+0.54`、e150 `≈+0.91`、e200 `≈+1.06`（強正向增強）；低 margin 仍為負。
- 觀察（熵與門控）
  - 熵均值隨 epoch 下降：e10 `~0.91` → e200 `~0.73`；`peaked_frac_gt0.7` 同步下降（注意力由極尖峰轉向「較集中但非單峰」的穩定模式）。
  - 門控在中 margin 的 `high_gate_frac` 隨 epoch 增長（見各 epoch gate_stats 與彙整表），對應中‑高 margin 貢獻提高。
- 結論：門控 + 長訓練同時帶來更強的高 margin 方向性與整體淨效益；停滯仍集中於低 margin 長尾。

小結與建議
- 針對低 margin：
  - 在門控上加入 margin‑aware 調度（低 margin 壓 gate，高/中 margin 放大）。
  - 增補以目標對齊的幾何損失（對 Δlog p(target) 直接正規化）。
