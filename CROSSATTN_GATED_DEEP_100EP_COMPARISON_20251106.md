# CrossAttn Gated‑100 vs Deep‑100（補充指標）對比摘要

- 來源
  - Gated‑100：`results/crossattn_k4_gate_100ep_20251105_221334/analysis/supplemental_summary_100ep.csv`
  - Deep‑100：`results/crossattn_k4_deep_100ep_20251105_221426/analysis/supplemental_summary_100ep.csv`
- 指標
  - 熵：`ent_mean`、`ent_p90`、`peaked_frac`（normalized entropy 越低→越尖銳）
  - 門控（僅 gated）：`low_gate_frac_lowbin`（低‑margin 抑制率）、`high_gate_frac_mid`（中‑margin 放大量）

注意：目前補充分析正在生成，其餘 epoch 將隨完成自動補齊。

