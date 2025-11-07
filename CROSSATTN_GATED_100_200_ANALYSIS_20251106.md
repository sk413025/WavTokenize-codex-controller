# CrossAttn Gated（100/200 epoch）補充分析章節（占位）

- Gate 分佈：`results/crossattn_k4_gate_100ep_*/analysis/gate_distribution/epoch_XX/gate_stats_epoch_XX.csv`，`results/crossattn_k4_gate_200ep_*/analysis/gate_distribution/epoch_XX/gate_stats_epoch_XX.csv`
- 注意力熵：`results/crossattn_k4_gate_100ep_*/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`，`results/crossattn_k4_gate_200ep_*/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
- 判讀：
  - 低‑margin：`low_gate_frac(<0.2)` 上升；熵下降（更聚焦）或穩定。
  - 中‑margin：`high_gate_frac(>0.8)` 上升；`ΔAcc` 與幾何改善相呼應。

100ep 已得結果（先行摘要）
- 行為（E1/E2/E3）：`epoch_summary.csv` 顯示 e100 時 `ΔAcc_zero≈−8.98pp`、`mid ΔAcc_zero≈−1.46pp`、`mid_coverage≈29.5%`。
- 熵（補充）：`supplemental_summary_100ep.csv` 目前 e10→e20 的 `ent_mean` 由 ~0.892 降至 ~0.875（更聚焦），其餘 epoch 正在生成。
- 解讀：整體與中‑margin 均呈「加入 speaker 更好」，覆蓋率隨訓練上升；注意力有輕微聚焦趨勢。
