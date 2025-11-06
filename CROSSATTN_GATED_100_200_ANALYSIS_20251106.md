# CrossAttn Gated（100/200 epoch）補充分析章節（占位）

- Gate 分佈：`results/crossattn_k4_gate_100ep_*/analysis/gate_distribution/epoch_XX/gate_stats_epoch_XX.csv`，`results/crossattn_k4_gate_200ep_*/analysis/gate_distribution/epoch_XX/gate_stats_epoch_XX.csv`
- 注意力熵：`results/crossattn_k4_gate_100ep_*/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`，`results/crossattn_k4_gate_200ep_*/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
- 判讀：
  - 低‑margin：`low_gate_frac(<0.2)` 上升；熵下降（更聚焦）或穩定。
  - 中‑margin：`high_gate_frac(>0.8)` 上升；`ΔAcc` 與幾何改善相呼應。

