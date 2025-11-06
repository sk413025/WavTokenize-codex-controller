# CrossAttn Gated（50 epoch）分析摘要

|epoch|W→C−C→W (zero) pp|ΔAcc (zero) pp|Mid ΔAcc (zero) pp|Mid Coverage (zero) %|cos_mean (mid)|Δmargin (mid)|
|---|---|---|---|---|---|---|
|10|-0.483|-0.483|||||
|20|-0.931|-0.931|||||
|30|-2.623|-2.623|||||
|40|-3.369|-3.369|||||
|50|-4.170|-4.170|||||

## Gate 分佈（補充）
- 產物：`results/crossattn_k4_gate_50ep_20251105_105730/analysis/gate_distribution/epoch_XX/gate_stats_epoch_XX.csv`
- 指標：各 margin bin 的 gate 均值/分位數、`low_gate_frac(<0.2)`（低‑margin 抑制率）、`high_gate_frac(>0.8)`（中‑margin 放大量）。
- 判讀：低‑margin 應提高低 gate 比例（抑制 C→W）；中‑margin 應提高高 gate 比例（放大 W→C）。

## 注意力熵（補充）
- 產物：`results/crossattn_k4_gate_50ep_20251105_105730/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
- 指標：normalized entropy（0→尖銳、1→均勻）的均值/分位數、`peaked_frac@0.7`。
- 判讀：隨 epoch，若中‑margin 的 `ΔAcc` 改善同時熵下降（更尖銳），多半代表注意力集中於有效 tokens；過度均勻或過尖銳需對照性能觀察是否不穩定。
