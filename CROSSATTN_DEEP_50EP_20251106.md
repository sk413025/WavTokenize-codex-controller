# CrossAttn Deep（50 epoch）分析摘要

|epoch|W→C−C→W (zero) pp|ΔAcc (zero) pp|Mid ΔAcc (zero) pp|Mid Coverage (zero) %|cos_mean (mid)|Δmargin (mid)|
|---|---|---|---|---|---|---|
|10|-0.061|-0.061|4.027|3.126|-0.004|-0.154|
|20|0.317|0.317|7.959|3.016|-0.008|-0.999|
|30|0.516|0.516|4.580|6.550|-0.007|-1.167|
|40|0.740|0.740|3.993|11.073|-0.009|-1.657|
|50|0.522|0.521|2.795|16.900|-0.010|-1.957|

## 注意力熵（補充）
- 產物：`results/crossattn_k4_deep_50ep_20251105_211848/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
- 指標：normalized entropy（0→尖銳、1→均勻）的均值/分位數、`peaked_frac@0.7`。
- 判讀：若中‑margin 的 `ΔAcc` 為正（移除更好），且熵上升（更分散），可能代表注意力聚焦在不穩定/無用的 speaker tokens；需配合門控或方向性損失調整。
