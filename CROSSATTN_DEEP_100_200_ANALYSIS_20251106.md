# CrossAttn Deep（100/200 epoch）補充分析章節（占位）

- 注意力熵：`results/crossattn_k4_deep_100ep_*/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`，`results/crossattn_k4_deep_200ep_*/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
- 判讀：若 `ΔAcc_zero` 仍為正且熵趨於升高，代表加入 speaker 導致注意力更分散/無效；建議疊加 margin‑aware gate 或方向性輔助損失。

100ep 已得結果（先行摘要）
- 行為（E1/E2/E3）：e100 時整體 `ΔAcc_zero≈−4.70pp`（加入更好），但中‑margin `ΔAcc_zero≈+1.07pp`（移除更好），幾何 `Δmargin_mid≈−3.53`（偏負）。
- 熵（補充）：`supplemental_summary_100ep.csv` 目前僅 e10（`ent_mean≈0.772`），其餘 epoch 正在生成，將與 E2/E3 對照解讀。
- 解讀：長程訓練將整體行為翻轉為淨正向，但中‑margin 的方向性仍需強化；建議疊加 margin‑aware gate/方向性輔助 loss。
