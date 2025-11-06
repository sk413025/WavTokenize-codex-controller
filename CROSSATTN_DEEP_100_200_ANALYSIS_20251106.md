# CrossAttn Deep（100/200 epoch）補充分析章節（占位）

- 注意力熵：`results/crossattn_k4_deep_100ep_*/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`，`results/crossattn_k4_deep_200ep_*/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
- 判讀：若 `ΔAcc_zero` 仍為正且熵趨於升高，代表加入 speaker 導致注意力更分散/無效；建議疊加 margin‑aware gate 或方向性輔助損失。

