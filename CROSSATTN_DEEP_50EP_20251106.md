# CrossAttn Deep（50 epoch）行為分析（完成版）

背景/動機/目的
- 問題：基線 K=4 修正退化後，Train/Val Acc 仍平臺。懷疑在決策邊界（中‑margin）附近，speaker 的 cross‑attn 擾動「方向性不足」，導致大量使用但淨效有限甚至為負。
- 目的：驗證「多層注入（Deep）」在 50 epoch 是否能提升中‑margin 的正向作用，並對齊基線觀察整體趨勢。

資料與指令（GPU0）
- Run：`results/crossattn_k4_deep_50ep_20251105_211848`
- 分析指令（E1/E2/E3）：見 `done/exp/run_behavior_analysis.sh`（或逐條執行 `analyze_*`）；彙整指令：`python -u done/exp/summarize_behavior_metrics.py --results_dir <RUN> --epochs 10 20 30 40 50`

主要結果（指標摘要）
- 表（zero 條件；pp 為百分點）

|epoch|W→C−C→W (zero) pp|ΔAcc (zero) pp|Mid ΔAcc (zero) pp|Mid Coverage (zero) %|cos_mean (mid)|Δmargin (mid)|
|---|---|---|---|---|---|---|
|10|-0.061|-0.061|4.027|3.126|-0.004|-0.154|
|20|0.317|0.317|7.959|3.016|-0.008|-0.999|
|30|0.516|0.516|4.580|6.550|-0.007|-1.167|
|40|0.740|0.740|3.993|11.073|-0.009|-1.657|
|50|0.522|0.521|2.795|16.900|-0.010|-1.957|

解讀與結論
- 整體（E1）：自 e20 起，`ΔAcc_zero>0` 且 `W→C−C→W>0`，代表「移除 speaker 反而更好」。
- 中‑margin（E2）：`mid ΔAcc_zero` 為正（e20 +7.96 → e50 +2.80pp），顯示 Deep 注入在決策邊界附近多數為「破壞性擾動」。
- 幾何（E3）：`cos_mean_mid` 小負、`Δmargin_mid` 顯著負，方向性偏離目標邊界的趨勢與 E1/E2 一致。
- 小結：單靠多層注入不足以解決平臺；需疊加「margin‑aware 門控」或「方向性輔助損失」來抑制低‑margin 的 C→W、強化中‑margin 的 W→C。

重現步驟（摘要）
- E1/E2/E3 串跑：`CUDA_VISIBLE_DEVICES=0 done/exp/run_behavior_analysis.sh 0 results/crossattn_k4_deep_50ep_20251105_211848 "10 20 30 40 50" 5 16`
- 彙整：`python -u done/exp/summarize_behavior_metrics.py --results_dir results/crossattn_k4_deep_50ep_20251105_211848 --epochs 10 20 30 40 50`

產物清單（部分）
- `analysis/epoch_summary.csv`
- `analysis/influence_breakdown/epoch_XX/breakdown_epoch_XX.csv`
- `analysis/margins_topk/epoch_XX/margins_bins_epoch_XX.csv`
- `analysis/logit_geometry/epoch_XX/geometry_epoch_XX.csv`

## 注意力熵（補充）
- 產物：`results/crossattn_k4_deep_50ep_20251105_211848/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
- 指標：normalized entropy（0→尖銳、1→均勻）的均值/分位數、`peaked_frac@0.7`。
- 判讀：若中‑margin 的 `ΔAcc` 為正（移除更好），且熵上升（更分散），可能代表注意力聚焦在不穩定/無用的 speaker tokens；需配合門控或方向性損失調整。
