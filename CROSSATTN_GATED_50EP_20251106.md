# CrossAttn Gated（50 epoch）行為分析（完成版）

背景/動機/目的
- 問題：speaker 使用率高但方向性不足，低‑margin 區造成破壞性擾動。
- 目的：在 cross‑attn 殘差上加入 per‑token gate，抑制低‑margin 的 C→W、放大中‑margin 的 W→C，觀察 50 epoch 的行為變化。

資料與指令（GPU0）
- Run：`results/crossattn_k4_gate_50ep_20251105_105730`
- 分析指令：同 Deep‑50（見 `done/exp/run_behavior_analysis.sh`）；彙整：`python -u done/exp/summarize_behavior_metrics.py --results_dir <RUN> --epochs 10 20 30 40 50`

主要結果（指標摘要；zero 條件，pp）

|epoch|W→C−C→W (zero) pp|ΔAcc (zero) pp|Mid ΔAcc (zero) pp|Mid Coverage (zero) %|cos_mean (mid)|Δmargin (mid)|
|---|---|---|---|---|---|---|
|10|-0.483|-0.483|||||
|20|-0.931|-0.931|||||
|30|-2.623|-2.623|||||
|40|-3.369|-3.369|||||
|50|-4.170|-4.170|||||

解讀與結論
- 整體（E1）：`ΔAcc_zero<0` 且幅度隨 epoch 增強（e50 ≈ −4.17pp），代表「移除 speaker 會變差」→ speaker 使用呈現淨正向。
- 中‑margin/幾何：對應的 E2/E3 已完成，數據將由補充章節與後續報告呈現；預期中‑margin `ΔAcc` 轉為負、覆蓋率上升。

重現步驟（摘要）
- E1/E2/E3 串跑：`CUDA_VISIBLE_DEVICES=0 done/exp/run_behavior_analysis.sh 0 results/crossattn_k4_gate_50ep_20251105_105730 "10 20 30 40 50" 5 16`
- 彙整：`python -u done/exp/summarize_behavior_metrics.py --results_dir results/crossattn_k4_gate_50ep_20251105_105730 --epochs 10 20 30 40 50`

## Gate 分佈（補充）
- 產物：`results/crossattn_k4_gate_50ep_20251105_105730/analysis/gate_distribution/epoch_XX/gate_stats_epoch_XX.csv`
- 指標：各 margin bin 的 gate 均值/分位數、`low_gate_frac(<0.2)`（低‑margin 抑制率）、`high_gate_frac(>0.8)`（中‑margin 放大量）。
- 判讀：低‑margin 應提高低 gate 比例（抑制 C→W）；中‑margin 應提高高 gate 比例（放大 W→C）。

## 注意力熵（補充）
- 產物：`results/crossattn_k4_gate_50ep_20251105_105730/analysis/attn_entropy/epoch_XX/entropy_epoch_XX.csv`
- 指標：normalized entropy（0→尖銳、1→均勻）的均值/分位數、`peaked_frac@0.7`。
- 判讀：隨 epoch，若中‑margin 的 `ΔAcc` 改善同時熵下降（更尖銳），多半代表注意力集中於有效 tokens；過度均勻或過尖銳需對照性能觀察是否不穩定。
