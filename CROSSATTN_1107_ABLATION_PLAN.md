# 2025‑11‑07 機轉驗證對照（100 epoch × 2）

**目的**
- 驗證 11/06 分析提出的兩條機轉修復方向：
  1) Margin‑aware 門控（低 margin 壓 gate=0；中 margin 放大；高 margin 保守）
  2) 方向性輔助 loss（最大化 `Δlog p(target) − Δlog p(c2)` 對比「關 speaker」）
- 觀察中 margin（決策邊界）是否出現更穩健的正向行為（mid‑ΔAcc 更負、幾何轉正）。

**與先前實驗的對照與連結**
- GateL2‑100：高 margin 已正向且有強淨效益；中/低 margin 幾何仍偏負 → 需要門控調度與幾何對齊。
- Gated‑200：長訓練可持續強化高 margin 幾何正向；低 margin 仍需抑制。
- Deep‑200：中/低 margin 幾何負向更明顯，說明僅多層注入不足以修正方向性。

**重現與啟動**
- 一鍵啟動（tmux + 指定 GPU）：`done/exp/launch_ablation.sh:1`
  - Margin‑aware：`tmux new -d -s ma100 "bash -lc 'bash done/exp/launch_ablation.sh 1 ma ablations_ma_100ep 32'"`
  - Directional：`tmux new -d -s dir100 "bash -lc 'bash done/exp/launch_ablation.sh 2 dir ablations_dir_100ep 32'"`
- 分析（出現 e10/e20 後）：
  - `bash done/exp/run_behavior_analysis.sh <gpu> <results_dir> "10 20 30 40 50 80 100" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`
- 觀察重點（E1/E2/E3）：
  - Mid ΔAcc：e10→e100 單調更負；中 margin 覆蓋率增加時仍維持負值。
  - 幾何：`cos_mean_mid > 0`、`dmargin_mean_mid > 0`；高 margin 保持正向。
  - 門控分布（僅 gated）：低 bin `low_gate_frac` 高且上升；中 bin `high_gate_frac` 上升。

**成功判準**
- Margin‑aware：mid ΔAcc 穩定為負且幅度優於 GateL2‑100；幾何中 margin 轉正。
- Directional：幾何中 margin 的 `cos_mean/dmargin_mean` 明顯轉正；整體 ΔAcc_zero 更負。

**實作位置**
- 模型/訓練：
  - `done/exp/model_zeroshot_crossattn_gated.py:1`（margin 排程與 gate override）
  - `done/exp/train_crossattn_gated_cached.py:1`（方向性 loss；CLI 旗標；gate_init）
- 啟動腳本：`done/exp/launch_ablation.sh:1`

**備註**
- 預設 batch=32，已加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 避免碎片 OOM。
- 可調參：`low_thr/mid_thr/mid_amp/high_amp`、`dir_loss_weight`、`gate_init`。

