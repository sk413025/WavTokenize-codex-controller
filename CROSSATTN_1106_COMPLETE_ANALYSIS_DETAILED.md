# 11/06 Cross‑Attention 完整機轉分析（詳細版）

目的
- 整合本輪 Gated/Deep/變體 GateL2 的關鍵分析結果（E1/E2/E3 + 補充），以驗證「方向性/長尾/門控」三軸機轉假說。
- 每節附上原始數據檔路徑與代表性指標，便於核對與再分析。

---

## Gated‑200（K=4）

關鍵數據（E3 幾何；方向性指標）
- 高 margin(0.4–1.01) 的 Δmargin 平均值 dmargin_mean（越大越好）
  - e80: results/crossattn_k4_gate_200ep_20251106_014033/analysis/logit_geometry/epoch_80/geometry_epoch_80.csv:1 → +0.123
  - e100: results/crossattn_k4_gate_200ep_20251106_014033/analysis/logit_geometry/epoch_100/geometry_epoch_100.csv:1 → +0.536
  - e150: results/crossattn_k4_gate_200ep_20251106_014033/analysis/logit_geometry/epoch_150/geometry_epoch_150.csv:1 → +0.913
  - e200: results/crossattn_k4_gate_200ep_20251106_014033/analysis/logit_geometry/epoch_200/geometry_epoch_200.csv:1 → +1.061
- 低/中 margin：dmargin_mean 仍為負值，幅度隨 epoch 稍增。

補充指標（注意力熵 + 門控彙整）
- results/crossattn_k4_gate_200ep_20251106_014033/analysis/supplemental_summary_200ep.csv:1
  - 熵均值 ent_mean：0.909(e10) → 0.726(e200)
  - peaked_frac_gt0.7：0.998(e10) → 0.738(e200)
  - 低 bin low_gate_frac_lowbin 維持高（~0.96→0.97），mid 高 gate 事件稀少且下降

解讀（機轉）
- 門控使高 margin 的擾動具明顯「朝目標方向」的分量（Δmargin>0 且隨 epoch 增強）。
- 注意力由極尖峰轉為更穩定集中，低 margin 長尾持續抑制（低 gate），有助降低破壞性翻轉。

---

## Deep‑200（K=4）

關鍵數據（E3 幾何）
- 高 margin dmargin_mean：
  - e80: results/crossattn_k4_deep_200ep_20251106_014239/analysis/logit_geometry/epoch_80/geometry_epoch_80.csv:1 → −1.095
  - e100: results/crossattn_k4_deep_200ep_20251106_014239/analysis/logit_geometry/epoch_100/geometry_epoch_100.csv:1 → −0.706
  - e150: results/crossattn_k4_deep_200ep_20251106_014239/analysis/logit_geometry/epoch_150/geometry_epoch_150.csv:1 → −0.709
  - e200: results/crossattn_k4_deep_200ep_20251106_014239/analysis/logit_geometry/epoch_200/geometry_epoch_200.csv:1 → −0.709
- 低 margin dmargin_mean：≈ −2.59(e80) → −3.09(e100) → −3.91(e150) → −4.60(e200)

補充指標（Deep 無門控/注意力熵）：
- results/crossattn_k4_deep_200ep_20251106_014239/analysis/supplemental_summary_200ep.csv:1（空表為預期）

解讀（機轉）
- 多層注入在長訓練下仍無法將擾動方向對齊目標，低/中 margin 長尾惡化明顯，解釋整體提升受限。

---

## GateL2‑100（K=4, 層數=2）

關鍵數據（E1 淨影響；移除/隨機化 speaker 對 Acc 影響）
- results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/influence_breakdown/epoch_10/breakdown_epoch_10.csv:1 → net ΔAcc_zero ≈ −0.83 pp
- results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/influence_breakdown/epoch_20/breakdown_epoch_20.csv:1 → ≈ −2.58 pp
- results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/influence_breakdown/epoch_30/breakdown_epoch_30.csv:1 → ≈ −5.38 pp
- results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/influence_breakdown/epoch_40/breakdown_epoch_40.csv:1 → ≈ −9.47 pp
- results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/influence_breakdown/epoch_50/breakdown_epoch_50.csv:1 → ≈ −10.50 pp
- results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/influence_breakdown/epoch_80/breakdown_epoch_80.csv:1 → ≈ −15.16 pp
- results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/influence_breakdown/epoch_100/breakdown_epoch_100.csv:1 → ≈ −15.91 pp

關鍵數據（E2 分桶；ΔAcc by margin）
- e10: results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/margins_topk/epoch_10/margins_bins_epoch_10.csv:1
  - 高 margin(0.4–1.01) ΔAcc_zero ≈ −1.15 pp（加入更好）
- e20: results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/margins_topk/epoch_20/margins_bins_epoch_20.csv:1
  - 高 margin(0.4–1.01) ΔAcc_zero ≈ −5.35 pp（加入更好）

進度說明
- 其餘 E2（30/40/50/80/100）與 E3 幾何（10→100）已在 GPU0 會話持續產出，完成後將補入對照圖表與指標摘要。

解讀（機轉）
- L2 門控變體更快、更強地依賴 speaker（net ΔAcc 更負），高 margin 的正貢獻提早顯現；待幾何完成後比對方向性與 Gated‑200 的一致性。

---

## 綜合結論（對應假設）

- H4 方向性：Gated 在高 margin 呈現強正向 Δmargin 且持續增強；Deep 在高/低 margin 多為負，尤其低 margin 惡化。
- H2 邊界行為：GateL2 與 Gated 在高 margin 的 ΔAcc 為負（移除更差），說明加入 speaker 提升正確率；低/中 margin 仍需針對性抑制。
- 注意力/門控：Gated 注意力由極尖峰走向穩定集中；低 margin 長期低 gate，符合「抑制破壞性擾動」的預期。

參考與連結
- Gated 補充報告：CROSSATTN_GATED_100_200_ANALYSIS_20251106.md:1
- Deep 補充報告：CROSSATTN_DEEP_100_200_ANALYSIS_20251106.md:1
- 重現指南：EXPERIMENT_REPRODUCTION_GUIDE.md:359

