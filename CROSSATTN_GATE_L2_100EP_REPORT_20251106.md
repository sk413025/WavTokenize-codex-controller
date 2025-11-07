# CrossAttn GateL2（K=4, 100 epoch）實驗報告（2025‑11‑06）

**Run 目錄**: `results/crossattn_k4_gateL2_100ep_20251106_000140`

**目的與動機**
- 背景：前一輪分析顯示平臺期主因在於「低/中 margin token 的方向性不足」，導致破壞性翻轉（C→W）與正負抵消；高 margin 區已有穩健收益（加入 speaker 更好）。
- 動機：在 Gated 類方法中測試較淺層（L2）門控變體的行為是否能更快拉出正向效益；觀察中/低 margin 區是否改善。
- 假設（機轉）：
  - H4（方向性）：高 margin 應呈現正向 Δmargin/cos；中/低 margin 容易負向。
  - H2（邊界行為）：中 margin 應能出現「加入 speaker 更好」的淨效益（ΔAcc_zero < 0）。

**重現步驟（取得 E1/E2/E3）**
- 一鍵分析（建議在 GPU0/1/2 任一空閒）：
  `bash done/exp/run_behavior_analysis.sh <gpu> results/crossattn_k4_gateL2_100ep_20251106_000140 "10 20 30 40 50 80 100" 5 16 /home/sbplab/ruizi/c_code/done/exp/data`
- 產物路徑：
  - E1 影響分解：`analysis/influence_breakdown/epoch_E/breakdown_epoch_E.csv`
  - E2 分桶：`analysis/margins_topk/epoch_E/margins_bins_epoch_E.csv`
  - E3 幾何：`analysis/logit_geometry/epoch_E/geometry_epoch_E.csv`

**關鍵結果（e100）**
- E1 淨影響（移除為 zero；數值為 ΔAcc_zero）
  - `results/crossattn_k4_gateL2_100ep_20251106_000140/analysis/influence_breakdown/epoch_100/breakdown_epoch_100.csv:1`
  - net_acc_delta_pct（zero）：約 −15.91 pp；（random）：約 −16.08 pp
  - 解讀：到 e100 已高度依賴 speaker，且整體效益為正（移除更差）。

- E2 分桶（ΔAcc by margin；加入更好 → 負值）
  - `results/.../margins_topk/epoch_100/margins_bins_epoch_100.csv:1`
  - 低 margin：0–0.02 `≈ −0.23pp`；0.02–0.05 `≈ −0.38pp`；0.05–0.1 `≈ −0.74pp`
  - 中 margin：0.1–0.2 `≈ −1.72pp`；0.2–0.3 `≈ −4.81pp`；0.3–0.4 `≈ −10.25pp`
  - 高 margin：0.4–1.01 `≈ −34.29pp`
  - 解讀：e100 時中/高 margin 已呈現明顯淨效益；低 margin 仍僅小幅收益。

- E3 幾何（方向性與 margin 變化）
  - `results/.../logit_geometry/epoch_100/geometry_epoch_100.csv:1`
  - 高 margin：`cos_mean ≈ +0.004`、`dmargin_mean ≈ +2.59`（強正向）
  - 中 margin（0.2–0.4 合併行）：`cos_mean ≈ −0.0081`、`dmargin_mean ≈ −1.89`
  - 低 margin（0–0.2）：`cos_mean ≈ −0.009~−0.010`、`dmargin_mean ≈ −2.00~−2.21`
  - 解讀：中/低 margin 幾何仍偏負（方向性不足），與 E2 顯示的「決策收益已出現」存在落差，顯示收益主要由高 margin 貢獻。

**對照與連結**
- Gated（100/200ep）：高 margin 幾何強正向；中 margin 在長訓練下持續改善；低 margin 偏負（有門控抑制）。`CROSSATTN_GATED_100_200_ANALYSIS_20251106.md:1`
- Deep（100/200ep）：高/低 margin 幾何偏負；低/中 margin 惡化更明顯。`CROSSATTN_DEEP_100_200_ANALYSIS_20251106.md:1`
- 綜合：GateL2‑100 表現與 Gated 系列一致的趨勢（高 margin 正向強），但中/低 margin 幾何仍需針對性對齊。

**結論與後續**
- 結論：GateL2 在 e100 時已展現強整體淨效益與高 margin 幾何正向；停滯與風險仍集中於中/低 margin 的方向性不足。
- 後續（已立題並啟動的對照）：
  - Margin‑aware 門控：低 margin gate→0；中 margin 放大；高 margin 保守（目標：mid‑ΔAcc 更負且幾何轉正）。
  - 方向性輔助 loss：最大化 `Δlog p(target) − Δlog p(c2)` 相對「關 speaker」的增益（目標：cos_mean_mid/Δmargin_mid > 0）。
- 重現與分析請參考：`EXPERIMENT_REPRODUCTION_GUIDE.md:243` 的 Quick Repro Index 與 `done/exp/run_behavior_analysis.sh:1`。

