# CrossAttn Gated‑50 vs Deep‑50 vs Baseline（對齊 10/20/30/40/50）

目的
- 匯總 Gated‑50 與 Deep‑50 在 E1/E2/E3 的關鍵指標，與基線 K=4（同 epoch）對齊，檢驗「低‑margin 抑制 × 中‑margin 強化 × 方向性提升」是否成立。

資料
- Gated‑50：`results/crossattn_k4_gate_50ep_20251105_105730`
- Deep‑50：`results/crossattn_k4_deep_50ep_20251105_211848`
- Baseline（K=4, 100ep run）：`results/crossattn_k4_100epochs_20251105_051626`

彙整規則
- E1：取每 epoch 的 `W→C − C→W`、net ΔAcc（zero/random）。
- E2：取中‑margin bin 的 `ΔAcc`、覆蓋率、flip 率；低‑margin 的 flip 率與 ΔAcc（期望下降）。
- E3：取中‑margin 的 `cos_mean`、`Δmargin`；備註 W→C/C→W 子集之 cos。

重現（完成 E1/E2/E3 後彙整）
- 以 Python/Pandas 將兩個 run 的 epoch 指標合併到 `analysis/epoch10_50_summary.csv`，再與基線同 epoch 指標對齊。
- 待分析程式跑完後補上具體彙整程式與輸出路徑。

初步結論（占位）
- 待 E1/E2/E3 完成後填入：
  - 哪一種介入在中‑margin 的 `W→C − C→W` 與 `ΔAcc` 提升更明顯？
  - 低‑margin 是否被 gated 有效抑制？
  - 幾何指標是否顯示方向性改善？

---

