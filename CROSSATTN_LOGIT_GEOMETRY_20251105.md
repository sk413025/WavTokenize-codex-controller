# CrossAttn K=4 ΔLogits 幾何分析（方向性驗證）

目的
- 驗證機轉 H4：speaker 注入造成的 logits 差分 Δz 是否缺乏「朝目標方向」的分量，導致大量翻轉但淨增益小。

數學定義
- 比較 Normal 與 Zero speaker 的 logits：\(\Delta z = z^N - z^0\)。
- 以 Normal 的 top-2 為基準，定義方向向量 \(d = e_{target} - e_{c2}\)。
- 餘弦相似度：\(\cos = \langle \Delta z, d \rangle / (\|\Delta z\|\,\|d\|)\)，其中 \(\|d\|=\sqrt{2}\)。
- 目標 margin 變化：\(\Delta m = (z_t - z_{c2})^N - (z_t - z_{c2})^0\)。

資料與程式
- Run 目錄：`results/crossattn_k4_100epochs_20251105_051626`
- 腳本：`done/exp/analyze_logit_shift_geometry.py`
- 產出：`analysis/logit_geometry/epoch_{20,40}/geometry_epoch_E.csv`

重現指令
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_logit_shift_geometry.py \
  --results_dir results/crossattn_k4_100epochs_20251105_051626 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 20 40 --batch_size 32
```

關鍵結果（cos/dmargin 依 margin 分箱）
- Epoch 20：
  - [0,0.02): cos_mean≈−0.00065、Δm≈−0.095（朝目標方向分量近 0 且負小）
  - [0.2,0.4]: cos_mean≈−0.0157、Δm≈−1.505（方向性偏負）
  - [0.4,1.01]: cos_mean≈−0.0014、Δm≈−0.127（高 margin 幾乎無方向影響）
- Epoch 40：
  - [0,0.02): cos_mean≈−0.00431、Δm≈−0.986（低 margin 大量翻轉但明顯非朝目標）
  - [0.1,0.2]: cos_mean≈−0.0144、Δm≈−1.712（方向性偏負）
  - [0.4,1.01]: cos_mean≈−0.00230、Δm≈−0.188（高 margin 幾乎不影響）
- W→C / C→W 子集（Epoch 40）：
  - 在各 bin 中，cos_w2c_mean 通常高於 cos_c2w_mean（例如 [0.4,1.01]：0.033 > −0.019），顯示當 Δz 與目標方向更對齊時更可能 W→C；
  - 但全體 cos_mean 仍偏負或接近 0，說明大多數 Δz 並未對齊目標方向（支持 H4）。

解讀
- 低/中 margin 區：Δz 的方向多為中性至負向，無法穩定拉大 (z_t − z_{c2})，因此雖翻轉多（見 Margins & Top‑k）但淨增益小。
- 高 margin 區：Δz 幾乎不改變決策（cos≈0, Δm≈0），與 flip 率極低一致。
- 結論：現行注入位置/強度/結構下，speaker 訊號的「方向性」不足，與 Influence/Margins 的發現一致。

後續（方向性介入）
- 門控殘差（margin‑aware）：抑制低 margin 區的負向/中性擾動，放大具正向 cos 的擾動。
- 多層注入/預置 speaker tokens：讓條件訊號在 encoder 多層被處理，增強朝目標方向分量。
- 方向性輔助損失：最大化 Δm 或增大 \(\langle \Delta z, d \rangle\)；可與 Top‑k 目標（target‑in‑top‑k 提升）聯動。
