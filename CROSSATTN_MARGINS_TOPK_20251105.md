# CrossAttn K=4 Margins & Top‑k Stability（決策邊界/穩定度）

目的
- 驗證機轉 H2：低決策邊界（小 margin）位置對 speaker 改變極度敏感，但翻轉多屬近鄰洗牌，對整體準確率貢獻有限。
- 以機率與排名的觀點，度量翻轉位置的「難度」與「有用性」。

數學定義
- 對每個位置 i，softmax 機率 \(p_i(c)=\text{softmax}(z_i)\)。令 \(c_1=\arg\max_c p_i(c), c_2=\arg\max_{c\ne c_1} p_i(c)\)。
- 決策邊界距離（margin）：\(m_i = p_i(c_1) - p_i(c_2)\in[0,1]\)。m 越小表示「近決策邊界」。
- Top‑k overlap：\(\text{overlap}_i = |K^N_i \cap K^V_i|/k\)（Normal/Variant 的前 k 候選集相交比例）。
- 以 margin 分箱：統計每個 bin 的 flip rate、ΔAcc、overlap、以及 target‑in‑top‑k 的變化。

資料與程式
- Run 目錄：`results/crossattn_k4_100epochs_20251105_051626`
- 腳本：`done/exp/analyze_margins_topk.py`
- 產出：`analysis/margins_topk/epoch_{10,20,30,40}/margins_bins_epoch_XX.csv` 與對應圖（flip_rate/delta_acc/topk_overlap）。

重現指令
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_margins_topk.py \
  --results_dir results/crossattn_k4_100epochs_20251105_051626 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 10 20 30 40 --k 5 --batch_size 32
```

關鍵結果（Epoch 40 範例，Zero Speaker）
- 低 margin [0,0.02): flip_rate=99.06%，ΔAcc=−0.1744pp，top‑k overlap≈3.72%（近乎完全洗牌），target‑in‑top‑k 下降（−1.48pp）。
- 中 margin [0.1,0.2]: flip_rate=64.32%，ΔAcc=+19.08pp，top‑k overlap≈57.78%（重排但更接近目標），target‑in‑top‑k 上升（+2.70pp）。
- 高 margin [0.4,1.01]: flip_rate=2.37%，ΔAcc≈0，top‑k overlap≈81.51%（對穩定預測影響極小）。
- Random Speaker 與 Zero 趨勢一致：低 margin 大洗牌但收益有限，中 margin 少量翻轉帶來較大增益，高 margin 幾乎不動。

解讀與機轉驗證
- 支持 H2：低 margin 翻轉多屬近鄰洗牌（overlap 低、ΔAcc 近零或負），顯示 speaker 的擾動未能有效「朝目標方向」推進。
- 搭配 Influence 分解可知：W→C 與 C→W 在全域互抵；而在中 margin 區，少數翻轉反而更「有方向性」，對 Acc 有顯著正貢獻。
- 支持 H4 的推論：需要改變條件化的「位置/強度」，使訊號在「可帶來正貢獻」的區域更有效。

物理/數學直覺（為何低 margin 易翻、但提升少）
- softmax 近決策邊界處 \(m\approx 0\) 時，兩類的 logit 差值很小；微小擾動（如 speaker 注入）足以改變排序（高 flip），
  但若擾動沒有「對齊目標方向」，就會在近鄰間反覆重排，未提升 target‑in‑top‑k（overlap 低、ΔAcc 近 0）。
- 反之，中 margin 區擾動必須「更具方向性」才能翻轉，因而一旦翻轉更可能是朝向目標（ΔAcc 大於 0、target‑in‑top‑k 上升）。

後續方向（對應機轉修正）
- 門控殘差（margin-aware）：以 margin 作隱性 proxy，低 margin 區抑制破壞性擾動、高 margin 區保守、中 margin 區放大有用擾動。
- 多層注入/預置 speaker tokens：避免單次前置注入被 LayerNorm 稀釋，讓條件訊號在 encoder 多層整合，提升方向性。
- 方向性輔助損失：鼓勵 \(\Delta z = z^N - z^0\) 在 target‑vs‑distractor 的投影方向上增加，提升中 margin 區的 W→C 機率。

