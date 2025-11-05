# CrossAttn K=4 Influence Breakdown（C→W/W→C）

目的
- 驗證機轉 H1/H3：為何 Speaker 改變造成大量 token 翻轉，但對整體準確率的淨影響極小？
- 以形式化分解 C→W/W→C/W→W/C→C，量化淨變化 ΔAcc。

數學定義
- 對每個 token 位置 i：有標籤 y_i，Normal 預測 \(\hat y^N_i\)，Variant 預測 \(\hat y^V_i\)（Zero/Random speaker）。
- 事件：
  - C→W: \(\hat y^N_i = y_i, \ \hat y^V_i \ne y_i\)
  - W→C: \(\hat y^N_i \ne y_i, \ \hat y^V_i = y_i\)
  - C→C, W→W 類推。
- 有效樣本數（非 padding）：\(N = |\{ i : y_i \ne 0 \}|\)。
- 準確率與淨變化：
  - \(\text{Acc}_N = \frac{1}{N}\sum_i 1[\hat y^N_i = y_i]\)，\(\text{Acc}_V\) 同理。
  - \(\Delta \text{Acc} = \text{Acc}_V - \text{Acc}_N = \frac{\#\text{W→C} - \#\text{C→W}}{N}\)。
  - 結論：若 W→C 與 C→W 近似相等，則淨變化接近 0（即便翻轉量很大）。

資料與程式
- Run 目錄：`results/crossattn_k4_100epochs_20251105_051626`
- 腳本：`done/exp/analyze_influence_breakdown.py`
- 產出：`analysis/influence_breakdown/epoch_{10,20,30,40}/breakdown_epoch_XX.csv`

重現指令
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_influence_breakdown.py \
  --results_dir results/crossattn_k4_100epochs_20251105_051626 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 10 20 30 40 --batch_size 32
```

關鍵結果（Epoch 40 範例）
- Normal→Zero:
  - Acc_N=37.7544%，Acc_V=38.1181%，ΔAcc=+0.3637pp
  - C→W=0.9174%，W→C=1.2810%，故 ΔAcc≈(1.2810−0.9174)=+0.3636pp
- Normal→Random:
  - Acc_N=37.7544%，Acc_V=37.5805%，ΔAcc=−0.1739pp
  - C→W=1.3763%，W→C=1.2023%，故 ΔAcc≈(1.2023−1.3763)=−0.1740pp
- 其他 epoch（10/20/30）亦呈現 ΔAcc 僅 ±0.1–0.36pp。

解讀與機轉驗證
- 支持 H1：翻轉結構中 W→C 與 C→W 彼此抵消，淨提升極小。
- 支持 H3：Speaker 改變造成的翻轉多屬「錯→錯」或「近鄰重排」，對正確率淨貢獻有限。
- 與邊界分析（見 Margins & Top‑k 報告）相呼應：低 margin 區翻轉極多，但淨 ΔAcc 不增反減或近零。

後續方向
- 針對「淨提升來源」鎖定提升 W→C、降低 C→W：
  - 門控殘差（margin-aware gate）：在低 margin 區抑制破壞性 C→W，放大具方向性的 W→C。
  - 多層注入/預置 speaker tokens：讓條件訊號更深入被 encoder 使用，提高方向性（見下一份報告）。

