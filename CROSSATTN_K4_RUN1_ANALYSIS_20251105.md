# Cross-Attention K=4 Run-1（Epoch 5–40）分析

目的
- 觀察 K=4 改版在真實資料（快取）上的行為，釐清：
  1) 退化是否解除？
  2) 為何 Speaker influence 很大，但移除/隨機 speaker 對 Val Acc 影響極小？

資料與路徑
- Run 目錄：`/home/sbplab/ruizi/WavTokenize-self-supervised/results/crossattn_k4_100epochs_20251105_051626`
- 驗證快取：`/home/sbplab/ruizi/c_code/done/exp/data/val_cache.pt`

分析腳本與指令
- Per-epoch 檢查（5,10,15,...,40）：
  - `python -u done/exp/analyze_crossattn_checkpoints.py --results_dir results/crossattn_k4_100epochs_20251105_051626 --use_real_cache --cache_dir /home/sbplab/ruizi/c_code/done/exp/data`
  - 產出：`results/.../analysis/summary.csv`
- 單一 checkpoint 深入（epoch 40）：
  - `python -u done/exp/quick_eval_crossattn.py --results_dir results/crossattn_k4_100epochs_20251105_051626 --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epoch 40`

關鍵結果
- 非退化：`attn_w_mean≈0.25, attn_w_std≈0.116–0.125`，注意力 token-wise 變異 >0。
- Epoch 40（val 批次）：
  - Speaker influence：Zero/Random 導致 ≈63% tokens 改變。
  - 準確率變化：acc_drop ≈ 0.33 個百分點（極小）。
  - Logit 分布：maxp≈0.386、entropy≈3.62（ln(4096)≈8.32），屬中等置信，未過度自信。
- 訓練曲線：Val Acc 約在 15–20 epoch 到 ~41–42% 後停滯；Train Acc ~60% 後趨緩。
- 參數範數：encoder/output head 持續上升；cross‑attn 相關權重變化較小。

解讀
- 模型確實「使用」speaker（大量翻轉），但多數翻轉屬於「錯→錯」的近鄰重排，對淨準確率幫助有限。
- 前置單次注入 + LayerNorm 可能稀釋了有效的 speaker 訊號；內容建模（encoder/output）仍是主力。
- 固定 LR 於 1e-4 使得早期快速提升後進入平臺期。

重現步驟
1) 每 epoch 分析：執行上方 per-epoch 指令，檢視 `analysis/summary.csv` 中 attn 權重統計。
2) Epoch 40 深入：執行 quick_eval 指令，取得 influence 與 entropy 指標。
3) 訓練日誌：`results/.../training.log` 可對照 Val Acc 停滯區間與 checkpoint 時間點。

