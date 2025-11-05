# Cross-Attention 退化診斷（S=1 舊實驗）

目的
- 確認舊版 Cross-Attention 使用單一 speaker key（S=1）導致 softmax 退化為常數 1，造成注意力無法隨 token 變化，speaker 訊號僅成為被 LayerNorm 抵消的常數偏置。

資料與路徑
- 舊 run 結果目錄：`/home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/results/crossattn_100epochs_20251105_025951`
- 驗證快取：`/home/sbplab/ruizi/c_code/done/exp/data/val_cache.pt`

方法
- 使用腳本：`done/exp/analyze_crossattn_checkpoints.py`
- 指令：
  - `python -u done/exp/analyze_crossattn_checkpoints.py --results_dir done/exp/results/crossattn_100epochs_20251105_025951 --use_real_cache --cache_dir /home/sbplab/ruizi/c_code/done/exp/data`
- 產出：`done/exp/results/crossattn_100epochs_20251105_025951/analysis/summary.csv`

關鍵結果（摘要）
- `attn_w_mean=1.0, attn_w_std=0.0, attn_w_min=1.0, attn_w_max=1.0`（各 epoch）
- `attn_output_token_var_mean=0.0`（注意力對序列完全不變）
- encoder/output head 的參數範數持續上升，cross‑attn 相關權重變化小。

解讀
- 使用 `speaker_emb.unsqueeze(1)` 使 Key/Value 長度 S=1，MultiheadAttention 在 S 維度 softmax 退化為 1，`attn_output` 對所有 token 相同。
- 殘差 + LayerNorm 使該常數偏置被弱化，speaker conditioning 幾乎無效。

重現步驟
1) 準備 val 快取：`/home/sbplab/ruizi/c_code/done/exp/data/val_cache.pt`
2) 執行上方指令產生 `summary.csv`
3) 檢視 `attn_w_std` 與 `attn_output_token_var_mean` 皆為 0 的證據。

