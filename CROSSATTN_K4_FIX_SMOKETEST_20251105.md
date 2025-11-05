# Cross-Attention K=4 修正與 Smoke Test（合成資料）

背景與目的
- 為避免 S=1 退化，將 speaker 向量展開為 `K>1` 個 speaker tokens，令 Cross‑Attention 的 Key/Value 長度 S=K（本次採用 K=4）。
- 先以合成小資料（不依賴快取）進行 3 epoch smoke test，驗證注意力不再退化並觀察 speaker influence。

程式變更
- `done/exp/model_zeroshot_crossattn.py`：
  - `CrossAttentionFusion(..., speaker_tokens=4)`；新增 `spk_expand` 與 `spk_pos`，生成 `(B, K, D)` 的 speaker tokens。
- `done/exp/train_crossattn_cached.py`：新增參數 `--speaker_tokens`。
- 新增 `done/exp/smoke_test_crossattn_k4.py`。

重現指令
- `python -u done/exp/smoke_test_crossattn_k4.py --epochs 3 --speaker_tokens 4`

關鍵結果（合成資料）
- Attention stats（K=4）：`mean≈0.25, std≈0.031, var_across_tokens_mean≈9.6e-4`（>0，非退化）
- Speaker influence：Normal→Zero/Random 時，≈75% tokens 改變；短訓練與隨機 codebook 下整體 acc 仍低（僅 sanity）。

結論
- K=4 成功避免注意力退化（attn 權重具變異且隨 token 變化）。
- 合成任務僅作結構驗證；正式效能需在真實快取上評估（見 Run1 分析）。

