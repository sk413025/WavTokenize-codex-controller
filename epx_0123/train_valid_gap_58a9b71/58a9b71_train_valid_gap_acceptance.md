# 驗收標準：commit `58a9b71`（Exp K v5）Train/Valid accuracy 落差分析

## 1. 驗收目的

這份驗收標準用來判斷：我們是否真的「找到了 train/valid accuracy 落差很大」的主要原因，且能用可重現的證據支持結論，而不是停留在猜測。

---

## 2. 必須達成（MUST）

### M1：度量一致性（可比性）被確認
- 能同時回報 `acc_batch_mean` 與 `acc_frame_weighted`（train/val 都要有）。
- 明確說明最終報告採用哪一種（建議以 frame-weighted 為主），並給出理由。

### M2：離線 eval 可重現「gap 存在」
- 使用 `best_model.pt`（至少一個 checkpoint）在 train/val split 都跑過離線 eval（`model.eval()`、`no_grad`）。
- 結果能清楚展示：gap 的量級（例如 % 點）與方向（train > val）。

### M3：H1–H7 假設至少完成「可判定」的證據鏈
至少要讓每個假設都有其一：
- 支持（有數據/圖證）
- 不支持（有反證）
- 證據不足（但要說明缺什麼資料/怎麼補）

### M4：主因排序 + 可行下一步
在 `CONCLUSION.md` 中給出：
- Top-3 可能主因排序（含每項的關鍵證據）
- 對應的最小改動建議（每項建議需可在 1–2 天內完成驗證或被否證）

---

## 3. 應該達成（SHOULD）

### S1：對齊敏感度被量化
- 有 `lag` 分佈（train vs val）與 `acc_tolerant_k`（k≥2）的對照。
- 能回答：「val 低是不是主要因為 offset？」（是/否 + 量化提升幅度）。

### S2：資料難度差異被量化
- 有 SNR 分佈（或其他難度 proxy）與 `acc vs difficulty` 曲線。
- 能回答：「val 是否系統性更難？」以及「gap 是否集中在某一段難度」。

### S3：token collapse / 多樣性診斷完成
- 至少提供 student/teacher 的 token histogram 統計（entropy、unique、top-k mass），並比較 train vs val。

---

## 4. 可選達成（COULD）

### C1：提供一個最小修正的「前後對照」結果
例如：
- 改用 frame-weighted accuracy 作為主要報告後，gap 的估計是否顯著改變；或
- 加入 tolerant 評估後，能解釋 strict gap 的大部分來源；或
- 修正資料流程（如對齊/metadata/split）後，val acc 有一致改善。

### C2：多 seed / 多 checkpoint 穩健性
- 以至少 2–3 個 seed 或 2 個不同 checkpoint 重覆關鍵結論（避免只對單一 run 適用）。

---

## 5. 驗收檢查清單（Reviewer Checklist）

- [ ] `metrics_summary.json` 存在且包含 strict/tolerant、batch-mean/frame-weighted、train/val  
- [ ] `gap_curve.png` 能看出 gap 隨 epoch 的行為（含 best epoch 標註）  
- [ ] `snr_hist_train_vs_val.png`（或等價難度分析）存在  
- [ ] `lag_hist_train_vs_val.png` 存在  
- [ ] `token_usage_stats.json`（或等價 token 多樣性統計）存在  
- [ ] `CONCLUSION.md` 完整逐條回答 H1–H7，並提供 Top-3 主因與下一步建議  

