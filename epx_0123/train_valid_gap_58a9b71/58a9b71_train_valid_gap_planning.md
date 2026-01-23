# 分析規劃：commit `58a9b71`（Exp K v5）Train/Valid accuracy 落差

## 0. 背景與觀測現象

**關聯 commit**：`58a9b71d2c9621b6485fbd019854b1526d9efea6`  
**關聯程式**：
- 訓練：`exp_0112_intermediate/train_v5.py`
- 啟動腳本：`exp_0112_intermediate/run_exp_k_v5.sh`

**已觀測到的現象（以既有 run 為例）**：
- Run：`exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848`
- `best val_masked_acc`：`0.0089896`（`0.899%`）@ epoch `141`
- 最終 `train_masked_acc`：`0.0304570`（`3.046%`）
- 最終 `val_masked_acc`：`0.0084521`（`0.845%`）
- Gap：在 epoch `141` 約 `1.67%`（train `2.57%` vs val `0.90%`），到最終約 `2.20%`

**本次要解釋的 accuracy**：以訓練腳本輸出的 `masked_acc` 為主（Student codes 與 Teacher codes 做 masked、frame-level 的嚴格相等比對；在本 repo 的既有分析中，這個指標常被稱作 token accuracy / match acc）。

---

## 1. 分析目標（要回答的問題）

1. **這個 train-valid gap 的主因是什麼？**（度量計算/聚合、資料分佈差、對齊偏移、token collapse、多樣性不足、真實過擬合、或資料切分/快取問題）
2. **哪些因素是「評估/資料流程」造成的假象**，哪些因素是**模型泛化能力不足**？
3. 產出一份**可重現、可驗證**的診斷結果，讓後續改動（loss、資料、curriculum、regularization）能被客觀比較。

---

## 2. 核心假設（Hypotheses）與驗證方式

### H1：accuracy 的「聚合方式」讓 train 與 val 不可比
**典型症狀**：train/val 使用不同的平均方式（per-batch mean vs frame-weighted），或因長度分佈差造成偏差。  
**驗證**：
- 以同一個 checkpoint、同一套 dataloader，在 train/val 同時計算：
  - `acc_batch_mean`（每個 batch 的 acc 取平均）
  - `acc_frame_weighted`（total correct / total frames）
- 若兩種聚合方式差異很大，優先統一到 frame-weighted 作為主報告指標。

### H2：train 在 `model.train()` 狀態下算 acc，val 在 `model.eval()` 狀態下算 acc
**典型症狀**：dropout / layernorm 行為不同導致可比性降低。  
**驗證**：
- 固定 checkpoint，在 `eval()` 模式下重新跑 train set 的 evaluation（不做 backprop），與訓練期 log 的 train acc 對照。

### H3：train/val 的噪音難度分佈不同（SNR、noise type、speaker 等）
**典型症狀**：val 較難（更低 SNR 或不同噪音類型）導致 val acc 明顯偏低。  
**驗證**：
- 統計 train/val 的 SNR 分佈、音檔長度分佈、noise 類型（若 cache 有 metadata）。
- 做 **accuracy vs SNR** 的曲線（或分桶），看 gap 是否主要集中在低 SNR 區。

### H4：noisy-clean 的時間對齊偏移在 val 較嚴重（strict token equality 對 offset 非常敏感）
**典型症狀**：strict acc 低，但 tolerant acc（允許 ±1/±2 frame shift）提升明顯；且 val 的 lag 分佈更差。  
**驗證**：
- 以 `exp_1226/diagnose_alignment.py` 的方法計算 train/val 的 lag 分佈（至少抽樣 50–200 筆）。
- 同時計算 tolerant accuracy（例如在 token 序列上允許 ±k frame 的最佳對齊）並比較 train/val 的提升幅度。

### H5：token 分佈/多樣性問題（mode collapse 或 codebook usage mismatch）
**典型症狀**：train acc 可上升，但 val acc 卡住；val 上 student codes 的 unique code 數、entropy 明顯低於 teacher 或低於 train。  
**驗證**：
- 對 train/val 分別統計 student/teacher 的 token histogram、top-k 集中度、entropy、unique count、與 teacher 的 KL divergence。
- 參考 `exp_1226/quick_token_acc_check.py` 的診斷輸出格式。

### H6：資料切分/快取問題（重複樣本、speaker overlap、cache 內容不一致）
**典型症狀**：train 指標異常高、val 異常低，或重新生成 cache 後現象改變。  
**驗證**：
- 檢查 train/val cache 的 key（utterance id、path、speaker id）是否有交集（若 cache 有記錄）。
- 對同一批 raw 檔案重建 cache，確認結果一致（hash/統計量）。

### H7：目標函數與指標不一致（feature alignment 變好，但 token strict equality 不對應）
**典型症狀**：feature loss/中間層 cosine loss 有改善，但 token acc 改善有限；音質指標（PESQ/STOI/SI-SDR）可能變好。  
**驗證**：
- 在 train/val 上同時計算：feature-space cosine/MSE、以及 token acc（strict/tolerant）。
- 若 feature 指標改善但 token acc 不動，需重新檢視「研究主指標」與 loss 設計的一致性。

---

## 3. 分析路線圖（以最小代價先排除假象）

### Phase 0：復現與度量稽核（先確定「比的東西」是對的）
- 以既有 run 的 `history.json` 先重建：best epoch、gap 曲線、是否隨 epoch 擴大。
- 對同一 checkpoint 做一個「離線 eval」：統一用 `model.eval()`、同一套 dataloader 計算 train/val 的 strict acc（同時做 batch-mean 與 frame-weighted）。

### Phase 1：資料難度/對齊分佈剖析（確認 val 是否天生更難或更偏移）
- SNR 分佈、長度分佈、noise 類型（如可得）
- lag 分佈（cross-correlation）
- accuracy vs SNR / lag 的關聯

### Phase 2：模型行為診斷（token collapse、representation mismatch）
- token 使用分佈（student vs teacher；train vs val）
- feature-space alignment（final layer + supervised intermediate layers）
- 針對少數樣本做可視化（top-k codes、錯誤位置、對齊後是否改善）

### Phase 3：提出「可驗證的修正」與最小 follow-up
- 若主因是對齊：加入 tolerant metric、或在資料/評估做對齊校正；觀察 val strict/tolerant 的改善幅度。
- 若主因是分佈差：做分桶訓練/重加權/補齊 noise types；或修正 curriculum（確保真的在用 SNR 排序）。
- 若主因是 overfit：降低 LoRA capacity/epoch、增加正則化、或加 stronger validation protocol（多 seed、多 split）。

---

## 4. 產出物（Artifacts）

最終希望得到可直接放進報告/簡報的結果：
- `metrics_summary.json`：strict/tolerant、batch-mean/frame-weighted、train/val 對照
- `gap_curve.png`：gap 隨 epoch 變化（標註 best epoch）
- `snr_hist_train_vs_val.png`、`acc_vs_snr.png`
- `lag_hist_train_vs_val.png`、`acc_vs_lag.png`
- `token_usage_train_vs_val.png`（含 entropy/unique/top-k 集中度）
- `CONCLUSION.md`：用 H1–H7 的形式逐一寫「證據→結論→下一步」

---

## 5. 為什麼這跟研究目標有關

Exp K 的研究動機是：**用中間層監督讓 noisy 輸入下的 Student 表徵/離散碼更接近 Teacher（clean）**，以提升噪音魯棒性並服務下游（如 audio LM 的 token 序列品質）。  
因此：
- **train-valid gap 是泛化能力的直接指標**：若 gap 主要來自流程/度量假象，我們的結論會「看起來有效但不可重現」。
- 若 gap 主要來自對齊/分佈差，代表我們現在的 loss 或資料假設與真實驗證條件不一致；釐清後才能做出「可發表/可交付」的改進。
- 把 gap 拆成可量化因素（SNR、lag、token diversity、objective mismatch）能讓後續每次修改都有可比較的因果證據，而不是只靠單一 acc 數字賭運氣。

