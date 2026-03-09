# TTT2 Training: Background, Objectives, Theory, and Code Evidence

本文件詳細記錄 ttt2.py 與 run_box_only_training_ttt2.sh 所涉及的訓練背景動機、目標、損失設計、背後的數學/信號處理直覺，以及與程式碼的一一對應證據，同時盤點合理性與潛在疑慮並提供改進建議。

---

## 背景與動機

- 預訓練的 WavTokenizer/Encodec 提供穩定的編解碼 latent 空間與解碼器映射。利用這個穩定基座，插入可訓練的「特徵增強器」(feature enhancer)，將輸入音訊的 encoder 特徵轉換成更接近目標 domain 的特徵（本腳本以 box→box2 為例），藉此改善解碼後的音質/材質匹配。
- 訓練時採「特徵對齊 + 內容一致性」：
  - 特徵對齊：讓增強後特徵 `enhanced_features` 接近目標音訊的 encoder 特徵 `target_features`（L2）。
  - 內容一致性：同一 `content_id` 的樣本在中間層特徵上保持相似（cosine），鼓勵模型分離「內容」與「材質/說話人」。
- 僅使用 `box` 材質作為輸入（環境變數 `ONLY_USE_BOX_MATERIAL=true`），目標對應 `box2`，以形成明確的 domain 對齊任務。

程式碼對應：
- `run_box_only_training_ttt2.sh` 設定 `ONLY_USE_BOX_MATERIAL=true` 與訓練旗標。
- `ttt2.py` 中的 `EnhancedFeatureExtractor` 與損失函數群 (`compute_feature_loss`, `compute_content_consistency_loss`, `compute_layered_hybrid_loss`) 負責上述流程。

---

## 訓練目標

- 目標 1：在預訓練 encoder/decoder 凍結情況下，學到一個中間的特徵增強器，使 decoder(固定) 對 `enhanced_features` 的輸出趨近對應 target domain 的音訊。
- 目標 2：學到在「中間層」具有內容不變性的表徵（同內容、不同說話人/材質應靠近），最後一層再對齊至目標的特徵分佈。

程式碼對應：
- `ttt2.py/EnhancedFeatureExtractor`: 凍結 Encodec encoder/decoder，僅訓練中間的卷積殘差堆疊。
- `ttt2.py/compute_layered_hybrid_loss`: 僅在第二層施加內容一致性（cosine），僅在最終層施加 L2 對齊。

---

## 數學與信號處理直覺

### 1) Latent 對齊與固定解碼器

- 設 `E` 為預訓練 encoder，`D` 為預訓練 decoder，`Fθ` 為可訓練 enhancer。
- 輸入音訊 `x`：
  - `z_in = E(x)`
  - `z_out = Fθ(z_in)`
  - `ŷ = D(z_out)`
- 目標音訊 `y` 的對應 latent：`z_tgt = E(y)`。
- 若 `z_out ≈ z_tgt`，則 `D(z_out)` 應趨近 `D(z_tgt) ≈ y`（在編解碼近似可逆與分佈穩定的假設下）。
- 這避免直接在時域做 MSE，轉而在 encoder latent 空間做對齊，通常更貼合感知（Encodec latent 與感知相關）。

程式碼對應：
- `ttt2.py/EnhancedFeatureExtractor.forward`: 先 `encode_infer` 取 `input_features`，經增強層得 `enhanced`，最後丟 `encodec.decoder`（凍結）得到 `decoded`。
- `ttt2.py/compute_feature_loss`: 用 L2 讓 `enhanced_features` 接近 `target_features`（即 `E(y)`)。

### 2) 內容一致性（cosine）

- 在中間層（指定第二層）對具有相同 `content_id` 的樣本，施加餘弦相似度損失，使其方向接近，忽略幅度差。
- 形式：對每個 content group，令該組特徵為 `{h_i}`，其均值 `μ = mean(h_i)`；計算 `1 - cos(h_i, μ)` 的平均。
- 直覺：鼓勵該層承載「內容」而減少「材質/說話人」干擾，類似於分解式表徵中的內容因子抽取。

程式碼對應：
- `ttt2.py/compute_content_consistency_loss`: 將同 content 的中間特徵與其組內均值做 cosine 距離平均。
- `ttt2.py/compute_layered_hybrid_loss`: 僅在 `intermediate_features_list[1]`（第二層）施加上述內容一致性，總損失 `total = α·content + β·L2`，預設 `α=0.01, β=0.99`。

### 3) 分層監督與自由中間層

- 第二層：內容一致性（cosine）
- 最後層：與 `E(y)` 的 L2 對齊
- 介於兩者的中間層：不施加損失，讓其自然形成過渡。

程式碼對應：
- `ttt2.py/compute_layered_hybrid_loss` 的實作與說明。

---

## 整體架構與資料流程（含程式碼證據）

- 預訓練模型載入與凍結：
  - `ttt2.py/EnhancedFeatureExtractor.__init__`
    - `base_model = WavTokenizer.from_pretrained0802(...)`
    - `self.encodec = base_model.feature_extractor.encodec`
    - 凍結 `self.encodec.encoder` 與 `self.encodec.decoder` 的參數（`requires_grad=False`）。
- 特徵增強器（可訓練）：
  - `LayerNorm(512)` → `Conv1d(512→256, k=1)` → `BN` → `GELU` → `Dropout` → 多個 `ResidualBlock(256)` → `Conv1d(256→512, k=1)` → `BN` → `GELU` → `LayerNorm(512)`。
  - 程式碼：`ttt2.py/EnhancedFeatureExtractor` 與 `ttt2.py/ResidualBlock`。
- 中間特徵擷取：
  - 每個 residual block 的輸出存入 `intermediate_features_list`（用於內容一致性損失選層）。
  - 程式碼：`ttt2.py/EnhancedFeatureExtractor.forward`。
- 損失：
  - 特徵 L2：`ttt2.py/compute_feature_loss`（只用 L2）。
  - 內容一致性：`ttt2.py/compute_content_consistency_loss`（cosine）。
  - 分層混合：`ttt2.py/compute_layered_hybrid_loss`（第二層 content + 最終層 L2）。
- 資料與抽樣：
  - 僅用 `box`：`ONLY_USE_BOX_MATERIAL=true`，在 `ttt2.py` 中路徑指向 `data/raw/box` 作為輸入、`data/clean/box2` 作為目標。
  - 內容感知批次：`ContentAwareBatchSampler`（確保每批至少 N 個相同 content_id 樣本）。
  - 程式碼：`ttt2.py/ContentAwareBatchSampler` 與 `ttt2.py` 下資料載入段落。
- 訓練與驗證：
  - `ttt2.py/train_model`：完整的迭代、記錄、驗證與學習率排程（`ReduceLROnPlateau`）。

---

## 合理性與潛在疑慮（含程式碼證據）

- Off-manifold 風險：
  - 只用 latent L2 並未保證 `enhanced_features` 落在 encoder 訓練分佈之內（缺少碼本一致性/先驗正則），decoder 對 off-manifold 輸入可能產生不可預期 artifacts。
  - 程式碼：目前未見對 `discrete_code`（`encode_infer` 回傳）做任何一致性監督。
- 時間對齊假設：
  - L2 與 cosine 都近似逐幀比較（對通道做 norm/cos，時間維保留後再平均），若 input-target 或同 content 的對齊有偏移（起點/語速），損失會懲罰正確內容。
  - 程式碼：`compute_feature_loss` 與 `compute_content_consistency_loss` 的逐時間框處理。
- 內容損失權重偏弱：
  - `alpha=0.01` 可能不足以驅動清楚的「內容因子」分離，尤其在 `beta=0.99` 的 L2 主導下。
  - 程式碼：`compute_layered_hybrid_loss` 固定 α/β。
- 中間層完全自由：
  - 無任何正則/監督，可能造成梯度不穩定或非理想分解（尤其在 BN+Dropout 下）。
  - 程式碼：僅第二層與最終層受損失約束。
- BN/Dropout 的穩定性：
  - 語音時序長度變化、批次小、內容感知抽樣 → BN 統計不穩定。序列任務常以 GroupNorm/LayerNorm 取代 BN。
  - 程式碼：`ResidualBlock` 與上下游 `BatchNorm1d` 廣泛使用。
- 旗標語義一致性：
  - `--first_two_blocks_only` 目前不影響實作路徑，實際 loss 固定為「第二層內容 + 最終層 L2」，易致實驗解讀混淆。
  - 程式碼：旗標僅印訊息，未改變 loss 邏輯分支。
- 殘差塊實作瑕疵（重要）：
  - `ResidualBlock.forward` 於 conv1 後的第二次卷積使用 `self.conv2(x)` 而非 `self.conv2(out)`，導致第一層輸出被丟棄，功能受限。
  - 程式碼：`ttt2.py/ResidualBlock.forward`

---

## 建議改進（高優先級在前）

- 修正殘差塊實作：
  - 將 `self.conv2(x)` 改為 `self.conv2(out)`；評估以 GroupNorm/LayerNorm 取代 BN，提升穩定性。
- On-manifold 正則：
  - 加碼本一致性：將 `enhanced_features` 經相同 quantizer 映射到離散碼，與 target 的 `discrete_code` 做一致性/交叉熵損失；或引入 latent 先驗/對抗器避免 off-manifold。
- 頻譜/感知輔助損失：
  - 加入 multi-resolution STFT 或 log-mel 損失輔助，降低只靠 latent L2 的 artifacts 風險。
- 對齊魯棒化：
  - 內容一致性採時間池化（均值/attention pooling）或 segment-level 對比學習（SupCon/InfoNCE），放寬逐幀對齊假設。
- 權重與消融：
  - 掃描 `alpha`（0.01, 0.05, 0.1…）與不同層覆蓋策略，觀察內容損失實際貢獻；讓 `--first_two_blocks_only` 真正切換 loss 覆蓋。
- 擴充資料：
  - 若目標是跨材質泛化，不應只限 box→box2；可多 domain paired 或條件化訓練。
- L2 形式微調：
  - 考慮改用標準 MSE（不含 sqrt）或在通道/時間上做權重化以改善梯度性質。

---

## 聲明與程式碼對應摘要

- 目標：latent 對齊 + 內容一致性 → `EnhancedFeatureExtractor`, `compute_feature_loss`, `compute_content_consistency_loss`, `compute_layered_hybrid_loss`。
- 凍結/可訓練：`self.encodec.encoder/decoder` 凍結；中間卷積殘差層可訓練。
- 分層監督：第二層 cosine、最終層 L2；中間層自由。
- 資料與批次：`ONLY_USE_BOX_MATERIAL=true`；`ContentAwareBatchSampler` 確保同 content 批次共現。
- 疑慮：off-manifold、時間對齊、BN/Dropout、權重偏弱、旗標語義、殘差塊 bug。

---

## 附：運行方式與旗標（概覽）

- 腳本：`run_box_only_training_ttt2.sh`
  - 設定：`--tsne_flow_with_content --use_layered_loss --first_two_blocks_only`（目前最後一個旗標不影響損失實作）。
  - `ONLY_USE_BOX_MATERIAL=true` 僅用 `data/raw/box` 作為輸入；target 為 `data/clean/box2`。

---

若需要，我可以：
- 直接修正 `ResidualBlock` 的實作瑕疵並提交補丁。
- 加入簡易碼本一致性/量化損失與 multi-resolution STFT 輔助損失的最小實作，便於做消融比較。

