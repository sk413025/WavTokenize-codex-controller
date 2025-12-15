# 關鍵問題分析報告：Feature Loss 與 Token Accuracy 脫鉤的真相

## 1. 專家分析驗證 (Verification of Expert Analysis)

經過查閱 `exp_1209/models.py`, `exp_1209/train_lora_expanded.py` 以及底層的 quantization 代碼，確認專家的發現 **完全正確 (True)**，並且擊中了問題的核心。

### 三個致命細節：

1.  **Feature Loss 確實是在 "VQ 前" 計算 (已驗證)**
    *   **位置**: `exp_1209/models.py` (lines 367-376)
    *   **代碼**:
        ```python
        student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)
        # ... VQ 是在後面才做 ...
        return {
            'student_encoder_out': student_encoder_out,  # 這就是 Feature Loss 用的 input
            # ...
        }
        ```
    *   **後果**: Feature Loss 拼命拉近 Encoder 的連續輸出，讓 Loss 下降，但**完全沒管 VQ 是否對齊**。

2.  **Student Codebook 真的在偷偷漂移 (已驗證 - 最嚴重!)**
    *   **機制**:
        *   在 `exp_1209/train_lora_expanded.py` line 95: `model.student.train()` **強制開啟了訓練模式**。
        *   在 `encoder/quantization/core_vq.py` line 217:
            ```python
            if self.training:  # 因為上面開了 train，這裡會進去
                # ... 計算 EMA ...
                self.embed.data.copy_(embed_normalized)  # <--- Codebook 被修改了！
            ```
    *   **後果**: Teacher 的 Codebook 是凍結的，但 Student 的 Codebook 每一輪都在變（試圖去適應那 18 層 LoRA 產生的新特徵）。導致 **Teacher 和 Student 的 Token ID (比如 "5") 代表的向量已經不一樣了**。您在用 "舊地圖 (Teacher)" 的標準去衡量 "新地圖 (Student)" 的座標，導致 VQ Distance (距離) 越來越大，Token Accuracy 極低。

3.  **n_q (Quantizer 數量) 的不一致 (已驗證)**
    *   **Student (Training)**: `vq.py` line 103 顯示隨機選 `[4, 6, 8]` 個 quantizers。
    *   **Teacher (Inference)**: `models.py` line 362 設定 `bandwidth=0.075`，換算後確實只用了 **1 個 quantizer**。
    *   **後果**: Student 依賴多個 quantizers 來擬合，而 Teacher 只要一個。這讓兩者的特徵空間更難對齊。

---

## 2. 新發現的連鎖問題 (Chain Reaction of Errors)

除了上述問題，這個漂移還對 `Triplet Loss` 和 `Hard Negative Mining` 造成了毀滅性的**連鎖反應**。

### A. Triplet Loss 在拿「明朝的劍」斬「清朝的官」
在 `exp_1209/losses.py` line 150:
```python
hard_negatives = self.get_hard_negatives(student_out, codebook, teacher_codes)
```
*   **這裡的 `codebook`**：是 Teacher 的 **凍結 Codebook** (2023年版字典)。
*   **這裡的 `student_out`**：是 Student 經過 LoRA 生產的特徵。
*   **問題**：Student 的 Quantizer 剛剛偷偷用 EMA 把自己的 Codebook 更新了 (2024年版)，Student 的特徵是為了適應那個新 Codebook 而產生的。
*   **結果**：你拿 Teacher 的舊 Codebook 去衡量 Student 的新特徵，計算出的距離完全是錯的。Student 覺得自己已經跑到了正確的 Code 位置，但在 Teacher 的座標系裡，它可能跑到了十萬八千里外。

### B. Hard Negative Mining 正在挑選「假」的敵人
在 `exp_1209/losses.py` line 122:
```python
hard_neg_idx = dists.argmin(dim=1)  # 挑選距離最近的錯誤 Code
```
*   **嚴重後果**：因為 `dists` 是基於舊 Codebook 算的，系統選出來的所謂 "Hard Negative" (最難區分的錯誤選項)，對於現在的 Student 來說可能根本就不是 Hard Negative，甚至可能是它在自己新 Codebook 裡正確對應的那個點！
*   **這會導致**：模型被強迫遠離它其實應該靠近的地方 (如果那個點剛好被誤判為 Negative)。模型會陷入精神分裂。

### C. Triplet Loss 的正樣本距離 (d_pos) 也是虛假的
```python
d_pos = self._compute_distance(anchor, positive)
```
*   **分析**：雖然這看起來只是 Feature Matching，但因為 Student 的目標是讓 `student_out` 能被 *Student Codebook* 量化，而 `teacher_out` 是被 *Teacher Codebook* 量化的。
*   **結果**：如果兩個 Codebook 不同步，即便 Feature Loss 把兩者拉得再近，量化出來的 Token ID 也會不同。**這解釋了為什麼 Feature Loss 很低 (特徵重疊了)，但 Token Accuracy 很低 (量化結果不同)。**

---

## 3. 結論與修復方案

這個系統目前處於一種 **「雙重標準」** 的混亂狀態：
1.  **Student 內部**：努力適應不斷 EMA 漂移的新 Codebook。
2.  **Loss Function**：強迫 Student 的輸出必須符合舊 Codebook 的幾何關係。
3.  **Result**：Student 學會了「偽裝」——它的特徵值跟 Teacher 很像 (Feature Loss 低)，但在自己的量子化空間裡，它指向了完全不同的 Token (Token Acc 低)。

### 極力推薦：立即採納專家的修復建議

這個問題不修復，訓練再久都是浪費時間（因為 Codebook 已經爛掉了）。

**請修改 `exp_1209/models.py` 中的 `TeacherStudentExpandedLoRA.forward`：**

**修改前**:
```python
# Student forward
student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)

# VQ (這裡默認用了 training mode!)
quantizer = self.student.feature_extractor.encodec.quantizer
student_vq = quantizer(student_encoder_out, frame_rate=75, bandwidth=0.075)
```

**修改後 (強制 Student Quantizer 進入 Eval 模式)**:
```python
# Student forward
student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)

# VQ - 強制 EVAL 模式避免 EMA 更新 Codebook
quantizer = self.student.feature_extractor.encodec.quantizer
quantizer.eval()  # <--- 關鍵！凍結 EMA 更新

# 確保 n_q = 1 與 Teacher 一致
with torch.no_grad(): 
     student_vq = quantizer(student_encoder_out, frame_rate=75, bandwidth=0.075)

student_codes = student_vq.codes
# 記得設回 train (雖然在 forward 結束就沒差了，因為下一輪 train_epoch 會再呼叫 model.student.train())
quantizer.train() 
```

**更優雅的修法 (在 `__init__` 徹底凍結)**:
在 `exp_1209/models.py` 的 `__init__` 裡：
```python
# ... 載入 Student 後 ...
self.student.feature_extractor.encodec.quantizer.eval()
for param in self.student.feature_extractor.encodec.quantizer.parameters():
    param.requires_grad = False
```
**注意**：因為 `train_epoch` 函數通常會呼叫 `model.train()`，這會把所有子模組（包括 quantizer）重新設為 `training` 模式。所以**必須在 `forward` 中手動 `quantizer.eval()`，或者重寫 Student 模型的 `train()` 方法來防止 Quantizer 被開啟。**
