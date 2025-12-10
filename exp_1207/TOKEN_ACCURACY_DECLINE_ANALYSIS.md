# Token Accuracy 下降原因分析

## 觀察到的現象

### 1. **初始準確率高，後續下降** (exp_1204/exp_curriculum.log)
```
Epoch 1: Train 27.58% → Val 15.65%
Epoch 2: Train 14.01% → Val 8.02%
Epoch 3: Train 9.98%  → Val 7.48%
...
Epoch 25: Train 14.17% → Val 4.92%
```

### 2. **Loss 與 Accuracy 脫鉤**
- Total Loss 持續上升: 0.04 → 9.62
- Token Accuracy 下降: 27% → 12%
- **矛盾**：Loss 最小化 ≠ Token Accuracy 提升

---

## 可能原因分析

### ⚠️ **原因 1: Loss 函數設計問題**

#### 問題：MSE Loss 與 Token Selection 的矛盾

```python
# 當前 Loss 設計
MSE Loss = ||student_emb - teacher_emb||²  # VQ 前的連續 embedding
CE Loss  = CrossEntropy(logits, teacher_codes)  # VQ 後的離散 token

Total Loss = λ_mse × MSE + λ_ce × CE
```

**矛盾點**：
1. **MSE** 要求 student embedding **靠近** teacher embedding（連續空間）
2. **CE** 要求 student 選到**跟 teacher 一樣的 token**（離散分類）
3. 但 teacher_emb 經過 VQ 量化成 teacher_code 會損失信息
4. 當 teacher_emb 靠近 Voronoi boundary 時，MSE 的「靠近」≠ CE 的「同 token」

**示例**：
```
Teacher 處理流程:
teacher_emb = [0.95, 0.05]  (VQ 前的連續值)
→ VQ 量化 → teacher_code = 0
→ 實際使用 codebook[0] = [1.0, 0.0]

假設 Voronoi boundary 在 [0.9, 0.1]:
Token 0 center: [1.0, 0.0]
Token 1 center: [0.8, 0.2]

Student 學習目標:
MSE: 靠近 teacher_emb = [0.95, 0.05]
CE:  選擇 teacher_code = 0 (對應 [1.0, 0.0])

問題：
如果 student_emb = [0.89, 0.11]  (很接近 teacher_emb，MSE 小)
→ 但可能選到 Token 1 (因為更接近 [0.8, 0.2])
→ CE Loss 爆炸！Token Accuracy 下降
```

---

### ⚠️ **原因 2: Curriculum Learning 策略反效果**

#### 當前策略 (exp_1204)
```python
# Epoch 1-5: MSE only, temp=2.0, ce_weight=0
# Epoch 6-25: 漸進增加 CE, 降低 temp
# Epoch 25+: CE dominant, temp=0.1
```

**問題**：
- **Early stage (MSE only)**: 學到錯誤的 embedding 空間分佈
- **Later stage (CE dominant)**: 需要 **推翻** 之前學到的知識
- 結果：**Catastrophic forgetting**

**證據**：
```
Epoch 6-10: Loss 突然跳升 (0.03 → 0.65)
            Token Acc 下降 (14% → 11%)
```
這是 CE 開始介入，與 MSE 學到的知識衝突！

---

### ⚠️ **原因 3: 梯度衝突 (Gradient Conflict)**

```python
# MSE 梯度方向
∇MSE = 2(student_emb - codebook[correct])
       → 指向 correct codebook center

# CE 梯度方向  
∇CE = softmax(logits) - one_hot(correct)
    → 推開所有錯誤 token centers
    → 拉近 correct token center

# 但如果 embedding 已經在 wrong cell：
# MSE: "往 correct 移動"
# CE:  "在 wrong cell 中心穩定下來"（因為 logit 最大）
```

**梯度互相抵消 → 學習停滯**

---

### ⚠️ **原因 4: Temperature Annealing 過快**

```python
# 當前設定
temp: 2.0 → 0.1 (20x 衰減，20 epochs)
```

**問題**：
- **高溫 (2.0)**: Softmax 平滑，梯度信號弱
- **低溫 (0.1)**: Softmax 尖銳，但容易陷入局部最優
- **快速衰減**: 沒時間學到穩定的 embedding space

**建議**：
- 更緩慢的 annealing: `temp = max(0.5, 2.0 - epoch*0.05)`
- 或維持中溫: `temp = 1.0` (固定)

---

### ⚠️ **原因 5: Voronoi Boundary 問題**

#### 從 exp_1204 診斷結果
```
mean_correct_distance: 3.75  (到正確 token)
mean_min_distance:     0.45  (到最近錯誤 token)

問題：Student embedding 離錯誤 token 更近！
```

**根本原因**：
- Codebook 是固定的 (teacher 的)
- Student 需要學習**跨過 Voronoi boundary**
- 但 MSE Loss 只關心「靠近」，不關心「在哪一側」

**比喻**：
```
Voronoi boundary 像一道牆：
- MSE: "越過牆靠近目標"
- CE:  "告訴你牆在哪"
  
如果只有 MSE → 靠著牆但在錯誤一側
如果只有 CE  → 知道要過牆但不知道往哪走
```

---

### ⚠️ **原因 6: LoRA 參數量不足**

```python
# 當前設定
lora_rank = 64
lora_alpha = 128
trainable_params = 154,048 (0.19%)
```

**推測**：
- 需要學習複雜的 Voronoi cell 映射
- LoRA 參數太少，表達能力受限
- 只能學到「大致方向」，無法精確跨過 boundary

**建議**：
- 增加 LoRA rank: 64 → 256
- 或使用 Full Fine-tuning (如果記憶體允許)

---

## 解決方案建議

### 🔧 **方案 1: 改進 Loss 設計**

```python
# 新 Loss: Margin-based Contrastive Loss
class MarginContrastiveLoss(nn.Module):
    def forward(self, student_emb, teacher_codes, codebook):
        # 到正確 token 的距離
        correct_dist = ||student_emb - codebook[correct]||
        
        # 到最近錯誤 token 的距離
        wrong_dist = min(||student_emb - codebook[i]|| for i != correct)
        
        # Margin Loss: 確保 correct 距離 < wrong 距離 - margin
        loss = max(0, correct_dist - wrong_dist + margin)
        
        return loss
```

**優點**：
- 直接優化 Voronoi cell 邊界
- 不關心絕對距離，只關心相對距離
- 避免 MSE 與 CE 的矛盾

---

### 🔧 **方案 2: 分階段訓練（重新設計 Curriculum）**

```python
# Stage 1 (epoch 1-10): 只用 CE，快速定位到正確 cell
loss = CE_loss

# Stage 2 (epoch 11-30): CE + MSE，在 cell 內部精細調整
loss = 0.5 * CE_loss + 1.0 * MSE_loss

# Stage 3 (epoch 31-50): 降低 CE 權重，穩定 embedding
loss = 0.1 * CE_loss + 1.0 * MSE_loss
```

**邏輯**：
1. 先用 CE「找到正確的 cell」
2. 再用 MSE「在 cell 內部優化」
3. 避免 MSE 把 embedding 推到錯誤 cell

---

### 🔧 **方案 3: Hard Negative Mining**

```python
# 找到 K 個最難區分的錯誤 token
hard_negatives = top_k_nearest_wrong_tokens(student_emb, codebook, k=10)

# 專注於這些難樣本
CE_loss = CrossEntropy(
    logits[hard_negatives + [correct]],
    correct
)
```

**優點**：
- 專注於 Voronoi boundary 附近的區域
- 避免浪費計算在遠離的 token

---

### 🔧 **方案 4: Codebook Alignment**

```python
# 問題：Teacher codebook 可能不適合 student
# 解決：Fine-tune codebook 來適應 student

# 1. 凍結 encoder，只訓練 codebook
for epoch in range(5):
    codebook.requires_grad = True
    student.requires_grad = False
    loss = MSE(student_emb, codebook[teacher_codes])
    
# 2. 凍結 codebook，訓練 encoder
for epoch in range(45):
    codebook.requires_grad = False
    student.requires_grad = True
    loss = CE(student_logits, teacher_codes)
```

---

### 🔧 **方案 5: 監控並早停**

```python
# 如果 token accuracy 連續下降 N epochs，停止訓練
patience = 5
best_acc = 0
no_improve = 0

for epoch in range(50):
    val_acc = validate()
    
    if val_acc > best_acc:
        best_acc = val_acc
        no_improve = 0
    else:
        no_improve += 1
        
    if no_improve >= patience:
        print("Token accuracy declining, early stop!")
        break
```

---

## 實驗驗證計畫

### 🧪 **Exp A: Margin Contrastive Loss**
```bash
python train.py --loss_type margin_contrastive --margin 0.5
```

### 🧪 **Exp B: 反向 Curriculum (CE first)**
```bash
python train.py \
    --stage1_epochs 10 --stage1_ce_weight 1.0 --stage1_mse_weight 0.0 \
    --stage2_epochs 20 --stage2_ce_weight 0.5 --stage2_mse_weight 1.0
```

### 🧪 **Exp C: 增加 LoRA Rank**
```bash
python train.py --lora_rank 256 --lora_alpha 512
```

### 🧪 **Exp D: 固定 Temperature**
```bash
python train.py --temperature 1.0 --no_annealing
```

---

## 診斷工具建議

### 📊 **1. 繪製 Embedding Space**
```python
# 每 5 epoch 繪製 t-SNE
def plot_embedding_space(student_emb, teacher_codes, epoch):
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2)
    emb_2d = tsne.fit_transform(student_emb)
    
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=teacher_codes, cmap='tab20')
    plt.title(f'Epoch {epoch} - Student Embedding Space')
    plt.savefig(f'tsne_epoch_{epoch}.png')
```

### 📊 **2. 監控 Voronoi Boundary**
```python
# 計算每個樣本到 boundary 的距離
def distance_to_boundary(student_emb, correct_code, codebook):
    correct_dist = ||student_emb - codebook[correct]||
    nearest_wrong_dist = min(||student_emb - codebook[i]|| for i != correct)
    
    # 正值 = 在正確 cell 內部，負值 = 在錯誤 cell
    margin = nearest_wrong_dist - correct_dist
    
    return margin
```

### 📊 **3. 梯度分析**
```python
# 檢查 MSE 和 CE 梯度是否衝突
def analyze_gradient_conflict():
    mse_grad = compute_mse_grad()
    ce_grad = compute_ce_grad()
    
    # 計算梯度夾角
    cosine_sim = (mse_grad · ce_grad) / (||mse_grad|| × ||ce_grad||)
    
    print(f"Gradient cosine similarity: {cosine_sim:.3f}")
    # 如果 < 0: 梯度方向相反（衝突）
    # 如果 ~ 0: 梯度正交（互不影響）
    # 如果 > 0: 梯度方向一致（協同）
```

---

## 結論

Token Accuracy 下降的**核心原因**是：

1. ✅ **MSE Loss 無法保證 Token 正確性**
   - MSE 優化「距離」，但 Token 由「argmin」決定
   - 需要 Margin-based loss 或純 CE

2. ✅ **Curriculum Learning 順序錯誤**
   - 應該先用 CE 定位 cell，再用 MSE 精細化
   - 而不是先 MSE 再 CE

3. ✅ **Voronoi Boundary 難以跨越**
   - Student embedding 卡在錯誤 cell 的邊緣
   - 需要更強的監督信號（CE）或更大的模型容量（LoRA rank）

**建議優先嘗試**：
- 🎯 **Exp B (反向 Curriculum)** - 最容易實現
- 🎯 **Exp D (固定 Temperature=1.0)** - 簡單且有效
- 🎯 **Exp C (增加 LoRA Rank)** - 提升模型表達能力
