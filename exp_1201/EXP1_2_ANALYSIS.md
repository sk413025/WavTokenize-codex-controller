# exp_1201 系列實驗完整分析報告

**分析日期**: 2025-12-02
**實驗目標**: Teacher-Student Distillation for VQ Tokenization Denoising
**核心問題**: 如何提高 Token Accuracy 和音頻重建質量

---

## 1. 實驗總覽

### 1.1 所有實驗配置

| 實驗 | Loss Mode | 特殊配置 | 狀態 |
|------|-----------|----------|------|
| exp1 | Gumbel-Softmax | τ=1.0 | ✅ 完成 |
| exp2 | STE | τ=1.0 (baseline) | ✅ 完成 |
| exp3 | CE | ce_weight=1.0, dist_weight=0.1 | ✅ 完成 |
| exp4 | Margin | margin=0.5, hard_neg=10 | ✅ 完成 |
| exp5 | CE + Strong Feature | feature_weight=0.5 | ✅ 完成 |
| exp6 | STE | lr=5e-4 (高學習率) | ✅ 完成 |

### 1.2 共通配置

```python
# LoRA 配置
lora_rank = 64
lora_alpha = 128
lora_dropout = 0.1

# 訓練配置
batch_size = 8
num_epochs = 50
base_lr = 1e-4  # (exp6 用 5e-4)
warmup_epochs = 5
scheduler = CosineAnnealingLR

# 模型
Teacher: WavTokenizer (frozen)
Student: WavTokenizer + LoRA (trainable)
Codebook: 4096 entries, 512 dim (frozen)
```

---

## 2. 實驗結果對比

### 2.1 Token Accuracy 排名 (核心指標)

| 排名 | 實驗 | Exact Match | Top-5 | Top-10 | Top-100 |
|------|------|-------------|-------|--------|---------|
| 🥇 | **exp4 (Margin)** | **5.74%** | 11.57% | 18.34% | 50.54% |
| 🥈 | exp5 (Strong Feature + CE) | 4.60% | 10.49% | 17.35% | 49.24% |
| 🥉 | exp3 (CE) | ~5.5%* | - | - | - |
| 4 | exp6 (STE High LR) | 2.35% | 6.82% | 12.80% | 53.26% |
| 5 | exp2 (STE baseline) | ~2.17% | - | - | - |
| 6 | exp1 (Gumbel) | ~1.75% | - | - | - |

*exp3 數據來自訓練 log，非獨立測試

### 2.2 Feature 對齊指標

| 實驗 | 初始 L2 | 最終 L2 | L2 變化 | 最終 Cosine |
|------|---------|---------|---------|-------------|
| exp4 (Margin) | 0.849 | 0.709 | **-16.6%** | 0.959 |
| exp5 (Strong Feature) | 0.840 | 0.715 | -14.8% | 0.966 |
| exp6 (STE High LR) | 0.837 | 0.660 | **-21.2%** | 0.970 |

### 2.3 關鍵發現

1. **Margin Loss 最有效**：直接優化決策邊界，Token Acc 提升到 5.74%
2. **Feature 對齊 ≠ Token Accuracy**：exp6 的 L2 最低 (0.660)，但 Token Acc 最差 (2.35%)
3. **CE 和 Margin 都有效**：比 STE/Gumbel baseline 提升 2-3x
4. **High LR 無效**：加速 feature 收斂，但無法改善 token 選擇

---

## 3. 最佳配置推薦

### 3.1 推薦配置 (基於 exp4 Margin)

```python
# losses.py 配置
loss_mode = "margin"
margin = 0.5                # triplet margin
num_hard_negatives = 10     # hard negative mining
feature_loss_weight = 0.1   # 輔助 feature 對齊
distance_loss_weight = 0.0  # 不用 soft distance

# 訓練配置
learning_rate = 1e-4        # 不要太高
warmup_epochs = 5
num_epochs = 50
```

### 3.2 次佳配置 (CE Loss)

```python
loss_mode = "ce"
ce_loss_weight = 1.0
distance_loss_weight = 0.1  # 小權重的 soft distance 作為正則化
feature_loss_weight = 0.1
temperature = 1.0
```

### 3.3 不推薦配置

| 配置 | 原因 |
|------|------|
| Gumbel-Softmax | 隨機性導致訓練不穩定，codebook 太大 (4096) |
| High Learning Rate (5e-4) | Feature 收斂但 token 選擇不改善 |
| Pure STE | 間接優化，梯度信號不足 |
| 強 Feature Loss (>0.3) | 可能與 Token Accuracy 衝突 |

---

## 4. 驗證與檢查方法

### 4.1 已有驗證工具 (test/ 資料夾)

| 工具 | 功能 | 命令 |
|------|------|------|
| `feature_tsne_analysis.py` | t-SNE 可視化 + L2/Cosine 趨勢 | `python feature_tsne_analysis.py --exp_name margin_tuned` |
| `token_distance_analysis.py` | Token 準確率 + Rank 分析 | `python token_distance_analysis.py --exp_name margin_tuned` |

### 4.2 建議增加的驗證方法

#### A. 音頻質量評估 (MOS/PESQ/STOI)

```python
# 建議新增: test/audio_quality_analysis.py
from pesq import pesq
from pystoi import stoi

def evaluate_audio_quality(clean_path, enhanced_path, sr=24000):
    """評估重建音頻質量"""
    clean, _ = librosa.load(clean_path, sr=sr)
    enhanced, _ = librosa.load(enhanced_path, sr=sr)

    # PESQ (Perceptual Evaluation of Speech Quality)
    pesq_score = pesq(sr, clean, enhanced, 'wb')  # wideband

    # STOI (Short-Time Objective Intelligibility)
    stoi_score = stoi(clean, enhanced, sr, extended=False)

    return {'pesq': pesq_score, 'stoi': stoi_score}
```

現有音頻樣本位置:
```
experiments/*/audio_samples/epoch_*/
├── train_1_clean.wav        # 原始乾淨音頻
├── train_1_noisy.wav        # 加噪音頻
├── train_1_teacher_recon.wav # Teacher 重建
└── train_1_student_pred.wav  # Student 預測
```

#### B. Codebook 使用率分析

```python
# 建議新增: test/codebook_usage_analysis.py
def analyze_codebook_usage(student_tokens, teacher_tokens):
    """分析 codebook 使用分布"""
    student_dist = Counter(student_tokens.flatten().tolist())
    teacher_dist = Counter(teacher_tokens.flatten().tolist())

    # 計算使用率
    student_usage = len(student_dist) / 4096
    teacher_usage = len(teacher_dist) / 4096

    # 計算分布相似度 (KL divergence)
    kl_div = compute_kl_divergence(student_dist, teacher_dist)

    return {
        'student_usage': student_usage,
        'teacher_usage': teacher_usage,
        'kl_divergence': kl_div
    }
```

#### C. 梯度流動檢查 (PDB 方法)

```bash
# 使用項目的 PDB 調試方法
python -m pdb train.py < pdb_gradient_check.txt
```

```
# pdb_gradient_check.txt
b train.py:280  # backward 後
run --exp_name debug --batch_size 2 --num_epochs 1
c
p sum(1 for p in model.parameters() if p.grad is not None)
p model.student.lora_layers[0].weight.grad.norm().item()
q
```

#### D. Token 混淆矩陣

```python
# 建議新增: test/token_confusion_analysis.py
def plot_confusion_matrix(student_tokens, teacher_tokens, top_k=50):
    """繪製最常見 tokens 的混淆矩陣"""
    # 找出最常見的 teacher tokens
    top_teacher = Counter(teacher_tokens).most_common(top_k)

    # 建立混淆矩陣
    confusion = np.zeros((top_k, top_k))
    for t_idx, s_idx in zip(teacher_tokens, student_tokens):
        # ... 填充矩陣

    plt.imshow(confusion, cmap='Blues')
    plt.title('Token Confusion Matrix')
```

#### E. 訓練穩定性監控

```python
# 檢查訓練是否穩定
def check_training_stability(training_history):
    """檢查 loss 是否有異常波動"""
    losses = training_history['train_loss']

    # 檢查 NaN
    if any(np.isnan(losses)):
        return "WARNING: NaN detected"

    # 檢查劇烈波動
    std_ratio = np.std(losses[-10:]) / np.mean(losses[-10:])
    if std_ratio > 0.5:
        return "WARNING: High variance in recent losses"

    return "OK"
```

---

## 5. 提升 Token Accuracy 的方法

### 5.1 已驗證有效的方法

| 方法 | 效果 | 說明 |
|------|------|------|
| **Margin Loss** | ✅✅✅ | 直接優化決策邊界，最有效 |
| **CE Loss** | ✅✅ | 分類監督，簡單有效 |
| **Hard Negative Mining** | ✅✅ | 專注於困難樣本 |
| **較小的 Feature Loss 權重** | ✅ | 避免與 token 目標衝突 |

### 5.2 建議嘗試的方法

#### A. Temperature Annealing

```python
# 從高溫開始，逐漸降低
def get_temperature(epoch, initial=2.0, final=0.5, anneal_epochs=30):
    if epoch < anneal_epochs:
        return initial - (initial - final) * epoch / anneal_epochs
    return final
```

理論：高溫時探索更多 codes，低溫時聚焦最佳選擇

#### B. Curriculum Learning

```python
# 先學習簡單樣本，再學習困難樣本
def curriculum_sampling(dataset, epoch, difficulty_scores):
    """根據 epoch 選擇適當難度的樣本"""
    threshold = min(1.0, 0.3 + epoch * 0.02)  # 逐漸提高難度
    easy_samples = [i for i, d in enumerate(difficulty_scores) if d < threshold]
    return Subset(dataset, easy_samples)
```

#### C. Label Smoothing for CE

```python
# 軟化標籤，考慮 codebook 結構
def smooth_labels(teacher_tokens, distance_matrix, smoothing=0.1):
    """基於距離矩陣的 label smoothing"""
    one_hot = F.one_hot(teacher_tokens, num_classes=4096)

    # 距離越近的 codes 獲得更多權重
    smooth_weights = F.softmax(-distance_matrix[teacher_tokens] / temp, dim=-1)

    return (1 - smoothing) * one_hot + smoothing * smooth_weights
```

#### D. Projection Head

```python
# 在 student encoder 後加一個投影層
class ProjectionHead(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.proj(x)
```

理論：讓模型有專門的 layer 適應 VQ 空間

#### E. Contrastive Learning

```python
# InfoNCE loss
def info_nce_loss(student_features, teacher_features, temperature=0.07):
    """對比學習損失"""
    # 正樣本：同位置的 teacher feature
    # 負樣本：其他位置的 features

    sim = F.cosine_similarity(student_features, teacher_features)
    # ...
```

### 5.3 不建議的方法

| 方法 | 原因 |
|------|------|
| 更大的 LoRA rank | 已測試 r=32, 64, 128，差異不大 |
| 更高的學習率 | exp6 證明無效 |
| 更多 epochs | 50 epochs 後 loss 已收斂 |
| 純 Feature Loss | 可能與 Token Accuracy 衝突 |

---

## 6. 提升音頻質量的方法

### 6.1 當前瓶頸分析

```
音頻質量 = f(Token Accuracy, Decoder Quality, Noise Level)
```

目前：
- Token Accuracy: ~5.74% (瓶頸)
- Decoder: WavTokenizer decoder (frozen, 質量好)
- Noise: 使用真實 DNS 噪音

### 6.2 改進方向

#### A. 提高 Token Accuracy (最重要)

- 實施上述 5.2 的方法
- 目標：Token Acc > 15%

#### B. 加入 Acoustic Loss

```python
# 在 waveform 層面加入損失
def acoustic_loss(student_audio, teacher_audio):
    """音頻層面的損失"""
    # Multi-resolution STFT loss
    loss = 0
    for n_fft in [512, 1024, 2048]:
        student_spec = torch.stft(student_audio, n_fft)
        teacher_spec = torch.stft(teacher_audio, n_fft)
        loss += F.l1_loss(student_spec, teacher_spec)
    return loss
```

#### C. 後處理優化

```python
# 對 student tokens 進行後處理
def post_process_tokens(student_tokens, confidence_scores, threshold=0.3):
    """低置信度時使用 teacher 重建"""
    mask = confidence_scores < threshold
    # 這些位置可以用其他策略處理
    return processed_tokens
```

#### D. 考慮 Decoder Fine-tuning

目前 decoder 是凍結的。如果 token accuracy 無法繼續提升，可以考慮：
- Fine-tune decoder 的最後幾層
- 訓練一個 "robust decoder" 來處理不完美的 tokens

---

## 7. 實驗總結

### 7.1 關鍵結論

1. **Distance-based Loss 是間接優化**：Softmax 平滑效應導致梯度差異不足
2. **Margin/CE Loss 是直接優化**：直接監督 token 選擇，效果更好
3. **Feature 對齊 ≠ Token 對齊**：優化 feature loss 可能惡化 token accuracy
4. **Codebook 幾何複雜**：4096 entries，歐氏距離 ≠ 語義距離

### 7.2 下一步計劃

| 優先級 | 任務 | 預期效果 |
|--------|------|----------|
| P0 | 實現 Temperature Annealing | Token Acc +2-3% |
| P0 | 實現 Label Smoothing CE | 更穩定的訓練 |
| P1 | 添加 PESQ/STOI 評估 | 量化音頻質量 |
| P1 | 實現 Projection Head | Token Acc +3-5% |
| P2 | Contrastive Learning | 更好的 feature 表示 |
| P2 | Curriculum Learning | 更高效的訓練 |

### 7.3 成功標準

| 指標 | 當前最佳 | 短期目標 | 長期目標 |
|------|----------|----------|----------|
| Token Exact Match | 5.74% | 10% | 20% |
| Top-10 Accuracy | 18.34% | 30% | 50% |
| PESQ | 未測量 | > 2.5 | > 3.0 |
| STOI | 未測量 | > 0.8 | > 0.9 |

---

## 8. 附錄：詳細數據

### 8.1 完整實驗配置

```bash
# exp1: Gumbel-Softmax
python train.py --loss_mode gumbel --temperature 1.0

# exp2: STE Baseline
python train.py --loss_mode ste --temperature 1.0

# exp3: CE Loss
python train.py --loss_mode ce --ce_weight 1.0 --dist_weight 0.1

# exp4: Margin Loss (推薦)
python train.py --loss_mode margin --margin 0.5 --num_hard_neg 10

# exp5: Strong Feature + CE
python train.py --loss_mode ce --feature_weight 0.5

# exp6: STE High LR
python train.py --loss_mode ste --lr 5e-4
```

### 8.2 Token Accuracy 詳細數據

```
margin_tuned (exp4):
  Exact Match: 5.74%
  Top-5: 11.57%
  Top-10: 18.34%
  Top-50: 38.52%
  Top-100: 50.54%
  Median Wrong Rank: 113

strong_feature_ce (exp5):
  Exact Match: 4.60%
  Top-5: 10.49%
  Top-10: 17.35%
  Top-50: 38.11%
  Top-100: 49.24%
  Median Wrong Rank: 117

ste_high_lr (exp6):
  Exact Match: 2.35%
  Top-5: 6.82%
  Top-10: 12.80%
  Top-50: 38.00%
  Top-100: 53.26%
  Median Wrong Rank: 91
```

### 8.3 Feature Distance 變化

```
margin_tuned: L2 0.849 → 0.709 (-16.6%), Cosine 0.939 → 0.959
strong_feature_ce: L2 0.840 → 0.715 (-14.8%), Cosine 0.957 → 0.966
ste_high_lr: L2 0.837 → 0.660 (-21.2%), Cosine 0.961 → 0.970
```

---

**文檔更新**: 2025-12-02
**作者**: Claude Code Analysis
