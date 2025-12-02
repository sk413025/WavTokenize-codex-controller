# LoRA Encoder Feature-Level Denoising 實作方案

## 🎯 任務定義

### **目標**
```
輸入: Noisy Audio
經過: Fine-tuned Encoder (LoRA)
輸出: Features/Codes 接近 Clean Audio 通過原始 Encoder 的輸出
```

### **核心策略**
- **Teacher**: 原始 WavTokenizer Encoder（凍結）處理 clean audio
- **Student**: LoRA-adapted Encoder 處理 noisy audio
- **目標**: Student 輸出 → Teacher 輸出（Feature-level Distillation）
- **優勢**: 使用 VQ Distance Matrix（commit 927880a）做 soft target

---

## 🔧 完整實作

### **Step 1: 準備兩個模型**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import sys
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
from decoder.pretrained import WavTokenizer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 載入預訓練模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

config_path = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
ckpt_path = "/home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt"

# Teacher: 原始 encoder (完全凍結)
teacher = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

# Student: 加 LoRA 的 encoder
student = WavTokenizer.from_pretrained0802(config_path, ckpt_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置 LoRA (只在 Encoder 的卷積層)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

lora_config = LoraConfig(
    r=16,                    # Rank (可調整: 8, 16, 32)
    lora_alpha=32,           # Scaling factor
    target_modules=[
        # WavTokenizer Encoder 的卷積層
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.2.conv.conv",
        "feature_extractor.encodec.encoder.model.4.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
    ],
    lora_dropout=0.1,
    bias="none",
)

student = get_peft_model(student, lora_config)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 凍結 VQ 和 Backbone (只訓練 Encoder LoRA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

for name, param in student.named_parameters():
    if 'lora' not in name:  # 非 LoRA 層全部凍結
        param.requires_grad = False

# 檢查
student.print_trainable_parameters()
# 預期輸出: trainable params: ~100K / 50M (< 1%)

print("✅ Teacher (frozen) and Student (LoRA) ready!")
```

---

### **Step 2: 定義 Loss Functions (使用 Distance Matrix)**

基於你的 commit 927880a，整合 distance-based loss：

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 載入 VQ Distance Matrix (from commit 927880a)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 假設你的預處理保存了 distance matrix
# 或者從 VQ codebook 計算
def compute_codebook_distances(wavtokenizer):
    """
    計算 VQ codebook 中所有 code 之間的距離

    Returns:
        distance_matrix: (4096, 4096) - pairwise distances
    """
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    # codebook shape: (4096, 512)

    # 計算 pairwise L2 distances
    # dist[i, j] = ||codebook[i] - codebook[j]||
    dist_matrix = torch.cdist(codebook, codebook, p=2)  # (4096, 4096)

    return dist_matrix

distance_matrix = compute_codebook_distances(teacher)
print(f"Distance matrix shape: {distance_matrix.shape}")
print(f"Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loss Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EncoderDistillationLoss(nn.Module):
    """
    Feature-level + Code-level Distillation Loss
    使用 VQ distance matrix 做 soft matching
    """

    def __init__(self, distance_matrix, temperature=2.0, alpha=0.5):
        """
        Args:
            distance_matrix: (4096, 4096) VQ codebook distances
            temperature: Softmax temperature for soft targets
            alpha: Weight for soft vs hard loss
        """
        super().__init__()
        self.register_buffer('distance_matrix', distance_matrix)
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_features, teacher_features,
                student_codes, teacher_codes):
        """
        Args:
            student_features: (B, 512, T) - Student encoder output
            teacher_features: (B, 512, T) - Teacher encoder output
            student_codes: (B, 1, T) - Student VQ codes
            teacher_codes: (B, 1, T) - Teacher VQ codes

        Returns:
            total_loss: scalar
            metrics: dict of individual losses
        """
        B, C, T = student_features.shape

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Loss 1: Feature-level MSE (在 VQ 之前)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        feature_loss = F.mse_loss(student_features, teacher_features)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Loss 2: Code-level Loss (Hard Matching)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        student_codes_flat = student_codes[:, 0, :].reshape(-1)  # (B*T)
        teacher_codes_flat = teacher_codes[:, 0, :].reshape(-1)  # (B*T)

        code_loss_hard = F.cross_entropy(
            student_codes_flat.float(),  # 需要轉成 logits，這裡簡化
            teacher_codes_flat.long()
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Loss 3: Distance-based Soft Loss
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 查表: student code 到 teacher code 的距離
        distances = self.distance_matrix[
            student_codes_flat.long(),
            teacher_codes_flat.long()
        ]  # (B*T)

        # 距離越大，loss 越大
        distance_loss = distances.mean()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Alternative: Soft Target from Distances
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 從 teacher code 的距離分佈構建 soft target
        teacher_distances_to_all = -self.distance_matrix[teacher_codes_flat.long()]  # (B*T, 4096)
        # 負號: 距離越小 = 越相似

        soft_targets = F.softmax(teacher_distances_to_all / self.temperature, dim=-1)

        # Student 需要輸出 logits（這裡假設有一個 projection head）
        # student_logits = self.projection_head(student_features)  # (B, T, 4096)
        # 為了簡化，這裡省略，實際使用時需要加上

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Total Loss
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        total_loss = (
            1.0 * feature_loss +      # Feature 匹配
            0.3 * distance_loss +     # Distance-based 匹配
            # + soft_target_loss if using projection head
        )

        metrics = {
            'feature_loss': feature_loss.item(),
            'distance_loss': distance_loss.item(),
            'avg_distance': distances.mean().item(),
        }

        return total_loss, metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 簡化版本: 只用 Feature + Distance Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simple_distillation_loss(student_features, teacher_features,
                             student_codes, teacher_codes, distance_matrix):
    """
    簡化的 distillation loss

    Args:
        student_features: (B, 512, T)
        teacher_features: (B, 512, T)
        student_codes: (B, 1, T)
        teacher_codes: (B, 1, T)
        distance_matrix: (4096, 4096)

    Returns:
        loss: scalar
    """
    # Feature-level MSE
    feature_loss = F.mse_loss(student_features, teacher_features)

    # Distance-based code loss
    B, _, T = student_codes.shape
    student_flat = student_codes[:, 0, :].reshape(-1).long()
    teacher_flat = teacher_codes[:, 0, :].reshape(-1).long()

    distances = distance_matrix[student_flat, teacher_flat]
    distance_loss = distances.mean()

    # Total
    total_loss = feature_loss + 0.1 * distance_loss

    return total_loss, {
        'feature_loss': feature_loss.item(),
        'distance_loss': distance_loss.item(),
    }
```

---

### **Step 3: 訓練循環**

```python
import torch
from torch.utils.data import DataLoader
# 假設你的 HDF5 dataset (commit 927880a)
# from done.exp.data_with_distances import YourDataset

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 設置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

device = 'cuda' if torch.cuda.is_available() else 'cpu'

teacher = teacher.to(device)
student = student.to(device)
distance_matrix = distance_matrix.to(device)

# Optimizer (只優化 LoRA 參數)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, student.parameters()),
    lr=5e-5,  # 小學習率
    weight_decay=0.01
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_epoch(student, teacher, dataloader, optimizer, distance_matrix, epoch):
    """
    訓練一個 epoch
    """
    student.train()
    teacher.eval()

    total_loss = 0
    total_feature_loss = 0
    total_distance_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # 假設 batch 包含:
        # - noisy_audio: (B, T_audio)
        # - clean_audio: (B, T_audio)
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        optimizer.zero_grad()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Forward Pass
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Teacher: Clean audio → features, codes
        with torch.no_grad():
            teacher_features, teacher_codes, _ = teacher.feature_extractor(
                clean_audio, bandwidth_id=0
            )

        # Student: Noisy audio → features, codes
        student_features, student_codes, vq_loss = student.feature_extractor(
            noisy_audio, bandwidth_id=0
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Compute Loss
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        loss, metrics = simple_distillation_loss(
            student_features,
            teacher_features,
            student_codes,
            teacher_codes,
            distance_matrix
        )

        # Add VQ commitment loss (optional)
        loss = loss + 0.01 * vq_loss

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Backward
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Logging
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        total_loss += loss.item()
        total_feature_loss += metrics['feature_loss']
        total_distance_loss += metrics['distance_loss']

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Feature Loss: {metrics['feature_loss']:.4f}")
            print(f"  Distance Loss: {metrics['distance_loss']:.4f}")

    # Epoch summary
    avg_loss = total_loss / len(dataloader)
    avg_feature_loss = total_feature_loss / len(dataloader)
    avg_distance_loss = total_distance_loss / len(dataloader)

    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Summary:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg Feature Loss: {avg_feature_loss:.4f}")
    print(f"  Avg Distance Loss: {avg_distance_loss:.4f}")
    print(f"{'='*60}\n")

    return avg_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 驗證函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate(student, teacher, dataloader, distance_matrix):
    """
    驗證：檢查 student 輸出是否接近 teacher
    """
    student.eval()
    teacher.eval()

    total_feature_loss = 0
    total_distance_loss = 0
    total_code_match = 0  # 完全匹配的比例
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # Teacher
            teacher_features, teacher_codes, _ = teacher.feature_extractor(
                clean_audio, bandwidth_id=0
            )

            # Student
            student_features, student_codes, _ = student.feature_extractor(
                noisy_audio, bandwidth_id=0
            )

            # Metrics
            feature_loss = F.mse_loss(student_features, teacher_features)

            student_flat = student_codes[:, 0, :].reshape(-1).long()
            teacher_flat = teacher_codes[:, 0, :].reshape(-1).long()
            distances = distance_matrix[student_flat, teacher_flat]

            code_match = (student_flat == teacher_flat).float().mean()

            total_feature_loss += feature_loss.item()
            total_distance_loss += distances.mean().item()
            total_code_match += code_match.item()
            total_samples += 1

    avg_feature_loss = total_feature_loss / total_samples
    avg_distance_loss = total_distance_loss / total_samples
    avg_code_match = total_code_match / total_samples

    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"  Feature Loss: {avg_feature_loss:.4f}")
    print(f"  Avg Code Distance: {avg_distance_loss:.4f}")
    print(f"  Code Match Rate: {avg_code_match*100:.2f}%")
    print(f"{'='*60}\n")

    return avg_feature_loss, avg_distance_loss, avg_code_match


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練循環
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

num_epochs = 50

for epoch in range(num_epochs):
    # Train
    train_loss = train_epoch(
        student, teacher, train_loader, optimizer, distance_matrix, epoch
    )

    # Validate
    val_feature_loss, val_distance_loss, val_code_match = validate(
        student, teacher, val_loader, distance_matrix
    )

    # Scheduler step
    scheduler.step()

    # Save checkpoint
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_feature_loss': val_feature_loss,
            'val_code_match': val_code_match,
        }, f'checkpoints/lora_encoder_epoch_{epoch}.pth')

    # Early stopping if code match rate is high
    if val_code_match > 0.95:
        print(f"✅ Reached 95% code match rate! Stopping.")
        break

print("✅ Training completed!")
```

---

### **Step 4: 評估與分析**

```python
def analyze_denoising_quality(student, teacher, test_audio_noisy, test_audio_clean):
    """
    分析去噪質量
    """
    student.eval()
    teacher.eval()

    with torch.no_grad():
        # Teacher: clean
        teacher_features, teacher_codes, _ = teacher.feature_extractor(
            test_audio_clean, bandwidth_id=0
        )

        # Student: noisy
        student_features, student_codes, _ = student.feature_extractor(
            test_audio_noisy, bandwidth_id=0
        )

        # Original encoder on noisy (for comparison)
        orig_features, orig_codes, _ = teacher.feature_extractor(
            test_audio_noisy, bandwidth_id=0
        )

    # Metrics
    print(f"\n{'='*60}")
    print(f"Denoising Quality Analysis")
    print(f"{'='*60}")

    # Feature distance
    student_to_clean = F.mse_loss(student_features, teacher_features)
    orig_to_clean = F.mse_loss(orig_features, teacher_features)

    print(f"Feature Distance to Clean:")
    print(f"  Student (Noisy → Denoised): {student_to_clean.item():.6f}")
    print(f"  Original (Noisy → Noisy):   {orig_to_clean.item():.6f}")
    print(f"  Improvement: {(orig_to_clean - student_to_clean).item():.6f} ✓")

    # Code match rate
    student_match = (student_codes == teacher_codes).float().mean()
    orig_match = (orig_codes == teacher_codes).float().mean()

    print(f"\nCode Match Rate to Clean:")
    print(f"  Student: {student_match.item()*100:.2f}%")
    print(f"  Original: {orig_match.item()*100:.2f}%")
    print(f"  Improvement: +{(student_match - orig_match).item()*100:.2f}% ✓")

    # Visualize
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Teacher (Clean)
    axes[0].imshow(teacher_features[0].cpu().numpy(), aspect='auto', cmap='viridis')
    axes[0].set_title('Teacher: Clean Audio Features')
    axes[0].set_ylabel('Feature Dim')

    # Student (Noisy → Denoised)
    axes[1].imshow(student_features[0].cpu().numpy(), aspect='auto', cmap='viridis')
    axes[1].set_title('Student: Noisy Audio → Denoised Features')
    axes[1].set_ylabel('Feature Dim')

    # Original (Noisy)
    axes[2].imshow(orig_features[0].cpu().numpy(), aspect='auto', cmap='viridis')
    axes[2].set_title('Original: Noisy Audio Features (no denoising)')
    axes[2].set_ylabel('Feature Dim')
    axes[2].set_xlabel('Time Frame')

    plt.tight_layout()
    plt.savefig('denoising_quality_analysis.png')
    print(f"\n✅ Visualization saved to denoising_quality_analysis.png")
```

---

## 📊 與你現有工作的整合

### **利用 commit 927880a 的資源**

```python
# 1. 使用你的 HDF5 dataset
from done.exp.train_with_distances import ZeroShotAudioDatasetHDF5

dataset = ZeroShotAudioDatasetHDF5(
    hdf5_path='path/to/your/preprocessed.h5',
    split='train'
)

# 2. Dataloader
train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 3. 每個 batch 包含:
for batch in train_loader:
    noisy_audio = batch['noisy_audio']        # (B, T_audio)
    clean_audio = batch['clean_audio']        # (B, T_audio) - 如果有
    token_ids = batch['token_ids']            # (B, T_frame)
    distance_scores = batch['distance_scores'] # (B, T_frame, 4096)
    # ... 其他
```

### **修改建議**

如果你的 HDF5 沒有 `clean_audio`，有兩種選擇：

**選項 A: 預處理時加入 clean audio**
```python
# 在預處理時同時保存 noisy 和 clean
# 這樣訓練時可以直接用 teacher-student distillation
```

**選項 B: 使用已有的 distance_scores**
```python
# 利用你已經保存的 distance_scores 作為 soft target
# 不需要 teacher model，直接用預計算的 distances

def train_with_precomputed_distances(student, batch):
    noisy_audio = batch['noisy_audio']
    target_distances = batch['distance_scores']  # (B, T, 4096)
    target_tokens = batch['token_ids']           # (B, T)

    # Student forward
    features, codes, _ = student.feature_extractor(noisy_audio)

    # 需要一個 projection head 將 features 轉成 logits
    # logits = projection_head(features)  # (B, T, 4096)

    # Soft target loss (from commit 927880a)
    from done.exp.losses_with_distances import SoftTargetLoss

    loss_fn = SoftTargetLoss(temperature=2.0, alpha=0.5)
    loss = loss_fn(logits, target_distances, target_tokens)

    return loss
```

---

## ✅ 總結與建議

### **這個方案非常適合你的情況，因為：**

1. ✅ **LoRA 保護原始能力**
   - Encoder 原始權重不變
   - 隨時可以恢復
   - 只有 < 1% 參數訓練

2. ✅ **Feature-level Distillation 直接**
   - 明確目標：noisy features → clean features
   - 不需要中間的 token prediction

3. ✅ **Distance Matrix 提供 Soft Target**
   - 比 hard token matching 更寬容
   - 利用 VQ 學到的語義相似度
   - 你已經在 commit 927880a 實現了相關 loss

4. ✅ **易於評估**
   - 直接比較 feature distance
   - 計算 code match rate
   - 可視化 feature maps

### **實驗流程建議**

```
Phase 1: Baseline
  └─ 用原始 encoder 處理 noisy audio
     記錄 feature distance 和 code match rate

Phase 2: LoRA (rank=8, 小試)
  └─ 訓練 10 epochs
     檢查是否有改善

Phase 3: LoRA (rank=16, 正式)
  └─ 如果 phase 2 有效，增加容量
     訓練 50 epochs

Phase 4: 分析
  └─ 比較不同 noise level 的效果
     檢查是否保留原始能力
```

### **下一步行動**

1. **準備數據**: 確保有配對的 noisy-clean audio
2. **實現 projection head**: 將 features 轉成 logits（如果用 soft target）
3. **開始訓練**: 從小 LoRA rank 開始
4. **監控指標**: Feature loss, Distance loss, Code match rate

**這個方案理論上非常合理，而且有你的 distance matrix 作為優勢，應該會比直接 token prediction 更有效！** 🚀
