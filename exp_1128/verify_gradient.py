#!/usr/bin/env python3
"""
驗證 distance_loss 是否有梯度的簡化測試

這個腳本不載入完整模型，直接測試 loss 計算的梯度流
"""

import torch
import torch.nn.functional as F

print("="*60)
print("Distance Loss 梯度驗證")
print("="*60)

# 模擬 VQ 流程
B, T = 2, 10  # batch size, time steps
codebook_size = 4096
feature_dim = 512

# 1. 創建可訓練的 encoder 輸出 (模擬 LoRA 參數影響)
student_features = torch.randn(B, feature_dim, T, requires_grad=True)
teacher_features = torch.randn(B, feature_dim, T)

# 2. 創建 codebook 和 distance matrix
codebook = torch.randn(codebook_size, feature_dim)
distance_matrix = torch.cdist(codebook, codebook, p=2)

print(f"\nStudent features: {student_features.shape}, requires_grad={student_features.requires_grad}")
print(f"Teacher features: {teacher_features.shape}")
print(f"Codebook: {codebook.shape}")
print(f"Distance matrix: {distance_matrix.shape}")

# 3. Feature Loss (可微)
feature_loss = F.mse_loss(student_features, teacher_features)
print(f"\n[Feature Loss]")
print(f"  Value: {feature_loss.item():.4f}")
print(f"  requires_grad: {feature_loss.requires_grad}")
print(f"  grad_fn: {feature_loss.grad_fn}")

# 4. VQ 過程 (argmax - 不可微)
# 重新排列 features: (B, C, T) -> (B*T, C)
student_flat = student_features.permute(0, 2, 1).reshape(-1, feature_dim)
teacher_flat = teacher_features.permute(0, 2, 1).reshape(-1, feature_dim)

# 找最近的 code (argmax)
student_distances = torch.cdist(student_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
teacher_distances = torch.cdist(teacher_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)

student_codes = student_distances.argmin(dim=-1)  # (B*T,) - argmax 結果
teacher_codes = teacher_distances.argmin(dim=-1)  # (B*T,)

print(f"\n[VQ Codes]")
print(f"  Student codes: {student_codes.shape}, dtype={student_codes.dtype}")
print(f"  Teacher codes: {teacher_codes.shape}")
print(f"  Code match rate: {(student_codes == teacher_codes).float().mean().item()*100:.1f}%")

# 5. Distance Loss (當前實現 - 不可微)
distances_current = distance_matrix[student_codes, teacher_codes]  # 純 indexing
distance_loss_current = distances_current.mean()

print(f"\n[Distance Loss - 當前實現 (不可微)]")
print(f"  Value: {distance_loss_current.item():.4f}")
print(f"  requires_grad: {distance_loss_current.requires_grad}")
print(f"  grad_fn: {distance_loss_current.grad_fn}")

# 嘗試 backward
print(f"\n  嘗試 backward...")
try:
    distance_loss_current.backward(retain_graph=True)
    if student_features.grad is not None:
        print(f"  ✅ 有梯度! grad norm: {student_features.grad.norm().item():.6f}")
    else:
        print(f"  ❌ 沒有梯度! student_features.grad = None")
except Exception as e:
    print(f"  ❌ Backward 失敗: {e}")

# 清除梯度
student_features.grad = None

# 6. Soft Distance Loss (可微版本)
print(f"\n[Soft Distance Loss - 可微版本]")

temperature = 1.0

# 用 softmax 取代 argmax
student_logits = -student_distances / temperature  # (B*T, 4096)
student_soft_codes = F.softmax(student_logits, dim=-1)  # (B*T, 4096)

# Teacher codes 對應的 distances
teacher_distances_row = distance_matrix[teacher_codes]  # (B*T, 4096)

# Weighted sum of distances
soft_distance_loss = (student_soft_codes * teacher_distances_row).sum(dim=-1).mean()

print(f"  Value: {soft_distance_loss.item():.4f}")
print(f"  requires_grad: {soft_distance_loss.requires_grad}")
print(f"  grad_fn: {soft_distance_loss.grad_fn}")

# 嘗試 backward
print(f"\n  嘗試 backward...")
try:
    soft_distance_loss.backward()
    if student_features.grad is not None:
        print(f"  ✅ 有梯度! grad norm: {student_features.grad.norm().item():.6f}")
    else:
        print(f"  ❌ 沒有梯度! student_features.grad = None")
except Exception as e:
    print(f"  ❌ Backward 失敗: {e}")

print("\n" + "="*60)
print("結論")
print("="*60)
print("""
1. Feature Loss: ✅ 可微 (MSE 操作保持梯度)
2. Distance Loss (當前): ❌ 不可微 (argmax + indexing 切斷梯度)
3. Soft Distance Loss: ✅ 可微 (softmax 保持梯度)

當前實驗中，distance_loss_weight=0.05/0.1 沒有實際作用，
因為梯度無法從 distance_loss 傳回到 LoRA 參數！
""")
