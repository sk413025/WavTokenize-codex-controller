"""
可视化 Triplet Loss 计算流程

执行:
    python families/compat_legacy/intermediate_stack/visualize_triplet_loss.py
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 模拟参数
B, D, T = 2, 4, 3  # 简化维度便于可视化
K = 8  # codebook size

print("=" * 70)
print("Triplet Loss 计算流程可视化")
print("=" * 70)

# 模拟输入
student_features = torch.randn(B, D, T)
teacher_codes = torch.tensor([
    [2, 5, 1],  # batch 0 的 3 个 frames
    [3, 2, 7]   # batch 1 的 3 个 frames
])
codebook = torch.randn(K, D)

print(f"\n输入 Shapes:")
print(f"  student_features: {student_features.shape}  (B={B}, D={D}, T={T})")
print(f"  teacher_codes:    {teacher_codes.shape}  (B={B}, T={T})")
print(f"  codebook:         {codebook.shape}  (K={K}, D={D})")

# ============= Step 1: Reshape =============
z = student_features.permute(0, 2, 1).reshape(-1, D)
print(f"\n[Step 1] Reshape student features:")
print(f"  (B, D, T) → (B, T, D) → (B*T, D)")
print(f"  {student_features.shape} → {z.shape}")

# ============= Step 2: Distance Matrix =============
dists = torch.cdist(z, codebook, p=2)
print(f"\n[Step 2] Compute distance matrix:")
print(f"  z: {z.shape} × codebook: {codebook.shape} → dists: {dists.shape}")
print(f"  每个 frame 到所有 codebook entries 的 L2 距离")

# ============= Step 3: Positive Distance =============
teacher_flat = teacher_codes.reshape(-1)
batch_indices = torch.arange(len(teacher_flat))
pos_dist = dists[batch_indices, teacher_flat]

print(f"\n[Step 3] Extract positive distances:")
print(f"  teacher_flat: {teacher_flat.tolist()}")
print(f"  pos_dist shape: {pos_dist.shape}")
for i in range(len(teacher_flat)):
    print(f"    Frame {i}: teacher_code={teacher_flat[i].item()}, pos_dist={pos_dist[i].item():.3f}")

# ============= Step 4: Negative Distance =============
dists_for_neg = dists.clone()
dists_for_neg[batch_indices, teacher_flat] = float('inf')
neg_dist = dists_for_neg.min(dim=1).values

print(f"\n[Step 4] Find hard negative distances:")
print(f"  neg_dist shape: {neg_dist.shape}")
for i in range(len(teacher_flat)):
    neg_code = dists_for_neg[i].argmin().item()
    print(f"    Frame {i}: hard_neg_code={neg_code}, neg_dist={neg_dist[i].item():.3f}")

# ============= Step 5: Triplet Loss =============
margin = 0.2
pos_dist_2d = pos_dist.reshape(B, T)
neg_dist_2d = neg_dist.reshape(B, T)
triplet = torch.relu(pos_dist_2d - neg_dist_2d + margin)

print(f"\n[Step 5] Compute triplet loss (margin={margin}):")
print(f"  triplet shape: {triplet.shape}")
for b in range(B):
    for t in range(T):
        print(f"    Batch {b}, Frame {t}: "
              f"pos={pos_dist_2d[b, t].item():.3f}, "
              f"neg={neg_dist_2d[b, t].item():.3f}, "
              f"triplet={triplet[b, t].item():.3f}")

# ============= Visualization =============
fig = plt.figure(figsize=(20, 12))

# --- Subplot 1: Shape Transformation ---
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_title("Step 1: Shape Transformation", fontsize=14, fontweight='bold')
ax1.axis('off')

y_pos = 0.9
ax1.text(0.5, y_pos, "student_features", ha='center', fontsize=12, fontweight='bold')
y_pos -= 0.1
ax1.add_patch(patches.Rectangle((0.2, y_pos-0.05), 0.6, 0.08,
                                 fill=True, facecolor='lightblue', edgecolor='black', linewidth=2))
ax1.text(0.5, y_pos, f"(B={B}, D={D}, T={T})", ha='center', fontsize=10)

y_pos -= 0.15
ax1.arrow(0.5, y_pos+0.05, 0, -0.05, head_width=0.05, head_length=0.02, fc='black', ec='black')
ax1.text(0.7, y_pos, "permute(0,2,1)", fontsize=9, style='italic')

y_pos -= 0.1
ax1.add_patch(patches.Rectangle((0.2, y_pos-0.05), 0.6, 0.08,
                                 fill=True, facecolor='lightgreen', edgecolor='black', linewidth=2))
ax1.text(0.5, y_pos, f"(B={B}, T={T}, D={D})", ha='center', fontsize=10)

y_pos -= 0.15
ax1.arrow(0.5, y_pos+0.05, 0, -0.05, head_width=0.05, head_length=0.02, fc='black', ec='black')
ax1.text(0.7, y_pos, "reshape(-1, D)", fontsize=9, style='italic')

y_pos -= 0.1
ax1.add_patch(patches.Rectangle((0.2, y_pos-0.05), 0.6, 0.08,
                                 fill=True, facecolor='salmon', edgecolor='black', linewidth=2))
ax1.text(0.5, y_pos, f"z: (B*T={B*T}, D={D})", ha='center', fontsize=10, fontweight='bold')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# --- Subplot 2: Distance Matrix ---
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title("Step 2: Distance Matrix", fontsize=14, fontweight='bold')
im = ax2.imshow(dists.detach().numpy(), cmap='viridis', aspect='auto')
ax2.set_xlabel(f"Codebook Index (K={K})", fontsize=11)
ax2.set_ylabel(f"Frame Index (B*T={B*T})", fontsize=11)
ax2.set_xticks(range(K))
ax2.set_yticks(range(B*T))
plt.colorbar(im, ax=ax2, label='L2 Distance')

# 标记 positive codes
for i in range(len(teacher_flat)):
    ax2.add_patch(patches.Rectangle((teacher_flat[i]-0.4, i-0.4), 0.8, 0.8,
                                     fill=False, edgecolor='red', linewidth=3))
    ax2.text(teacher_flat[i], i, '✓', ha='center', va='center',
             fontsize=14, color='red', fontweight='bold')

# --- Subplot 3: Positive vs Negative Distance ---
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title("Step 3-4: Positive vs Negative Distance", fontsize=14, fontweight='bold')

frames = np.arange(B*T)
width = 0.35

bars1 = ax3.bar(frames - width/2, pos_dist.detach().numpy(), width,
                label='Positive Distance', color='salmon', alpha=0.8)
bars2 = ax3.bar(frames + width/2, neg_dist.detach().numpy(), width,
                label='Negative Distance (Hard)', color='lightgreen', alpha=0.8)

ax3.set_xlabel("Frame Index", fontsize=11)
ax3.set_ylabel("L2 Distance", fontsize=11)
ax3.set_xticks(frames)
ax3.set_xticklabels([f"F{i}" for i in frames])
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# 标注 teacher codes
for i, (p, n) in enumerate(zip(pos_dist.detach().numpy(), neg_dist.detach().numpy())):
    ax3.text(i - width/2, p + 0.1, f"T:{teacher_flat[i].item()}",
             ha='center', fontsize=8, color='darkred')

# --- Subplot 4: Triplet Loss Computation ---
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_title("Step 5: Triplet Loss (ReLU)", fontsize=14, fontweight='bold')

triplet_flat = triplet.reshape(-1).detach().numpy()
colors = ['red' if t > 0 else 'green' for t in triplet_flat]

bars = ax4.bar(frames, triplet_flat, color=colors, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.axhline(y=margin, color='orange', linestyle='--', linewidth=2, label=f'Margin={margin}')

ax4.set_xlabel("Frame Index", fontsize=11)
ax4.set_ylabel("Triplet Loss", fontsize=11)
ax4.set_xticks(frames)
ax4.set_xticklabels([f"F{i}" for i in frames])
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# 标注 loss 值
for i, (bar, val) in enumerate(zip(bars, triplet_flat)):
    if val > 0:
        ax4.text(i, val + 0.02, f"{val:.3f}", ha='center', fontsize=9, color='darkred')

# --- Subplot 5: Formula Breakdown ---
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title("Formula Breakdown (Example: Frame 0)", fontsize=14, fontweight='bold')
ax5.axis('off')

# 选择 Frame 0 作为示例
frame_idx = 0
pos_val = pos_dist[frame_idx].item()
neg_val = neg_dist[frame_idx].item()
trip_val = triplet.reshape(-1)[frame_idx].item()
teacher_code = teacher_flat[frame_idx].item()

y = 0.9
ax5.text(0.5, y, f"Frame {frame_idx} Analysis", ha='center', fontsize=13, fontweight='bold')

y -= 0.15
ax5.text(0.1, y, f"Teacher Code:", fontsize=11)
ax5.text(0.6, y, f"{teacher_code}", fontsize=11, color='blue', fontweight='bold')

y -= 0.1
ax5.text(0.1, y, f"d_pos (to code {teacher_code}):", fontsize=11)
ax5.text(0.6, y, f"{pos_val:.4f}", fontsize=11, color='red', fontweight='bold')

y -= 0.1
neg_code = dists_for_neg[frame_idx].argmin().item()
ax5.text(0.1, y, f"d_neg (to code {neg_code}):", fontsize=11)
ax5.text(0.6, y, f"{neg_val:.4f}", fontsize=11, color='green', fontweight='bold')

y -= 0.1
ax5.text(0.1, y, f"Margin:", fontsize=11)
ax5.text(0.6, y, f"{margin}", fontsize=11, color='orange', fontweight='bold')

y -= 0.15
ax5.text(0.5, y, "─" * 50, ha='center', fontsize=10)

y -= 0.1
ax5.text(0.5, y, "Loss = ReLU(d_pos - d_neg + margin)", ha='center',
         fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

y -= 0.12
ax5.text(0.5, y, f"= ReLU({pos_val:.3f} - {neg_val:.3f} + {margin})",
         ha='center', fontsize=10)

y -= 0.1
ax5.text(0.5, y, f"= ReLU({pos_val - neg_val + margin:.3f})",
         ha='center', fontsize=10)

y -= 0.1
ax5.text(0.5, y, f"= {trip_val:.4f}", ha='center', fontsize=12,
         fontweight='bold', color='red' if trip_val > 0 else 'green')

y -= 0.15
if trip_val > 0:
    ax5.text(0.5, y, "⚠️ Need Optimization", ha='center', fontsize=11,
             color='red', fontweight='bold')
else:
    ax5.text(0.5, y, "✓ Satisfied Constraint", ha='center', fontsize=11,
             color='green', fontweight='bold')

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)

# --- Subplot 6: Geometric Interpretation ---
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_title("Geometric Interpretation (2D Projection)", fontsize=14, fontweight='bold')

# 2D 投影（取前两个维度）
z_2d = z[:, :2].detach().numpy()
codebook_2d = codebook[:, :2].detach().numpy()

# 绘制 codebook
ax6.scatter(codebook_2d[:, 0], codebook_2d[:, 1],
           c='gray', s=100, alpha=0.5, marker='s', label='Codebook')

# 标注 codebook indices
for i in range(K):
    ax6.text(codebook_2d[i, 0], codebook_2d[i, 1], str(i),
            ha='center', va='center', fontsize=8, fontweight='bold')

# 绘制 student features
colors_frames = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
for i in range(B*T):
    ax6.scatter(z_2d[i, 0], z_2d[i, 1],
               c=colors_frames[i], s=200, marker='o',
               edgecolors='black', linewidth=2, label=f'Frame {i}', zorder=5)

    # 连线到 positive code
    pos_code = teacher_flat[i].item()
    ax6.plot([z_2d[i, 0], codebook_2d[pos_code, 0]],
            [z_2d[i, 1], codebook_2d[pos_code, 1]],
            'r--', linewidth=2, alpha=0.7, label=f'Positive' if i == 0 else '')

    # 连线到 hard negative code
    neg_code = dists_for_neg[i].argmin().item()
    ax6.plot([z_2d[i, 0], codebook_2d[neg_code, 0]],
            [z_2d[i, 1], codebook_2d[neg_code, 1]],
            'g:', linewidth=2, alpha=0.7, label=f'Hard Negative' if i == 0 else '')

ax6.set_xlabel("Dimension 0", fontsize=11)
ax6.set_ylabel("Dimension 1", fontsize=11)
ax6.legend(fontsize=8, loc='upper right', ncol=2)
ax6.grid(alpha=0.3)

plt.tight_layout()

# 保存图片
output_dir = Path(__file__).parent
output_path = output_dir / "triplet_loss_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化图表已保存至: {output_path}")

print("\n" + "=" * 70)
print("完成！请查看生成的图表。")
print("=" * 70)
