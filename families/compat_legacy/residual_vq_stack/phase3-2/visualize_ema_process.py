"""
可视化 EMA + Dead-Code Reset 过程

执行:
    python exp_0128/phase3-2/visualize_ema_process.py
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("EMA + Dead-Code Reset 流程可视化")
print("=" * 70)

# 模拟参数
K = 20  # codebook size (简化)
D = 4   # feature dimension (简化)
N = 50  # batch size (frames)
decay = 0.99
threshold = 2
n_steps = 10

# 初始化 EMA buffers
ema_cluster_size = torch.zeros(K)
ema_embed_avg = torch.zeros(K, D)
codebook = torch.randn(K, D) * 0.1  # 初始 codebook

# 模拟训练过程
history = {
    'ema_cluster_size': [],
    'used_codes': [],
    'dead_codes': [],
    'top5_mass': [],
}

print(f"\n模拟参数:")
print(f"  Codebook size: {K}")
print(f"  Feature dim: {D}")
print(f"  Batch size: {N} frames")
print(f"  EMA decay: {decay}")
print(f"  Dead-code threshold: {threshold}")
print(f"  Training steps: {n_steps}")

for step in range(n_steps):
    # 模拟当前 batch 的 residuals 和 code assignments
    # 让前几个 codes 更容易被选中 (模拟不均匀分布)
    residuals = torch.randn(N, D)

    # 模拟 code assignment (前 5 个 codes 高频，后面低频)
    if step < 3:
        # 早期：非常不均匀
        probs = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1] + [0] * (K - 5))
    elif step < 6:
        # 中期：逐渐均匀
        probs = torch.tensor([0.15, 0.12, 0.10, 0.08, 0.05] + [0.01] * (K - 5))
        probs = probs / probs.sum()
    else:
        # 后期：相对均匀（EMA + Dead-code reset 生效）
        probs = torch.ones(K) / K
        probs[:5] *= 2  # 前 5 个仍稍高
        probs = probs / probs.sum()

    indices = torch.multinomial(probs, N, replacement=True)

    # Step 1-2: 统计当前 batch
    counts = torch.bincount(indices, minlength=K).float()
    embed_sum = torch.zeros(K, D)
    embed_sum.index_add_(0, indices, residuals)

    # Step 3-4: EMA 更新
    ema_cluster_size = decay * ema_cluster_size + (1 - decay) * counts
    ema_embed_avg = decay * ema_embed_avg + (1 - decay) * embed_sum

    # Step 5: 计算新 codebook
    eps = 1e-5
    n = ema_cluster_size.sum()
    cluster_size = (ema_cluster_size + eps) / (n + K * eps) * n
    codebook_new = ema_embed_avg / cluster_size.unsqueeze(1).clamp(min=1e-12)

    # Step 6: Dead-code reset
    dead = ema_cluster_size < threshold
    if dead.any():
        dead_idx = dead.nonzero(as_tuple=False).squeeze(1)
        num_dead = int(dead_idx.numel())
        if num_dead > 0 and N > 0:
            rand = torch.randint(0, N, (num_dead,))
            sampled = residuals[rand]
            codebook_new[dead_idx] = sampled
            ema_cluster_size[dead_idx] = 1.0
            ema_embed_avg[dead_idx] = sampled

    codebook = codebook_new

    # 记录历史
    history['ema_cluster_size'].append(ema_cluster_size.clone())
    history['used_codes'].append((ema_cluster_size > 0.1).sum().item())
    history['dead_codes'].append(dead.sum().item())

    # 计算 top-5 mass
    total = ema_cluster_size.sum()
    top5_mass = ema_cluster_size.topk(5).values.sum() / (total + 1e-8)
    history['top5_mass'].append(top5_mass.item())

    print(f"\nStep {step:2d}:")
    print(f"  Current batch counts (top 5): {counts[:5].numpy()}")
    print(f"  EMA cluster_size (top 5): {ema_cluster_size[:5].numpy()}")
    print(f"  Dead codes: {dead.sum().item()}/{K}")
    print(f"  Used codes: {history['used_codes'][-1]}/{K}")
    print(f"  Top-5 mass: {history['top5_mass'][-1]:.3f}")

# ============= Visualization =============
fig = plt.figure(figsize=(20, 12))

# --- Subplot 1: EMA Cluster Size Evolution ---
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_title("EMA Cluster Size Evolution", fontsize=14, fontweight='bold')

ema_history = torch.stack(history['ema_cluster_size']).T.numpy()  # (K, n_steps)
for k in range(min(10, K)):  # 只画前 10 个 codes
    ax1.plot(range(n_steps), ema_history[k], marker='o', label=f'Code {k}', linewidth=2)

ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Dead Threshold={threshold}')
ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("EMA Cluster Size", fontsize=11)
ax1.legend(fontsize=8, loc='upper left', ncol=2)
ax1.grid(alpha=0.3)

# --- Subplot 2: Used Codes & Dead Codes ---
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title("Codebook Usage Over Time", fontsize=14, fontweight='bold')

steps_x = range(n_steps)
ax2_twin = ax2.twinx()

line1 = ax2.plot(steps_x, history['used_codes'], 'g-o', linewidth=3, markersize=8, label='Used Codes')
line2 = ax2_twin.plot(steps_x, history['dead_codes'], 'r-s', linewidth=3, markersize=8, label='Dead Codes')

ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("Used Codes", fontsize=11, color='g')
ax2_twin.set_ylabel("Dead Codes", fontsize=11, color='r')
ax2.tick_params(axis='y', labelcolor='g')
ax2_twin.tick_params(axis='y', labelcolor='r')
ax2.set_ylim([0, K+2])
ax2_twin.set_ylim([0, K+2])

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize=10, loc='upper right')
ax2.grid(alpha=0.3)

# --- Subplot 3: Top-5 Mass ---
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title("Top-5 Mass (Concentration)", fontsize=14, fontweight='bold')

ax3.plot(steps_x, history['top5_mass'], 'b-o', linewidth=3, markersize=8)
ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Target <0.5')
ax3.fill_between(steps_x, 0, history['top5_mass'], alpha=0.3)

ax3.set_xlabel("Training Step", fontsize=11)
ax3.set_ylabel("Top-5 Mass", fontsize=11)
ax3.set_ylim([0, 1])
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# --- Subplot 4: EMA Update Formula ---
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_title("EMA Update Mechanism", fontsize=14, fontweight='bold')
ax4.axis('off')

y = 0.9
ax4.text(0.5, y, "Exponential Moving Average (EMA)", ha='center', fontsize=13, fontweight='bold')

y -= 0.12
ax4.text(0.5, y, "每个 training step:", ha='center', fontsize=11, style='italic')

y -= 0.1
ax4.text(0.1, y, "1️⃣ 统计当前 batch:", fontsize=10, fontweight='bold')
y -= 0.06
ax4.text(0.15, y, "counts[k] = 当前 batch 中 code k 被使用的次数", fontsize=9)

y -= 0.1
ax4.text(0.1, y, "2️⃣ EMA 更新:", fontsize=10, fontweight='bold')
y -= 0.06
ax4.text(0.15, y, f"ema_cluster_size[k] = {decay} × ema_old[k] + {1-decay} × counts[k]", fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
y -= 0.06
ax4.text(0.15, y, "↳ 99% 历史 + 1% 当前 → 平滑稳定", fontsize=8, color='blue')

y -= 0.1
ax4.text(0.1, y, "3️⃣ 计算新 codebook:", fontsize=10, fontweight='bold')
y -= 0.06
ax4.text(0.15, y, "codebook[k] = ema_embed_avg[k] / ema_cluster_size[k]", fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
y -= 0.06
ax4.text(0.15, y, "↳ 向量累积和 / 累积次数 = 平均向量", fontsize=8, color='green')

y -= 0.1
ax4.text(0.1, y, f"4️⃣ Dead-Code Reset (threshold={threshold}):", fontsize=10, fontweight='bold')
y -= 0.06
ax4.text(0.15, y, "if ema_cluster_size[k] < threshold:", fontsize=9,
         bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))
y -= 0.04
ax4.text(0.18, y, "codebook[k] = random_sample(residuals)", fontsize=8)
y -= 0.04
ax4.text(0.18, y, "ema_cluster_size[k] = 1.0", fontsize=8)
y -= 0.06
ax4.text(0.15, y, "↳ 重新初始化未使用的 codes", fontsize=8, color='red')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

# --- Subplot 5: Heatmap of EMA Cluster Size ---
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title("EMA Cluster Size Heatmap", fontsize=14, fontweight='bold')

im = ax5.imshow(ema_history, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax5.set_xlabel("Training Step", fontsize=11)
ax5.set_ylabel("Code Index", fontsize=11)
ax5.set_yticks(range(K))
ax5.set_xticks(range(n_steps))
plt.colorbar(im, ax=ax5, label='EMA Cluster Size')

# 标记 dead codes
for step in range(n_steps):
    dead_mask = ema_history[:, step] < threshold
    for k in range(K):
        if dead_mask[k]:
            ax5.add_patch(patches.Rectangle((step-0.4, k-0.4), 0.8, 0.8,
                                            fill=False, edgecolor='blue', linewidth=2))

# --- Subplot 6: Example: Single Code Evolution ---
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_title("Example: Code 0 vs Code 10 Evolution", fontsize=14, fontweight='bold')

ax6.plot(steps_x, ema_history[0], 'g-o', linewidth=3, markersize=8, label='Code 0 (High Freq)')
ax6.plot(steps_x, ema_history[10], 'r-s', linewidth=3, markersize=8, label='Code 10 (Low Freq/Dead)')
ax6.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Dead Threshold={threshold}')

# 标记 dead-code reset 事件
for step in range(n_steps):
    if ema_history[10, step] < threshold and step > 0:
        ax6.axvline(x=step, color='red', linestyle=':', alpha=0.5)
        ax6.text(step, threshold+0.5, 'Reset', fontsize=8, color='red', ha='center')

ax6.set_xlabel("Training Step", fontsize=11)
ax6.set_ylabel("EMA Cluster Size", fontsize=11)
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

plt.tight_layout()

# 保存图片
output_dir = Path(__file__).parent
output_path = output_dir / "ema_dead_code_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化图表已保存至: {output_path}")

print("\n" + "=" * 70)
print("完成！请查看生成的图表。")
print("=" * 70)
