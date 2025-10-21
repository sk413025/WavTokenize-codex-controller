#!/usr/bin/env python3
"""
檢查新的損失權重配置是否合理
"""

# 新配置
ce_weight = 15.0
l2_weight = 0.5
coherence_weight = 0.2
manifold_weight = 0.1

# 假設 raw loss 大約值（基於 Epoch 400 的測量）
raw_ce = 8.59  # 接近隨機，訓練後應該降到 2.0-4.0
raw_l2 = 0.93
raw_coherence = 12.15
raw_manifold = 15.39

# 計算加權後的損失
weighted_ce = ce_weight * raw_ce
weighted_l2 = l2_weight * raw_l2
weighted_coherence = coherence_weight * raw_coherence
weighted_manifold = manifold_weight * raw_manifold

total = weighted_ce + weighted_l2 + weighted_coherence + weighted_manifold

print("="*70)
print("新損失權重配置分析")
print("="*70)

print(f"\n配置:")
print(f"  CE weight:        {ce_weight}")
print(f"  L2 weight:        {l2_weight}")
print(f"  Coherence weight: {coherence_weight}")
print(f"  Manifold weight:  {manifold_weight}")

print(f"\n假設 Raw Loss 值（基於 Epoch 400）:")
print(f"  Raw CE:        {raw_ce:.4f}")
print(f"  Raw L2:        {raw_l2:.4f}")
print(f"  Raw Coherence: {raw_coherence:.4f}")
print(f"  Raw Manifold:  {raw_manifold:.4f}")

print(f"\n加權後的損失:")
print(f"  Weighted CE:        {weighted_ce:10.4f} ({weighted_ce/total*100:5.1f}%)")
print(f"  Weighted L2:        {weighted_l2:10.4f} ({weighted_l2/total*100:5.1f}%)")
print(f"  Weighted Coherence: {weighted_coherence:10.4f} ({weighted_coherence/total*100:5.1f}%)")
print(f"  Weighted Manifold:  {weighted_manifold:10.4f} ({weighted_manifold/total*100:5.1f}%)")
print(f"  " + "-"*60)
print(f"  Total:              {total:10.4f} (100.0%)")

print(f"\n✅ 評估:")

# CE Loss 占比
ce_ratio = weighted_ce / total
if ce_ratio > 0.9:
    print(f"  ✅ CE Loss 占比 {ce_ratio*100:.1f}% - 非常好！主導訓練")
elif ce_ratio > 0.85:
    print(f"  ✅ CE Loss 占比 {ce_ratio*100:.1f}% - 良好")
elif ce_ratio > 0.7:
    print(f"  ⚠️  CE Loss 占比 {ce_ratio*100:.1f}% - 可能仍不足")
else:
    print(f"  ❌ CE Loss 占比 {ce_ratio*100:.1f}% - 太低！")

# L2 Loss 占比
l2_ratio = weighted_l2 / total
print(f"\n  📊 L2 Embed 占比 {l2_ratio*100:.2f}%")
if l2_ratio < 0.5:
    print(f"     ⚠️  語者相似性權重較低")
    print(f"     💡 如果需要更好的語者保持，可以考慮增加到 1.0-2.0")
else:
    print(f"     ✅ 語者相似性權重適中")

print(f"\n建議:")
print(f"  1. 當前配置: CE 主導 ({ce_ratio*100:.1f}%)，適合修復 token 準確率")
print(f"  2. L2 weight = 0.5 對於語者相似性可能偏低")
print(f"  3. 如果訓練後發現語者特徵丟失，建議:")
print(f"     - 將 L2 weight 從 0.5 增加到 1.0-2.0")
print(f"     - 或者在 CE Loss 下降後（<4.0）再增加 L2 weight")

print("\n" + "="*70)
