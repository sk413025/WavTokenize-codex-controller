#!/usr/bin/env python3
"""
Diagnostic: 分析 token_noisy → token_clean 的轉換統計
目標: 看是否存在可利用的 pattern 做 VQ 後 steering
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# 載入 val_cache
CACHE_PATH = "/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt"
cache = torch.load(CACHE_PATH, weights_only=False)
print(f"Loaded {len(cache)} samples from val_cache")

# ============================================================
# 1. 基本統計
# ============================================================
total_tokens = 0
match_tokens = 0
all_transitions = Counter()  # (noisy_token, clean_token) -> count
noisy_token_counts = Counter()
clean_token_counts = Counter()

# 每個 noisy token 對應的 clean token 分佈
noisy_to_clean_dist = defaultdict(Counter)  # noisy_token -> {clean_token: count}

for item in cache:
    noisy_tokens = item['noisy_tokens']
    clean_tokens = item['clean_tokens']

    # 確保是 tensor
    if isinstance(noisy_tokens, torch.Tensor):
        noisy_tokens = noisy_tokens.flatten().tolist()
    if isinstance(clean_tokens, torch.Tensor):
        clean_tokens = clean_tokens.flatten().tolist()

    # 對齊長度
    min_len = min(len(noisy_tokens), len(clean_tokens))
    noisy_tokens = noisy_tokens[:min_len]
    clean_tokens = clean_tokens[:min_len]

    for t_n, t_c in zip(noisy_tokens, clean_tokens):
        total_tokens += 1
        if t_n == t_c:
            match_tokens += 1
        all_transitions[(t_n, t_c)] += 1
        noisy_token_counts[t_n] += 1
        clean_token_counts[t_c] += 1
        noisy_to_clean_dist[t_n][t_c] += 1

print(f"\n{'='*60}")
print("1. 基本統計")
print(f"{'='*60}")
print(f"Total tokens: {total_tokens:,}")
print(f"Matched tokens: {match_tokens:,} ({100*match_tokens/total_tokens:.2f}%)")
print(f"Unique noisy tokens: {len(noisy_token_counts)}")
print(f"Unique clean tokens: {len(clean_token_counts)}")
print(f"Unique transitions: {len(all_transitions)}")

# ============================================================
# 2. 轉換集中度分析
# ============================================================
print(f"\n{'='*60}")
print("2. 轉換集中度分析 (每個 noisy token 對應多少種 clean token)")
print(f"{'='*60}")

# 計算每個 noisy token 對應的 clean token 數量
n_clean_per_noisy = []
entropy_per_noisy = []

for noisy_tok, clean_dist in noisy_to_clean_dist.items():
    n_clean = len(clean_dist)
    n_clean_per_noisy.append(n_clean)

    # 計算 entropy
    total = sum(clean_dist.values())
    probs = np.array([c / total for c in clean_dist.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    entropy_per_noisy.append(entropy)

n_clean_per_noisy = np.array(n_clean_per_noisy)
entropy_per_noisy = np.array(entropy_per_noisy)

print(f"每個 noisy token 對應的 clean token 數量:")
print(f"  Min: {n_clean_per_noisy.min()}")
print(f"  Max: {n_clean_per_noisy.max()}")
print(f"  Mean: {n_clean_per_noisy.mean():.1f}")
print(f"  Median: {np.median(n_clean_per_noisy):.1f}")
print(f"  只對應 1 個 clean token: {np.sum(n_clean_per_noisy == 1)} ({100*np.mean(n_clean_per_noisy == 1):.1f}%)")
print(f"  對應 ≤5 個 clean token: {np.sum(n_clean_per_noisy <= 5)} ({100*np.mean(n_clean_per_noisy <= 5):.1f}%)")
print(f"  對應 ≤10 個 clean token: {np.sum(n_clean_per_noisy <= 10)} ({100*np.mean(n_clean_per_noisy <= 10):.1f}%)")

print(f"\nEntropy 統計 (越低越集中):")
print(f"  Min: {entropy_per_noisy.min():.2f}")
print(f"  Max: {entropy_per_noisy.max():.2f}")
print(f"  Mean: {entropy_per_noisy.mean():.2f}")
print(f"  Median: {np.median(entropy_per_noisy):.2f}")

# ============================================================
# 3. Top-K 準確率分析
# ============================================================
print(f"\n{'='*60}")
print("3. Top-K 準確率分析 (如果選最常見的 K 個，能覆蓋多少)")
print(f"{'='*60}")

def compute_topk_coverage(noisy_to_clean_dist, k):
    """計算 top-k 覆蓋率"""
    covered = 0
    total = 0
    for noisy_tok, clean_dist in noisy_to_clean_dist.items():
        # 取最常見的 k 個 clean token
        top_k_clean = set([t for t, _ in clean_dist.most_common(k)])
        for clean_tok, count in clean_dist.items():
            total += count
            if clean_tok in top_k_clean:
                covered += count
    return covered / total

for k in [1, 3, 5, 10, 20, 50]:
    coverage = compute_topk_coverage(noisy_to_clean_dist, k)
    print(f"  Top-{k}: {100*coverage:.2f}%")

# ============================================================
# 4. 「可修正」token 分析
# ============================================================
print(f"\n{'='*60}")
print("4. 「可修正」token 分析")
print(f"{'='*60}")

# 如果 noisy_token != clean_token 且 top-1 預測正確
correctable_by_top1 = 0
total_mismatched = 0

for noisy_tok, clean_dist in noisy_to_clean_dist.items():
    for clean_tok, count in clean_dist.items():
        if noisy_tok != clean_tok:
            total_mismatched += count
            # 檢查 top-1 是否就是 clean_tok
            top1 = clean_dist.most_common(1)[0][0]
            if top1 == clean_tok:
                correctable_by_top1 += count

print(f"Mismatched tokens: {total_mismatched:,}")
print(f"Top-1 預測正確 (可用查表修正): {correctable_by_top1:,} ({100*correctable_by_top1/total_mismatched:.2f}%)")

# ============================================================
# 5. 具體例子：最常見的轉換
# ============================================================
print(f"\n{'='*60}")
print("5. 最常見的 token 轉換 (noisy → clean)")
print(f"{'='*60}")

# 排除相同的 (noisy == clean)
diff_transitions = {k: v for k, v in all_transitions.items() if k[0] != k[1]}
print(f"Top 20 轉換 (noisy ≠ clean):")
for (t_n, t_c), count in Counter(diff_transitions).most_common(20):
    # 計算這個 noisy token 總共出現幾次
    total_noisy = noisy_token_counts[t_n]
    # 這個轉換佔該 noisy token 的比例
    ratio = count / total_noisy
    print(f"  {t_n:4d} → {t_c:4d}: {count:6d} times ({100*ratio:.1f}% of token {t_n})")

# ============================================================
# 6. 高置信度可修正 token
# ============================================================
print(f"\n{'='*60}")
print("6. 高置信度可修正 token (dominant clean target ≥ 80%)")
print(f"{'='*60}")

high_conf_tokens = []
for noisy_tok, clean_dist in noisy_to_clean_dist.items():
    total = sum(clean_dist.values())
    top1_clean, top1_count = clean_dist.most_common(1)[0]
    ratio = top1_count / total
    if ratio >= 0.8 and noisy_tok != top1_clean:
        high_conf_tokens.append((noisy_tok, top1_clean, ratio, total))

high_conf_tokens.sort(key=lambda x: -x[3])  # 按出現次數排序
print(f"找到 {len(high_conf_tokens)} 個高置信度可修正 token")
print(f"Top 20:")
for noisy_tok, clean_tok, ratio, total in high_conf_tokens[:20]:
    print(f"  {noisy_tok:4d} → {clean_tok:4d}: {100*ratio:.1f}% confidence, {total:5d} occurrences")

# 計算這些 token 佔總 mismatch 的比例
high_conf_correctable = sum(
    noisy_to_clean_dist[t[0]][t[1]]
    for t in high_conf_tokens
)
print(f"\n高置信度可修正 token 覆蓋的 mismatch: {high_conf_correctable:,} ({100*high_conf_correctable/total_mismatched:.2f}%)")

# ============================================================
# 7. 結論
# ============================================================
print(f"\n{'='*60}")
print("7. 結論")
print(f"{'='*60}")
if compute_topk_coverage(noisy_to_clean_dist, 1) > 0.5:
    print("✅ Top-1 覆蓋率 > 50%，有潛力用查表方式做 token steering")
else:
    print("⚠️  Top-1 覆蓋率較低，轉換關係分散")

if len(high_conf_tokens) > 100:
    print(f"✅ 有 {len(high_conf_tokens)} 個高置信度可修正 token，可嘗試 rule-based 修正")
else:
    print(f"⚠️  高置信度可修正 token 數量較少 ({len(high_conf_tokens)})")

print("\n建議:")
if entropy_per_noisy.mean() < 3.0:
    print("  - 轉換 entropy 較低，pattern 相對集中，可嘗試學習映射")
else:
    print("  - 轉換 entropy 較高，pattern 分散，可能需要 context-aware 方法")
