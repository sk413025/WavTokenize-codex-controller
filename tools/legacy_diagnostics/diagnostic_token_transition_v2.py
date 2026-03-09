#!/usr/bin/env python3
"""
Diagnostic v2: 按語者/句子分層分析 token 轉換
目標: 看同一人說同一句話時，token 轉換是否有規律
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
# 0. 先看一個具體例子
# ============================================================
print(f"\n{'='*60}")
print("0. 具體例子：第一個樣本的 token 比較")
print(f"{'='*60}")

item = cache[0]
print(f"Speaker: {item['speaker_id']}")
print(f"Sentence: {item['sentence_id']}")
print(f"Material: {item['material']}")

noisy_tokens = item['noisy_tokens'].flatten().tolist()[:20]
clean_tokens = item['clean_tokens'].flatten().tolist()[:20]

print(f"\n前 20 個 token 比較:")
print(f"{'位置':<6} {'Noisy':<8} {'Clean':<8} {'Match':<6}")
print("-" * 30)
for i, (t_n, t_c) in enumerate(zip(noisy_tokens, clean_tokens)):
    match = "✓" if t_n == t_c else "✗"
    print(f"{i:<6} {t_n:<8} {t_c:<8} {match:<6}")

match_rate = sum(1 for t_n, t_c in zip(noisy_tokens, clean_tokens) if t_n == t_c) / len(noisy_tokens)
print(f"\n此樣本 match rate: {100*match_rate:.1f}%")

# ============================================================
# 1. 按語者分析
# ============================================================
print(f"\n{'='*60}")
print("1. 按語者 (Speaker) 分析")
print(f"{'='*60}")

speaker_stats = defaultdict(lambda: {'match': 0, 'total': 0, 'transitions': Counter()})

for item in cache:
    speaker = item['speaker_id']
    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        speaker_stats[speaker]['total'] += 1
        if t_n == t_c:
            speaker_stats[speaker]['match'] += 1
        else:
            speaker_stats[speaker]['transitions'][(t_n, t_c)] += 1

print(f"\n{'Speaker':<12} {'Match Rate':<12} {'Samples':<10} {'Unique Trans':<15}")
print("-" * 50)
for speaker in sorted(speaker_stats.keys()):
    stats = speaker_stats[speaker]
    match_rate = stats['match'] / stats['total']
    n_trans = len(stats['transitions'])
    print(f"{speaker:<12} {100*match_rate:>6.1f}%      {stats['total']:<10} {n_trans:<15}")

# ============================================================
# 2. 按句子分析
# ============================================================
print(f"\n{'='*60}")
print("2. 按句子 (Sentence) 分析")
print(f"{'='*60}")

sentence_stats = defaultdict(lambda: {'match': 0, 'total': 0, 'transitions': Counter()})

for item in cache:
    sentence = item['sentence_id']
    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        sentence_stats[sentence]['total'] += 1
        if t_n == t_c:
            sentence_stats[sentence]['match'] += 1
        else:
            sentence_stats[sentence]['transitions'][(t_n, t_c)] += 1

print(f"\n前 10 個句子:")
print(f"{'Sentence':<12} {'Match Rate':<12} {'Samples':<10} {'Unique Trans':<15}")
print("-" * 50)
for sentence in sorted(sentence_stats.keys())[:10]:
    stats = sentence_stats[sentence]
    match_rate = stats['match'] / stats['total']
    n_trans = len(stats['transitions'])
    print(f"{sentence:<12} {100*match_rate:>6.1f}%      {stats['total']:<10} {n_trans:<15}")

# ============================================================
# 3. 按 (語者, 句子) 組合分析 - 最細粒度
# ============================================================
print(f"\n{'='*60}")
print("3. 按 (Speaker, Sentence) 組合分析 - 同一人說同一句話")
print(f"{'='*60}")

combo_stats = defaultdict(lambda: {
    'match': 0, 'total': 0,
    'transitions': Counter(),
    'samples': []  # 存每個樣本的 token 序列
})

for item in cache:
    speaker = item['speaker_id']
    sentence = item['sentence_id']
    combo = (speaker, sentence)

    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))
    noisy_tokens = noisy_tokens[:min_len]
    clean_tokens = clean_tokens[:min_len]

    combo_stats[combo]['samples'].append({
        'noisy': noisy_tokens,
        'clean': clean_tokens,
        'material': item['material']
    })

    for t_n, t_c in zip(noisy_tokens, clean_tokens):
        combo_stats[combo]['total'] += 1
        if t_n == t_c:
            combo_stats[combo]['match'] += 1
        else:
            combo_stats[combo]['transitions'][(t_n, t_c)] += 1

# 統計每個組合有多少樣本
combo_sample_counts = [(k, len(v['samples'])) for k, v in combo_stats.items()]
combo_sample_counts.sort(key=lambda x: -x[1])

print(f"\n總共 {len(combo_stats)} 個 (speaker, sentence) 組合")
print(f"\n樣本數最多的 10 個組合:")
print(f"{'Speaker':<10} {'Sentence':<10} {'#Samples':<10} {'Match Rate':<12} {'Unique Trans':<12}")
print("-" * 60)
for (speaker, sentence), n_samples in combo_sample_counts[:10]:
    stats = combo_stats[(speaker, sentence)]
    match_rate = stats['match'] / stats['total']
    n_trans = len(stats['transitions'])
    print(f"{speaker:<10} {sentence:<10} {n_samples:<10} {100*match_rate:>6.1f}%      {n_trans:<12}")

# ============================================================
# 4. 深入分析：同一人說同一句話，多次錄音的 token 轉換是否一致？
# ============================================================
print(f"\n{'='*60}")
print("4. 深入分析：同一人同一句話的多次錄音")
print(f"{'='*60}")

# 找一個有多個樣本的組合
for (speaker, sentence), n_samples in combo_sample_counts[:5]:
    if n_samples >= 2:
        print(f"\n>>> 分析: {speaker} 說 '{sentence}' ({n_samples} 個樣本)")

        stats = combo_stats[(speaker, sentence)]
        samples = stats['samples']

        # 比較前兩個樣本
        print(f"\n樣本材質: {[s['material'] for s in samples]}")

        # 看每個位置的 token 是否穩定
        min_len = min(len(s['noisy']) for s in samples)
        min_len = min(min_len, 30)  # 只看前 30 個

        print(f"\n前 {min_len} 個位置的 token 比較:")
        print(f"{'Pos':<5}", end="")
        for i, s in enumerate(samples[:3]):
            print(f"{'S'+str(i)+' Noisy':<10} {'S'+str(i)+' Clean':<10}", end="")
        print(f"{'Noisy一致':<12} {'Clean一致':<12} {'轉換一致':<12}")
        print("-" * 100)

        noisy_consistent = 0
        clean_consistent = 0
        transition_consistent = 0

        for pos in range(min_len):
            noisy_tokens_at_pos = [s['noisy'][pos] for s in samples]
            clean_tokens_at_pos = [s['clean'][pos] for s in samples]

            noisy_same = len(set(noisy_tokens_at_pos)) == 1
            clean_same = len(set(clean_tokens_at_pos)) == 1

            # 轉換是否一致 (noisy→clean 的 pair 是否相同)
            transitions_at_pos = [(s['noisy'][pos], s['clean'][pos]) for s in samples]
            trans_same = len(set(transitions_at_pos)) == 1

            if noisy_same:
                noisy_consistent += 1
            if clean_same:
                clean_consistent += 1
            if trans_same:
                transition_consistent += 1

            # 只印前 15 個
            if pos < 15:
                print(f"{pos:<5}", end="")
                for s in samples[:3]:
                    print(f"{s['noisy'][pos]:<10} {s['clean'][pos]:<10}", end="")
                print(f"{'✓' if noisy_same else '✗':<12} {'✓' if clean_same else '✗':<12} {'✓' if trans_same else '✗':<12}")

        print(f"\n一致性統計 (前 {min_len} 個位置):")
        print(f"  Noisy token 一致: {noisy_consistent}/{min_len} ({100*noisy_consistent/min_len:.1f}%)")
        print(f"  Clean token 一致: {clean_consistent}/{min_len} ({100*clean_consistent/min_len:.1f}%)")
        print(f"  轉換 (N→C) 一致:  {transition_consistent}/{min_len} ({100*transition_consistent/min_len:.1f}%)")

        break  # 只分析一個組合

# ============================================================
# 5. 分析同一材質 (錄音環境) 的規律性
# ============================================================
print(f"\n{'='*60}")
print("5. 按材質 (Material/錄音環境) 分析")
print(f"{'='*60}")

material_stats = defaultdict(lambda: {'match': 0, 'total': 0, 'transitions': Counter()})

for item in cache:
    material = item['material']
    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        material_stats[material]['total'] += 1
        if t_n == t_c:
            material_stats[material]['match'] += 1
        else:
            material_stats[material]['transitions'][(t_n, t_c)] += 1

print(f"\n{'Material':<15} {'Match Rate':<12} {'Tokens':<12} {'Unique Trans':<15} {'Trans集中度':<15}")
print("-" * 70)
for material in sorted(material_stats.keys()):
    stats = material_stats[material]
    match_rate = stats['match'] / stats['total']
    n_trans = len(stats['transitions'])

    # 計算 top-10 轉換佔所有轉換的比例 (集中度)
    if stats['transitions']:
        top10_count = sum(c for _, c in stats['transitions'].most_common(10))
        total_trans = sum(stats['transitions'].values())
        concentration = top10_count / total_trans
    else:
        concentration = 0

    print(f"{material:<15} {100*match_rate:>6.1f}%      {stats['total']:<12} {n_trans:<15} {100*concentration:>6.1f}%")

# ============================================================
# 6. 最終結論：如果限定 (speaker, sentence, material)，轉換是否有規律？
# ============================================================
print(f"\n{'='*60}")
print("6. 最細粒度: (Speaker, Sentence, Material) 組合")
print(f"{'='*60}")

finest_stats = defaultdict(lambda: {'match': 0, 'total': 0, 'transitions': Counter(), 'count': 0})

for item in cache:
    key = (item['speaker_id'], item['sentence_id'], item['material'])
    finest_stats[key]['count'] += 1

    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        finest_stats[key]['total'] += 1
        if t_n == t_c:
            finest_stats[key]['match'] += 1
        else:
            finest_stats[key]['transitions'][(t_n, t_c)] += 1

# 計算 Top-1 可修正率
def compute_top1_correctable(transitions):
    """在 mismatch 中，top-1 預測正確的比例"""
    if not transitions:
        return 0, 0

    # 按 noisy token 分組
    noisy_to_clean = defaultdict(Counter)
    for (t_n, t_c), count in transitions.items():
        noisy_to_clean[t_n][t_c] += count

    correctable = 0
    total_mismatch = 0
    for t_n, clean_dist in noisy_to_clean.items():
        top1 = clean_dist.most_common(1)[0][0]
        for t_c, count in clean_dist.items():
            total_mismatch += count
            if t_c == top1:
                correctable += count

    return correctable, total_mismatch

print(f"\n總共 {len(finest_stats)} 個 (speaker, sentence, material) 組合")

# 統計
match_rates = []
top1_rates = []

for key, stats in finest_stats.items():
    if stats['total'] > 0:
        match_rates.append(stats['match'] / stats['total'])
        correctable, total_mismatch = compute_top1_correctable(stats['transitions'])
        if total_mismatch > 0:
            top1_rates.append(correctable / total_mismatch)

print(f"\nMatch Rate 分佈:")
print(f"  Mean: {100*np.mean(match_rates):.1f}%")
print(f"  Std:  {100*np.std(match_rates):.1f}%")
print(f"  Min:  {100*np.min(match_rates):.1f}%")
print(f"  Max:  {100*np.max(match_rates):.1f}%")

print(f"\nTop-1 可修正率 (在 mismatch 中):")
print(f"  Mean: {100*np.mean(top1_rates):.1f}%")
print(f"  Std:  {100*np.std(top1_rates):.1f}%")
print(f"  Min:  {100*np.min(top1_rates):.1f}%")
print(f"  Max:  {100*np.max(top1_rates):.1f}%")

# 找一個 top1 可修正率高的例子
best_key = None
best_rate = 0
for key, stats in finest_stats.items():
    correctable, total_mismatch = compute_top1_correctable(stats['transitions'])
    if total_mismatch > 50:  # 至少 50 個 mismatch
        rate = correctable / total_mismatch
        if rate > best_rate:
            best_rate = rate
            best_key = key

if best_key:
    print(f"\n最佳例子: {best_key}")
    print(f"  Top-1 可修正率: {100*best_rate:.1f}%")
    stats = finest_stats[best_key]
    print(f"  Match rate: {100*stats['match']/stats['total']:.1f}%")
    print(f"  Top 5 轉換:")
    for (t_n, t_c), count in stats['transitions'].most_common(5):
        print(f"    {t_n} → {t_c}: {count} times")
