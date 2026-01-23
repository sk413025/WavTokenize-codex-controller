#!/usr/bin/env python3
"""
Diagnostic v3: 從 filename 解析正確的資訊，重新分析 token 轉換

filename 格式: nor_girl9_box_LDV_100.wav
- nor: 無意義
- girl9: speaker_id
- box: 噪音類型 (box, plastic, papercup 等)
- LDV: 無意義
- 100: 句子編號
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
# 0. 解析 filename 獲取正確資訊
# ============================================================
print(f"\n{'='*60}")
print("0. 從 filename 解析正確資訊")
print(f"{'='*60}")

def parse_filename(filename):
    """
    解析 filename: nor_girl9_box_LDV_100.wav
    返回: (speaker, noise_type, sentence_id)
    """
    name = filename.replace('.wav', '')
    parts = name.split('_')

    if 'clean' in parts:
        # clean 檔案: nor_girl9_clean_100
        speaker = parts[1]
        noise_type = 'clean'
        sentence_id = parts[3]
    else:
        # noisy 檔案: nor_girl9_box_LDV_100
        speaker = parts[1]
        noise_type = parts[2]  # box, plastic, papercup 等
        sentence_id = parts[4]

    return speaker, noise_type, sentence_id

# 測試解析
print("\n測試 filename 解析:")
for i in range(3):
    fn = cache[i]['noisy_path']
    parsed = parse_filename(fn)
    print(f"  {fn} -> speaker={parsed[0]}, noise_type={parsed[1]}, sentence_id={parsed[2]}")

# 解析所有樣本
for item in cache:
    speaker, noise_type, sentence_id = parse_filename(item['noisy_path'])
    item['parsed_noise_type'] = noise_type
    item['parsed_sentence_id'] = sentence_id

# 統計
print(f"\n唯一值統計:")
print(f"  Noise types: {set(item['parsed_noise_type'] for item in cache)}")
sentence_ids = sorted(set(item['parsed_sentence_id'] for item in cache))
print(f"  Sentence IDs: {sentence_ids[:10]}... (共 {len(sentence_ids)} 個)")
print(f"  Speakers: {set(item['speaker_id'] for item in cache)}")

# ============================================================
# 1. 按 (speaker, sentence_id) 分析 - 同一人說同一句話
# ============================================================
print(f"\n{'='*60}")
print("1. 按 (Speaker, Sentence_ID) 分析 - 同一人說同一句話")
print(f"{'='*60}")

speaker_sentence_stats = defaultdict(lambda: {
    'match': 0, 'total': 0,
    'transitions': Counter(),
    'samples': []
})

for item in cache:
    speaker = item['speaker_id']
    sentence_id = item['parsed_sentence_id']
    key = (speaker, sentence_id)

    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    speaker_sentence_stats[key]['samples'].append({
        'noisy': noisy_tokens[:min_len],
        'clean': clean_tokens[:min_len],
        'noise_type': item['parsed_noise_type']
    })

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        speaker_sentence_stats[key]['total'] += 1
        if t_n == t_c:
            speaker_sentence_stats[key]['match'] += 1
        else:
            speaker_sentence_stats[key]['transitions'][(t_n, t_c)] += 1

print(f"\n總共 {len(speaker_sentence_stats)} 個 (speaker, sentence_id) 組合")

# 統計每個組合的樣本數
combo_sample_counts = [(k, len(v['samples'])) for k, v in speaker_sentence_stats.items()]
combo_sample_counts.sort(key=lambda x: -x[1])

print(f"\n樣本數分佈:")
sample_counts = [len(v['samples']) for v in speaker_sentence_stats.values()]
print(f"  Min: {min(sample_counts)}, Max: {max(sample_counts)}, Mean: {np.mean(sample_counts):.1f}")

print(f"\n前 10 個組合:")
for (speaker, sentence_id), n_samples in combo_sample_counts[:10]:
    stats = speaker_sentence_stats[(speaker, sentence_id)]
    match_rate = stats['match'] / stats['total'] if stats['total'] > 0 else 0
    print(f"  ({speaker}, {sentence_id}): {n_samples} 樣本, match rate={100*match_rate:.1f}%")

# ============================================================
# 2. 深入分析：同一人同一句話的多個樣本 (不同噪音類型)
# ============================================================
print(f"\n{'='*60}")
print("2. 深入分析：同一人同一句話，不同噪音類型")
print(f"{'='*60}")

# 找一個有多個樣本的組合
for (speaker, sentence_id), n_samples in combo_sample_counts[:1]:
    print(f"\n>>> 分析: {speaker} 說句子 '{sentence_id}' ({n_samples} 個樣本)")

    stats = speaker_sentence_stats[(speaker, sentence_id)]
    samples = stats['samples']

    # 顯示各樣本的噪音類型
    noise_types = [s['noise_type'] for s in samples]
    print(f"\n噪音類型分佈: {Counter(noise_types)}")

    # 比較 clean tokens 是否一致 (同一人說同一句話，clean 應該一樣)
    min_len = min(len(s['clean']) for s in samples)
    min_len = min(min_len, 50)

    print(f"\n前 20 個位置的 Clean Token 一致性檢查:")
    clean_consistent_count = 0
    for pos in range(min_len):
        clean_tokens_at_pos = [s['clean'][pos] for s in samples]
        is_consistent = len(set(clean_tokens_at_pos)) == 1
        if is_consistent:
            clean_consistent_count += 1
        if pos < 20:
            unique_count = len(set(clean_tokens_at_pos))
            print(f"  位置 {pos:2d}: {unique_count:3d} 種不同的 clean token",
                  "✓" if is_consistent else "✗")

    print(f"\nClean token 一致率: {clean_consistent_count}/{min_len} ({100*clean_consistent_count/min_len:.1f}%)")

# ============================================================
# 3. 檢查：同一 Clean 音檔產生的 Token 是否一致？
# ============================================================
print(f"\n{'='*60}")
print("3. 關鍵問題：同一 Clean 音檔產生的 Token 是否一致？")
print(f"{'='*60}")

# 按 clean_path 分組
clean_path_tokens = defaultdict(list)
for item in cache:
    clean_path = item['clean_path']
    clean_tokens = tuple(item['clean_tokens'].flatten().tolist())
    clean_path_tokens[clean_path].append(clean_tokens)

print(f"\n唯一 clean_path 數量: {len(clean_path_tokens)}")

# 檢查每個 clean_path 對應的 token 是否都一樣
consistent_count = 0
inconsistent_count = 0
for clean_path, token_lists in clean_path_tokens.items():
    if len(token_lists) > 1:
        if len(set(token_lists)) == 1:
            consistent_count += 1
        else:
            inconsistent_count += 1

print(f"有多個樣本的 clean_path: {consistent_count + inconsistent_count}")
print(f"  Token 一致: {consistent_count}")
print(f"  Token 不一致: {inconsistent_count}")

if inconsistent_count > 0:
    print(f"\n⚠️ 發現問題：同一個 clean 音檔產生了不同的 token！")
else:
    print(f"\n✓ 同一個 clean 音檔的 token 是一致的")

# ============================================================
# 4. 按噪音類型 (Noise Type) 分析
# ============================================================
print(f"\n{'='*60}")
print("4. 按噪音類型 (Noise Type) 分析")
print(f"{'='*60}")

noise_stats = defaultdict(lambda: {'match': 0, 'total': 0, 'transitions': Counter()})

for item in cache:
    noise_type = item['parsed_noise_type']
    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        noise_stats[noise_type]['total'] += 1
        if t_n == t_c:
            noise_stats[noise_type]['match'] += 1
        else:
            noise_stats[noise_type]['transitions'][(t_n, t_c)] += 1

print(f"\n{'Noise Type':<15} {'Match Rate':<12} {'Tokens':<12} {'Unique Trans':<15}")
print("-" * 55)
for noise_type in sorted(noise_stats.keys()):
    stats = noise_stats[noise_type]
    match_rate = stats['match'] / stats['total']
    n_trans = len(stats['transitions'])
    print(f"{noise_type:<15} {100*match_rate:>6.1f}%      {stats['total']:<12} {n_trans:<15}")

# ============================================================
# 5. 最細粒度: (Speaker, Sentence_ID, Noise_Type)
# ============================================================
print(f"\n{'='*60}")
print("5. 最細粒度: (Speaker, Sentence_ID, Noise_Type)")
print(f"{'='*60}")

finest_stats = defaultdict(lambda: {
    'match': 0, 'total': 0,
    'transitions': Counter(),
    'n_samples': 0
})

for item in cache:
    key = (item['speaker_id'], item['parsed_sentence_id'], item['parsed_noise_type'])
    finest_stats[key]['n_samples'] += 1

    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        finest_stats[key]['total'] += 1
        if t_n == t_c:
            finest_stats[key]['match'] += 1
        else:
            finest_stats[key]['transitions'][(t_n, t_c)] += 1

print(f"\n總共 {len(finest_stats)} 個 (speaker, sentence_id, noise_type) 組合")

# 計算 Top-1 可修正率
def compute_top1_correctable(transitions):
    if not transitions:
        return 0, 0

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

if top1_rates:
    print(f"\nTop-1 可修正率 (在 mismatch 中):")
    print(f"  Mean: {100*np.mean(top1_rates):.1f}%")
    print(f"  Std:  {100*np.std(top1_rates):.1f}%")
    print(f"  Min:  {100*np.min(top1_rates):.1f}%")
    print(f"  Max:  {100*np.max(top1_rates):.1f}%")

# 找高 top1 可修正率的例子
print(f"\n高 Top-1 可修正率的組合 (>50%):")
high_top1_combos = []
for key, stats in finest_stats.items():
    correctable, total_mismatch = compute_top1_correctable(stats['transitions'])
    if total_mismatch > 20:  # 至少 20 個 mismatch
        rate = correctable / total_mismatch
        if rate > 0.5:
            high_top1_combos.append((key, rate, total_mismatch, stats['n_samples']))

high_top1_combos.sort(key=lambda x: -x[1])
for key, rate, n_mismatch, n_samples in high_top1_combos[:10]:
    print(f"  {key}: {100*rate:.1f}% ({n_mismatch} mismatches, {n_samples} samples)")

if not high_top1_combos:
    print("  (無)")

# ============================================================
# 6. 結論
# ============================================================
print(f"\n{'='*60}")
print("6. 結論")
print(f"{'='*60}")

avg_match = np.mean(match_rates)
avg_top1 = np.mean(top1_rates) if top1_rates else 0

print(f"\n平均 Match Rate: {100*avg_match:.1f}%")
print(f"平均 Top-1 可修正率: {100*avg_top1:.1f}%")

if avg_top1 > 0.5:
    print("\n✅ Top-1 可修正率較高，有潛力做 token steering")
elif avg_top1 > 0.2:
    print("\n⚠️ Top-1 可修正率中等，可能需要更多 context")
else:
    print("\n❌ Top-1 可修正率很低，簡單查表方式不可行")
