#!/usr/bin/env python3
"""
分析易錯 Token，輸出給 LoRA 訓練使用

輸出:
1. token_error_rates.pt: 每個 token 的 error rate
2. noise_type_difficulty.json: 各噪音類型的難度排序
3. error_token_analysis.json: 詳細分析報告
"""

import torch
import numpy as np
import json
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# 配置
# ============================================================
CACHE_PATH = "/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt"
OUTPUT_DIR = Path(__file__).parent / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 載入資料
# ============================================================
print("Loading val_cache...")
cache = torch.load(CACHE_PATH, weights_only=False)
print(f"Loaded {len(cache)} samples")

def parse_filename(filename):
    """解析 filename: nor_girl9_box_LDV_100.wav"""
    name = filename.replace('.wav', '')
    parts = name.split('_')
    if 'clean' in parts:
        return parts[1], 'clean', parts[3]
    else:
        return parts[1], parts[2], parts[4]

# 解析所有樣本
for item in cache:
    speaker, noise_type, sentence_id = parse_filename(item['noisy_path'])
    item['parsed_noise_type'] = noise_type
    item['parsed_sentence_id'] = sentence_id

# ============================================================
# 1. 計算每個 Token 的 Error Rate
# ============================================================
print("\n計算每個 Token 的 Error Rate...")

# 統計每個 clean token 被正確/錯誤預測的次數
token_correct = Counter()  # clean_token -> correct count
token_total = Counter()    # clean_token -> total count

for item in cache:
    noise_type = item['parsed_noise_type']
    if noise_type == 'clean':
        continue

    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        token_total[t_c] += 1
        if t_n == t_c:
            token_correct[t_c] += 1

# 計算 error rate
token_error_rates = {}
for token_id in token_total:
    total = token_total[token_id]
    correct = token_correct[token_id]
    error_rate = 1.0 - (correct / total)
    token_error_rates[token_id] = {
        'error_rate': error_rate,
        'total_count': total,
        'correct_count': correct,
    }

# 轉成 tensor 格式 (4096 維，未出現的 token 設為 0.5)
error_rate_tensor = torch.full((4096,), 0.5)  # 預設 0.5
for token_id, stats in token_error_rates.items():
    if token_id < 4096:
        error_rate_tensor[token_id] = stats['error_rate']

# 儲存
torch.save(error_rate_tensor, OUTPUT_DIR / "token_error_rates.pt")
print(f"Saved token_error_rates.pt (shape: {error_rate_tensor.shape})")

# 統計
error_rates = [s['error_rate'] for s in token_error_rates.values()]
print(f"\nError Rate 統計:")
print(f"  Min: {min(error_rates):.3f}")
print(f"  Max: {max(error_rates):.3f}")
print(f"  Mean: {np.mean(error_rates):.3f}")
print(f"  Median: {np.median(error_rates):.3f}")

# ============================================================
# 2. 找出最易錯的 Token
# ============================================================
print("\n最易錯的 20 個 Token (至少出現 100 次):")
sorted_tokens = sorted(
    [(k, v) for k, v in token_error_rates.items() if v['total_count'] >= 100],
    key=lambda x: -x[1]['error_rate']
)

top_error_tokens = []
for token_id, stats in sorted_tokens[:20]:
    print(f"  Token {token_id:4d}: error_rate={stats['error_rate']:.3f}, "
          f"total={stats['total_count']:5d}, correct={stats['correct_count']:5d}")
    top_error_tokens.append({
        'token_id': token_id,
        **stats
    })

# ============================================================
# 3. 按噪音類型統計難度
# ============================================================
print("\n按噪音類型統計...")

noise_type_stats = defaultdict(lambda: {'match': 0, 'total': 0})

for item in cache:
    noise_type = item['parsed_noise_type']
    if noise_type == 'clean':
        continue

    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        noise_type_stats[noise_type]['total'] += 1
        if t_n == t_c:
            noise_type_stats[noise_type]['match'] += 1

noise_type_difficulty = {}
print("\n噪音類型難度排序 (Match Rate 低 = 更難):")
for noise_type in sorted(noise_type_stats.keys(),
                         key=lambda x: noise_type_stats[x]['match']/noise_type_stats[x]['total']):
    stats = noise_type_stats[noise_type]
    match_rate = stats['match'] / stats['total']
    error_rate = 1 - match_rate
    noise_type_difficulty[noise_type] = {
        'match_rate': match_rate,
        'error_rate': error_rate,
        'total_tokens': stats['total'],
    }
    print(f"  {noise_type:<12}: match_rate={match_rate:.3f}, error_rate={error_rate:.3f}")

# ============================================================
# 4. 分析「易錯 token 在哪些噪音下特別容易錯」
# ============================================================
print("\n分析易錯 token 在各噪音類型下的表現...")

# 對 top 10 易錯 token 做詳細分析
token_by_noise = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

for item in cache:
    noise_type = item['parsed_noise_type']
    if noise_type == 'clean':
        continue

    noisy_tokens = item['noisy_tokens'].flatten().tolist()
    clean_tokens = item['clean_tokens'].flatten().tolist()
    min_len = min(len(noisy_tokens), len(clean_tokens))

    for t_n, t_c in zip(noisy_tokens[:min_len], clean_tokens[:min_len]):
        token_by_noise[t_c][noise_type]['total'] += 1
        if t_n == t_c:
            token_by_noise[t_c][noise_type]['correct'] += 1

# 輸出 top 10 易錯 token 的詳細分析
print("\nTop 10 易錯 Token 在各噪音類型下的 Error Rate:")
detailed_analysis = []
for token_id, stats in sorted_tokens[:10]:
    print(f"\n  Token {token_id} (overall error_rate={stats['error_rate']:.3f}):")
    token_detail = {'token_id': token_id, 'overall_error_rate': stats['error_rate'], 'by_noise': {}}
    for noise_type in ['plastic', 'box', 'papercup']:
        ns = token_by_noise[token_id][noise_type]
        if ns['total'] > 0:
            er = 1 - (ns['correct'] / ns['total'])
            print(f"    {noise_type:<12}: error_rate={er:.3f} ({ns['correct']}/{ns['total']})")
            token_detail['by_noise'][noise_type] = {'error_rate': er, 'total': ns['total']}
    detailed_analysis.append(token_detail)

# ============================================================
# 5. 儲存分析結果
# ============================================================
analysis_output = {
    'summary': {
        'total_samples': len(cache),
        'total_tokens_analyzed': sum(s['total_count'] for s in token_error_rates.values()),
        'unique_tokens': len(token_error_rates),
        'overall_error_rate': np.mean(error_rates),
    },
    'noise_type_difficulty': noise_type_difficulty,
    'top_error_tokens': top_error_tokens,
    'detailed_analysis': detailed_analysis,
}

with open(OUTPUT_DIR / "error_token_analysis.json", 'w') as f:
    json.dump(analysis_output, f, indent=2)

with open(OUTPUT_DIR / "noise_type_difficulty.json", 'w') as f:
    json.dump(noise_type_difficulty, f, indent=2)

print(f"\n分析結果已儲存至 {OUTPUT_DIR}/")
print("  - token_error_rates.pt: 每個 token 的 error rate tensor")
print("  - error_token_analysis.json: 詳細分析報告")
print("  - noise_type_difficulty.json: 噪音類型難度")

# ============================================================
# 6. 給 LoRA 訓練的建議
# ============================================================
print("\n" + "="*60)
print("給 LoRA 訓練的建議")
print("="*60)

print("\n1. Token-Weighted Loss 建議權重:")
print("   - 對 error_rate > 0.8 的 token: weight = 2.0")
print("   - 對 error_rate > 0.6 的 token: weight = 1.5")
print("   - 其他 token: weight = 1.0")

high_error_count = sum(1 for s in token_error_rates.values() if s['error_rate'] > 0.8)
mid_error_count = sum(1 for s in token_error_rates.values() if 0.6 < s['error_rate'] <= 0.8)
print(f"\n   高錯誤率 token 數量 (>0.8): {high_error_count}")
print(f"   中錯誤率 token 數量 (0.6-0.8): {mid_error_count}")

print("\n2. Noise-Type Aware 建議:")
print(f"   - plastic: 最難 (error_rate={noise_type_difficulty['plastic']['error_rate']:.3f})")
print(f"   - box: 中等 (error_rate={noise_type_difficulty['box']['error_rate']:.3f})")
print(f"   - papercup: 較易 (error_rate={noise_type_difficulty['papercup']['error_rate']:.3f})")
print("   建議: 對 plastic 類型樣本給更高的 loss weight")

print("\n3. 敏感層 (來自 exp_1231_feature):")
print("   - model[4] ResBlock2: 噪音敏感度 0.80")
print("   - model[6] Downsample2: 噪音敏感度 0.79")
print("   建議: 優先在這些層加 LoRA")
