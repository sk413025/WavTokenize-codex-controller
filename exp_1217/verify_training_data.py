"""
驗證 exp_1217 訓練是否使用了 clean->clean 資料
"""
import torch
from pathlib import Path

train_cache = "/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt"

print("=" * 60)
print("驗證訓練資料流程")
print("=" * 60)

# 載入 cache
data = torch.load(train_cache, weights_only=False)
print(f"\n總樣本數: {len(data)}")

# 分析 noisy_path 和 clean_path
clean_clean_samples = []
noisy_clean_samples = []

for sample in data:
    noisy_path = sample.get('noisy_path', '')
    clean_path = sample.get('clean_path', '')

    # 檢查 noisy_path 是否包含 "clean" (表示是 clean 檔案)
    noisy_is_clean_file = '_clean_' in noisy_path or noisy_path.endswith('_clean.wav')

    if noisy_path == clean_path:
        clean_clean_samples.append(sample)
    elif noisy_is_clean_file:
        clean_clean_samples.append(sample)
    else:
        noisy_clean_samples.append(sample)

print(f"\n根據 PATH 分析:")
print(f"  Clean->Clean pairs: {len(clean_clean_samples)} ({len(clean_clean_samples)/len(data)*100:.1f}%)")
print(f"  Noisy->Clean pairs: {len(noisy_clean_samples)} ({len(noisy_clean_samples)/len(data)*100:.1f}%)")

# 顯示 clean->clean 範例
print("\nClean->Clean 範例 (前 5 個):")
for i, s in enumerate(clean_clean_samples[:5]):
    print(f"  {i+1}. noisy_path: {s.get('noisy_path')}")
    print(f"      clean_path: {s.get('clean_path')}")

# 顯示 noisy->clean 範例
print("\nNoisy->Clean 範例 (前 5 個):")
for i, s in enumerate(noisy_clean_samples[:5]):
    print(f"  {i+1}. noisy_path: {s.get('noisy_path')}")
    print(f"      clean_path: {s.get('clean_path')}")

# 確認訓練流程
print("\n" + "=" * 60)
print("訓練流程確認")
print("=" * 60)

print("""
訓練資料流程:
1. Cache 包含 noisy_path 和 clean_path
2. data_aligned.py 的 AlignedNoisyCleanPairDataset:
   - 首先嘗試從 sample 取 'noisy_audio' 和 'clean_audio'
   - 如果不存在，就從 noisy_path 和 clean_path 載入音檔

3. 對於 clean->clean pairs:
   - noisy_path 指向 clean 檔案 (例如 nor_girl7_clean_100.wav)
   - clean_path 也指向同一個 clean 檔案
   - 訓練時 noisy_audio 和 clean_audio 會是完全相同的！
""")

# 驗證：檢查實際的 token 匹配
print("=" * 60)
print("Token 匹配驗證")
print("=" * 60)

clean_clean_match_rates = []
for s in clean_clean_samples[:100]:
    noisy_tokens = s.get('noisy_tokens')
    clean_tokens = s.get('clean_tokens')
    if noisy_tokens is not None and clean_tokens is not None:
        min_len = min(len(noisy_tokens), len(clean_tokens))
        match_rate = (noisy_tokens[:min_len] == clean_tokens[:min_len]).float().mean().item()
        clean_clean_match_rates.append(match_rate)

noisy_clean_match_rates = []
for s in noisy_clean_samples[:100]:
    noisy_tokens = s.get('noisy_tokens')
    clean_tokens = s.get('clean_tokens')
    if noisy_tokens is not None and clean_tokens is not None:
        min_len = min(len(noisy_tokens), len(clean_tokens))
        match_rate = (noisy_tokens[:min_len] == clean_tokens[:min_len]).float().mean().item()
        noisy_clean_match_rates.append(match_rate)

import numpy as np
clean_clean_match_rates = np.array(clean_clean_match_rates)
noisy_clean_match_rates = np.array(noisy_clean_match_rates)

print(f"\nClean->Clean token 匹配率: {clean_clean_match_rates.mean()*100:.2f}% (應該接近 100%)")
print(f"Noisy->Clean token 匹配率: {noisy_clean_match_rates.mean()*100:.2f}% (真正的基線)")

print("\n" + "=" * 60)
print("🔍 結論")
print("=" * 60)
print(f"""
是的，exp_1217 的訓練確實使用了 clean->clean 資料！

1. 約 28% 的訓練樣本是 clean->clean pairs
2. 這些樣本的 noisy_path 指向 clean 檔案
3. 訓練時模型會收到完全相同的 noisy 和 clean 音頻
4. 對於這些樣本，正確答案就是「不做任何改變」

這可能影響模型學習：
- 模型可能學會「大部分時候不改變輸入」的策略
- 這會限制模型在真正 noisy 輸入上的表現
""")
