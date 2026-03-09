"""
Token Distribution 分析工具

用途: 分析 Train/Val set 的 token 分布差異，診斷 distribution mismatch 問題

實驗日期: 2025-11-05
相關報告: PLATEAU_MECHANISM_ANALYSIS.md
"""

import torch
from collections import Counter
import numpy as np
import sys
from pathlib import Path

def analyze_token_distribution(cache_path, dataset_name="Dataset"):
    """
    分析指定數據集的 token 分布
    
    Args:
        cache_path: 緩存數據路徑 (e.g., './data/train_cache.pt')
        dataset_name: 數據集名稱 (用於輸出)
    
    Returns:
        dict: {
            'total_tokens': int,
            'unique_tokens': int,
            'token_counter': Counter,
            'top_20': list of (token_id, count, percentage)
        }
    """
    print(f"\n{'='*70}")
    print(f"分析 {dataset_name}")
    print(f"{'='*70}\n")
    
    # 載入數據
    data = torch.load(cache_path, weights_only=False)
    print(f"載入 {len(data)} 個樣本")
    
    # 提取所有非 padding tokens
    all_tokens = []
    for sample in data:
        tokens = sample['clean_tokens'].tolist()
        # 移除 padding (假設 padding value = 0)
        non_pad_tokens = [t for t in tokens if t != 0]
        all_tokens.extend(non_pad_tokens)
    
    # 統計
    token_counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(token_counter)
    
    print(f"總 token 數: {total_tokens:,}")
    print(f"唯一 token 數: {unique_tokens:,} / 4096 ({unique_tokens/4096*100:.1f}%)")
    
    # Top-20 tokens
    top_20 = []
    for token_id, count in token_counter.most_common(20):
        percentage = count / total_tokens * 100
        top_20.append((token_id, count, percentage))
    
    print(f"\nTop-20 最常見 Tokens:")
    print(f"{'Rank':<6} {'Token':<8} {'Count':<12} {'Percentage':<10}")
    print("-" * 40)
    for rank, (token_id, count, pct) in enumerate(top_20, 1):
        print(f"{rank:<6} {token_id:<8} {count:<12,} {pct:<10.2f}%")
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'token_counter': token_counter,
        'top_20': top_20
    }


def compare_distributions(train_stats, val_stats, threshold=0.3):
    """
    比較 Train 和 Val 的 token 分布差異
    
    Args:
        train_stats: analyze_token_distribution 的返回值 (train set)
        val_stats: analyze_token_distribution 的返回值 (val set)
        threshold: 差異閾值 (百分點)，超過此值視為顯著差異
    
    Returns:
        list: 有顯著差異的 tokens [(token_id, train_pct, val_pct, diff), ...]
    """
    print(f"\n{'='*70}")
    print(f"Token 分布比較 (Train vs Val)")
    print(f"{'='*70}\n")
    
    train_counter = train_stats['token_counter']
    val_counter = val_stats['token_counter']
    train_total = train_stats['total_tokens']
    val_total = val_stats['total_tokens']
    
    # 找出所有出現過的 tokens
    all_tokens = set(train_counter.keys()) | set(val_counter.keys())
    
    # 計算每個 token 的分布差異
    token_diffs = []
    for token_id in all_tokens:
        train_pct = train_counter.get(token_id, 0) / train_total * 100
        val_pct = val_counter.get(token_id, 0) / val_total * 100
        diff = val_pct - train_pct
        abs_diff = abs(diff)
        
        # 計算該 token 在 Train+Val 的平均重要性
        avg_pct = (train_pct + val_pct) / 2
        
        token_diffs.append({
            'token': token_id,
            'train_pct': train_pct,
            'val_pct': val_pct,
            'diff': diff,
            'abs_diff': abs_diff,
            'avg_pct': avg_pct
        })
    
    # 按平均重要性排序 (找出 Top-20)
    token_diffs.sort(key=lambda x: x['avg_pct'], reverse=True)
    top_20 = token_diffs[:20]
    
    print(f"Top-20 Tokens 的分布比較:\n")
    print(f"{'Rank':<6} {'Token':<8} {'Train %':<10} {'Val %':<10} {'Diff':<12} {'Abs Diff':<10}")
    print("=" * 70)
    
    mismatch_tokens = []
    for rank, item in enumerate(top_20, 1):
        diff_str = f"{item['diff']:+.2f}"
        print(f"{rank:<6} {item['token']:<8} {item['train_pct']:<10.2f} {item['val_pct']:<10.2f} {diff_str:<12} {item['abs_diff']:<10.2f}")
        
        if item['abs_diff'] > threshold:
            mismatch_tokens.append(item)
    
    # 統計 mismatch
    total_abs_diff = sum(item['abs_diff'] for item in mismatch_tokens)
    
    print(f"\n{'='*70}")
    print(f"發現 {len(mismatch_tokens)} 個 Top-20 tokens 有顯著分布差異 (>{threshold}%)")
    print(f"累計絕對差異: {total_abs_diff:.2f}%")
    print(f"{'='*70}")
    
    if mismatch_tokens:
        print(f"\n這些 Tokens 的詳細差異:")
        for item in mismatch_tokens[:10]:
            direction = "↑" if item['diff'] > 0 else "↓"
            print(f"  Token {item['token']:4d}: Train {item['train_pct']:5.2f}% → Val {item['val_pct']:5.2f}% ({direction} {item['abs_diff']:.2f}%)")
    
    return mismatch_tokens


def main():
    """主函式：執行完整的 token distribution 分析"""
    
    # 分析訓練集
    train_stats = analyze_token_distribution(
        './data/train_cache.pt',
        'Train Set'
    )
    
    # 分析驗證集
    val_stats = analyze_token_distribution(
        './data/val_cache.pt',
        'Val Set'
    )
    
    # 比較分布
    mismatch_tokens = compare_distributions(train_stats, val_stats, threshold=0.3)
    
    # Token 453 專門分析
    train_453_pct = train_stats['token_counter'][453] / train_stats['total_tokens'] * 100
    val_453_pct = val_stats['token_counter'][453] / val_stats['total_tokens'] * 100
    
    print(f"\n{'='*70}")
    print(f"Token 453 深度分析")
    print(f"{'='*70}\n")
    print(f"Train Set: {train_453_pct:.2f}%")
    print(f"Val Set:   {val_453_pct:.2f}%")
    print(f"差異:      {val_453_pct - train_453_pct:+.2f}% (絕對)")
    print(f"相對增幅:  {(val_453_pct / train_453_pct - 1) * 100:+.1f}%")
    
    # 計算 Token 453 對錯誤的最大貢獻
    # 假設 Train Acc = 54.7%, Val Acc = 36.75%
    train_error_rate = 1 - 0.547
    val_error_rate = 1 - 0.3675
    
    train_453_max_contribution = (train_453_pct / 100) / train_error_rate * 100
    val_453_max_contribution = (val_453_pct / 100) / val_error_rate * 100
    
    print(f"\nToken 453 對錯誤的最大貢獻 (假設完全預測錯誤):")
    print(f"  Train: {train_453_max_contribution:.1f}% of total errors")
    print(f"  Val:   {val_453_max_contribution:.1f}% of total errors")
    
    print(f"\n{'='*70}")
    print("分析完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
