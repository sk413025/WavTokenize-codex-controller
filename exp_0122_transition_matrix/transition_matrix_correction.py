#!/usr/bin/env python3
"""
Transition Matrix Token Correction

方向 A: 使用統計方法建立 transition matrix，不需要訓練神經網路

原理:
1. 從 train_cache 建立 P[token_clean | token_noisy] 轉換矩陣
2. 推理時對每個 noisy token 查表取 argmax 得到 corrected token
3. 評估修正後的 token accuracy 和音質 (PESQ/STOI)
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import argparse


def build_transition_matrix(cache_path: str, vocab_size: int = 4096):
    """
    從 cache 建立 transition matrix

    Returns:
        transition_matrix: (vocab_size, vocab_size)
            transition_matrix[noisy_token, clean_token] = count
        transition_prob: (vocab_size, vocab_size)
            transition_prob[noisy_token, clean_token] = probability
    """
    cache = torch.load(cache_path, weights_only=False)
    print(f"Loaded {len(cache)} samples from {cache_path}")

    # 建立計數矩陣
    transition_count = np.zeros((vocab_size, vocab_size), dtype=np.int64)

    total_tokens = 0
    for item in cache:
        noisy_tokens = item['noisy_tokens']
        clean_tokens = item['clean_tokens']

        # 轉換為 numpy
        if isinstance(noisy_tokens, torch.Tensor):
            noisy_tokens = noisy_tokens.flatten().numpy()
        if isinstance(clean_tokens, torch.Tensor):
            clean_tokens = clean_tokens.flatten().numpy()

        # 對齊長度
        min_len = min(len(noisy_tokens), len(clean_tokens))
        noisy_tokens = noisy_tokens[:min_len]
        clean_tokens = clean_tokens[:min_len]

        # 累計計數
        for t_n, t_c in zip(noisy_tokens, clean_tokens):
            transition_count[int(t_n), int(t_c)] += 1
            total_tokens += 1

    print(f"Total tokens processed: {total_tokens}")

    # 計算機率 (row-wise normalization)
    row_sums = transition_count.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)  # 避免除以零
    transition_prob = transition_count / row_sums

    return transition_count, transition_prob


def analyze_transition_matrix(transition_count, transition_prob):
    """分析 transition matrix 的特性"""
    vocab_size = transition_count.shape[0]

    # 1. 對角線比例 (原本就正確的)
    diagonal_sum = np.trace(transition_count)
    total_sum = transition_count.sum()
    match_rate = diagonal_sum / total_sum if total_sum > 0 else 0

    # 2. 每個 noisy token 的 top-1 clean token
    top1_indices = transition_prob.argmax(axis=1)
    top1_probs = transition_prob.max(axis=1)

    # 3. 統計 top-1 是否等於自己 (不需要修正)
    no_change_count = (top1_indices == np.arange(vocab_size)).sum()

    # 4. 統計有效的 noisy token (出現過的)
    active_tokens = (transition_count.sum(axis=1) > 0).sum()

    # 5. Top-1 可修正率
    # 對於 mismatch 的 token，如果用 top-1 替換能修正多少
    mismatch_total = 0
    mismatch_correctable = 0
    for noisy_tok in range(vocab_size):
        for clean_tok in range(vocab_size):
            count = transition_count[noisy_tok, clean_tok]
            if count > 0 and noisy_tok != clean_tok:
                mismatch_total += count
                if top1_indices[noisy_tok] == clean_tok:
                    mismatch_correctable += count

    correction_rate = mismatch_correctable / mismatch_total if mismatch_total > 0 else 0

    print(f"\n{'='*60}")
    print("Transition Matrix Analysis")
    print(f"{'='*60}")
    print(f"Total tokens: {total_sum:,}")
    print(f"Match rate (diagonal): {match_rate*100:.2f}%")
    print(f"Active tokens: {active_tokens}/{vocab_size}")
    print(f"No-change tokens (top1 = self): {no_change_count}")
    print(f"Mismatch tokens: {mismatch_total:,}")
    print(f"Top-1 correctable: {mismatch_correctable:,}")
    print(f"Top-1 correction rate: {correction_rate*100:.2f}%")

    return {
        'match_rate': match_rate,
        'correction_rate': correction_rate,
        'active_tokens': active_tokens,
        'top1_indices': top1_indices,
        'top1_probs': top1_probs,
    }


def correct_tokens(noisy_tokens, top1_indices, threshold=0.0):
    """
    使用 transition matrix 修正 tokens

    Args:
        noisy_tokens: (N,) noisy token ids
        top1_indices: (vocab_size,) 每個 noisy token 的最可能 clean token
        threshold: 只有當 noisy != top1 時才修正 (threshold=0 表示全部修正)

    Returns:
        corrected_tokens: (N,) 修正後的 token ids
    """
    if isinstance(noisy_tokens, torch.Tensor):
        noisy_tokens = noisy_tokens.numpy()

    corrected = top1_indices[noisy_tokens.astype(int)]
    return corrected


def evaluate_correction(cache_path: str, top1_indices, split='val'):
    """
    評估 token 修正的效果
    """
    cache = torch.load(cache_path, weights_only=False)
    print(f"\nEvaluating on {len(cache)} samples from {split}")

    total_tokens = 0
    original_correct = 0
    corrected_correct = 0

    for item in cache:
        noisy_tokens = item['noisy_tokens']
        clean_tokens = item['clean_tokens']

        if isinstance(noisy_tokens, torch.Tensor):
            noisy_tokens = noisy_tokens.flatten().numpy()
        if isinstance(clean_tokens, torch.Tensor):
            clean_tokens = clean_tokens.flatten().numpy()

        min_len = min(len(noisy_tokens), len(clean_tokens))
        noisy_tokens = noisy_tokens[:min_len]
        clean_tokens = clean_tokens[:min_len]

        # 修正
        corrected_tokens = correct_tokens(noisy_tokens, top1_indices)

        # 統計
        total_tokens += min_len
        original_correct += (noisy_tokens == clean_tokens).sum()
        corrected_correct += (corrected_tokens == clean_tokens).sum()

    original_acc = original_correct / total_tokens * 100
    corrected_acc = corrected_correct / total_tokens * 100
    improvement = corrected_acc - original_acc

    print(f"\n{'='*60}")
    print(f"Token Correction Results ({split})")
    print(f"{'='*60}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Original accuracy: {original_acc:.3f}%")
    print(f"Corrected accuracy: {corrected_acc:.3f}%")
    print(f"Improvement: {improvement:+.3f}%")

    return {
        'original_acc': original_acc,
        'corrected_acc': corrected_acc,
        'improvement': improvement,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cache', type=str,
                        default='/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt')
    parser.add_argument('--val_cache', type=str,
                        default='/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. 從 train_cache 建立 transition matrix
    print("Building transition matrix from train_cache...")
    transition_count, transition_prob = build_transition_matrix(args.train_cache)

    # 2. 分析 transition matrix
    analysis = analyze_transition_matrix(transition_count, transition_prob)

    # 3. 在 train_cache 上評估 (應該會很好，因為是用它建的)
    print("\n" + "="*60)
    print("Evaluating on TRAIN set (sanity check)")
    train_results = evaluate_correction(args.train_cache, analysis['top1_indices'], 'train')

    # 4. 在 val_cache 上評估 (真正的測試)
    print("\n" + "="*60)
    print("Evaluating on VALIDATION set")
    val_results = evaluate_correction(args.val_cache, analysis['top1_indices'], 'val')

    # 5. 儲存結果
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj

    results = {
        'analysis': convert_to_serializable({k: v for k, v in analysis.items()
                                              if k not in ['top1_indices', 'top1_probs']}),
        'train_results': convert_to_serializable(train_results),
        'val_results': convert_to_serializable(val_results),
    }

    # 儲存 transition matrix
    np.save(output_dir / 'transition_count.npy', transition_count)
    np.save(output_dir / 'transition_prob.npy', transition_prob)
    np.save(output_dir / 'top1_indices.npy', analysis['top1_indices'])

    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # 6. 總結
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Train: {train_results['original_acc']:.3f}% → {train_results['corrected_acc']:.3f}% ({train_results['improvement']:+.3f}%)")
    print(f"Val:   {val_results['original_acc']:.3f}% → {val_results['corrected_acc']:.3f}% ({val_results['improvement']:+.3f}%)")


if __name__ == '__main__':
    main()
