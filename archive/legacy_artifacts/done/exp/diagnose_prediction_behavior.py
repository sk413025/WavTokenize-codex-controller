"""
診斷關鍵問題：模型預測行為分析

重點檢查：
1. 模型是否只是在"預測眾數"？
2. Speaker embedding 是否真的有影響？
3. 預測信心度如何？
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader

from model_zeroshot import ZeroShotDenoisingTransformer
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def analyze_prediction_distribution(model, dataloader, device, dataset_name="Train"):
    """
    分析模型預測的 token 分布
    """
    print(f"\n{'='*80}")
    print(f"預測分布分析: {dataset_name}")
    print(f"{'='*80}")
    
    model.eval()
    
    all_predictions = []
    all_clean_tokens = []
    all_noisy_tokens = []
    all_max_probs = []
    all_entropy = []
    
    with torch.no_grad():
        for batch in dataloader:
            noisy_tokens = batch['noisy_tokens'].to(device)
            clean_tokens = batch['clean_tokens'].to(device)
            speaker_embeddings = batch['speaker_embeddings'].to(device)
            
            logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
            predictions = logits.argmax(dim=-1)
            
            # 立即計算 probs 統計，不保存整個 logits
            B, T, V = logits.shape
            probs = torch.softmax(logits.reshape(B * T, V), dim=-1)
            max_probs, _ = probs.max(dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
            all_predictions.append(predictions.cpu().flatten())
            all_clean_tokens.append(clean_tokens.cpu().flatten())
            all_noisy_tokens.append(noisy_tokens.cpu().flatten())
            all_max_probs.append(max_probs.cpu())
            all_entropy.append(entropy.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_clean_tokens = torch.cat(all_clean_tokens)
    all_noisy_tokens = torch.cat(all_noisy_tokens)
    all_max_probs = torch.cat(all_max_probs)
    all_entropy = torch.cat(all_entropy)
    
    # 1. Token 分布統計
    pred_counter = Counter(all_predictions.tolist())
    true_counter = Counter(all_clean_tokens.tolist())
    noisy_counter = Counter(all_noisy_tokens.tolist())
    
    print(f"\nTop-20 Predicted Tokens:")
    print(f"{'Token ID':>10} {'Count':>12} {'Percentage':>12} {'True %':>12} {'Noisy %':>12}")
    print("-" * 65)
    
    total_preds = len(all_predictions)
    for token_id, count in pred_counter.most_common(20):
        pred_pct = (count / total_preds) * 100
        true_pct = (true_counter[token_id] / total_preds) * 100
        noisy_pct = (noisy_counter[token_id] / total_preds) * 100
        print(f"{token_id:>10} {count:>12} {pred_pct:>11.2f}% {true_pct:>11.2f}% {noisy_pct:>11.2f}%")
    
    # 2. 準確率
    accuracy = (all_predictions == all_clean_tokens).float().mean().item() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # 3. Token 0 和 Token 453 的特殊分析
    print(f"\n特殊 Token 分析:")
    for special_token in [0, 453]:
        pred_count = pred_counter[special_token]
        true_count = true_counter[special_token]
        noisy_count = noisy_counter[special_token]
        
        # Token accuracy (在 true 中是這個 token 的情況下，預測對的比例)
        mask = (all_clean_tokens == special_token)
        if mask.sum() > 0:
            token_acc = (all_predictions[mask] == special_token).float().mean().item() * 100
        else:
            token_acc = 0.0
        
        print(f"\n  Token {special_token}:")
        print(f"    - Predicted: {pred_count:>7} ({pred_count/total_preds*100:>6.2f}%)")
        print(f"    - True:      {true_count:>7} ({true_count/total_preds*100:>6.2f}%)")
        print(f"    - Noisy:     {noisy_count:>7} ({noisy_count/total_preds*100:>6.2f}%)")
        print(f"    - Token Accuracy: {token_acc:>6.2f}%")
    
    # 4. Logits 信心度分析
    print(f"\n預測信心度:")
    print(f"  - 平均最大機率: {all_max_probs.mean().item():.4f}")
    print(f"  - 最大機率中位數: {all_max_probs.median().item():.4f}")
    print(f"  - 高信心預測 (p>0.9): {(all_max_probs > 0.9).float().mean().item()*100:.2f}%")
    print(f"  - 低信心預測 (p<0.5): {(all_max_probs < 0.5).float().mean().item()*100:.2f}%")
    print(f"  - 平均熵: {all_entropy.mean().item():.4f} (理論最大: {np.log(4096):.4f})")
    
    return {
        'predictions': all_predictions,
        'clean_tokens': all_clean_tokens,
        'noisy_tokens': all_noisy_tokens,
        'accuracy': accuracy,
        'pred_counter': pred_counter,
        'true_counter': true_counter,
        'max_probs': all_max_probs,
        'entropy': all_entropy
    }


def analyze_speaker_influence(model, dataloader, device):
    """
    分析 Speaker Embedding 的實際影響
    """
    print(f"\n{'='*80}")
    print(f"Speaker Embedding 影響力分析")
    print(f"{'='*80}")
    
    model.eval()
    
    batch = next(iter(dataloader))
    noisy_tokens = batch['noisy_tokens'].to(device)
    clean_tokens = batch['clean_tokens'].to(device)
    speaker_embeddings = batch['speaker_embeddings'].to(device)
    
    with torch.no_grad():
        # 1. 正常預測
        logits_normal = model(noisy_tokens, speaker_embeddings, return_logits=True)
        pred_normal = logits_normal.argmax(dim=-1)
        
        # 2. Zero speaker embedding
        zero_speaker = torch.zeros_like(speaker_embeddings)
        logits_zero = model(noisy_tokens, zero_speaker, return_logits=True)
        pred_zero = logits_zero.argmax(dim=-1)
        
        # 3. Random speaker embedding
        random_speaker = torch.randn_like(speaker_embeddings)
        logits_random = model(noisy_tokens, random_speaker, return_logits=True)
        pred_random = logits_random.argmax(dim=-1)
        
        # 4. Swapped speaker embeddings
        B = speaker_embeddings.shape[0]
        if B > 1:
            swapped_speaker = speaker_embeddings.roll(shifts=1, dims=0)
            logits_swapped = model(noisy_tokens, swapped_speaker, return_logits=True)
            pred_swapped = logits_swapped.argmax(dim=-1)
        else:
            pred_swapped = pred_normal
    
    # 統計差異
    total_tokens = noisy_tokens.numel()
    
    diff_zero = (pred_normal != pred_zero).sum().item()
    diff_random = (pred_normal != pred_random).sum().item()
    diff_swapped = (pred_normal != pred_swapped).sum().item()
    
    diff_zero_pct = (diff_zero / total_tokens) * 100
    diff_random_pct = (diff_random / total_tokens) * 100
    diff_swapped_pct = (diff_swapped / total_tokens) * 100
    
    print(f"\nSpeaker Embedding 變化對預測的影響:")
    print(f"  - Zero speaker:    {diff_zero:>6} / {total_tokens} tokens 改變 ({diff_zero_pct:>6.2f}%)")
    print(f"  - Random speaker:  {diff_random:>6} / {total_tokens} tokens 改變 ({diff_random_pct:>6.2f}%)")
    print(f"  - Swapped speaker: {diff_swapped:>6} / {total_tokens} tokens 改變 ({diff_swapped_pct:>6.2f}%)")
    
    print(f"\n🔍 診斷結果:")
    if diff_zero_pct < 1.0:
        print(f"  ⚠️  Speaker embedding 幾乎無影響 (<1% tokens 改變)")
        print(f"      → 模型可能完全忽略了 speaker information")
    elif diff_zero_pct < 5.0:
        print(f"  ⚠️  Speaker embedding 影響很小 (<5% tokens 改變)")
        print(f"      → Speaker conditioning 太弱")
    elif diff_zero_pct < 20.0:
        print(f"  ⚠️  Speaker embedding 影響中等 (<20% tokens 改變)")
        print(f"      → Speaker conditioning 有作用但不強")
    else:
        print(f"  ✅ Speaker embedding 有顯著影響 ({diff_zero_pct:.2f}% tokens 改變)")
    
    return {
        'diff_zero_pct': diff_zero_pct,
        'diff_random_pct': diff_random_pct,
        'diff_swapped_pct': diff_swapped_pct
    }


def analyze_per_token_accuracy(predictions, clean_tokens, top_k=20):
    """
    分析每個 token 的準確率
    """
    print(f"\n{'='*80}")
    print(f"Per-Token 準確率分析")
    print(f"{'='*80}")
    
    # 計算每個 token 的準確率
    token_stats = {}
    for token_id in range(4096):
        mask = (clean_tokens == token_id)
        count = mask.sum().item()
        if count > 0:
            correct = (predictions[mask] == token_id).sum().item()
            accuracy = (correct / count) * 100
            token_stats[token_id] = {
                'count': count,
                'correct': correct,
                'accuracy': accuracy
            }
    
    # 按出現次數排序
    sorted_by_count = sorted(token_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print(f"\nTop-{top_k} 最常見 Token 的準確率:")
    print(f"{'Token ID':>10} {'Count':>10} {'Correct':>10} {'Accuracy':>12}")
    print("-" * 50)
    
    for token_id, stats in sorted_by_count[:top_k]:
        print(f"{token_id:>10} {stats['count']:>10} {stats['correct']:>10} {stats['accuracy']:>11.2f}%")
    
    # 按準確率排序（最差的）
    sorted_by_acc = sorted(
        [(tid, s) for tid, s in token_stats.items() if s['count'] >= 100],
        key=lambda x: x[1]['accuracy']
    )
    
    print(f"\nTop-{top_k} 準確率最差的 Token (count >= 100):")
    print(f"{'Token ID':>10} {'Count':>10} {'Correct':>10} {'Accuracy':>12}")
    print("-" * 50)
    
    for token_id, stats in sorted_by_acc[:top_k]:
        print(f"{token_id:>10} {stats['count']:>10} {stats['correct']:>10} {stats['accuracy']:>11.2f}%")
    
    return token_stats


def main():
    """主診斷流程"""
    print("="*80)
    print("關鍵診斷：預測行為分析")
    print("="*80)
    
    # 設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 載入數據
    print("\n載入數據...")
    train_dataset = ZeroShotAudioDatasetCached(Path('data/train_cache.pt'))
    val_dataset = ZeroShotAudioDatasetCached(Path('data/val_cache.pt'))
    
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=False,
        collate_fn=cached_collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        collate_fn=cached_collate_fn, num_workers=0
    )
    
    # 載入模型
    print("\n載入模型...")
    checkpoint_path = 'results/zeroshot_100epochs_20251105_002300/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    codebook_tensor = checkpoint['model_state_dict']['codebook']
    model = ZeroShotDenoisingTransformer(
        codebook=codebook_tensor,
        speaker_embed_dim=256,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ 模型載入完成 (Epoch {checkpoint.get('epoch', 'unknown')})")
    
    # 診斷 1: 預測分布分析 (Train)
    train_results = analyze_prediction_distribution(model, train_loader, device, "Train")
    
    # 診斷 2: 預測分布分析 (Val)
    val_results = analyze_prediction_distribution(model, val_loader, device, "Val")
    
    # 診斷 3: Speaker Embedding 影響
    speaker_results = analyze_speaker_influence(model, train_loader, device)
    
    # 診斷 4: Per-Token 準確率
    print(f"\nTrain Set:")
    train_token_stats = analyze_per_token_accuracy(
        train_results['predictions'],
        train_results['clean_tokens'],
        top_k=20
    )
    
    print(f"\nVal Set:")
    val_token_stats = analyze_per_token_accuracy(
        val_results['predictions'],
        val_results['clean_tokens'],
        top_k=20
    )
    
    # 總結
    print(f"\n{'='*80}")
    print(f"診斷總結")
    print(f"{'='*80}")
    
    print(f"\n1. 預測行為:")
    train_top1 = train_results['pred_counter'].most_common(1)[0]
    val_top1 = val_results['pred_counter'].most_common(1)[0]
    print(f"   Train: 最常預測 Token {train_top1[0]} ({train_top1[1]/len(train_results['predictions'])*100:.2f}%)")
    print(f"   Val:   最常預測 Token {val_top1[0]} ({val_top1[1]/len(val_results['predictions'])*100:.2f}%)")
    
    print(f"\n2. Speaker Embedding 影響:")
    print(f"   改變 speaker → {speaker_results['diff_zero_pct']:.2f}% tokens 改變")
    
    print(f"\n3. 預測信心:")
    print(f"   Train: 平均最大機率 {train_results['max_probs'].mean().item():.4f}")
    print(f"   Val:   平均最大機率 {val_results['max_probs'].mean().item():.4f}")
    
    print("\n診斷完成！")


if __name__ == '__main__':
    main()
