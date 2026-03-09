"""
Cross-Attention Weights 分析

目的: 驗證假設 1 - Cross-Attention 是否有效利用 Speaker Information

分析內容:
1. Attention weights 的分布和變化
2. 不同 tokens 的 attention weights variance
3. Speaker Influence 測試 (zero/random speaker)
4. Attention weights 隨訓練的演進

參考: Commit 9f544600d176324b7bd296992043edbad5537ece
實驗: EXP-20251105-CrossAttn (Commit 2d699aa)
日期: 2025-11-05
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================

RESULT_DIR = Path("results/crossattn_100epochs_20251105_025951")
OUTPUT_DIR = Path("analysis_outputs/crossattn_weights")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 要分析的 epochs
EPOCHS_TO_ANALYZE = [10, 20, 30, 40, 50]

# ==================== 工具函數 ====================

def load_model_and_data(epoch):
    """
    載入指定 epoch 的模型和數據
    """
    from torch.utils.data import DataLoader
    from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn
    from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn
    
    print(f"\n{'='*60}")
    print(f"載入 Epoch {epoch} 模型...")
    print(f"{'='*60}")
    
    # 載入模型
    checkpoint_path = RESULT_DIR / f"checkpoint_epoch_{epoch}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    
    # 建立模型
    model = ZeroShotDenoisingTransformerCrossAttn(
        codebook_size=4096,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        speaker_dim=256
    ).cuda()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型載入成功 (Epoch {checkpoint['epoch']})")
    
    # 載入數據
    val_cache_path = Path("data/val_cache.pt")
    val_dataset = ZeroShotAudioDatasetCached(str(val_cache_path))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=cached_collate_fn,
        pin_memory=True
    )
    
    return model, val_loader


def extract_attention_weights(model, dataloader, num_samples=100):
    """
    提取 attention weights
    
    Args:
        model: 模型
        dataloader: 數據 loader
        num_samples: 提取的樣本數
        
    Returns:
        attention_stats: Dict with attention weights statistics
    """
    print(f"\n提取 attention weights (前 {num_samples} 個樣本)...")
    
    all_attn_weights = []
    all_predictions = []
    all_clean_tokens = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader, desc="Extracting attention"):
            if sample_count >= num_samples:
                break
            
            noisy_tokens = batch['noisy_tokens'].cuda()  # (B, T)
            clean_tokens = batch['clean_tokens'].cuda()  # (B, T)
            speaker_emb = batch['speaker_emb'].cuda()    # (B, 256)
            
            # 前向傳播，返回 attention weights
            logits, attn_weights = model(
                noisy_tokens, 
                speaker_emb, 
                return_attention=True
            )  # logits: (B, T, 4096), attn_weights: (B, T, 1)
            
            predictions = logits.argmax(dim=-1)  # (B, T)
            
            # 收集數據
            all_attn_weights.append(attn_weights.cpu())
            all_predictions.append(predictions.cpu())
            all_clean_tokens.append(clean_tokens.cpu())
            
            sample_count += noisy_tokens.size(0)
    
    # 合併
    all_attn_weights = torch.cat(all_attn_weights, dim=0)  # (N, T, 1)
    all_predictions = torch.cat(all_predictions, dim=0)    # (N, T)
    all_clean_tokens = torch.cat(all_clean_tokens, dim=0)  # (N, T)
    
    print(f"✓ 提取完成")
    print(f"  樣本數: {all_attn_weights.size(0)}")
    print(f"  序列長度: {all_attn_weights.size(1)}")
    
    # 計算統計量
    attn_weights_flat = all_attn_weights.squeeze(-1)  # (N, T)
    
    attention_stats = {
        'weights': all_attn_weights[:100].numpy(),  # 只保存前 100 個
        'mean': attn_weights_flat.mean().item(),
        'std': attn_weights_flat.std().item(),
        'min': attn_weights_flat.min().item(),
        'max': attn_weights_flat.max().item(),
        'variance_across_tokens': attn_weights_flat.var(dim=1).mean().item(),  # 每個樣本內的 variance
        'accuracy': (all_predictions == all_clean_tokens).float().mean().item(),
    }
    
    print(f"\n統計量:")
    print(f"  Mean: {attention_stats['mean']:.6f}")
    print(f"  Std:  {attention_stats['std']:.6f}")
    print(f"  Min:  {attention_stats['min']:.6f}")
    print(f"  Max:  {attention_stats['max']:.6f}")
    print(f"  Variance across tokens: {attention_stats['variance_across_tokens']:.6f}")
    
    return attention_stats, all_attn_weights, all_predictions, all_clean_tokens


def test_speaker_influence(model, dataloader, num_samples=50):
    """
    測試 Speaker Influence
    
    測試 3 種情況:
    1. Normal speaker
    2. Zero speaker (全零向量)
    3. Random speaker (隨機向量)
    
    Args:
        model: 模型
        dataloader: 數據 loader
        num_samples: 測試樣本數
        
    Returns:
        influence_stats: Dict with speaker influence statistics
    """
    print(f"\n測試 Speaker Influence (前 {num_samples} 個樣本)...")
    
    normal_predictions = []
    zero_predictions = []
    random_predictions = []
    clean_tokens_list = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader, desc="Testing speaker influence"):
            if sample_count >= num_samples:
                break
            
            noisy_tokens = batch['noisy_tokens'].cuda()  # (B, T)
            clean_tokens = batch['clean_tokens'].cuda()  # (B, T)
            speaker_emb = batch['speaker_emb'].cuda()    # (B, 256)
            
            # 1. Normal speaker
            logits_normal = model(noisy_tokens, speaker_emb)
            pred_normal = logits_normal.argmax(dim=-1)
            
            # 2. Zero speaker
            zero_speaker = torch.zeros_like(speaker_emb)
            logits_zero = model(noisy_tokens, zero_speaker)
            pred_zero = logits_zero.argmax(dim=-1)
            
            # 3. Random speaker
            random_speaker = torch.randn_like(speaker_emb)
            logits_random = model(noisy_tokens, random_speaker)
            pred_random = logits_random.argmax(dim=-1)
            
            # 收集
            normal_predictions.append(pred_normal.cpu())
            zero_predictions.append(pred_zero.cpu())
            random_predictions.append(pred_random.cpu())
            clean_tokens_list.append(clean_tokens.cpu())
            
            sample_count += noisy_tokens.size(0)
    
    # 合併
    normal_predictions = torch.cat(normal_predictions, dim=0)
    zero_predictions = torch.cat(zero_predictions, dim=0)
    random_predictions = torch.cat(random_predictions, dim=0)
    clean_tokens = torch.cat(clean_tokens_list, dim=0)
    
    # 計算影響力
    total_tokens = normal_predictions.numel()
    
    # Normal vs Zero
    changed_zero = (normal_predictions != zero_predictions).sum().item()
    influence_zero = changed_zero / total_tokens
    
    # Normal vs Random
    changed_random = (normal_predictions != random_predictions).sum().item()
    influence_random = changed_random / total_tokens
    
    # Accuracy
    acc_normal = (normal_predictions == clean_tokens).float().mean().item()
    acc_zero = (zero_predictions == clean_tokens).float().mean().item()
    acc_random = (random_predictions == clean_tokens).float().mean().item()
    
    influence_stats = {
        'influence_zero_speaker': influence_zero,
        'influence_random_speaker': influence_random,
        'acc_normal': acc_normal,
        'acc_zero': acc_zero,
        'acc_random': acc_random,
        'acc_drop_zero': acc_normal - acc_zero,
        'acc_drop_random': acc_normal - acc_random,
    }
    
    print(f"\n✓ Speaker Influence 測試完成")
    print(f"  Normal speaker accuracy:  {acc_normal:.4f}")
    print(f"  Zero speaker accuracy:    {acc_zero:.4f} (Δ = {acc_normal - acc_zero:+.4f})")
    print(f"  Random speaker accuracy:  {acc_random:.4f} (Δ = {acc_normal - acc_random:+.4f})")
    print(f"  \n  Influence (prediction change):")
    print(f"    Normal vs Zero:   {influence_zero:.2%} tokens changed")
    print(f"    Normal vs Random: {influence_random:.2%} tokens changed")
    
    # 判定
    if influence_zero < 0.10:
        print(f"\n  ⚠️  Speaker influence 很弱 (<10%) - 假設 1 可能成立")
    elif influence_zero > 0.20:
        print(f"\n  ✅ Speaker influence 顯著 (>20%) - 假設 1 不成立")
    else:
        print(f"\n  ⚠️  Speaker influence 中等 (10-20%) - 需進一步分析")
    
    return influence_stats


# ==================== 主要分析函數 ====================

def analyze_all_epochs():
    """
    分析所有 epochs 的 attention weights
    
    Returns:
        results: Dict with analysis results
    """
    results = {
        'epochs': EPOCHS_TO_ANALYZE,
        'attention_stats': {},
        'speaker_influence': {},
    }
    
    for epoch in EPOCHS_TO_ANALYZE:
        print(f"\n{'#'*60}")
        print(f"# 分析 Epoch {epoch}")
        print(f"{'#'*60}")
        
        # 載入模型
        model, val_loader = load_model_and_data(epoch)
        
        # 提取 attention weights
        attn_stats, attn_weights, predictions, clean_tokens = extract_attention_weights(
            model, val_loader, num_samples=100
        )
        
        # 測試 speaker influence
        influence_stats = test_speaker_influence(
            model, val_loader, num_samples=50
        )
        
        # 保存結果
        results['attention_stats'][epoch] = attn_stats
        results['speaker_influence'][epoch] = influence_stats
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} 完成")
        print(f"{'='*60}")
    
    # 保存原始數據
    save_path = OUTPUT_DIR / "crossattn_weights_analysis.pt"
    torch.save(results, save_path)
    print(f"\n✓ 原始數據已保存: {save_path}")
    
    # 保存 JSON (不含 weights 數組)
    json_results = {
        'epochs': results['epochs'],
        'attention_stats': {
            str(e): {k: v for k, v in stats.items() if k != 'weights'}
            for e, stats in results['attention_stats'].items()
        },
        'speaker_influence': {
            str(e): stats
            for e, stats in results['speaker_influence'].items()
        },
    }
    
    json_path = OUTPUT_DIR / "crossattn_weights_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ JSON 數據已保存: {json_path}")
    
    return results


# ==================== 視覺化 ====================

def plot_attention_stats(results):
    """
    繪製 attention weights 統計量隨 epoch 變化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = results['epochs']
    
    # Mean
    ax = axes[0, 0]
    means = [results['attention_stats'][e]['mean'] for e in epochs]
    ax.plot(epochs, means, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean Attention Weight', fontsize=11)
    ax.set_title('Attention Weight Mean', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Expected (1 key)')
    ax.legend()
    
    # Std
    ax = axes[0, 1]
    stds = [results['attention_stats'][e]['std'] for e in epochs]
    ax.plot(epochs, stds, 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Std of Attention Weight', fontsize=11)
    ax.set_title('Attention Weight Std', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Variance across tokens
    ax = axes[1, 0]
    variances = [results['attention_stats'][e]['variance_across_tokens'] for e in epochs]
    ax.plot(epochs, variances, '^-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Variance (within sample)', fontsize=11)
    ax.set_title('Attention Weight Variance Across Tokens', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='No variance (all same)')
    ax.legend()
    
    # Range (max - min)
    ax = axes[1, 1]
    ranges = [results['attention_stats'][e]['max'] - results['attention_stats'][e]['min'] for e in epochs]
    ax.plot(epochs, ranges, 'd-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Range (Max - Min)', fontsize=11)
    ax.set_title('Attention Weight Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "attention_stats_evolution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 圖表已保存: {save_path}")
    plt.close()


def plot_speaker_influence(results):
    """
    繪製 speaker influence 隨 epoch 變化
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = results['epochs']
    
    # Influence (prediction change)
    ax = axes[0]
    influence_zero = [results['speaker_influence'][e]['influence_zero_speaker'] * 100 for e in epochs]
    influence_random = [results['speaker_influence'][e]['influence_random_speaker'] * 100 for e in epochs]
    
    ax.plot(epochs, influence_zero, 'o-', label='Zero Speaker', linewidth=2, markersize=8)
    ax.plot(epochs, influence_random, 's-', label='Random Speaker', linewidth=2, markersize=8)
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Target: 20%')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Weak: 10%')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Speaker Influence (%)', fontsize=12)
    ax.set_title('Speaker Influence (Prediction Change Rate)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Accuracy drop
    ax = axes[1]
    acc_drop_zero = [results['speaker_influence'][e]['acc_drop_zero'] * 100 for e in epochs]
    acc_drop_random = [results['speaker_influence'][e]['acc_drop_random'] * 100 for e in epochs]
    
    ax.plot(epochs, acc_drop_zero, 'o-', label='Zero Speaker', linewidth=2, markersize=8)
    ax.plot(epochs, acc_drop_random, 's-', label='Random Speaker', linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Accuracy Drop from Removing Speaker Info', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "speaker_influence_evolution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 圖表已保存: {save_path}")
    plt.close()


def plot_attention_distribution(results, epoch=50):
    """
    繪製特定 epoch 的 attention weight 分布
    """
    attn_weights = results['attention_stats'][epoch]['weights']  # (100, T, 1)
    attn_weights_flat = attn_weights.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(attn_weights_flat, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Expected (1.0)')
    ax.set_xlabel('Attention Weight', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Attention Weight Distribution (Epoch {epoch})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Box plot (per sample)
    ax = axes[1]
    attn_per_sample = attn_weights.squeeze(-1)  # (100, T)
    ax.boxplot([attn_per_sample[i] for i in range(min(20, len(attn_per_sample)))], 
               showfliers=False)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Expected (1.0)')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(f'Attention Weight Distribution per Sample (Epoch {epoch}, first 20)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"attention_distribution_epoch{epoch}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 圖表已保存: {save_path}")
    plt.close()


# ==================== 主程式 ====================

def main():
    """
    主函數：執行完整的 Cross-Attention weights 分析
    
    驗證假設: 假設 1 - Cross-Attention 是否有效利用 Speaker Information
    
    判定標準:
    - Attention weights variance < 0.01 → weights 幾乎相同 → 假設成立
    - Speaker influence < 10% → speaker 影響力弱 → 假設成立
    - Attention weights mean ≈ 1.0 ± 0.05 → 退化成簡單平均 → 假設成立
    """
    print(f"\n{'#'*70}")
    print(f"# Cross-Attention Weights 分析")
    print(f"# 目的: 驗證假設 1 - Cross-Attention 是否有效利用 Speaker Information")
    print(f"{'#'*70}")
    
    # 步驟 1: 分析所有 epochs
    print(f"\n{'='*70}")
    print(f"步驟 1: 分析所有 epochs 的 attention weights")
    print(f"{'='*70}")
    results = analyze_all_epochs()
    
    # 步驟 2: 視覺化
    print(f"\n{'='*70}")
    print(f"步驟 2: 繪製分析圖表")
    print(f"{'='*70}")
    plot_attention_stats(results)
    plot_speaker_influence(results)
    plot_attention_distribution(results, epoch=50)
    
    # 最終結論
    print(f"\n{'#'*70}")
    print(f"# 分析完成")
    print(f"{'#'*70}")
    print(f"\n輸出檔案:")
    print(f"  1. {OUTPUT_DIR}/crossattn_weights_analysis.pt")
    print(f"  2. {OUTPUT_DIR}/crossattn_weights_analysis.json")
    print(f"  3. {OUTPUT_DIR}/attention_stats_evolution.png")
    print(f"  4. {OUTPUT_DIR}/speaker_influence_evolution.png")
    print(f"  5. {OUTPUT_DIR}/attention_distribution_epoch50.png")
    
    # 判定假設
    epoch_50_influence = results['speaker_influence'][50]['influence_zero_speaker']
    epoch_50_variance = results['attention_stats'][50]['variance_across_tokens']
    epoch_50_mean = results['attention_stats'][50]['mean']
    
    print(f"\n假設驗證 (基於 Epoch 50):")
    print(f"  Attention weights mean: {epoch_50_mean:.6f}")
    print(f"  Attention weights variance: {epoch_50_variance:.6f}")
    print(f"  Speaker influence (zero): {epoch_50_influence:.2%}")
    
    hypothesis_1_holds = False
    reasons = []
    
    if abs(epoch_50_mean - 1.0) < 0.05:
        reasons.append("✅ Attention weights ≈ 1.0 (退化成簡單平均)")
        hypothesis_1_holds = True
    
    if epoch_50_variance < 0.01:
        reasons.append("✅ Variance < 0.01 (所有 token 權重幾乎相同)")
        hypothesis_1_holds = True
    
    if epoch_50_influence < 0.10:
        reasons.append("✅ Speaker influence < 10% (speaker 影響力很弱)")
        hypothesis_1_holds = True
    
    if hypothesis_1_holds:
        print(f"\n結論: 假設 1 **成立** ⚠️")
        print(f"  Cross-Attention 未能有效利用 Speaker Information")
        for reason in reasons:
            print(f"    {reason}")
    else:
        print(f"\n結論: 假設 1 **不成立** ✅")
        print(f"  Cross-Attention 有效利用了 Speaker Information")


if __name__ == "__main__":
    main()
