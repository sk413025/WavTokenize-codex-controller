"""
Per-Token Accuracy 演進分析

目的: 驗證假設 2 - 模型是否學會「忽略困難 Tokens」

分析內容:
1. 每個 token 在不同 epoch 的準確率變化
2. Top-20 mismatch tokens (來自 Commit 9f54460) 的學習曲線
3. 簡單 tokens vs 困難 tokens 的學習速度
4. Token 453 的特別分析

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
OUTPUT_DIR = Path("analysis_outputs/pertoken_accuracy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 要分析的 epochs
EPOCHS_TO_ANALYZE = [10, 20, 30, 40, 50]

# Top-20 Mismatch Tokens (來自 Commit 9f54460)
# 這些 tokens 在 Train/Val 分布有顯著差異
MISMATCH_TOKENS = [
    453,   # +5.08% (最大差異)
    1145,  # +0.73%
    1750,  # +0.52%
    1016,  # +0.51%
    1764,  # +0.45%
    1019,  # +0.42%
    1749,  # +0.41%
    1731,  # +0.40%
    1017,  # +0.40%
    1746,  # +0.39%
    1756,  # +0.38%
    1755,  # +0.37%
    1018,  # +0.36%
    1732,  # +0.35%
    1757,  # +0.34%
]

# ==================== 工具函數 ====================

def load_model_and_data(epoch):
    """
    載入指定 epoch 的模型和數據
    
    Args:
        epoch: Epoch 編號
        
    Returns:
        model: 載入的模型
        train_loader: 訓練數據 loader
        val_loader: 驗證數據 loader
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
    
    print(f"✓ 模型載入成功")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Acc: {checkpoint.get('train_acc', 'N/A')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    
    # 載入數據
    print(f"\n載入數據...")
    train_cache_path = Path("data/train_cache.pt")
    val_cache_path = Path("data/val_cache.pt")
    
    train_dataset = ZeroShotAudioDatasetCached(str(train_cache_path))
    val_dataset = ZeroShotAudioDatasetCached(str(val_cache_path))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,  # 分析時不 shuffle
        num_workers=4,
        collate_fn=cached_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=cached_collate_fn,
        pin_memory=True
    )
    
    print(f"✓ 數據載入成功")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return model, train_loader, val_loader


def compute_pertoken_accuracy(model, dataloader, split_name, max_batches=None):
    """
    計算每個 token 的準確率
    
    Args:
        model: 模型
        dataloader: 數據 loader
        split_name: 'train' 或 'val'
        max_batches: 最大處理的 batch 數（None = 全部）
        
    Returns:
        token_accuracy: (4096,) 每個 token 的準確率
        token_count: (4096,) 每個 token 的出現次數
        overall_acc: 整體準確率
    """
    print(f"\n計算 {split_name} set 的 per-token accuracy...")
    
    token_correct = torch.zeros(4096, dtype=torch.long).cuda()
    token_total = torch.zeros(4096, dtype=torch.long).cuda()
    
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"{split_name} batches")
        for batch_idx, batch in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break
            
            noisy_tokens = batch['noisy_tokens'].cuda()  # (B, T)
            clean_tokens = batch['clean_tokens'].cuda()  # (B, T)
            speaker_emb = batch['speaker_emb'].cuda()    # (B, 256)
            
            # 前向傳播
            logits = model(noisy_tokens, speaker_emb)  # (B, T, 4096)
            predictions = logits.argmax(dim=-1)  # (B, T)
            
            # 整體準確率
            correct_mask = (predictions == clean_tokens)
            total_correct += correct_mask.sum().item()
            total_tokens += clean_tokens.numel()
            
            # Per-token 統計
            clean_flat = clean_tokens.view(-1)  # (B*T,)
            pred_flat = predictions.view(-1)    # (B*T,)
            
            for token_id in range(4096):
                mask = (clean_flat == token_id)
                if mask.sum() > 0:
                    token_total[token_id] += mask.sum()
                    token_correct[token_id] += (pred_flat[mask] == token_id).sum()
            
            # 更新進度條
            current_acc = total_correct / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix({'acc': f'{current_acc:.4f}'})
    
    # 計算準確率
    token_accuracy = torch.zeros(4096).cuda()
    valid_mask = (token_total > 0)
    token_accuracy[valid_mask] = token_correct[valid_mask].float() / token_total[valid_mask].float()
    
    overall_acc = total_correct / total_tokens if total_tokens > 0 else 0
    
    print(f"✓ {split_name} set 完成")
    print(f"  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Unique tokens: {valid_mask.sum().item()} / 4096")
    
    return token_accuracy.cpu(), token_total.cpu(), overall_acc


# ==================== 主要分析函數 ====================

def analyze_all_epochs():
    """
    分析所有 epochs 的 per-token accuracy
    
    Returns:
        results: Dict with analysis results
    """
    results = {
        'epochs': EPOCHS_TO_ANALYZE,
        'train_acc': {},
        'val_acc': {},
        'train_count': {},
        'val_count': {},
        'overall_train_acc': [],
        'overall_val_acc': [],
    }
    
    for epoch in EPOCHS_TO_ANALYZE:
        print(f"\n{'#'*60}")
        print(f"# 分析 Epoch {epoch}")
        print(f"{'#'*60}")
        
        # 載入模型和數據
        model, train_loader, val_loader = load_model_and_data(epoch)
        
        # 計算 Train set per-token accuracy
        train_acc, train_count, overall_train = compute_pertoken_accuracy(
            model, train_loader, 'train', max_batches=100  # 取樣 100 batches
        )
        
        # 計算 Val set per-token accuracy
        val_acc, val_count, overall_val = compute_pertoken_accuracy(
            model, val_loader, 'val', max_batches=None  # 全部
        )
        
        # 保存結果
        results['train_acc'][epoch] = train_acc.numpy()
        results['val_acc'][epoch] = val_acc.numpy()
        results['train_count'][epoch] = train_count.numpy()
        results['val_count'][epoch] = val_count.numpy()
        results['overall_train_acc'].append(overall_train)
        results['overall_val_acc'].append(overall_val)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} 完成")
        print(f"  Train Acc: {overall_train:.4f}")
        print(f"  Val Acc: {overall_val:.4f}")
        print(f"{'='*60}")
    
    # 保存原始數據
    save_path = OUTPUT_DIR / "pertoken_accuracy_raw.pt"
    torch.save(results, save_path)
    print(f"\n✓ 原始數據已保存: {save_path}")
    
    return results


def analyze_mismatch_tokens(results):
    """
    分析 Top-20 Mismatch Tokens 的學習曲線
    
    Args:
        results: analyze_all_epochs() 的輸出
        
    Returns:
        mismatch_analysis: Dict with mismatch token analysis
    """
    print(f"\n{'#'*60}")
    print(f"# 分析 Mismatch Tokens")
    print(f"{'#'*60}")
    
    epochs = results['epochs']
    mismatch_analysis = {
        'token_id': [],
        'epochs': epochs,
        'train_acc_evolution': [],
        'val_acc_evolution': [],
        'train_count_avg': [],
        'val_count_avg': [],
        'learning_stagnation': [],  # Epoch 30-50 的提升幅度
    }
    
    for token_id in MISMATCH_TOKENS:
        print(f"\nToken {token_id}:")
        
        train_accs = [results['train_acc'][e][token_id] for e in epochs]
        val_accs = [results['val_acc'][e][token_id] for e in epochs]
        train_counts = [results['train_count'][e][token_id] for e in epochs]
        val_counts = [results['val_count'][e][token_id] for e in epochs]
        
        # 計算學習停滯指標 (Epoch 30-50 的提升)
        idx_30 = epochs.index(30)
        idx_50 = epochs.index(50)
        
        train_stagnation = train_accs[idx_50] - train_accs[idx_30]
        val_stagnation = val_accs[idx_50] - val_accs[idx_30]
        
        mismatch_analysis['token_id'].append(token_id)
        mismatch_analysis['train_acc_evolution'].append(train_accs)
        mismatch_analysis['val_acc_evolution'].append(val_accs)
        mismatch_analysis['train_count_avg'].append(np.mean(train_counts))
        mismatch_analysis['val_count_avg'].append(np.mean(val_counts))
        mismatch_analysis['learning_stagnation'].append({
            'train': train_stagnation,
            'val': val_stagnation
        })
        
        print(f"  Train Acc: {train_accs[0]:.3f} → {train_accs[-1]:.3f} (Δ={train_accs[-1]-train_accs[0]:.3f})")
        print(f"  Val Acc:   {val_accs[0]:.3f} → {val_accs[-1]:.3f} (Δ={val_accs[-1]-val_accs[0]:.3f})")
        print(f"  Stagnation (Epoch 30-50): Train Δ={train_stagnation:.3f}, Val Δ={val_stagnation:.3f}")
        
        if abs(val_stagnation) < 0.02:
            print(f"  ⚠️  Val Accuracy 幾乎無進步 (< 2%)")
    
    # 保存分析結果
    save_path = OUTPUT_DIR / "mismatch_tokens_analysis.json"
    with open(save_path, 'w') as f:
        # Convert to JSON-serializable format
        json_data = {
            'token_id': mismatch_analysis['token_id'],
            'epochs': mismatch_analysis['epochs'],
            'train_acc_evolution': [list(map(float, acc)) for acc in mismatch_analysis['train_acc_evolution']],
            'val_acc_evolution': [list(map(float, acc)) for acc in mismatch_analysis['val_acc_evolution']],
            'train_count_avg': [float(c) for c in mismatch_analysis['train_count_avg']],
            'val_count_avg': [float(c) for c in mismatch_analysis['val_count_avg']],
            'learning_stagnation': mismatch_analysis['learning_stagnation']
        }
        json.dump(json_data, f, indent=2)
    
    print(f"\n✓ Mismatch tokens 分析已保存: {save_path}")
    
    return mismatch_analysis


def analyze_token453(results):
    """
    特別分析 Token 453 (最大 mismatch)
    
    Args:
        results: analyze_all_epochs() 的輸出
    """
    print(f"\n{'#'*60}")
    print(f"# Token 453 特別分析")
    print(f"{'#'*60}")
    
    token_id = 453
    epochs = results['epochs']
    
    print(f"\nToken 453 是 Train/Val 分布差異最大的 token (+5.08%)")
    print(f"根據 Commit 9f54460:")
    print(f"  Train: 13.57% 出現率")
    print(f"  Val:   18.65% 出現率")
    print(f"  Token 453 對錯誤的貢獻: Train 30%, Val 29.5%")
    
    print(f"\n準確率演進:")
    for epoch in epochs:
        train_acc = results['train_acc'][epoch][token_id]
        val_acc = results['val_acc'][epoch][token_id]
        train_count = results['train_count'][epoch][token_id]
        val_count = results['val_count'][epoch][token_id]
        
        print(f"  Epoch {epoch:2d}: Train {train_acc:.4f} (n={train_count:5d}), "
              f"Val {val_acc:.4f} (n={val_count:5d})")
    
    # 計算改善幅度
    train_improvement = results['train_acc'][50][token_id] - results['train_acc'][10][token_id]
    val_improvement = results['val_acc'][50][token_id] - results['val_acc'][10][token_id]
    
    print(f"\n改善幅度 (Epoch 10 → 50):")
    print(f"  Train: {train_improvement:+.4f}")
    print(f"  Val:   {val_improvement:+.4f}")
    
    if abs(val_improvement) < 0.05:
        print(f"\n⚠️  Token 453 的 Val Accuracy 幾乎無改善 (< 5%)")
        print(f"     這支持「假設 2: 模型學會忽略困難 Tokens」")


def categorize_tokens(results):
    """
    將 tokens 分類為「簡單」和「困難」
    
    Args:
        results: analyze_all_epochs() 的輸出
        
    Returns:
        categories: Dict with token categories
    """
    print(f"\n{'#'*60}")
    print(f"# Tokens 分類分析")
    print(f"{'#'*60}")
    
    epoch_50_train_acc = results['train_acc'][50]
    epoch_50_val_acc = results['val_acc'][50]
    epoch_50_val_count = results['val_count'][50]
    
    # 只考慮在 Val set 中出現 >10 次的 tokens
    valid_tokens = (epoch_50_val_count > 10)
    
    # 分類標準:
    # 簡單 tokens: Val Acc > 0.6
    # 困難 tokens: Val Acc < 0.3
    easy_tokens = valid_tokens & (epoch_50_val_acc > 0.6)
    difficult_tokens = valid_tokens & (epoch_50_val_acc < 0.3)
    
    print(f"\n分類結果 (Epoch 50, Val set, count > 10):")
    print(f"  簡單 tokens (Acc > 0.6):  {easy_tokens.sum()} tokens")
    print(f"  困難 tokens (Acc < 0.3):  {difficult_tokens.sum()} tokens")
    print(f"  中等 tokens (0.3-0.6):    {(valid_tokens & ~easy_tokens & ~difficult_tokens).sum()} tokens")
    
    # 分析學習速度
    easy_learning = []
    difficult_learning = []
    
    for epoch in results['epochs']:
        easy_avg = epoch_50_val_acc[easy_tokens].mean()
        difficult_avg = epoch_50_val_acc[difficult_tokens].mean()
        easy_learning.append(easy_avg)
        difficult_learning.append(difficult_avg)
    
    print(f"\n學習速度比較:")
    print(f"  簡單 tokens: {easy_learning[0]:.4f} → {easy_learning[-1]:.4f} (Δ={easy_learning[-1]-easy_learning[0]:.4f})")
    print(f"  困難 tokens: {difficult_learning[0]:.4f} → {difficult_learning[-1]:.4f} (Δ={difficult_learning[-1]-difficult_learning[0]:.4f})")
    
    categories = {
        'easy_tokens': torch.where(easy_tokens)[0].numpy(),
        'difficult_tokens': torch.where(difficult_tokens)[0].numpy(),
        'easy_learning': easy_learning,
        'difficult_learning': difficult_learning,
    }
    
    return categories


# ==================== 視覺化 ====================

def plot_overall_accuracy(results):
    """
    繪製整體準確率曲線
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = results['epochs']
    train_accs = results['overall_train_acc']
    val_accs = results['overall_val_acc']
    
    ax.plot(epochs, train_accs, 'o-', label='Train Accuracy', linewidth=2)
    ax.plot(epochs, val_accs, 's-', label='Val Accuracy', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Overall Accuracy Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 標註平台期
    ax.axvspan(30, 50, alpha=0.1, color='red', label='Plateau Region')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "overall_accuracy_evolution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 圖表已保存: {save_path}")
    plt.close()


def plot_mismatch_tokens(mismatch_analysis):
    """
    繪製 Mismatch Tokens 的學習曲線
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs = mismatch_analysis['epochs']
    
    # Train Accuracy
    ax = axes[0]
    for i, token_id in enumerate(mismatch_analysis['token_id'][:10]):  # 只畫前 10 個
        train_accs = mismatch_analysis['train_acc_evolution'][i]
        ax.plot(epochs, train_accs, 'o-', label=f'Token {token_id}', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Accuracy', fontsize=12)
    ax.set_title('Top-10 Mismatch Tokens - Train Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvspan(30, 50, alpha=0.1, color='red')
    
    # Val Accuracy
    ax = axes[1]
    for i, token_id in enumerate(mismatch_analysis['token_id'][:10]):
        val_accs = mismatch_analysis['val_acc_evolution'][i]
        ax.plot(epochs, val_accs, 's-', label=f'Token {token_id}', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val Accuracy', fontsize=12)
    ax.set_title('Top-10 Mismatch Tokens - Val Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvspan(30, 50, alpha=0.1, color='red')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "mismatch_tokens_evolution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 圖表已保存: {save_path}")
    plt.close()


def plot_token453(results):
    """
    繪製 Token 453 的特別分析圖
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = results['epochs']
    token_id = 453
    
    train_accs = [results['train_acc'][e][token_id] for e in epochs]
    val_accs = [results['val_acc'][e][token_id] for e in epochs]
    
    ax.plot(epochs, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=8)
    ax.plot(epochs, val_accs, 's-', label='Val Accuracy', linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Token 453 Accuracy Evolution\n(Largest Train/Val Distribution Mismatch: +5.08%)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvspan(30, 50, alpha=0.1, color='red', label='Plateau Region')
    
    # 標註數值
    for i, (e, train_acc, val_acc) in enumerate(zip(epochs, train_accs, val_accs)):
        if i % 2 == 0:  # 只標註偶數 epoch
            ax.text(e, train_acc + 0.02, f'{train_acc:.3f}', ha='center', fontsize=9)
            ax.text(e, val_acc - 0.02, f'{val_acc:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "token453_evolution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 圖表已保存: {save_path}")
    plt.close()


def plot_easy_vs_difficult(categories, results):
    """
    繪製簡單 vs 困難 tokens 的學習速度比較
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = results['epochs']
    easy_learning = categories['easy_learning']
    difficult_learning = categories['difficult_learning']
    
    ax.plot(epochs, easy_learning, 'o-', label='Easy Tokens (Acc > 0.6)', linewidth=2, markersize=8)
    ax.plot(epochs, difficult_learning, 's-', label='Difficult Tokens (Acc < 0.3)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Easy vs Difficult Tokens Learning Speed', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvspan(30, 50, alpha=0.1, color='red', label='Plateau Region')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "easy_vs_difficult_tokens.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 圖表已保存: {save_path}")
    plt.close()


# ==================== 主程式 ====================

def main():
    """
    主函數：執行完整的 per-token accuracy 分析
    
    驗證假設: 假設 2 - 模型學會「忽略困難 Tokens」
    
    判定標準:
    - 困難 tokens (如 Token 453) 在 Epoch 30-50 的準確率提升 < 2% → 假設成立
    - 簡單 tokens 的準確率持續提升至 80%+ → 假設成立
    """
    print(f"\n{'#'*70}")
    print(f"# Per-Token Accuracy 演進分析")
    print(f"# 目的: 驗證假設 2 - 模型是否學會「忽略困難 Tokens」")
    print(f"{'#'*70}")
    
    # 步驟 1: 分析所有 epochs
    print(f"\n{'='*70}")
    print(f"步驟 1: 分析所有 epochs 的 per-token accuracy")
    print(f"{'='*70}")
    results = analyze_all_epochs()
    
    # 步驟 2: 分析 Mismatch Tokens
    print(f"\n{'='*70}")
    print(f"步驟 2: 分析 Mismatch Tokens 學習曲線")
    print(f"{'='*70}")
    mismatch_analysis = analyze_mismatch_tokens(results)
    
    # 步驟 3: Token 453 特別分析
    print(f"\n{'='*70}")
    print(f"步驟 3: Token 453 特別分析")
    print(f"{'='*70}")
    analyze_token453(results)
    
    # 步驟 4: Tokens 分類
    print(f"\n{'='*70}")
    print(f"步驟 4: 簡單 vs 困難 Tokens 分類")
    print(f"{'='*70}")
    categories = categorize_tokens(results)
    
    # 步驟 5: 視覺化
    print(f"\n{'='*70}")
    print(f"步驟 5: 繪製分析圖表")
    print(f"{'='*70}")
    plot_overall_accuracy(results)
    plot_mismatch_tokens(mismatch_analysis)
    plot_token453(results)
    plot_easy_vs_difficult(categories, results)
    
    # 最終結論
    print(f"\n{'#'*70}")
    print(f"# 分析完成")
    print(f"{'#'*70}")
    print(f"\n輸出檔案:")
    print(f"  1. {OUTPUT_DIR}/pertoken_accuracy_raw.pt")
    print(f"  2. {OUTPUT_DIR}/mismatch_tokens_analysis.json")
    print(f"  3. {OUTPUT_DIR}/overall_accuracy_evolution.png")
    print(f"  4. {OUTPUT_DIR}/mismatch_tokens_evolution.png")
    print(f"  5. {OUTPUT_DIR}/token453_evolution.png")
    print(f"  6. {OUTPUT_DIR}/easy_vs_difficult_tokens.png")
    
    print(f"\n假設驗證:")
    print(f"  查看以上圖表和數據，判定「假設 2」是否成立")
    print(f"  判定標準:")
    print(f"    - 困難 tokens (Epoch 30-50 提升 < 2%) → 假設成立 ✅")
    print(f"    - 簡單 tokens 持續提升至 80%+ → 假設成立 ✅")
    print(f"    - Token 453 Val Acc 無改善 (<5%) → 假設成立 ✅")


if __name__ == "__main__":
    main()
