"""
簡化版 Per-Token Accuracy 分析 - 只分析 Epoch 50

這個簡化版本:
1. 只分析最新的 Epoch 50（最關鍵）
2. 快速驗證假設 2
3. 避免重複載入模型（節省時間）

完整版: analyze_pertoken_accuracy.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

# 設定路徑
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize')
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer

# ==================== 配置 ====================

RESULT_DIR = Path("results/crossattn_100epochs_20251105_025951")
EPOCH_TO_ANALYZE = 50  # 只分析 Epoch 50

# Top-20 Mismatch Tokens (來自 Commit 9f54460)
MISMATCH_TOKENS = [
    453, 1145, 1750, 1016, 1764, 1019, 1749, 1731, 1017, 1746,
    1756, 1755, 1018, 1732, 1757
]

# ==================== 主要函數 ====================

def load_model(epoch):
    """載入模型"""
    from torch.utils.data import DataLoader
    from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn
    from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn
    
    print(f"\n{'='*60}")
    print(f"載入 Epoch {epoch} 模型...")
    print(f"{'='*60}")
    
    # 1. 載入 WavTokenizer 獲取 codebook
    print("步驟 1: 載入 WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(
        config_path="../../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        model_path="../../models/wavtokenizer_large_speech_320_24k.ckpt"
    )
    wavtokenizer = wavtokenizer.cuda()
    wavtokenizer.eval()
    
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    print(f"✓ Codebook shape: {codebook.shape}")  # (4096, 512)
    
    # 2. 載入 checkpoint
    print(f"\n步驟 2: 載入 Epoch {epoch} checkpoint...")
    checkpoint_path = RESULT_DIR / f"checkpoint_epoch_{epoch}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    
    # 3. 建立模型
    print(f"\n步驟 3: 建立模型...")
    model = ZeroShotDenoisingTransformerCrossAttn(
        codebook=codebook,
        speaker_embed_dim=256,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1
    ).cuda()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型載入成功")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Acc: {checkpoint.get('train_acc', 'N/A')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    
    # 4. 載入數據
    print(f"\n步驟 4: 載入數據...")
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
    
    print(f"✓ Val set: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return model, val_loader


def compute_pertoken_accuracy(model, dataloader):
    """計算每個 token 的準確率"""
    print(f"\n{'='*60}")
    print(f"計算 Per-Token Accuracy...")
    print(f"{'='*60}")
    
    token_correct = torch.zeros(4096, dtype=torch.long).cuda()
    token_total = torch.zeros(4096, dtype=torch.long).cuda()
    
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            noisy_tokens = batch['noisy_tokens'].cuda()
            clean_tokens = batch['clean_tokens'].cuda()
            speaker_emb = batch['speaker_embeddings'].cuda()
            
            # 前向傳播
            logits = model(noisy_tokens, speaker_emb, return_logits=True)
            predictions = logits.argmax(dim=-1)
            
            # 整體準確率
            correct_mask = (predictions == clean_tokens)
            total_correct += correct_mask.sum().item()
            total_tokens += clean_tokens.numel()
            
            # Per-token 統計
            clean_flat = clean_tokens.view(-1)
            pred_flat = predictions.view(-1)
            
            for token_id in range(4096):
                mask = (clean_flat == token_id)
                if mask.sum() > 0:
                    token_total[token_id] += mask.sum()
                    token_correct[token_id] += (pred_flat[mask] == token_id).sum()
    
    # 計算準確率
    token_accuracy = torch.zeros(4096).cuda()
    valid_mask = (token_total > 0)
    token_accuracy[valid_mask] = token_correct[valid_mask].float() / token_total[valid_mask].float()
    
    overall_acc = total_correct / total_tokens
    
    print(f"\n✓ 完成")
    print(f"  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Unique tokens: {valid_mask.sum().item()} / 4096")
    
    return token_accuracy.cpu().numpy(), token_total.cpu().numpy(), overall_acc


def analyze_mismatch_tokens(token_acc, token_count):
    """分析 mismatch tokens"""
    print(f"\n{'='*60}")
    print(f"分析 Top-15 Mismatch Tokens")
    print(f"{'='*60}")
    
    print(f"\n這些 tokens 在 Train/Val 分布有顯著差異 (來自 Commit 9f54460)")
    print(f"\n{'Token ID':>10} {'Val Acc':>10} {'Count':>10} {'狀態'}")
    print(f"{'-'*45}")
    
    for token_id in MISMATCH_TOKENS:
        acc = token_acc[token_id]
        count = token_count[token_id]
        
        if count == 0:
            status = "未出現"
        elif acc < 0.3:
            status = "困難 ⚠️"
        elif acc > 0.6:
            status = "簡單 ✅"
        else:
            status = "中等"
        
        print(f"{token_id:>10} {acc:>10.4f} {count:>10} {status}")
    
    # Token 453 特別分析
    print(f"\n{'='*60}")
    print(f"Token 453 特別分析 (最大 mismatch: +5.08%)")
    print(f"{'='*60}")
    
    token_453_acc = token_acc[453]
    token_453_count = token_count[453]
    
    print(f"\n根據 Commit 9f54460:")
    print(f"  Train 出現率: 13.57%")
    print(f"  Val 出現率:   18.65%")
    print(f"  差異:         +5.08% (37.5% 相對增幅)")
    print(f"  對錯誤貢獻:   Train 30%, Val 29.5%")
    
    print(f"\nEpoch 50 結果:")
    print(f"  Val Accuracy: {token_453_acc:.4f}")
    print(f"  Val Count:    {token_453_count}")
    
    if token_453_acc < 0.3:
        print(f"\n⚠️  Token 453 的準確率很低 (<30%)")
        print(f"     這支持「假設 2: 模型學會忽略困難 Tokens」")


def categorize_tokens(token_acc, token_count):
    """將 tokens 分類"""
    print(f"\n{'='*60}")
    print(f"Tokens 分類分析")
    print(f"{'='*60}")
    
    # 只考慮出現 >10 次的 tokens
    valid_mask = (token_count > 10)
    
    easy_mask = valid_mask & (token_acc > 0.6)
    difficult_mask = valid_mask & (token_acc < 0.3)
    medium_mask = valid_mask & ~easy_mask & ~difficult_mask
    
    easy_tokens = np.where(easy_mask)[0]
    difficult_tokens = np.where(difficult_mask)[0]
    medium_tokens = np.where(medium_mask)[0]
    
    print(f"\n分類結果 (Val set, count > 10):")
    print(f"  簡單 tokens (Acc > 0.6):  {len(easy_tokens):4d} tokens")
    print(f"  中等 tokens (0.3-0.6):    {len(medium_tokens):4d} tokens")
    print(f"  困難 tokens (Acc < 0.3):  {len(difficult_tokens):4d} tokens")
    
    if len(easy_tokens) > 0:
        print(f"\n簡單 tokens 平均準確率: {token_acc[easy_tokens].mean():.4f}")
    if len(difficult_tokens) > 0:
        print(f"困難 tokens 平均準確率: {token_acc[difficult_tokens].mean():.4f}")
    
    # 分析 mismatch tokens 分類
    mismatch_in_difficult = [t for t in MISMATCH_TOKENS if t in difficult_tokens]
    mismatch_in_easy = [t for t in MISMATCH_TOKENS if t in easy_tokens]
    
    print(f"\nMismatch Tokens 分佈:")
    print(f"  在困難類別: {len(mismatch_in_difficult)}/15")
    print(f"  在簡單類別: {len(mismatch_in_easy)}/15")
    
    if len(mismatch_in_difficult) > len(mismatch_in_easy):
        print(f"\n⚠️  大部分 mismatch tokens 屬於困難類別")
        print(f"     這支持「假設 2」")
    
    return {
        'easy': easy_tokens,
        'difficult': difficult_tokens,
        'medium': medium_tokens
    }


def save_results(token_acc, token_count, overall_acc, categories):
    """保存結果"""
    output_dir = Path("analysis_outputs/pertoken_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'epoch': int(EPOCH_TO_ANALYZE),
        'overall_acc': float(overall_acc),
        'token_accuracy': [float(x) for x in token_acc],
        'token_count': [int(x) for x in token_count],
        'mismatch_tokens': [int(t) for t in MISMATCH_TOKENS],
        'mismatch_acc': [float(token_acc[t]) for t in MISMATCH_TOKENS],
        'mismatch_count': [int(token_count[t]) for t in MISMATCH_TOKENS],
        'categories': {
            'easy': [int(x) for x in categories['easy']],
            'difficult': [int(x) for x in categories['difficult']],
            'medium': [int(x) for x in categories['medium']]
        }
    }
    
    save_path = output_dir / "pertoken_analysis_epoch50.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 結果已保存: {save_path}")


# ==================== 主程式 ====================

def main():
    print(f"\n{'#'*70}")
    print(f"# 簡化版 Per-Token Accuracy 分析 (Epoch {EPOCH_TO_ANALYZE})")
    print(f"# 目的: 快速驗證假設 2 - 模型是否忽略困難 Tokens")
    print(f"{'#'*70}")
    
    # 載入模型
    model, val_loader = load_model(EPOCH_TO_ANALYZE)
    
    # 計算 per-token accuracy
    token_acc, token_count, overall_acc = compute_pertoken_accuracy(model, val_loader)
    
    # 分析 mismatch tokens
    analyze_mismatch_tokens(token_acc, token_count)
    
    # 分類 tokens
    categories = categorize_tokens(token_acc, token_count)
    
    # 保存結果
    save_results(token_acc, token_count, overall_acc, categories)
    
    # 最終結論
    print(f"\n{'#'*70}")
    print(f"# 假設驗證")
    print(f"{'#'*70}")
    
    token_453_acc = token_acc[453]
    difficult_count = len(categories['difficult'])
    mismatch_in_difficult = sum(1 for t in MISMATCH_TOKENS 
                                 if t in categories['difficult'])
    
    print(f"\n檢查點:")
    print(f"  1. Token 453 (最大 mismatch) 準確率: {token_453_acc:.4f}")
    print(f"  2. 困難 tokens 數量: {difficult_count}")
    print(f"  3. Mismatch tokens 在困難類別: {mismatch_in_difficult}/15")
    
    hypothesis_holds = False
    if token_453_acc < 0.3:
        print(f"\n✅ Token 453 準確率 < 30% → 假設 2 成立")
        hypothesis_holds = True
    
    if mismatch_in_difficult >= 10:
        print(f"✅ 大部分 mismatch tokens 屬於困難類別 → 假設 2 成立")
        hypothesis_holds = True
    
    if hypothesis_holds:
        print(f"\n結論: 假設 2 **成立** ⚠️")
        print(f"  模型確實學會「忽略困難 Tokens」")
        print(f"\n改進方向:")
        print(f"  1. 使用 Focal Loss 或 Class-Balanced Loss")
        print(f"  2. 增加困難 tokens 的訓練權重")
        print(f"  3. 使用 Hard Example Mining")
    else:
        print(f"\n結論: 假設 2 **不成立** ✅")
        print(f"  模型沒有明顯忽略困難 Tokens")


if __name__ == "__main__":
    main()
