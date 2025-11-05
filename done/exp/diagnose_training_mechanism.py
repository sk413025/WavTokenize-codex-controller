"""
深入診斷訓練機轉：為何 Training Loss 無法下降？

檢查項目：
1. 梯度流動：是否有梯度消失/爆炸？
2. 權重變化：參數是否在更新？
3. 每層輸出：activation 是否合理？
4. Loss landscape：是否陷入局部最優？
5. Speaker embedding：是否真的有影響？
6. Token embedding：frozen codebook 是否合適？
7. Logits 分布：是否過度集中？
8. 學習率：是否太小或太大？
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from model_zeroshot import ZeroShotDenoisingTransformer
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def diagnose_gradients(model, dataloader, criterion, device):
    """
    診斷 1: 梯度流動分析
    
    檢查:
    - 每層梯度的 L2 norm
    - 是否有梯度消失 (grad_norm < 1e-6)
    - 是否有梯度爆炸 (grad_norm > 100)
    """
    print("\n" + "="*80)
    print("診斷 1: 梯度流動分析")
    print("="*80)
    
    model.train()
    
    # 取一個 batch
    batch = next(iter(dataloader))
    noisy_tokens = batch['noisy_tokens'].to(device)
    clean_tokens = batch['clean_tokens'].to(device)
    speaker_embeddings = batch['speaker_embeddings'].to(device)
    
    # Forward
    logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
    B, T, vocab = logits.shape
    loss = criterion(logits.reshape(B * T, vocab), clean_tokens.reshape(B * T))
    
    # Backward
    model.zero_grad()
    loss.backward()
    
    # 收集梯度統計
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()
            
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max,
                'shape': param.shape
            }
    
    # 顯示梯度統計
    print(f"\n梯度統計（共 {len(grad_stats)} 層）:")
    print(f"{'Layer Name':<60} {'Norm':>12} {'Mean':>12} {'Std':>12} {'Max':>12}")
    print("-" * 110)
    
    for name, stats in grad_stats.items():
        print(f"{name:<60} {stats['norm']:>12.6f} {stats['mean']:>12.6f} "
              f"{stats['std']:>12.6f} {stats['max']:>12.6f}")
    
    # 診斷結論
    vanishing_layers = [name for name, stats in grad_stats.items() if stats['norm'] < 1e-6]
    exploding_layers = [name for name, stats in grad_stats.items() if stats['norm'] > 100]
    
    print(f"\n🔍 診斷結果:")
    if vanishing_layers:
        print(f"  ⚠️  梯度消失層: {len(vanishing_layers)} 層")
        for layer in vanishing_layers[:5]:
            print(f"      - {layer}")
    else:
        print(f"  ✅ 無梯度消失")
    
    if exploding_layers:
        print(f"  ⚠️  梯度爆炸層: {len(exploding_layers)} 層")
        for layer in exploding_layers[:5]:
            print(f"      - {layer}")
    else:
        print(f"  ✅ 無梯度爆炸")
    
    return grad_stats


def diagnose_weight_updates(model, dataloader, optimizer, criterion, device, num_steps=10):
    """
    診斷 2: 權重更新分析
    
    檢查:
    - 訓練 N 步後，參數是否真的改變？
    - 改變幅度是否合理？
    """
    print("\n" + "="*80)
    print("診斷 2: 權重更新分析")
    print("="*80)
    
    model.train()
    
    # 記錄初始權重
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()
    
    # 訓練 N 步
    print(f"\n訓練 {num_steps} 步...")
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        noisy_tokens = batch['noisy_tokens'].to(device)
        clean_tokens = batch['clean_tokens'].to(device)
        speaker_embeddings = batch['speaker_embeddings'].to(device)
        
        # Forward
        logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
        B, T, vocab = logits.shape
        loss = criterion(logits.reshape(B * T, vocab), clean_tokens.reshape(B * T))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # 計算權重變化
    weight_changes = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_weights:
            delta = (param.data - initial_weights[name]).abs()
            weight_changes[name] = {
                'mean_delta': delta.mean().item(),
                'max_delta': delta.max().item(),
                'relative_change': (delta.mean() / (initial_weights[name].abs().mean() + 1e-8)).item()
            }
    
    # 顯示權重變化
    print(f"\n權重變化統計（{num_steps} 步後）:")
    print(f"{'Layer Name':<60} {'Mean Δ':>12} {'Max Δ':>12} {'Relative %':>12}")
    print("-" * 100)
    
    for name, stats in weight_changes.items():
        print(f"{name:<60} {stats['mean_delta']:>12.8f} {stats['max_delta']:>12.8f} "
              f"{stats['relative_change']*100:>11.6f}%")
    
    # 診斷結論
    no_change_layers = [name for name, stats in weight_changes.items() 
                        if stats['relative_change'] < 1e-6]
    
    print(f"\n🔍 診斷結果:")
    if no_change_layers:
        print(f"  ⚠️  未更新層: {len(no_change_layers)} 層")
        for layer in no_change_layers[:5]:
            print(f"      - {layer}")
    else:
        print(f"  ✅ 所有層都有更新")
    
    return weight_changes


def diagnose_activations(model, dataloader, device):
    """
    診斷 3: 每層激活值分析
    
    檢查:
    - Token embedding 輸出
    - Speaker embedding 輸出
    - Transformer 每層輸出
    - Final logits 分布
    """
    print("\n" + "="*80)
    print("診斷 3: 每層激活值分析")
    print("="*80)
    
    model.eval()
    
    # 註冊 hooks 來捕獲中間層輸出
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook
    
    # 註冊 hooks
    hooks = []
    hooks.append(model.speaker_proj.register_forward_hook(get_activation('speaker_proj')))
    hooks.append(model.pos_encoding.register_forward_hook(get_activation('pos_encoding')))
    hooks.append(model.transformer_encoder.register_forward_hook(get_activation('transformer_output')))
    hooks.append(model.output_proj.register_forward_hook(get_activation('logits')))
    
    # Forward 一個 batch
    with torch.no_grad():
        batch = next(iter(dataloader))
        noisy_tokens = batch['noisy_tokens'].to(device)
        clean_tokens = batch['clean_tokens'].to(device)
        speaker_embeddings = batch['speaker_embeddings'].to(device)
        
        # 手動執行 forward 來捕獲 token_emb
        token_emb = model.codebook[noisy_tokens]
        activations['token_emb'] = token_emb
        
        # 完整 forward
        logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    # 分析激活值
    print(f"\n激活值統計:")
    print(f"{'Layer Name':<30} {'Shape':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 100)
    
    for name, act in activations.items():
        print(f"{name:<30} {str(tuple(act.shape)):<20} "
              f"{act.mean().item():>12.6f} {act.std().item():>12.6f} "
              f"{act.min().item():>12.6f} {act.max().item():>12.6f}")
    
    # 特別檢查 logits 分布
    logits = activations['logits']
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    
    print(f"\nLogits 分布分析:")
    print(f"  - Logits 範圍: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  - 最大機率平均: {max_probs.mean().item():.4f}")
    print(f"  - 預測熵平均: {entropy.mean().item():.4f} (理論最大: {np.log(4096):.4f})")
    print(f"  - 預測集中度: {(max_probs > 0.5).float().mean().item() * 100:.2f}% 的 token 預測機率 > 50%")
    
    return activations


def diagnose_speaker_embedding_influence(model, dataloader, criterion, device):
    """
    診斷 4: Speaker Embedding 影響力
    
    檢查:
    - 有 speaker embedding vs 無 speaker embedding 的損失差異
    - 改變 speaker embedding 對預測的影響
    """
    print("\n" + "="*80)
    print("診斷 4: Speaker Embedding 影響力分析")
    print("="*80)
    
    model.eval()
    
    batch = next(iter(dataloader))
    noisy_tokens = batch['noisy_tokens'].to(device)
    clean_tokens = batch['clean_tokens'].to(device)
    speaker_embeddings = batch['speaker_embeddings'].to(device)
    
    with torch.no_grad():
        # 1. 正常預測
        logits_normal = model(noisy_tokens, speaker_embeddings, return_logits=True)
        B, T, vocab = logits_normal.shape
        loss_normal = criterion(logits_normal.reshape(B * T, vocab), 
                               clean_tokens.reshape(B * T))
        pred_normal = logits_normal.argmax(dim=-1)
        
        # 2. Zero speaker embedding
        zero_speaker = torch.zeros_like(speaker_embeddings)
        logits_zero = model(noisy_tokens, zero_speaker, return_logits=True)
        loss_zero = criterion(logits_zero.reshape(B * T, vocab), 
                             clean_tokens.reshape(B * T))
        pred_zero = logits_zero.argmax(dim=-1)
        
        # 3. Random speaker embedding
        random_speaker = torch.randn_like(speaker_embeddings)
        logits_random = model(noisy_tokens, random_speaker, return_logits=True)
        loss_random = criterion(logits_random.reshape(B * T, vocab), 
                               clean_tokens.reshape(B * T))
        pred_random = logits_random.argmax(dim=-1)
        
        # 4. 交換 speaker embeddings (batch 內)
        if B > 1:
            swapped_speaker = speaker_embeddings.roll(shifts=1, dims=0)
            logits_swapped = model(noisy_tokens, swapped_speaker, return_logits=True)
            loss_swapped = criterion(logits_swapped.reshape(B * T, vocab), 
                                    clean_tokens.reshape(B * T))
            pred_swapped = logits_swapped.argmax(dim=-1)
        else:
            loss_swapped = loss_normal
            pred_swapped = pred_normal
    
    # 統計差異
    diff_zero = (pred_normal != pred_zero).float().mean().item() * 100
    diff_random = (pred_normal != pred_random).float().mean().item() * 100
    diff_swapped = (pred_normal != pred_swapped).float().mean().item() * 100
    
    print(f"\nSpeaker Embedding 影響力:")
    print(f"  正常 speaker embedding:")
    print(f"    - Loss: {loss_normal.item():.4f}")
    print(f"\n  Zero speaker embedding:")
    print(f"    - Loss: {loss_zero.item():.4f} (Δ = {loss_zero.item() - loss_normal.item():+.4f})")
    print(f"    - 預測改變: {diff_zero:.2f}% tokens")
    print(f"\n  Random speaker embedding:")
    print(f"    - Loss: {loss_random.item():.4f} (Δ = {loss_random.item() - loss_normal.item():+.4f})")
    print(f"    - 預測改變: {diff_random:.2f}% tokens")
    print(f"\n  Swapped speaker embedding:")
    print(f"    - Loss: {loss_swapped.item():.4f} (Δ = {loss_swapped.item() - loss_normal.item():+.4f})")
    print(f"    - 預測改變: {diff_swapped:.2f}% tokens")
    
    print(f"\n🔍 診斷結果:")
    if diff_zero < 1.0:
        print(f"  ⚠️  Speaker embedding 幾乎無影響（改變 <1% tokens）")
        print(f"      → 模型可能忽略了 speaker information")
    elif diff_zero < 5.0:
        print(f"  ⚠️  Speaker embedding 影響很小（改變 <5% tokens）")
        print(f"      → Speaker conditioning 可能太弱")
    else:
        print(f"  ✅ Speaker embedding 有明顯影響（改變 {diff_zero:.2f}% tokens）")
    
    return {
        'loss_normal': loss_normal.item(),
        'loss_zero': loss_zero.item(),
        'loss_random': loss_random.item(),
        'loss_swapped': loss_swapped.item(),
        'diff_zero': diff_zero,
        'diff_random': diff_random,
        'diff_swapped': diff_swapped
    }


def diagnose_loss_landscape(model, dataloader, criterion, device):
    """
    診斷 5: Loss Landscape 分析
    
    檢查:
    - Loss 對參數擾動的敏感度
    - 是否陷入銳利的局部最優（sharp minima）
    """
    print("\n" + "="*80)
    print("診斷 5: Loss Landscape 分析")
    print("="*80)
    
    model.eval()
    
    batch = next(iter(dataloader))
    noisy_tokens = batch['noisy_tokens'].to(device)
    clean_tokens = batch['clean_tokens'].to(device)
    speaker_embeddings = batch['speaker_embeddings'].to(device)
    
    # 計算當前 loss
    with torch.no_grad():
        logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
        B, T, vocab = logits.shape
        loss_original = criterion(logits.reshape(B * T, vocab), 
                                 clean_tokens.reshape(B * T)).item()
    
    # 測試不同擾動幅度
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1]
    losses_perturbed = []
    
    print(f"\n當前 Loss: {loss_original:.4f}")
    print(f"\n參數擾動測試:")
    print(f"{'Epsilon':>10} {'Loss':>12} {'Δ Loss':>12} {'Δ Loss %':>12}")
    print("-" * 50)
    
    for eps in epsilons:
        # 保存原始參數
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()
        
        # 添加隨機擾動
        for name, param in model.named_parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * eps
                param.data.add_(noise)
        
        # 計算擾動後的 loss
        with torch.no_grad():
            logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
            loss_pert = criterion(logits.reshape(B * T, vocab), 
                                clean_tokens.reshape(B * T)).item()
        
        delta_loss = loss_pert - loss_original
        delta_pct = (delta_loss / loss_original) * 100
        
        print(f"{eps:>10.6f} {loss_pert:>12.4f} {delta_loss:>+12.4f} {delta_pct:>+11.2f}%")
        losses_perturbed.append(loss_pert)
        
        # 恢復原始參數
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original_params[name])
    
    print(f"\n🔍 診斷結果:")
    # 如果小擾動導致大的 loss 變化，說明在 sharp minima
    if abs(losses_perturbed[0] - loss_original) / loss_original > 0.1:
        print(f"  ⚠️  Sharp Minima: 小擾動 (1e-4) 導致 loss 變化 > 10%")
        print(f"      → 可能難以泛化，建議增加 weight decay 或使用 SAM optimizer")
    else:
        print(f"  ✅ Flat Minima: 對小擾動不敏感")


def diagnose_learning_rate(model, dataloader, optimizer, criterion, device):
    """
    診斷 6: 學習率診斷
    
    檢查:
    - 當前學習率是否合適
    - Loss 對學習率的敏感度
    """
    print("\n" + "="*80)
    print("診斷 6: 學習率診斷")
    print("="*80)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\n當前學習率: {current_lr:.2e}")
    
    # 測試不同學習率
    test_lrs = [current_lr * 0.1, current_lr, current_lr * 10, current_lr * 100]
    
    print(f"\n學習率掃描:")
    print(f"{'LR':>12} {'Loss (Step 0)':>15} {'Loss (Step 1)':>15} {'Δ Loss':>12}")
    print("-" * 60)
    
    model.train()
    
    for test_lr in test_lrs:
        # 保存當前狀態
        model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # 設置測試 LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = test_lr
        
        # 取一個 batch
        batch = next(iter(dataloader))
        noisy_tokens = batch['noisy_tokens'].to(device)
        clean_tokens = batch['clean_tokens'].to(device)
        speaker_embeddings = batch['speaker_embeddings'].to(device)
        
        # Step 0: 初始 loss
        with torch.no_grad():
            logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
            B, T, vocab = logits.shape
            loss_0 = criterion(logits.reshape(B * T, vocab), 
                             clean_tokens.reshape(B * T)).item()
        
        # Step 1: 訓練一步
        logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
        loss = criterion(logits.reshape(B * T, vocab), clean_tokens.reshape(B * T))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step 1: 更新後的 loss
        with torch.no_grad():
            logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
            loss_1 = criterion(logits.reshape(B * T, vocab), 
                             clean_tokens.reshape(B * T)).item()
        
        delta = loss_1 - loss_0
        print(f"{test_lr:>12.2e} {loss_0:>15.4f} {loss_1:>15.4f} {delta:>+12.4f}")
        
        # 恢復狀態
        model.load_state_dict(model_state)
    
    # 恢復原始 LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    print(f"\n🔍 診斷結果:")
    print(f"  - 如果所有 LR 都導致 loss 上升，可能:")
    print(f"      1. 梯度方向有問題")
    print(f"      2. 已達到模型能力上限")
    print(f"  - 如果只有大 LR 能降低 loss，說明當前 LR 太小")
    print(f"  - 如果只有小 LR 能降低 loss，說明當前 LR 太大")


def main():
    """
    主診斷流程
    """
    print("="*80)
    print("深入診斷訓練機轉")
    print("="*80)
    
    # 設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 載入數據
    print("\n載入數據...")
    train_cache_path = Path('data/train_cache.pt')
    val_cache_path = Path('data/val_cache.pt')
    
    train_dataset = ZeroShotAudioDatasetCached(train_cache_path)
    val_dataset = ZeroShotAudioDatasetCached(val_cache_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=cached_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=cached_collate_fn,
        num_workers=0
    )
    
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    
    # 載入模型
    print("\n載入模型...")
    
    # 方法 1: 從 best_model checkpoint 直接載入
    checkpoint_path = 'results/zeroshot_100epochs_20251105_002300/best_model.pth'
    if Path(checkpoint_path).exists():
        print(f"  從 checkpoint 載入模型...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 從 checkpoint 提取 codebook
        codebook_tensor = checkpoint['model_state_dict']['codebook']
        print(f"  Codebook shape: {codebook_tensor.shape}")
        
        # 創建模型
        model = ZeroShotDenoisingTransformer(
            codebook=codebook_tensor,
            speaker_embed_dim=256,
            d_model=512,
            nhead=8,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.1
        ).to(device)
        
        # 載入完整狀態
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ 模型載入完成 (Epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"  ⚠️  Checkpoint 不存在: {checkpoint_path}")
        print(f"  請確認訓練是否已開始")
        return None
    
    # 創建 optimizer 和 criterion
    import torch.optim as optim
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss()
    
    # 執行診斷
    results = {}
    
    # 1. 梯度分析
    results['gradients'] = diagnose_gradients(model, train_loader, criterion, device)
    
    # 2. 權重更新
    results['weight_updates'] = diagnose_weight_updates(
        model, train_loader, optimizer, criterion, device, num_steps=10
    )
    
    # 3. 激活值分析
    results['activations'] = diagnose_activations(model, train_loader, device)
    
    # 4. Speaker embedding 影響
    results['speaker_influence'] = diagnose_speaker_embedding_influence(
        model, train_loader, criterion, device
    )
    
    # 5. Loss landscape
    diagnose_loss_landscape(model, train_loader, criterion, device)
    
    # 6. 學習率
    diagnose_learning_rate(model, train_loader, optimizer, criterion, device)
    
    print("\n" + "="*80)
    print("診斷完成！")
    print("="*80)
    
    return results


if __name__ == '__main__':
    results = main()
