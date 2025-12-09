"""
梯度衝突分析工具

檢查 MSE Loss 和 CE Loss 的梯度是否相互衝突
- cosine similarity > 0: 梯度方向一致（協同）
- cosine similarity ≈ 0: 梯度正交（互不影響）
- cosine similarity < 0: 梯度方向相反（衝突）

實驗編號: exp_1207_gradient_analysis
日期: 2025-12-08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Import from train_with_ce.py
import sys
sys.path.append(str(Path(__file__).parent))
from train_with_ce import (
    TeacherStudentWithCE, 
    set_seed
)

# Paths (使用 exp_1201 的配置)
WAVTOK_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"
DISTANCE_MATRIX = "/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201/distance_matrix.pt"


def compute_gradient_cosine_similarity(grad1, grad2):
    """
    計算兩個梯度張量的 cosine similarity
    
    Args:
        grad1: 第一個梯度張量
        grad2: 第二個梯度張量
        
    Returns:
        cosine_sim: 梯度夾角的 cosine 值
        angle_deg: 梯度夾角（度數）
        
    解讀:
        cosine_sim > 0.5: 梯度基本一致，協同學習
        cosine_sim > 0: 有一定協同作用
        cosine_sim ≈ 0: 梯度正交，互不影響
        cosine_sim < 0: 梯度衝突，互相抵消
        cosine_sim < -0.5: 嚴重衝突，互相對抗
    """
    # Flatten gradients
    g1_flat = grad1.reshape(-1)
    g2_flat = grad2.reshape(-1)
    
    # Compute cosine similarity
    dot_product = torch.dot(g1_flat, g2_flat)
    norm1 = torch.norm(g1_flat)
    norm2 = torch.norm(g2_flat)
    
    cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
    
    # Convert to angle (degrees)
    angle_rad = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0))
    angle_deg = angle_rad * 180.0 / np.pi
    
    return cosine_sim.item(), angle_deg.item()


def analyze_gradient_conflict(
    model,
    dataloader,
    device,
    num_batches=50,
    feature_weight=1.0,
    ce_weight=1.0,
    ce_temperature=0.1
):
    """
    分析 MSE 和 CE 梯度的衝突情況
    
    Args:
        model: TeacherStudentWithCE 模型
        dataloader: 資料載入器
        device: 運算裝置
        num_batches: 分析的批次數量
        feature_weight: Feature Loss 權重
        ce_weight: CE Loss 權重
        ce_temperature: CE Loss 溫度參數
        
    Returns:
        dict 包含:
        - cosine_similarities: 每個 batch 的 cosine similarity
        - angles: 每個 batch 的夾角（度）
        - mean_cosine: 平均 cosine similarity
        - std_cosine: cosine similarity 標準差
        - conflict_ratio: 衝突比例（cosine < 0）
    """
    model.train()  # 必須是訓練模式才能計算梯度
    
    cosine_sims = []
    angles = []
    mse_grad_norms = []
    ce_grad_norms = []
    
    print(f"\n分析 {num_batches} 個 batches 的梯度衝突...")
    
    with tqdm(total=num_batches, desc="Analyzing gradients") as pbar:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            # exp_1201 的 batch 格式是 {'noisy_audio', 'clean_audio'}
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)
            
            # Forward pass
            output = model(noisy_audio, clean_audio)
            student_encoder_out = output['student_encoder_out']
            teacher_encoder_out = output['teacher_encoder_out']
            teacher_codes = output['teacher_codes']
            
            # ===== Compute MSE gradient =====
            model.zero_grad()
            mse_loss = F.mse_loss(student_encoder_out, teacher_encoder_out)
            mse_loss.backward(retain_graph=True)
            
            # Collect MSE gradients
            mse_grads = []
            for name, param in model.named_parameters():
                if param.grad is not None and 'student' in name:
                    mse_grads.append(param.grad.clone())
            
            # Concatenate all gradients
            mse_grad_concat = torch.cat([g.reshape(-1) for g in mse_grads])
            mse_grad_norm = torch.norm(mse_grad_concat).item()
            mse_grad_norms.append(mse_grad_norm)
            
            # ===== Compute CE gradient =====
            model.zero_grad()
            
            # Compute CE loss
            logits = model.compute_ce_logits(student_encoder_out)
            B, T, num_codes = logits.shape
            
            if teacher_codes.dim() == 3:
                t_codes = teacher_codes[0]
            else:
                t_codes = teacher_codes.squeeze(1)
            
            logits_scaled = logits / ce_temperature
            logits_flat = logits_scaled.reshape(B * T, num_codes)
            targets_flat = t_codes.reshape(B * T).long()
            ce_loss = F.cross_entropy(logits_flat, targets_flat)
            ce_loss.backward()
            
            # Collect CE gradients
            ce_grads = []
            for name, param in model.named_parameters():
                if param.grad is not None and 'student' in name:
                    ce_grads.append(param.grad.clone())
            
            # Concatenate all gradients
            ce_grad_concat = torch.cat([g.reshape(-1) for g in ce_grads])
            ce_grad_norm = torch.norm(ce_grad_concat).item()
            ce_grad_norms.append(ce_grad_norm)
            
            # ===== Compute cosine similarity =====
            cosine_sim, angle = compute_gradient_cosine_similarity(
                mse_grad_concat, ce_grad_concat
            )
            
            cosine_sims.append(cosine_sim)
            angles.append(angle)
            
            pbar.update(1)
            pbar.set_postfix({
                'cosine': f'{cosine_sim:.3f}',
                'angle': f'{angle:.1f}°',
                'mse_norm': f'{mse_grad_norm:.2e}',
                'ce_norm': f'{ce_grad_norm:.2e}'
            })
    
    # Statistics
    cosine_sims = np.array(cosine_sims)
    angles = np.array(angles)
    
    results = {
        'cosine_similarities': cosine_sims.tolist(),
        'angles_deg': angles.tolist(),
        'mse_grad_norms': mse_grad_norms,
        'ce_grad_norms': ce_grad_norms,
        'mean_cosine': float(np.mean(cosine_sims)),
        'std_cosine': float(np.std(cosine_sims)),
        'median_cosine': float(np.median(cosine_sims)),
        'mean_angle': float(np.mean(angles)),
        'conflict_ratio': float(np.sum(cosine_sims < 0) / len(cosine_sims)),
        'orthogonal_ratio': float(np.sum(np.abs(cosine_sims) < 0.1) / len(cosine_sims)),
        'aligned_ratio': float(np.sum(cosine_sims > 0.5) / len(cosine_sims)),
    }
    
    return results


def plot_gradient_analysis(results, save_path):
    """
    繪製梯度分析結果
    
    Args:
        results: analyze_gradient_conflict 的返回結果
        save_path: 儲存路徑
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Cosine similarity distribution
    ax = axes[0, 0]
    ax.hist(results['cosine_similarities'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Orthogonal (0)')
    ax.axvline(results['mean_cosine'], color='green', linestyle='-', linewidth=2, 
               label=f"Mean ({results['mean_cosine']:.3f})")
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('MSE vs CE Gradient Cosine Similarity Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Angle distribution
    ax = axes[0, 1]
    ax.hist(results['angles_deg'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(90, color='red', linestyle='--', linewidth=2, label='Orthogonal (90°)')
    ax.axvline(results['mean_angle'], color='green', linestyle='-', linewidth=2,
               label=f"Mean ({results['mean_angle']:.1f}°)")
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Gradient Angle Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Cosine similarity over batches
    ax = axes[1, 0]
    batch_indices = range(len(results['cosine_similarities']))
    ax.plot(batch_indices, results['cosine_similarities'], 'b-', alpha=0.6, linewidth=1)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Orthogonal')
    ax.axhline(results['mean_cosine'], color='green', linestyle='-', linewidth=2, label='Mean')
    ax.fill_between(batch_indices, -1, 0, alpha=0.2, color='red', label='Conflict zone')
    ax.fill_between(batch_indices, 0, 1, alpha=0.2, color='green', label='Aligned zone')
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Cosine Similarity Over Batches', fontsize=14, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Gradient norm comparison
    ax = axes[1, 1]
    x = np.arange(len(results['mse_grad_norms']))
    ax.plot(x, results['mse_grad_norms'], 'b-', label='MSE Grad Norm', alpha=0.7, linewidth=1.5)
    ax.plot(x, results['ce_grad_norms'], 'r-', label='CE Grad Norm', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Magnitude Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"梯度分析圖表已儲存至: {save_path}")


def main():
    """
    主函式：執行梯度衝突分析
    """
    # 固定隨機種子
    set_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # Output directory
    output_dir = Path(__file__).parent / 'gradient_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("載入模型...")
    print("="*60)
    
    # Create model
    model = TeacherStudentWithCE(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=64,
        lora_alpha=128,
        device=device,
    )
    
    print("\n" + "="*60)
    print("載入資料...")
    print("="*60)
    
    # Create dataloader (使用 exp_1201 的方法)
    sys.path.insert(0, str(Path(__file__).parent.parent / 'exp_1201'))
    from data import create_dataloaders
    
    class SimpleConfig:
        def __init__(self):
            self.use_hdf5 = False
            self.batch_size = 4  # 減少 batch size 避免 OOM
            self.num_workers = 2
            self.pin_memory = True
    
    train_loader, val_loader = create_dataloaders(SimpleConfig())
    dataloader = train_loader
    
    print(f"\n資料集大小: {len(dataloader.dataset)}")
    print(f"批次數量: {len(dataloader)}")
    
    # Run analysis
    print("\n" + "="*60)
    print("執行梯度衝突分析...")
    print("="*60)
    
    results = analyze_gradient_conflict(
        model=model,
        dataloader=dataloader,
        device=device,
        num_batches=50,  # 減少分析批次數
        feature_weight=1.0,
        ce_weight=1.0,
        ce_temperature=0.1
    )
    
    # Print results
    print("\n" + "="*60)
    print("分析結果:")
    print("="*60)
    print(f"平均 Cosine Similarity: {results['mean_cosine']:.4f}")
    print(f"Cosine Similarity 標準差: {results['std_cosine']:.4f}")
    print(f"中位數 Cosine Similarity: {results['median_cosine']:.4f}")
    print(f"平均夾角: {results['mean_angle']:.2f}°")
    print(f"\n衝突比例 (cosine < 0): {results['conflict_ratio']*100:.1f}%")
    print(f"正交比例 (|cosine| < 0.1): {results['orthogonal_ratio']*100:.1f}%")
    print(f"對齊比例 (cosine > 0.5): {results['aligned_ratio']*100:.1f}%")
    
    # Interpretation
    print("\n" + "="*60)
    print("結果解讀:")
    print("="*60)
    
    if results['mean_cosine'] > 0.5:
        print("✅ 梯度基本對齊，MSE 和 CE 協同學習")
    elif results['mean_cosine'] > 0:
        print("⚠️  梯度有一定協同作用，但不強")
    elif results['mean_cosine'] > -0.2:
        print("⚠️  梯度接近正交，MSE 和 CE 互不影響")
    else:
        print("❌ 梯度嚴重衝突！MSE 和 CE 互相對抗")
    
    if results['conflict_ratio'] > 0.5:
        print(f"❌ 超過 50% 的樣本存在梯度衝突！")
    elif results['conflict_ratio'] > 0.2:
        print(f"⚠️  有 {results['conflict_ratio']*100:.1f}% 的樣本存在梯度衝突")
    else:
        print(f"✅ 只有 {results['conflict_ratio']*100:.1f}% 的樣本存在梯度衝突")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f'gradient_analysis_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n分析結果已儲存至: {json_path}")
    
    # Plot
    plot_path = output_dir / f'gradient_analysis_{timestamp}.png'
    plot_gradient_analysis(results, plot_path)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == '__main__':
    main()
