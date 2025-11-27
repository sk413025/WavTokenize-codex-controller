"""
驗證 LoRA 修改後的 features 與 VQ codebook 對齊情況

分析內容:
1. Teacher vs Student features 到最近 codebook entry 的距離
2. VQ 選擇的 code 是否改變
3. Feature 空間的漂移程度
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import json

# 添加路徑
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, str(Path(__file__).parent))

from model import TeacherStudentModel
from config import TrainConfig, WAVTOK_CONFIG, WAVTOK_CKPT
from data import create_dataloaders
from wavtok_lora_patch import apply_lora_patch


def create_teacher_student_model(config, device='cuda'):
    """創建 Teacher-Student 模型"""
    apply_lora_patch()
    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        device=device,
    )
    return model


def analyze_vq_alignment(model, dataloader, num_batches=10, device='cuda'):
    """
    分析 VQ 對齊情況

    Returns:
        dict with alignment metrics
    """
    model.eval()

    # 獲取 codebook
    codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    codebook = codebook.to(device)  # (4096, 512)

    metrics = {
        'teacher_to_nearest_code_dist': [],
        'student_to_nearest_code_dist': [],
        'teacher_to_selected_code_dist': [],
        'student_to_selected_code_dist': [],
        'feature_mse': [],
        'code_match_rate': [],
        'student_nearest_vs_selected_match': [],
        'teacher_nearest_vs_selected_match': [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)

            # Forward pass
            output = model(noisy, clean)

            teacher_feat = output['teacher_features']  # (B, 512, T)
            student_feat = output['student_features']  # (B, 512, T)
            teacher_codes = output['teacher_codes']    # (n_q, B, T) or (B, n_q, T)
            student_codes = output['student_codes']

            B, C, T = teacher_feat.shape

            # Debug shapes
            if batch_idx == 0:
                print(f"  teacher_feat: {teacher_feat.shape}")
                print(f"  teacher_codes: {teacher_codes.shape}")

            # Handle different code shapes - codes 可能是 (n_q, B, T) 或 (B, n_q, T)
            if teacher_codes.dim() == 3:
                if teacher_codes.shape[0] == B:
                    # (B, n_q, T) -> take first quantizer
                    teacher_codes = teacher_codes[:, 0, :]  # (B, T)
                    student_codes = student_codes[:, 0, :]
                else:
                    # (n_q, B, T) -> take first quantizer
                    teacher_codes = teacher_codes[0]  # (B, T)
                    student_codes = student_codes[0]

            # Reshape features: (B, 512, T) -> (B*T, 512)
            teacher_feat_flat = teacher_feat.permute(0, 2, 1).reshape(-1, C)
            student_feat_flat = student_feat.permute(0, 2, 1).reshape(-1, C)

            # 計算到所有 codebook entries 的距離
            # (B*T, 4096)
            teacher_to_all = torch.cdist(teacher_feat_flat, codebook, p=2)
            student_to_all = torch.cdist(student_feat_flat, codebook, p=2)

            # 最近的 code 距離
            teacher_min_dist, teacher_nearest = teacher_to_all.min(dim=1)
            student_min_dist, student_nearest = student_to_all.min(dim=1)

            metrics['teacher_to_nearest_code_dist'].append(teacher_min_dist.mean().item())
            metrics['student_to_nearest_code_dist'].append(student_min_dist.mean().item())

            # 實際選擇的 code 的距離 - codes now (B, T)
            teacher_codes_flat = teacher_codes.reshape(-1).long()
            student_codes_flat = student_codes.reshape(-1).long()

            # 取出選擇的 code embedding
            teacher_selected_embed = codebook[teacher_codes_flat]  # (B*T, 512)
            student_selected_embed = codebook[student_codes_flat]  # (B*T, 512)

            # 計算 feature 到選擇的 code 的距離
            teacher_to_selected = (teacher_feat_flat - teacher_selected_embed).norm(dim=1)
            student_to_selected = (student_feat_flat - student_selected_embed).norm(dim=1)

            metrics['teacher_to_selected_code_dist'].append(teacher_to_selected.mean().item())
            metrics['student_to_selected_code_dist'].append(student_to_selected.mean().item())

            # Feature MSE
            feature_mse = F.mse_loss(student_feat, teacher_feat)
            metrics['feature_mse'].append(feature_mse.item())

            # Code match rate
            code_match = (teacher_codes_flat == student_codes_flat).float().mean()
            metrics['code_match_rate'].append(code_match.item())

            # Nearest vs Selected match (VQ 是否選了最近的 code)
            teacher_nearest_match = (teacher_nearest == teacher_codes_flat).float().mean()
            student_nearest_match = (student_nearest == student_codes_flat).float().mean()

            metrics['teacher_nearest_vs_selected_match'].append(teacher_nearest_match.item())
            metrics['student_nearest_vs_selected_match'].append(student_nearest_match.item())

            print(f"Batch {batch_idx + 1}/{num_batches}: "
                  f"Code Match={code_match.item():.3f}, "
                  f"Teacher Dist={teacher_to_selected.mean().item():.3f}, "
                  f"Student Dist={student_to_selected.mean().item():.3f}")

    # 計算平均值
    results = {k: np.mean(v) for k, v in metrics.items()}
    return results


def analyze_feature_drift(model, dataloader, num_batches=10, device='cuda'):
    """
    分析 feature 漂移程度
    """
    model.eval()

    drifts = {
        'l2_norm': [],
        'cosine_sim': [],
        'relative_change': [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)

            output = model(noisy, clean)

            teacher_feat = output['teacher_features']
            student_feat = output['student_features']

            # L2 距離
            l2_diff = (teacher_feat - student_feat).norm(dim=1).mean()
            drifts['l2_norm'].append(l2_diff.item())

            # Cosine similarity
            B, C, T = teacher_feat.shape
            t_flat = teacher_feat.permute(0, 2, 1).reshape(-1, C)
            s_flat = student_feat.permute(0, 2, 1).reshape(-1, C)
            cos_sim = F.cosine_similarity(t_flat, s_flat, dim=1).mean()
            drifts['cosine_sim'].append(cos_sim.item())

            # Relative change
            teacher_norm = teacher_feat.norm(dim=1).mean()
            relative = l2_diff / (teacher_norm + 1e-8)
            drifts['relative_change'].append(relative.item())

    return {k: np.mean(v) for k, v in drifts.items()}


def compare_with_baseline(model, dataloader, num_batches=10, device='cuda'):
    """
    比較 Student (有 LoRA) vs Teacher (無 LoRA) 處理相同輸入

    這裡讓 Teacher 也處理 noisy audio，看 LoRA 帶來的變化
    """
    model.eval()

    metrics = {
        'teacher_on_noisy_code_match': [],  # Teacher 處理 noisy 時的 code match
        'student_on_noisy_code_match': [],  # Student 處理 noisy 時的 code match
        'lora_improvement': [],              # Student 是否比 Teacher 處理 noisy 更好
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)

            # Teacher 處理 clean (ground truth codes)
            teacher_clean_feat, teacher_clean_codes = model.teacher_forward(clean)

            # Teacher 處理 noisy (baseline)
            teacher_noisy_feat, teacher_noisy_codes = model.teacher_forward(noisy)

            # Student 處理 noisy
            student_noisy_feat, student_noisy_codes, _ = model.student_forward(noisy)

            # 計算 code match rates
            gt_codes = teacher_clean_codes[:, 0, :].reshape(-1)
            teacher_noisy_codes_flat = teacher_noisy_codes[:, 0, :].reshape(-1)
            student_noisy_codes_flat = student_noisy_codes[:, 0, :].reshape(-1)

            teacher_match = (teacher_noisy_codes_flat == gt_codes).float().mean()
            student_match = (student_noisy_codes_flat == gt_codes).float().mean()

            metrics['teacher_on_noisy_code_match'].append(teacher_match.item())
            metrics['student_on_noisy_code_match'].append(student_match.item())
            metrics['lora_improvement'].append((student_match - teacher_match).item())

            print(f"Batch {batch_idx + 1}: "
                  f"Teacher(noisy)={teacher_match.item():.3f}, "
                  f"Student(noisy)={student_match.item():.3f}, "
                  f"Δ={student_match.item() - teacher_match.item():+.3f}")

    return {k: np.mean(v) for k, v in metrics.items()}


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    從 checkpoint 載入 LoRA 權重
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 提取 student 相關的 state dict
    state_dict = ckpt['model_state_dict']

    # 只載入 student 部分
    student_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('student.'):
            # 去掉 'student.' 前綴
            new_key = k[len('student.'):]
            student_state_dict[new_key] = v

    # 載入到 model.student
    model.student.load_state_dict(student_state_dict, strict=False)
    print(f"✓ Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")

    return ckpt.get('epoch', -1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Student checkpoint path (LoRA weights)')
    parser.add_argument('--num_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 創建配置
    config = TrainConfig(
        batch_size=args.batch_size,
        num_workers=2,
    )

    # 創建模型
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    model = create_teacher_student_model(config, device=device)

    # 載入 checkpoint (如果有的話)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    else:
        print("\n⚠️  No checkpoint specified, using initialized LoRA weights")

    # 創建 dataloader
    print("\n" + "="*60)
    print("Creating dataloader...")
    print("="*60)
    train_loader, val_loader = create_dataloaders(config)

    # 分析 VQ 對齊
    print("\n" + "="*60)
    print("1. VQ Alignment Analysis")
    print("="*60)
    vq_metrics = analyze_vq_alignment(model, val_loader, args.num_batches, device)

    print("\n📊 VQ Alignment Results:")
    print("-" * 40)
    print(f"Teacher → Nearest Code Dist:   {vq_metrics['teacher_to_nearest_code_dist']:.4f}")
    print(f"Student → Nearest Code Dist:   {vq_metrics['student_to_nearest_code_dist']:.4f}")
    print(f"Teacher → Selected Code Dist:  {vq_metrics['teacher_to_selected_code_dist']:.4f}")
    print(f"Student → Selected Code Dist:  {vq_metrics['student_to_selected_code_dist']:.4f}")
    print(f"Feature MSE:                   {vq_metrics['feature_mse']:.6f}")
    print(f"Code Match Rate:               {vq_metrics['code_match_rate']:.4f} ({vq_metrics['code_match_rate']*100:.1f}%)")
    print(f"Teacher: Nearest=Selected:     {vq_metrics['teacher_nearest_vs_selected_match']:.4f}")
    print(f"Student: Nearest=Selected:     {vq_metrics['student_nearest_vs_selected_match']:.4f}")

    # 分析 Feature Drift
    print("\n" + "="*60)
    print("2. Feature Drift Analysis")
    print("="*60)
    drift_metrics = analyze_feature_drift(model, val_loader, args.num_batches, device)

    print("\n📊 Feature Drift Results:")
    print("-" * 40)
    print(f"L2 Norm (Student - Teacher):   {drift_metrics['l2_norm']:.4f}")
    print(f"Cosine Similarity:             {drift_metrics['cosine_sim']:.4f}")
    print(f"Relative Change:               {drift_metrics['relative_change']:.4f} ({drift_metrics['relative_change']*100:.1f}%)")

    # 比較 baseline
    print("\n" + "="*60)
    print("3. LoRA Effect Analysis (vs Baseline)")
    print("="*60)
    baseline_metrics = compare_with_baseline(model, val_loader, args.num_batches, device)

    print("\n📊 LoRA Effect Results:")
    print("-" * 40)
    print(f"Teacher on Noisy (baseline):   {baseline_metrics['teacher_on_noisy_code_match']:.4f} ({baseline_metrics['teacher_on_noisy_code_match']*100:.1f}%)")
    print(f"Student on Noisy (with LoRA):  {baseline_metrics['student_on_noisy_code_match']:.4f} ({baseline_metrics['student_on_noisy_code_match']*100:.1f}%)")
    print(f"LoRA Improvement:              {baseline_metrics['lora_improvement']:+.4f} ({baseline_metrics['lora_improvement']*100:+.1f}%)")

    # 診斷結論
    print("\n" + "="*60)
    print("4. Diagnosis")
    print("="*60)

    # 判斷 VQ 對齊問題
    dist_ratio = vq_metrics['student_to_selected_code_dist'] / (vq_metrics['teacher_to_selected_code_dist'] + 1e-8)

    print(f"\n🔍 Feature → Code Distance Ratio: {dist_ratio:.2f}x")

    if dist_ratio > 1.5:
        print("   ❌ PROBLEM: Student features 與 VQ codebook 嚴重失配!")
        print("   → LoRA 修改了 encoder 輸出，但 VQ 沒有適配")
    elif dist_ratio > 1.1:
        print("   ⚠️  WARNING: Student features 與 VQ codebook 有輕微失配")
    else:
        print("   ✅ OK: Student features 與 VQ codebook 對齊良好")

    nearest_match_diff = vq_metrics['teacher_nearest_vs_selected_match'] - vq_metrics['student_nearest_vs_selected_match']
    print(f"\n🔍 Nearest=Selected Match Diff: {nearest_match_diff:.4f}")

    if nearest_match_diff > 0.1:
        print("   ❌ PROBLEM: Student 的 VQ 選擇偏離最近鄰!")
        print("   → Feature 空間漂移導致 VQ 選擇不再是最近的 code")
    else:
        print("   ✅ OK: VQ 選擇仍然是最近的 code")

    # 保存結果
    results = {
        'vq_alignment': vq_metrics,
        'feature_drift': drift_metrics,
        'baseline_comparison': baseline_metrics,
        'diagnosis': {
            'dist_ratio': dist_ratio,
            'nearest_match_diff': nearest_match_diff,
        }
    }

    output_path = Path('vq_alignment_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
