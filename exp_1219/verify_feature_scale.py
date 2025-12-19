"""
exp_1219: 驗證特徵尺度問題

用於在訓練過程中驗證 Student 和 Teacher 特徵的尺度是否匹配。

使用方式：
1. 在訓練腳本中 import 此模組
2. 在每個 epoch 或每 N 個 batch 調用 analyze_feature_scale()
3. 記錄結果到 history

或者獨立運行此腳本來分析保存的 checkpoint
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import json
from pathlib import Path


def analyze_feature_scale(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    return_details: bool = False
) -> Dict[str, float]:
    """
    分析 Student 和 Teacher 特徵的尺度關係

    Args:
        student_features: (B, D, T) Student encoder output
        teacher_features: (B, D, T) Teacher encoder output
        return_details: 是否返回詳細統計

    Returns:
        dict: 包含尺度分析指標
    """
    B, D, T = student_features.shape

    # Reshape to (B*T, D) for per-frame analysis
    stu = student_features.permute(0, 2, 1).reshape(-1, D)
    tea = teacher_features.permute(0, 2, 1).reshape(-1, D)

    with torch.no_grad():
        # 1. 基本統計
        stu_flat = student_features.reshape(-1)
        tea_flat = teacher_features.reshape(-1)

        stu_mean = stu_flat.mean().item()
        stu_std = stu_flat.std().item()
        stu_min = stu_flat.min().item()
        stu_max = stu_flat.max().item()

        tea_mean = tea_flat.mean().item()
        tea_std = tea_flat.std().item()
        tea_min = tea_flat.min().item()
        tea_max = tea_flat.max().item()

        # 2. Per-frame L2 norm
        stu_norms = stu.norm(dim=1)
        tea_norms = tea.norm(dim=1)

        stu_norm_mean = stu_norms.mean().item()
        tea_norm_mean = tea_norms.mean().item()
        norm_ratio = stu_norm_mean / (tea_norm_mean + 1e-8)

        # 3. Cosine Similarity
        cos_sim = F.cosine_similarity(stu, tea, dim=1)
        cos_sim_mean = cos_sim.mean().item()
        cos_sim_std = cos_sim.std().item()

        # 4. MSE 分解
        diff = stu - tea
        mse_per_frame = (diff ** 2).sum(dim=1)
        norm_stu_sq = (stu ** 2).sum(dim=1)
        norm_tea_sq = (tea ** 2).sum(dim=1)
        inner_prod = (stu * tea).sum(dim=1)

        # 5. 尺度問題指標
        # 如果 Student 特徵被壓縮到很小範圍，norm_ratio 會遠小於 1
        # 如果方向不一致，cos_sim 會接近 0

    result = {
        'norm_ratio': norm_ratio,
        'cos_sim_mean': cos_sim_mean,
        'cos_sim_std': cos_sim_std,
        'stu_norm_mean': stu_norm_mean,
        'tea_norm_mean': tea_norm_mean,
        'mse_mean': mse_per_frame.mean().item(),
        'inner_prod_mean': inner_prod.mean().item(),
    }

    if return_details:
        result.update({
            'stu_mean': stu_mean,
            'stu_std': stu_std,
            'stu_min': stu_min,
            'stu_max': stu_max,
            'tea_mean': tea_mean,
            'tea_std': tea_std,
            'tea_min': tea_min,
            'tea_max': tea_max,
            'norm_stu_sq_mean': norm_stu_sq.mean().item(),
            'norm_tea_sq_mean': norm_tea_sq.mean().item(),
        })

    return result


def diagnose_scale_issue(metrics: Dict[str, float]) -> Tuple[bool, str]:
    """
    根據指標診斷是否有尺度問題

    Returns:
        (has_issue, diagnosis): 是否有問題，診斷說明
    """
    issues = []

    # 檢查 norm ratio
    if metrics['norm_ratio'] < 0.5:
        issues.append(f"Student 特徵 norm 偏小 (ratio={metrics['norm_ratio']:.3f})")
    elif metrics['norm_ratio'] > 2.0:
        issues.append(f"Student 特徵 norm 偏大 (ratio={metrics['norm_ratio']:.3f})")

    # 檢查 cosine similarity
    if metrics['cos_sim_mean'] < 0.5:
        issues.append(f"方向相似度低 (cos_sim={metrics['cos_sim_mean']:.3f})")

    if issues:
        return True, "發現尺度問題: " + "; ".join(issues)
    else:
        return False, f"尺度正常 (norm_ratio={metrics['norm_ratio']:.3f}, cos_sim={metrics['cos_sim_mean']:.3f})"


def print_analysis(metrics: Dict[str, float]):
    """打印分析結果"""
    print("\n--- 特徵尺度分析 ---")
    print(f"  Norm Ratio (Stu/Tea): {metrics['norm_ratio']:.4f}")
    print(f"  Cosine Similarity:    {metrics['cos_sim_mean']:.4f} ± {metrics['cos_sim_std']:.4f}")
    print(f"  Student Norm Mean:    {metrics['stu_norm_mean']:.4f}")
    print(f"  Teacher Norm Mean:    {metrics['tea_norm_mean']:.4f}")
    print(f"  MSE Mean:             {metrics['mse_mean']:.4f}")
    print(f"  Inner Product Mean:   {metrics['inner_prod_mean']:.4f}")

    has_issue, diagnosis = diagnose_scale_issue(metrics)
    print(f"\n診斷: {diagnosis}")


# ==================== 獨立運行測試 ====================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
    sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

    print("=" * 80)
    print("特徵尺度驗證工具")
    print("=" * 80)

    # 嘗試載入模型並分析
    # 如果 GPU 不可用，使用模擬數據

    try:
        from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT
        from exp_1217.models import TeacherStudentConfigurableLoRA
        from exp_1212.data_aligned import create_aligned_dataloaders

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # 載入 checkpoint
        exp_dir = Path('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217/runs/exp48_best_config')
        checkpoint_path = exp_dir / 'best_model.pt'

        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")

            model = TeacherStudentConfigurableLoRA(
                wavtok_config=str(WAVTOK_CONFIG),
                wavtok_ckpt=str(WAVTOK_CKPT),
                lora_rank=128,
                lora_alpha=256,
                lora_dropout=0.2,
                lora_layers='all_18',
                device=device,
            )

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # 載入數據
            class DataConfig:
                batch_size = 4
                num_workers = 0
                pin_memory = False

            _, val_loader = create_aligned_dataloaders(DataConfig())

            # 分析
            print("\n分析實際特徵...")
            with torch.no_grad():
                batch = next(iter(val_loader))
                noisy_audio = batch['noisy_audio'].to(device)
                clean_audio = batch['clean_audio'].to(device)

                output = model(noisy_audio, clean_audio)

                metrics = analyze_feature_scale(
                    output['student_encoder_out'],
                    output['teacher_encoder_out'],
                    return_details=True
                )

                print_analysis(metrics)

        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Using simulated data for testing...")
            raise FileNotFoundError()

    except Exception as e:
        print(f"\n無法載入模型 ({e})")
        print("使用模擬數據進行功能測試...\n")

        # 模擬測試
        B, D, T = 4, 512, 75

        # 情況 1: 正常尺度
        print("--- 測試情況 1: 正常尺度 ---")
        student = torch.randn(B, D, T) * 1.0
        teacher = torch.randn(B, D, T) * 1.0
        metrics = analyze_feature_scale(student, teacher, return_details=True)
        print_analysis(metrics)

        # 情況 2: Student 尺度偏小
        print("\n--- 測試情況 2: Student 尺度偏小 ---")
        student = torch.randn(B, D, T) * 0.1
        teacher = torch.randn(B, D, T) * 1.0
        metrics = analyze_feature_scale(student, teacher, return_details=True)
        print_analysis(metrics)

        # 情況 3: 你假設的極端情況
        print("\n--- 測試情況 3: 你假設的極端情況 ---")
        print("Zstu ∈ [-0.01, 0.02], Ztea ∈ [0.3, 0.8]")
        student = torch.rand(B, D, T) * 0.03 - 0.01  # [-0.01, 0.02]
        teacher = torch.rand(B, D, T) * 0.5 + 0.3    # [0.3, 0.8]
        metrics = analyze_feature_scale(student, teacher, return_details=True)
        print_analysis(metrics)
