"""
評估 Exp48 模型 - 簡化版
只使用已保存的 history 和 checkpoint 進行評估
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

# 設定 GPU (CUDA_VISIBLE_DEVICES=1 對應實際 GPU 2)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1212.data_aligned import create_aligned_dataloaders
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX


def load_model_for_eval():
    """載入模型進行評估"""
    from exp_1217.models import TeacherStudentConfigurableLoRA

    device = torch.device('cuda:0')
    print(f"Using device: {device}")

    # 載入配置
    model_path = Path(__file__).parent.parent / 'exp_1217/runs/exp48_best_config/best_model.pt'
    config_path = model_path.parent / 'config.json'

    with open(config_path) as f:
        config = json.load(f)

    print(f"Config: lora_rank={config['lora_rank']}, lora_layers={config['lora_layers']}")

    # 創建模型
    model = TeacherStudentConfigurableLoRA(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        lora_layers=config['lora_layers'],
        device='cuda:0'
    )

    # 載入權重
    checkpoint = torch.load(model_path, map_location='cuda:0')
    # checkpoint 包含完整模型狀態
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, device


def compute_topk_accuracy(student_features, codebook, teacher_codes, k_values=[1, 5, 10, 50]):
    """計算 Top-K Accuracy"""
    B, D, T = student_features.shape

    # student_features: (B, D, T) -> (B*T, D)
    features_flat = student_features.permute(0, 2, 1).reshape(-1, D)

    # 計算與 codebook 的距離: (B*T, C)
    distances = torch.cdist(features_flat, codebook, p=2)

    # 獲取 top-k 最近的 indices
    _, topk_indices = distances.topk(max(k_values), dim=1, largest=False)

    # Ground truth
    teacher_flat = teacher_codes.reshape(-1)

    results = {}
    for k in k_values:
        topk_k = topk_indices[:, :k]
        hits = (topk_k == teacher_flat.unsqueeze(1)).any(dim=1)
        results[k] = hits.float().mean().item() * 100

    return results


def compute_distance_stats(student_codes, teacher_codes, distance_matrix):
    """計算距離統計"""
    s_flat = student_codes.reshape(-1).long()
    t_flat = teacher_codes.reshape(-1).long()

    distances = distance_matrix[s_flat, t_flat]

    return {
        'mean': distances.mean().item(),
        'std': distances.std().item(),
        'median': distances.median().item(),
        'zero_ratio': (distances == 0).float().mean().item() * 100,
    }


def main():
    print("="*60)
    print("Exp48 模型評估")
    print("="*60)

    # 載入模型
    print("\n載入模型...")
    model, device = load_model_for_eval()

    # 獲取 codebook
    codebook = model._get_codebook()
    print(f"Codebook shape: {codebook.shape}")

    # 載入距離矩陣
    print("\n載入距離矩陣...")
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)

    # 載入資料
    print("\n載入驗證資料...")
    from dataclasses import dataclass

    @dataclass
    class EvalConfig:
        batch_size: int = 8
        num_workers: int = 2

    _, val_loader = create_aligned_dataloaders(EvalConfig())
    print(f"Val batches: {len(val_loader)}")

    # 評估
    print("\n開始評估...")

    all_topk = {1: [], 5: [], 10: [], 50: [], 100: []}
    all_dist_stats = []
    all_exact_acc = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            # lengths = batch['lengths'].to(device)  # model forward 不需要 lengths

            output = model(noisy, clean)

            # model 輸出的 key 是 student_encoder_out, 不是 student_features
            student_features = output['student_encoder_out']  # (B, D, T)
            student_codes = output['student_codes']
            teacher_codes = output['teacher_codes']

            if student_codes.dim() == 3:
                student_codes = student_codes[0]
            if teacher_codes.dim() == 3:
                teacher_codes = teacher_codes[0]

            # Top-K
            topk = compute_topk_accuracy(
                student_features, codebook, teacher_codes,
                k_values=[1, 5, 10, 50, 100]
            )
            for k, v in topk.items():
                all_topk[k].append(v)

            # Distance
            dist = compute_distance_stats(student_codes, teacher_codes, distance_matrix)
            all_dist_stats.append(dist)

            # Exact Acc
            exact_acc = (student_codes == teacher_codes).float().mean().item() * 100
            all_exact_acc.append(exact_acc)

    # 結果
    print("\n" + "="*60)
    print("評估結果")
    print("="*60)

    print("\n### Token Accuracy ###")
    for k in [1, 5, 10, 50, 100]:
        print(f"  Top-{k:3d}: {np.mean(all_topk[k]):6.3f}%")

    print("\n### Distance Statistics ###")
    mean_dist = np.mean([d['mean'] for d in all_dist_stats])
    zero_ratio = np.mean([d['zero_ratio'] for d in all_dist_stats])
    print(f"  Mean Distance:       {mean_dist:.4f}")
    print(f"  Exact Match (dist=0): {zero_ratio:.3f}%")

    # 與隨機比較
    print("\n### 與隨機基準比較 ###")
    random_dist = distance_matrix.mean().item()
    print(f"  隨機 Top-1:  {1/4096*100:.4f}% → 模型: {np.mean(all_topk[1]):.3f}% ({np.mean(all_topk[1])/(1/4096*100):.1f}x)")
    print(f"  隨機距離:    {random_dist:.4f} → 模型: {mean_dist:.4f} (改善 {(random_dist-mean_dist)/random_dist*100:.1f}%)")

    # 儲存
    results = {
        'topk_accuracy': {str(k): float(np.mean(v)) for k, v in all_topk.items()},
        'distance': {'mean': mean_dist, 'zero_ratio': zero_ratio},
        'improvement': {
            'topk_over_random': np.mean(all_topk[1]) / (1/4096*100),
            'distance_reduction': (random_dist - mean_dist) / random_dist * 100
        }
    }

    output_path = Path(__file__).parent / 'exp48_evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n結果已儲存至: {output_path}")


if __name__ == '__main__':
    main()
