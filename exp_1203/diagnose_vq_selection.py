"""
診斷腳本：分析 VQ 選擇行為

目的：
理解為什麼 Feature Loss 下降但 Token Accuracy 也下降

假設：
- Student features 確實在接近 Teacher features
- 但 VQ 的 argmin 選擇對微小差異很敏感
- 導致選擇「接近但不相同」的 codes

診斷指標：
1. Feature L2 Distance: ||student_feat - teacher_feat||
2. Token Accuracy: student_codes == teacher_codes 的比例
3. Code Distance: distance_matrix[student_codes, teacher_codes] 的平均值
4. Rank of Correct Code: 正確 code 在 student 的距離排序中的位置
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import TeacherStudentModel
from data import create_dataloaders
from config import get_train_config, WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX


def diagnose_vq_selection(checkpoint_path=None, num_batches=10):
    """
    診斷 VQ 選擇行為

    Args:
        checkpoint_path: 如果提供，載入訓練過的 checkpoint
        num_batches: 分析的 batch 數量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 載入模型
    print("Loading model...")
    config = get_train_config(
        exp_name='diagnose',
        batch_size=16,
        num_epochs=1,
    )

    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=64,
        lora_alpha=128,
    ).to(device)

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()

    # 載入 distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)

    # 載入 codebook
    codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    codebook = codebook.to(device)

    # 載入數據
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(config)

    # 診斷統計
    stats = {
        'feature_l2_distances': [],
        'token_accuracies': [],
        'code_distances': [],
        'correct_code_ranks': [],
        'top5_accuracies': [],
    }

    print(f"\nAnalyzing {num_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # Forward
            output = model(noisy_audio, clean_audio)

            student_features = output['student_features']  # (B, 512, T)
            teacher_features = output['teacher_features']  # (B, 512, T)
            student_codes = output['student_codes'][0]     # (B, T)
            teacher_codes = output['teacher_codes'][0]     # (B, T)

            B, C, T = student_features.shape

            # 1. Feature L2 Distance (per frame)
            feat_diff = student_features - teacher_features  # (B, 512, T)
            feat_l2 = torch.norm(feat_diff, dim=1).mean()    # scalar
            stats['feature_l2_distances'].append(feat_l2.item())

            # 2. Token Accuracy
            token_acc = (student_codes == teacher_codes).float().mean()
            stats['token_accuracies'].append(token_acc.item())

            # 3. Code Distance (in codebook space)
            student_flat = student_codes.reshape(-1).long()
            teacher_flat = teacher_codes.reshape(-1).long()
            code_dist = distance_matrix[student_flat, teacher_flat].mean()
            stats['code_distances'].append(code_dist.item())

            # 4. Rank of Correct Code
            # 計算 student features 到所有 codes 的距離
            features_flat = student_features.permute(0, 2, 1).reshape(-1, C)  # (B*T, 512)
            distances = torch.cdist(features_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)  # (B*T, 4096)

            # 對每個位置，計算正確 code 的 rank
            sorted_indices = distances.argsort(dim=-1)  # (B*T, 4096)，越小越前

            # 找到 teacher code 在排序中的位置
            ranks = []
            for i in range(len(teacher_flat)):
                correct_code = teacher_flat[i].item()
                rank = (sorted_indices[i] == correct_code).nonzero(as_tuple=True)[0].item()
                ranks.append(rank)

            avg_rank = sum(ranks) / len(ranks)
            stats['correct_code_ranks'].append(avg_rank)

            # 5. Top-5 Accuracy
            top5_preds = distances.topk(5, dim=-1, largest=False).indices  # (B*T, 5)
            top5_correct = (top5_preds == teacher_flat.unsqueeze(-1)).any(dim=-1)
            top5_acc = top5_correct.float().mean()
            stats['top5_accuracies'].append(top5_acc.item())

            print(f"  Batch {batch_idx+1}: "
                  f"Feat L2={feat_l2.item():.4f}, "
                  f"Token Acc={token_acc.item()*100:.1f}%, "
                  f"Code Dist={code_dist.item():.4f}, "
                  f"Avg Rank={avg_rank:.1f}, "
                  f"Top5 Acc={top5_acc.item()*100:.1f}%")

    # 總結
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)

    print(f"\n1. Feature L2 Distance (should be low if Feature Loss works):")
    print(f"   Mean: {sum(stats['feature_l2_distances'])/len(stats['feature_l2_distances']):.4f}")

    print(f"\n2. Token Accuracy (main metric):")
    print(f"   Mean: {sum(stats['token_accuracies'])/len(stats['token_accuracies'])*100:.2f}%")

    print(f"\n3. Code Distance (low = choosing 'nearby' codes):")
    print(f"   Mean: {sum(stats['code_distances'])/len(stats['code_distances']):.4f}")

    print(f"\n4. Average Rank of Correct Code (should be 0 if perfect):")
    print(f"   Mean: {sum(stats['correct_code_ranks'])/len(stats['correct_code_ranks']):.1f}")
    print(f"   (0 = always picks correct, 1 = picks 2nd closest, etc.)")

    print(f"\n5. Top-5 Accuracy (correct code in top 5 choices):")
    print(f"   Mean: {sum(stats['top5_accuracies'])/len(stats['top5_accuracies'])*100:.2f}%")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    avg_rank = sum(stats['correct_code_ranks'])/len(stats['correct_code_ranks'])
    avg_token_acc = sum(stats['token_accuracies'])/len(stats['token_accuracies'])
    avg_top5_acc = sum(stats['top5_accuracies'])/len(stats['top5_accuracies'])

    if avg_rank < 5 and avg_token_acc < 0.2:
        print("\n⚠️  Correct code is close but not top-1!")
        print("   → VQ selection is sensitive to small feature differences")
        print("   → Consider: sharper temperature, different loss design")

    if avg_top5_acc > 0.5 and avg_token_acc < 0.2:
        print("\n⚠️  High Top-5 but low Top-1 accuracy!")
        print("   → Student is learning the right 'region' of codebook")
        print("   → But not precise enough for exact match")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: use untrained model)')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='Number of batches to analyze')
    args = parser.parse_args()

    diagnose_vq_selection(args.checkpoint, args.num_batches)
