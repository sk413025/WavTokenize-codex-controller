#!/usr/bin/env python3
"""
測試 exp_1203 實驗結果

分析 exp7 (feature_correct_vq) 和 exp8 (emb_distillation) 的結果：
1. Token Accuracy (exact match)
2. Top-K Accuracy
3. 與 baseline 的比較
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent))


def collate_fn(batch):
    """處理不同長度的音頻"""
    # 找到最長的長度
    max_len = max(
        max(b['noisy_audio'].shape[-1], b['clean_audio'].shape[-1])
        for b in batch
    )

    # 對齊到 320 的倍數 (WavTokenizer 需要)
    max_len = ((max_len + 319) // 320) * 320

    noisy_audios = []
    clean_audios = []

    for b in batch:
        noisy = b['noisy_audio']
        clean = b['clean_audio']

        # Pad to max_len
        if noisy.shape[-1] < max_len:
            noisy = F.pad(noisy, (0, max_len - noisy.shape[-1]))
        if clean.shape[-1] < max_len:
            clean = F.pad(clean, (0, max_len - clean.shape[-1]))

        noisy_audios.append(noisy)
        clean_audios.append(clean)

    return {
        'noisy_audio': torch.stack(noisy_audios),
        'clean_audio': torch.stack(clean_audios),
    }

from config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE, DISTANCE_MATRIX
from model import TeacherStudentModel
from data import NoisyCleanPairDataset


def load_checkpoint(checkpoint_path, device='cuda', lora_rank=64, lora_alpha=128):
    """載入 checkpoint 並返回模型"""
    print(f"Loading checkpoint: {checkpoint_path}")

    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        device=device,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    return model


def compute_token_accuracy(model, dataloader, codebook, device='cuda'):
    """計算 Token Accuracy"""
    all_student_tokens = []
    all_teacher_tokens = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Token Accuracy"):
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # Forward pass
            output = model.forward_with_emb(noisy_audio, clean_audio)

            student_emb = output['student_emb']  # (B, C, T)
            teacher_codes = output['teacher_codes']  # (1, B, T) or (B, T)

            # 處理 teacher_codes 維度
            if teacher_codes.dim() == 3:
                teacher_codes = teacher_codes[0]  # (B, T)
            elif teacher_codes.dim() == 2 and teacher_codes.shape[0] == 1:
                teacher_codes = teacher_codes.squeeze(0)

            B, C, T_emb = student_emb.shape
            T_code = teacher_codes.shape[-1]
            T = min(T_emb, T_code)

            # 計算 student tokens (argmin to codebook)
            for b in range(B):
                emb = student_emb[b, :, :T].transpose(0, 1)  # (T, C)
                distances = torch.cdist(emb.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)  # (T, 4096)
                student_tokens = distances.argmin(dim=-1)  # (T,)

                teacher_tok = teacher_codes[b, :T]  # (T,)

                all_student_tokens.append(student_tokens.cpu().numpy())
                all_teacher_tokens.append(teacher_tok.cpu().numpy())

    return all_student_tokens, all_teacher_tokens


def compute_baseline_token_accuracy(model, dataloader, codebook, device='cuda'):
    """計算 Baseline Token Accuracy (Teacher noisy vs Teacher clean)"""
    all_noisy_tokens = []
    all_clean_tokens = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Baseline"):
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            # Teacher 處理 noisy audio
            noisy_output = model.teacher.feature_extractor(noisy_audio, bandwidth_id=0)
            noisy_codes = noisy_output[1]  # codes

            # Teacher 處理 clean audio
            clean_output = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            clean_codes = clean_output[1]  # codes

            # 處理維度
            if noisy_codes.dim() == 3:
                noisy_codes = noisy_codes[0]
            if clean_codes.dim() == 3:
                clean_codes = clean_codes[0]

            B = noisy_codes.shape[0]
            T = min(noisy_codes.shape[-1], clean_codes.shape[-1])

            for b in range(B):
                all_noisy_tokens.append(noisy_codes[b, :T].cpu().numpy())
                all_clean_tokens.append(clean_codes[b, :T].cpu().numpy())

    return all_noisy_tokens, all_clean_tokens


def analyze_accuracy(student_tokens, teacher_tokens, distance_matrix=None):
    """分析準確率"""
    total_tokens = 0
    exact_match = 0
    top_k_matches = {1: 0, 5: 0, 10: 0, 50: 0, 100: 0}

    # 如果有 distance matrix，預計算最近鄰
    nearest_neighbors = None
    if distance_matrix is not None:
        nearest_neighbors = {}
        for code in range(distance_matrix.shape[0]):
            sorted_indices = torch.argsort(distance_matrix[code])
            nearest_neighbors[code] = sorted_indices.numpy()

    for st_batch, tt_batch in zip(student_tokens, teacher_tokens):
        for st, tt in zip(st_batch, tt_batch):
            total_tokens += 1

            if st == tt:
                exact_match += 1

            # Top-K accuracy
            if nearest_neighbors is not None:
                for k in top_k_matches.keys():
                    if st in nearest_neighbors[tt][:k]:
                        top_k_matches[k] += 1

    results = {
        'total_tokens': total_tokens,
        'exact_match': exact_match,
        'exact_match_rate': exact_match / total_tokens if total_tokens > 0 else 0,
    }

    if nearest_neighbors is not None:
        results['top_k_accuracy'] = {k: v / total_tokens for k, v in top_k_matches.items()}

    return results


def main():
    parser = argparse.ArgumentParser(description='Test exp_1203 experiment results')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name (feature_correct_vq or emb_distillation)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to test')
    parser.add_argument('--use_val', action='store_true',
                       help='Use validation set')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default='best',
                       help='Checkpoint to use: latest, best')
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)

    # 設置路徑
    exp_dir = Path(__file__).parent / 'experiments' / args.exp_name
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)

    # 確定 checkpoint
    checkpoint_dir = exp_dir / 'checkpoints'
    if args.checkpoint == 'latest':
        ckpt_path = checkpoint_dir / 'latest.pt'
    elif args.checkpoint == 'best':
        ckpt_path = checkpoint_dir / 'best.pt'
    else:
        ckpt_path = checkpoint_dir / args.checkpoint

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"Testing: {args.exp_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print("=" * 70)

    # 載入 distance matrix
    print("\nLoading distance matrix...")
    distance_matrix = torch.load(DISTANCE_MATRIX).to(device)
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # 載入數據
    print("\nLoading data...")
    cache_path = VAL_CACHE if args.use_val else TRAIN_CACHE
    dataset = NoisyCleanPairDataset(cache_path, max_samples=args.num_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # 載入模型
    print("\nLoading model...")
    model = load_checkpoint(ckpt_path, device=args.device,
                           lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

    # 獲取 codebook
    codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    codebook = codebook.to(device)
    print(f"Codebook shape: {codebook.shape}")

    # 計算 Baseline (Teacher noisy vs Teacher clean)
    print("\n" + "=" * 70)
    print("Computing Baseline (Teacher noisy vs Teacher clean)...")
    print("=" * 70)

    noisy_tokens, clean_tokens = compute_baseline_token_accuracy(
        model, dataloader, codebook, device=args.device
    )
    baseline_results = analyze_accuracy(noisy_tokens, clean_tokens, distance_matrix.cpu())

    print(f"\nBaseline Results:")
    print(f"  Exact Match Rate: {baseline_results['exact_match_rate']*100:.2f}%")
    if 'top_k_accuracy' in baseline_results:
        for k, acc in baseline_results['top_k_accuracy'].items():
            print(f"  Top-{k}: {acc*100:.2f}%")

    # 計算 Student Token Accuracy
    print("\n" + "=" * 70)
    print("Computing Student Token Accuracy...")
    print("=" * 70)

    student_tokens, teacher_tokens = compute_token_accuracy(
        model, dataloader, codebook, device=args.device
    )
    student_results = analyze_accuracy(student_tokens, teacher_tokens, distance_matrix.cpu())

    print(f"\nStudent Results:")
    print(f"  Exact Match Rate: {student_results['exact_match_rate']*100:.2f}%")
    if 'top_k_accuracy' in student_results:
        for k, acc in student_results['top_k_accuracy'].items():
            print(f"  Top-{k}: {acc*100:.2f}%")

    # 比較
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    baseline_acc = baseline_results['exact_match_rate'] * 100
    student_acc = student_results['exact_match_rate'] * 100
    improvement = student_acc - baseline_acc

    print(f"\n{'Metric':<25} {'Baseline':<15} {'Student':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Exact Match':<25} {baseline_acc:>12.2f}% {student_acc:>12.2f}% {improvement:>+12.2f}%")

    if 'top_k_accuracy' in student_results and 'top_k_accuracy' in baseline_results:
        for k in [5, 10, 50]:
            b_acc = baseline_results['top_k_accuracy'][k] * 100
            s_acc = student_results['top_k_accuracy'][k] * 100
            imp = s_acc - b_acc
            print(f"{'Top-' + str(k):<25} {b_acc:>12.2f}% {s_acc:>12.2f}% {imp:>+12.2f}%")

    # 保存結果
    output_dir = exp_dir / 'test_results'
    output_dir.mkdir(exist_ok=True)

    results = {
        'experiment': args.exp_name,
        'checkpoint': str(ckpt_path),
        'num_samples': args.num_samples,
        'use_val': args.use_val,
        'baseline': {
            'exact_match_rate': baseline_results['exact_match_rate'],
            'top_k_accuracy': baseline_results.get('top_k_accuracy', {}),
        },
        'student': {
            'exact_match_rate': student_results['exact_match_rate'],
            'top_k_accuracy': student_results.get('top_k_accuracy', {}),
        },
        'improvement': {
            'exact_match': improvement,
        }
    }

    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # 總結
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nExperiment: {args.exp_name}")
    print(f"Baseline (Teacher noisy vs clean): {baseline_acc:.2f}%")
    print(f"Student: {student_acc:.2f}%")
    print(f"Improvement: {improvement:+.2f}%")

    if improvement > 1:
        print("\n✅ Student shows improvement over baseline!")
    elif improvement > 0:
        print("\n⚠️  Student shows marginal improvement.")
    else:
        print("\n❌ Student does not improve over baseline.")


if __name__ == '__main__':
    main()
