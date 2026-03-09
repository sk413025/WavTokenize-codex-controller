import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

import sys
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1219.losses import create_length_mask
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE


def select_device(requested: str) -> str:
    if requested and requested != 'auto':
        return requested
    if torch.cuda.is_available():
        try:
            for idx in range(torch.cuda.device_count()):
                major, _ = torch.cuda.get_device_capability(idx)
                if major >= 7:
                    return f'cuda:{idx}'
        except Exception:
            pass
    return 'cpu'


def build_loader(cache_path, batch_size, num_workers, max_samples=None):
    dataset = CurriculumDataset(cache_path, max_samples=max_samples, compute_snr=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    return loader


def masked_cosine(student_feat, teacher_feat, lengths, encoder_stride=320):
    # student_feat, teacher_feat: (B, C, T) or (B, T, C)
    if student_feat.shape != teacher_feat.shape:
        min_len = min(student_feat.shape[-1], teacher_feat.shape[-1])
        student_feat = student_feat[..., :min_len]
        teacher_feat = teacher_feat[..., :min_len]

    if student_feat.dim() == 3 and student_feat.shape[1] != teacher_feat.shape[1]:
        # fallback: assume (B, T, C)
        pass

    if student_feat.dim() == 3:
        B, C, T = student_feat.shape
        s = student_feat
        t = teacher_feat
        mask = create_length_mask(lengths, T * encoder_stride, encoder_stride, device=student_feat.device)
        mask = mask.unsqueeze(1)  # (B,1,T)
        s = torch.nn.functional.normalize(s, dim=1)
        t = torch.nn.functional.normalize(t, dim=1)
        cos = (s * t) * mask
        cos = cos.sum(dim=1)
        return float(cos.sum().item() / (mask.sum().item() + 1e-8))

    return 0.0


def masked_mse(student_feat, teacher_feat, lengths, encoder_stride=320):
    if student_feat.shape != teacher_feat.shape:
        min_len = min(student_feat.shape[-1], teacher_feat.shape[-1])
        student_feat = student_feat[..., :min_len]
        teacher_feat = teacher_feat[..., :min_len]

    if student_feat.dim() == 3:
        B, C, T = student_feat.shape
        mask = create_length_mask(lengths, T * encoder_stride, encoder_stride, device=student_feat.device)
        mask = mask.unsqueeze(1)
        mse = ((student_feat - teacher_feat) ** 2) * mask
        return float(mse.sum().item() / (mask.sum().item() + 1e-8))

    return 0.0


@torch.no_grad()
def eval_alignment(model, loader, device, encoder_stride=320, layers=(3,4,6), progress_every=50):
    results = {
        'final': {'cos': [], 'mse': []},
    }
    for l in layers:
        results[f'layer_{l}'] = {'cos': [], 'mse': []}

    for idx, batch in enumerate(loader, start=1):
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        output = model(noisy, clean)

        s_final = output['student_encoder_out']
        t_final = output['teacher_encoder_out']
        results['final']['cos'].append(masked_cosine(s_final, t_final, lengths, encoder_stride))
        results['final']['mse'].append(masked_mse(s_final, t_final, lengths, encoder_stride))

        s_inter = output['student_intermediates']
        t_inter = output['teacher_intermediates']
        for l in layers:
            key = f'layer_{l}'
            s_feat = s_inter.get(l, s_inter.get(str(l), None))
            t_feat = t_inter.get(l, t_inter.get(str(l), None))
            if s_feat is None or t_feat is None:
                continue
            results[key]['cos'].append(masked_cosine(s_feat, t_feat, lengths, encoder_stride))
            results[key]['mse'].append(masked_mse(s_feat, t_feat, lengths, encoder_stride))

        if progress_every and idx % progress_every == 0:
            print(f"[align] {idx} batches processed...")

    # aggregate
    summary = {}
    for key, vals in results.items():
        cos_list = vals['cos']
        mse_list = vals['mse']
        summary[key] = {
            'cos_mean': float(np.mean(cos_list)) if cos_list else 0.0,
            'cos_std': float(np.std(cos_list)) if cos_list else 0.0,
            'mse_mean': float(np.mean(mse_list)) if mse_list else 0.0,
            'mse_std': float(np.std(mse_list)) if mse_list else 0.0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--output_dir', type=str,
                        default='exp_0112_intermediate/analysis/train_valid_gap_58a9b71')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=1000)
    parser.add_argument('--max_val_samples', type=int, default=500)
    parser.add_argument('--encoder_stride', type=int, default=320)
    parser.add_argument('--progress_every', type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)

    run_dir = Path(args.run_dir)
    with open(run_dir / 'config.json') as f:
        config = json.load(f)

    model = TeacherStudentIntermediate(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=config.get('lora_rank', 256),
        lora_alpha=config.get('lora_alpha', 512),
        lora_dropout=config.get('lora_dropout', 0.2),
        intermediate_indices=[3, 4, 6],
        device=device,
    )
    ckpt = torch.load(run_dir / args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    train_loader = build_loader(TRAIN_CACHE, args.batch_size, args.num_workers, args.max_train_samples)
    val_loader = build_loader(VAL_CACHE, args.batch_size, args.num_workers, args.max_val_samples)

    train_stats = eval_alignment(model, train_loader, device, encoder_stride=args.encoder_stride,
                                 progress_every=args.progress_every)
    val_stats = eval_alignment(model, val_loader, device, encoder_stride=args.encoder_stride,
                               progress_every=args.progress_every)

    stats = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'device': device,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'train': train_stats,
        'val': val_stats,
    }

    with open(out_dir / 'feature_alignment_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
