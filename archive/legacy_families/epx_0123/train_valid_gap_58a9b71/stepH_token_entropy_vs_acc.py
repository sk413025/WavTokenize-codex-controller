import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, VAL_CACHE
from exp_1219.losses import create_length_mask


ENCODER_STRIDE = 320


def select_device(requested: str) -> str:
    if requested and requested != 'auto':
        return requested

    if torch.cuda.is_available():
        try:
            for idx in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(idx)
                if major >= 7:
                    return f'cuda:{idx}'
        except Exception:
            pass
    return 'cpu'


def build_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )


def per_sample_entropy(codes, mask, codebook_size=4096):
    # codes: (B,T), mask: (B,T)
    entropies = []
    topk_mass = []
    for i in range(codes.shape[0]):
        valid = mask[i] > 0
        tokens = codes[i][valid]
        if tokens.numel() == 0:
            entropies.append(0.0)
            topk_mass.append(0.0)
            continue
        counts = torch.bincount(tokens, minlength=codebook_size).float()
        probs = counts / counts.sum()
        entropy = -(probs * (probs + 1e-12).log2()).sum().item()
        topk = torch.topk(probs, k=min(10, probs.numel())).values.sum().item()
        entropies.append(entropy)
        topk_mass.append(topk)
    return entropies, topk_mass


def compute_correlations(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    pearson = np.corrcoef(x, y)[0, 1]
    # Spearman via rank
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    spearman = np.corrcoef(rx, ry)[0, 1]
    return float(pearson), float(spearman)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    checkpoint = args.checkpoint or str(run_dir / 'best_model.pt')

    device = select_device(args.device)
    print(f'Using device: {device}')

    # model
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        lora_dropout=0.2,
    )
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)

    dataset = CurriculumDataset(VAL_CACHE, max_samples=None, compute_snr=False)

    if args.max_val_samples is not None and args.max_val_samples < len(dataset):
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(dataset), size=args.max_val_samples, replace=False)
        dataset = Subset(dataset, indices.tolist())

    loader = build_loader(dataset, args.batch_size, args.num_workers)

    per_acc = []
    per_entropy = []
    per_topk = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)
            lengths = batch['lengths'].to(device)

            output = model(noisy_audio, clean_audio)
            s_codes = output['student_codes']
            t_codes = output['teacher_codes']

            if s_codes.dim() == 3:
                s_codes = s_codes[0]
            if t_codes.dim() == 3:
                t_codes = t_codes[0]

            B, T = s_codes.shape
            max_audio_len = T * ENCODER_STRIDE
            mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=device)

            correct = (s_codes == t_codes).float() * mask
            correct_per_sample = correct.sum(dim=1)
            total_per_sample = mask.sum(dim=1)
            acc = (correct_per_sample / (total_per_sample + 1e-8)).detach().cpu().numpy().tolist()

            entropy, topk = per_sample_entropy(s_codes.detach().cpu(), mask.detach().cpu(), codebook_size=4096)

            per_acc.extend(acc)
            per_entropy.extend(entropy)
            per_topk.extend(topk)

            if batch_idx % 200 == 0:
                print(f'[val] batch {batch_idx}/{len(loader)}')

    pearson, spearman = compute_correlations(per_entropy, per_acc)

    out_dir = Path(__file__).resolve().parent
    out_stats = out_dir / 'token_usage_stats_val_full.json'
    out_corr = out_dir / 'token_entropy_vs_acc_val.json'
    out_plot = out_dir / 'token_entropy_vs_acc_val.png'

    stats_payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'device': device,
        'max_val_samples': args.max_val_samples,
        'num_samples': len(per_acc),
        'entropy': {
            'mean': float(np.mean(per_entropy)),
            'std': float(np.std(per_entropy)),
            'min': float(np.min(per_entropy)),
            'max': float(np.max(per_entropy)),
        },
        'topk_mass': {
            'mean': float(np.mean(per_topk)),
            'std': float(np.std(per_topk)),
            'min': float(np.min(per_topk)),
            'max': float(np.max(per_topk)),
        },
    }

    corr_payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'num_samples': len(per_acc),
        'pearson_entropy_acc': pearson,
        'spearman_entropy_acc': spearman,
    }

    out_stats.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False))
    out_corr.write_text(json.dumps(corr_payload, indent=2, ensure_ascii=False))

    plt.figure(figsize=(6, 4))
    plt.scatter(per_entropy, per_acc, s=6, alpha=0.5)
    plt.xlabel('Student token entropy (per-sample)')
    plt.ylabel('Strict acc (per-sample)')
    plt.title('Val entropy vs strict acc')
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()

    print(f'Wrote: {out_stats}')
    print(f'Wrote: {out_corr}')
    print(f'Wrote: {out_plot}')


if __name__ == '__main__':
    main()
