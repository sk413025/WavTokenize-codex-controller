import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def counts_to_stats(counts, top_k=10):
    total = counts.sum()
    if total == 0:
        return {
            'total': 0,
            'unique': 0,
            'entropy': 0.0,
            'top_k_mass': 0.0,
        }
    probs = counts / total
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0] + 1e-12)).sum())
    top_k_mass = float(np.sort(counts)[-top_k:].sum() / total)
    unique = int((counts > 0).sum())
    return {
        'total': int(total),
        'unique': unique,
        'entropy': entropy,
        'top_k_mass': top_k_mass,
    }


def kl_div(p_counts, q_counts):
    p = p_counts / (p_counts.sum() + 1e-12)
    q = q_counts / (q_counts.sum() + 1e-12)
    p = p + 1e-12
    q = q + 1e-12
    return float((p * np.log(p / q)).sum())


@torch.no_grad()
def collect_counts(model, loader, device, num_codes, encoder_stride=320, progress_every=50, split_name='split'):
    student_counts = np.zeros(num_codes, dtype=np.int64)
    teacher_counts = np.zeros(num_codes, dtype=np.int64)

    for idx, batch in enumerate(loader, start=1):
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        output = model(noisy, clean)
        s_codes = output['student_codes']
        t_codes = output['teacher_codes']
        if s_codes.dim() == 3:
            s_codes = s_codes[0]
        if t_codes.dim() == 3:
            t_codes = t_codes[0]

        B, T = s_codes.shape
        mask = create_length_mask(lengths, T * encoder_stride, encoder_stride, device=s_codes.device).bool()

        s_flat = s_codes[mask].view(-1)
        t_flat = t_codes[mask].view(-1)

        s_count = torch.bincount(s_flat, minlength=num_codes).cpu().numpy()
        t_count = torch.bincount(t_flat, minlength=num_codes).cpu().numpy()

        student_counts += s_count
        teacher_counts += t_count

        if progress_every and idx % progress_every == 0:
            print(f"[{split_name}] {idx} batches processed...")

    return student_counts, teacher_counts


def plot_rank_frequency(train_counts_s, train_counts_t, val_counts_s, val_counts_t, out_path):
    def norm_sort(counts):
        total = counts.sum()
        if total == 0:
            return np.array([])
        freq = counts / total
        return np.sort(freq)[::-1]

    train_s = norm_sort(train_counts_s)
    train_t = norm_sort(train_counts_t)
    val_s = norm_sort(val_counts_s)
    val_t = norm_sort(val_counts_t)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(train_s, label='student')
    ax.plot(train_t, label='teacher')
    ax.set_title('Train token rank-frequency')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.legend()

    ax = axes[1]
    ax.plot(val_s, label='student')
    ax.plot(val_t, label='teacher')
    ax.set_title('Val token rank-frequency')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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
    parser.add_argument('--max_train_samples', type=int, default=2000)
    parser.add_argument('--max_val_samples', type=int, default=500)
    parser.add_argument('--encoder_stride', type=int, default=320)
    parser.add_argument('--top_k', type=int, default=10)
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

    num_codes = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook.shape[0]

    train_loader = build_loader(TRAIN_CACHE, args.batch_size, args.num_workers, args.max_train_samples)
    val_loader = build_loader(VAL_CACHE, args.batch_size, args.num_workers, args.max_val_samples)

    train_s, train_t = collect_counts(
        model, train_loader, device, num_codes,
        encoder_stride=args.encoder_stride,
        progress_every=args.progress_every,
        split_name='train'
    )
    val_s, val_t = collect_counts(
        model, val_loader, device, num_codes,
        encoder_stride=args.encoder_stride,
        progress_every=args.progress_every,
        split_name='val'
    )

    stats = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'device': device,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'student': {
            'train': counts_to_stats(train_s, top_k=args.top_k),
            'val': counts_to_stats(val_s, top_k=args.top_k),
        },
        'teacher': {
            'train': counts_to_stats(train_t, top_k=args.top_k),
            'val': counts_to_stats(val_t, top_k=args.top_k),
        },
        'kl_divergence': {
            'train': kl_div(train_s, train_t),
            'val': kl_div(val_s, val_t),
        }
    }

    with open(out_dir / 'token_usage_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    plot_rank_frequency(train_s, train_t, val_s, val_t,
                        out_path=out_dir / 'token_usage_train_vs_val.png')


if __name__ == '__main__':
    main()
