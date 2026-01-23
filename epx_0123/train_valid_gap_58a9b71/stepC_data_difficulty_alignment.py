import argparse
import json
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Paths
import sys
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1219.losses import create_length_mask
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum, estimate_snr
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE


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


def compute_cross_correlation(noisy: torch.Tensor, clean: torch.Tensor, max_lag: int = 4800, step: int = 80):
    noisy = (noisy - noisy.mean()) / (noisy.std() + 1e-8)
    clean = (clean - clean.mean()) / (clean.std() + 1e-8)

    noisy_np = noisy.cpu().numpy()
    clean_np = clean.cpu().numpy()

    correlations = []
    for lag in range(-max_lag, max_lag + 1, step):
        if lag < 0:
            n_aligned = noisy_np[-lag:]
            c_aligned = clean_np[:len(n_aligned)]
        elif lag > 0:
            c_aligned = clean_np[lag:]
            n_aligned = noisy_np[:len(c_aligned)]
        else:
            n_aligned = noisy_np
            c_aligned = clean_np[:len(n_aligned)]

        min_len = min(len(n_aligned), len(c_aligned))
        if min_len > 1000:
            corr = np.corrcoef(n_aligned[:min_len], c_aligned[:min_len])[0, 1]
            correlations.append((lag, corr if not np.isnan(corr) else 0))

    if not correlations:
        return 0, 0.0

    best_lag, max_corr = max(correlations, key=lambda x: x[1])
    return best_lag, max_corr


def per_sample_accuracy(student_codes, teacher_codes, lengths, encoder_stride=320):
    if student_codes.dim() == 3:
        student_codes = student_codes[0]
    if teacher_codes.dim() == 3:
        teacher_codes = teacher_codes[0]

    B, T = student_codes.shape
    max_audio_len = T * encoder_stride
    mask = create_length_mask(lengths, max_audio_len, encoder_stride, device=student_codes.device)

    correct = (student_codes == teacher_codes).float() * mask
    per_correct = correct.sum(dim=1)
    per_total = mask.sum(dim=1)
    per_acc = (per_correct / (per_total + 1e-8)).detach().cpu().numpy().tolist()
    return per_acc


def build_loader(cache_path, batch_size, num_workers, max_samples=None, compute_snr_flag=True):
    dataset = CurriculumDataset(cache_path, max_samples=max_samples, compute_snr=compute_snr_flag)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    return dataset, loader


def compute_snr_hist(dataset):
    if dataset.snr_values is None:
        # fallback: compute on-the-fly
        snrs = []
        for i in range(len(dataset)):
            item = dataset[i]
            snrs.append(estimate_snr(item['noisy_audio'], item['clean_audio']))
        return np.array(snrs)
    return np.array(dataset.snr_values)


def acc_vs_snr(model, loader, device, encoder_stride=320):
    snrs = []
    accs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            lengths = batch['lengths'].to(device)
            output = model(noisy, clean)
            s_codes = output['student_codes']
            t_codes = output['teacher_codes']
            per_acc = per_sample_accuracy(s_codes, t_codes, lengths, encoder_stride)
            accs.extend(per_acc)
            if 'snr' in batch:
                snrs.extend(batch['snr'].cpu().numpy().tolist())
            else:
                # fallback
                for i in range(noisy.shape[0]):
                    snrs.append(estimate_snr(noisy[i].cpu(), clean[i].cpu()))
    return np.array(snrs), np.array(accs)


def acc_vs_lag(model, dataset, device, encoder_stride=320, max_samples=200, max_lag=4800, step=80):
    indices = list(range(len(dataset)))
    if max_samples is not None and len(indices) > max_samples:
        indices = random.sample(indices, max_samples)

    lags = []
    accs = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            noisy = item['noisy_audio'].to(device)
            clean = item['clean_audio'].to(device)
            length_val = item.get('lengths', item.get('length', None))
            if length_val is None:
                raise KeyError("Sample missing 'length'/'lengths' for lag eval")
            lengths = torch.tensor([length_val], device=device)

            if noisy.dim() == 1:
                noisy = noisy.unsqueeze(0)
            if clean.dim() == 1:
                clean = clean.unsqueeze(0)

            output = model(noisy, clean)
            s_codes = output['student_codes']
            t_codes = output['teacher_codes']
            per_acc = per_sample_accuracy(s_codes, t_codes, lengths, encoder_stride)[0]

            lag, corr = compute_cross_correlation(noisy.squeeze(0), clean.squeeze(0), max_lag=max_lag, step=step)
            lags.append(lag / 24.0)  # ms
            accs.append(per_acc)

    return np.array(lags), np.array(accs)


def plot_hist(train_vals, val_vals, bins, xlabel, title, out_path):
    plt.figure(figsize=(6,4))
    plt.hist(train_vals, bins=bins, alpha=0.6, label='train', edgecolor='black')
    plt.hist(val_vals, bins=bins, alpha=0.6, label='val', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_acc_vs(x, acc, bins, xlabel, title, out_path):
    # bin and average
    bin_edges = np.linspace(x.min(), x.max(), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_acc = []
    for i in range(bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if mask.sum() == 0:
            bin_acc.append(np.nan)
        else:
            bin_acc.append(np.mean(acc[mask]))

    plt.figure(figsize=(6,4))
    plt.plot(bin_centers, bin_acc, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel('Strict Acc')
    plt.title(title)
    plt.grid(True, alpha=0.3)
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
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--snr_samples_train', type=int, default=None)
    parser.add_argument('--snr_samples_val', type=int, default=None)
    parser.add_argument('--acc_snr_samples_train', type=int, default=1000)
    parser.add_argument('--acc_snr_samples_val', type=int, default=500)
    parser.add_argument('--lag_samples_train', type=int, default=200)
    parser.add_argument('--lag_samples_val', type=int, default=200)
    parser.add_argument('--lag_max', type=int, default=4800)
    parser.add_argument('--lag_step', type=int, default=80)
    parser.add_argument('--encoder_stride', type=int, default=320)
    parser.add_argument('--bins', type=int, default=20)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)

    # Load model
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

    # SNR hist (train/val)
    train_dataset_snr, _ = build_loader(TRAIN_CACHE, args.batch_size, args.num_workers,
                                        max_samples=args.snr_samples_train, compute_snr_flag=True)
    val_dataset_snr, _ = build_loader(VAL_CACHE, args.batch_size, args.num_workers,
                                      max_samples=args.snr_samples_val, compute_snr_flag=True)

    train_snrs = compute_snr_hist(train_dataset_snr)
    val_snrs = compute_snr_hist(val_dataset_snr)

    plot_hist(train_snrs, val_snrs, bins=args.bins,
              xlabel='SNR (dB)', title='SNR Distribution',
              out_path=out_dir / 'snr_hist_train_vs_val.png')

    # Acc vs SNR
    train_dataset_acc, train_loader_acc = build_loader(TRAIN_CACHE, args.batch_size, args.num_workers,
                                                       max_samples=args.acc_snr_samples_train, compute_snr_flag=True)
    val_dataset_acc, val_loader_acc = build_loader(VAL_CACHE, args.batch_size, args.num_workers,
                                                   max_samples=args.acc_snr_samples_val, compute_snr_flag=True)

    train_snr_acc, train_accs = acc_vs_snr(model, train_loader_acc, device, encoder_stride=args.encoder_stride)
    val_snr_acc, val_accs = acc_vs_snr(model, val_loader_acc, device, encoder_stride=args.encoder_stride)

    plot_acc_vs(train_snr_acc, train_accs, bins=args.bins,
                xlabel='SNR (dB)', title='Train Acc vs SNR',
                out_path=out_dir / 'acc_vs_snr_train.png')
    plot_acc_vs(val_snr_acc, val_accs, bins=args.bins,
                xlabel='SNR (dB)', title='Val Acc vs SNR',
                out_path=out_dir / 'acc_vs_snr_val.png')

    # Lag hist + acc vs lag
    train_dataset_lag = CurriculumDataset(TRAIN_CACHE, max_samples=None, compute_snr=False)
    val_dataset_lag = CurriculumDataset(VAL_CACHE, max_samples=None, compute_snr=False)

    train_lags, train_accs_lag = acc_vs_lag(model, train_dataset_lag, device,
                                            encoder_stride=args.encoder_stride,
                                            max_samples=args.lag_samples_train,
                                            max_lag=args.lag_max, step=args.lag_step)
    val_lags, val_accs_lag = acc_vs_lag(model, val_dataset_lag, device,
                                        encoder_stride=args.encoder_stride,
                                        max_samples=args.lag_samples_val,
                                        max_lag=args.lag_max, step=args.lag_step)

    plot_hist(train_lags, val_lags, bins=args.bins,
              xlabel='Best Lag (ms)', title='Lag Distribution',
              out_path=out_dir / 'lag_hist_train_vs_val.png')

    plot_acc_vs(train_lags, train_accs_lag, bins=args.bins,
                xlabel='Best Lag (ms)', title='Train Acc vs Lag',
                out_path=out_dir / 'acc_vs_lag_train.png')
    plot_acc_vs(val_lags, val_accs_lag, bins=args.bins,
                xlabel='Best Lag (ms)', title='Val Acc vs Lag',
                out_path=out_dir / 'acc_vs_lag_val.png')

    # Save raw stats
    stats = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'device': device,
        'snr_samples_train': args.snr_samples_train,
        'snr_samples_val': args.snr_samples_val,
        'acc_snr_samples_train': args.acc_snr_samples_train,
        'acc_snr_samples_val': args.acc_snr_samples_val,
        'lag_samples_train': args.lag_samples_train,
        'lag_samples_val': args.lag_samples_val,
        'snr_train': {
            'mean': float(np.mean(train_snrs)),
            'std': float(np.std(train_snrs)),
            'min': float(np.min(train_snrs)),
            'max': float(np.max(train_snrs)),
        },
        'snr_val': {
            'mean': float(np.mean(val_snrs)),
            'std': float(np.std(val_snrs)),
            'min': float(np.min(val_snrs)),
            'max': float(np.max(val_snrs)),
        },
        'lag_train': {
            'mean_ms': float(np.mean(train_lags)),
            'std_ms': float(np.std(train_lags)),
            'min_ms': float(np.min(train_lags)),
            'max_ms': float(np.max(train_lags)),
        },
        'lag_val': {
            'mean_ms': float(np.mean(val_lags)),
            'std_ms': float(np.std(val_lags)),
            'min_ms': float(np.min(val_lags)),
            'max_ms': float(np.max(val_lags)),
        },
    }
    with open(out_dir / 'snr_lag_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
