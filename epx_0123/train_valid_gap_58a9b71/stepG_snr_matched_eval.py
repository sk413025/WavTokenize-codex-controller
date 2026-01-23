import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
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


def build_dataset(cache_path, compute_snr=True):
    return CurriculumDataset(cache_path, max_samples=None, compute_snr=compute_snr)


def build_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )


def strict_eval(model, loader, device, progress_every=200):
    model.eval()
    total_correct = 0.0
    total_frames = 0.0
    batch_acc_sum = 0.0
    n_samples = 0

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
            acc_per_sample = correct_per_sample / (total_per_sample + 1e-8)

            total_correct += correct_per_sample.sum().item()
            total_frames += total_per_sample.sum().item()
            batch_acc_sum += acc_per_sample.sum().item()
            n_samples += B

            if progress_every and batch_idx % progress_every == 0:
                print(f'[eval] batch {batch_idx}/{len(loader)}')

    return {
        'acc_batch_mean': batch_acc_sum / max(1, n_samples),
        'acc_frame_weighted': total_correct / max(1e-8, total_frames),
        'num_samples': n_samples,
        'num_frames': int(total_frames),
    }


def match_train_to_val(train_snr, val_snr, bin_edges, seed=42):
    rng = np.random.RandomState(seed)
    train_indices = np.arange(len(train_snr))
    val_indices = np.arange(len(val_snr))

    matched_train_indices = []
    bin_stats = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        train_bin = train_indices[(train_snr >= lo) & (train_snr < hi)]
        val_bin = val_indices[(val_snr >= lo) & (val_snr < hi)]

        take = min(len(train_bin), len(val_bin))
        if take > 0:
            sampled = rng.choice(train_bin, size=take, replace=False)
            matched_train_indices.extend(sampled.tolist())

        bin_stats.append({
            'bin': [float(lo), float(hi)],
            'train_count': int(len(train_bin)),
            'val_count': int(len(val_bin)),
            'matched_train': int(take),
        })

    matched_train_indices = np.array(matched_train_indices, dtype=int)
    return matched_train_indices, bin_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--bin_width', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress_every', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    checkpoint = args.checkpoint or str(run_dir / 'best_model.pt')

    device = select_device(args.device)
    print(f'Using device: {device}')

    # build model
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

    # compute SNR arrays
    train_dataset = build_dataset(TRAIN_CACHE, compute_snr=True)
    val_dataset = build_dataset(VAL_CACHE, compute_snr=True)
    train_snr = train_dataset.snr_values
    val_snr = val_dataset.snr_values

    snr_min = float(min(train_snr.min(), val_snr.min()))
    snr_max = float(max(train_snr.max(), val_snr.max()))
    bin_edges = np.arange(np.floor(snr_min), np.ceil(snr_max) + args.bin_width, args.bin_width)

    matched_train_indices, bin_stats = match_train_to_val(train_snr, val_snr, bin_edges, seed=args.seed)

    # evaluate strict acc on matched train subset and full val
    matched_train_ds = Subset(train_dataset, matched_train_indices.tolist())
    train_loader = build_loader(matched_train_ds, args.batch_size, args.num_workers)
    val_loader = build_loader(val_dataset, args.batch_size, args.num_workers)

    train_metrics = strict_eval(model, train_loader, device, progress_every=args.progress_every)
    val_metrics = strict_eval(model, val_loader, device, progress_every=args.progress_every)

    out_dir = Path(__file__).resolve().parent
    out_json = out_dir / 'snr_matched_stats.json'
    out_md = out_dir / 'snr_matched_eval.md'

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'device': device,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'bin_width': args.bin_width,
        'bin_edges': [float(x) for x in bin_edges.tolist()],
        'train_len': len(train_dataset),
        'val_len': len(val_dataset),
        'matched_train_len': int(len(matched_train_indices)),
        'bin_stats': bin_stats,
        'train_matched_strict': train_metrics,
        'val_strict': val_metrics,
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = [
        '# SNR-matched evaluation',
        '',
        f'- timestamp: {payload["timestamp"]}',
        f'- seed: {args.seed}',
        f'- bin_width: {args.bin_width} dB',
        f'- train_len: {len(train_dataset)}',
        f'- val_len: {len(val_dataset)}',
        f'- matched_train_len: {len(matched_train_indices)}',
        '',
        '## Strict accuracy',
        f'- train_matched acc_batch_mean: {train_metrics["acc_batch_mean"]:.6f}',
        f'- train_matched acc_frame_weighted: {train_metrics["acc_frame_weighted"]:.6f}',
        f'- val acc_batch_mean: {val_metrics["acc_batch_mean"]:.6f}',
        f'- val acc_frame_weighted: {val_metrics["acc_frame_weighted"]:.6f}',
        '',
        '## Bin stats',
    ]

    for b in bin_stats:
        lines.append(
            f'- bin {b["bin"][0]:.2f}–{b["bin"][1]:.2f}: train={b["train_count"]}, val={b["val_count"]}, matched_train={b["matched_train"]}'
        )

    out_md.write_text('\n'.join(lines) + '\n')

    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_md}')


if __name__ == '__main__':
    main()
