import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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


def _compute_offset_matches(student_codes, teacher_codes, mask, offsets):
    """Return accs and correct counts for each offset.

    accs: (O,B) tensor, corrects: (O,B) tensor, total: (B) tensor
    """
    if student_codes.dim() == 3:
        student_codes = student_codes[0]
    if teacher_codes.dim() == 3:
        teacher_codes = teacher_codes[0]

    B, T = student_codes.shape
    total = mask.sum(dim=1)
    accs = []
    corrects = []

    strict = (student_codes == teacher_codes).float()

    for offset in offsets:
        if offset < 0:
            s_slice = student_codes[:, -offset:]
            t_slice = teacher_codes[:, :T + offset]
            match = (s_slice == t_slice).float()
            match_padded = F.pad(match, (0, -offset), value=0)
        elif offset > 0:
            s_slice = student_codes[:, :T - offset]
            t_slice = teacher_codes[:, offset:]
            match = (s_slice == t_slice).float()
            match_padded = F.pad(match, (offset, 0), value=0)
        else:
            match_padded = strict

        masked = match_padded * mask
        correct = masked.sum(dim=1)
        acc = correct / (total + 1e-8)
        accs.append(acc)
        corrects.append(correct)

    accs = torch.stack(accs, dim=0)
    corrects = torch.stack(corrects, dim=0)
    return accs, corrects, total


def evaluate_split(model, loader, device, max_k=3, progress_every=200, split_name='split'):
    offsets = list(range(-max_k, max_k + 1))
    offset_tensor = torch.tensor(offsets, device=device)

    # accumulators
    metrics = {
        'num_batches': 0,
        'num_samples': 0,
        'num_frames': 0,
        'strict': {
            'acc_batch_mean': 0.0,
            'acc_frame_weighted': 0.0,
        },
        'global_shift': {},
    }

    # per-k accumulators
    k_values = [1, 2, 3]
    k_acc_sum = {k: 0.0 for k in k_values}
    k_correct_sum = {k: 0.0 for k in k_values}
    k_total_sum = {k: 0.0 for k in k_values}

    strict_acc_sum = 0.0
    strict_correct_sum = 0.0
    strict_total_sum = 0.0

    best_offsets = []

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

            accs, corrects, total = _compute_offset_matches(s_codes, t_codes, mask, offsets)

            # strict (offset=0)
            zero_idx = offsets.index(0)
            strict_acc = accs[zero_idx]
            strict_correct = corrects[zero_idx]

            strict_acc_sum += strict_acc.sum().item()
            strict_correct_sum += strict_correct.sum().item()
            strict_total_sum += total.sum().item()

            # global shift for each k
            for k in k_values:
                k_indices = list(range(max_k - k, max_k + k + 1))
                accs_k = accs[k_indices]
                corrects_k = corrects[k_indices]

                best_acc, best_idx = accs_k.max(dim=0)
                best_correct = corrects_k.gather(0, best_idx.unsqueeze(0)).squeeze(0)

                k_acc_sum[k] += best_acc.sum().item()
                k_correct_sum[k] += best_correct.sum().item()
                k_total_sum[k] += total.sum().item()

                if k == max_k:
                    best_offset = offset_tensor[k_indices][best_idx]
                    best_offsets.extend(best_offset.detach().cpu().tolist())

            metrics['num_batches'] += 1
            metrics['num_samples'] += B
            metrics['num_frames'] += int(total.sum().item())

            if progress_every and batch_idx % progress_every == 0:
                print(f'[{split_name}] batch {batch_idx}/{len(loader)}')

    # finalize metrics
    metrics['strict']['acc_batch_mean'] = strict_acc_sum / max(1, metrics['num_samples'])
    metrics['strict']['acc_frame_weighted'] = strict_correct_sum / max(1e-8, strict_total_sum)

    for k in k_values:
        metrics['global_shift'][f'k{k}'] = {
            'acc_batch_mean': k_acc_sum[k] / max(1, metrics['num_samples']),
            'acc_frame_weighted': k_correct_sum[k] / max(1e-8, k_total_sum[k]),
        }

    return metrics, best_offsets


def plot_offset_hist(train_offsets, val_offsets, output_path, max_k=3):
    bins = np.arange(-max_k - 0.5, max_k + 1.5, 1.0)
    plt.figure(figsize=(6, 4))
    plt.hist(train_offsets, bins=bins, alpha=0.6, label='train')
    plt.hist(val_offsets, bins=bins, alpha=0.6, label='val')
    plt.xticks(range(-max_k, max_k + 1))
    plt.xlabel('Best global shift (frames)')
    plt.ylabel('Count')
    plt.title('Best global shift distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--max_k', type=int, default=3)
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

    train_loader = build_loader(TRAIN_CACHE, args.batch_size, args.num_workers, args.max_train_samples)
    val_loader = build_loader(VAL_CACHE, args.batch_size, args.num_workers, args.max_val_samples)

    train_metrics, train_offsets = evaluate_split(model, train_loader, device, max_k=args.max_k, progress_every=args.progress_every, split_name='train')
    val_metrics, val_offsets = evaluate_split(model, val_loader, device, max_k=args.max_k, progress_every=args.progress_every, split_name='val')

    out_dir = Path(__file__).resolve().parent
    out_json = out_dir / 'metrics_global_shift.json'
    out_png = out_dir / 'global_shift_hist_train_vs_val.png'

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'device': device,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'encoder_stride': ENCODER_STRIDE,
        'max_k': args.max_k,
        'splits': {
            'train': train_metrics,
            'val': val_metrics,
        }
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    plot_offset_hist(train_offsets, val_offsets, out_png, max_k=args.max_k)

    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_png}')


if __name__ == '__main__':
    main()
