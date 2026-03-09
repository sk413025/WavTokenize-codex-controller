import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1219.losses import compute_masked_accuracy, create_length_mask
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE


def compute_tolerant_correct(student_codes, teacher_codes, lengths, tolerance, encoder_stride=320):
    if student_codes.dim() == 3:
        student_codes = student_codes[0]
    if teacher_codes.dim() == 3:
        teacher_codes = teacher_codes[0]

    B, T = student_codes.shape
    max_audio_len = T * encoder_stride
    mask = create_length_mask(lengths, max_audio_len, encoder_stride, device=student_codes.device)

    strict_correct = (student_codes == teacher_codes).float()
    tolerant_correct = torch.zeros_like(strict_correct)

    for offset in range(-tolerance, tolerance + 1):
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
            match_padded = strict_correct

        tolerant_correct = torch.maximum(tolerant_correct, match_padded)

    masked_correct = tolerant_correct * mask
    num_correct = masked_correct.sum().item()
    num_total = mask.sum().item()
    acc = num_correct / (num_total + 1e-8)
    return acc, int(num_correct), int(num_total)


def build_dataloader(cache_path, batch_size, num_workers, max_samples=None):
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


def select_device(requested: str) -> str:
    if requested and requested != 'auto':
        return requested

    if torch.cuda.is_available():
        # Prefer a supported CUDA device (capability >= 7.0)
        try:
            for idx in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(idx)
                if major >= 7:
                    return f'cuda:{idx}'
        except Exception:
            pass
    return 'cpu'


def evaluate_split(model, loader, device, encoder_stride=320, tolerances=(1, 2, 3), progress_every=50, split_name='split'):
    model.eval()
    split_metrics = {
        'num_batches': 0,
        'num_frames': 0,
        'strict': {
            'acc_batch_mean': 0.0,
            'acc_frame_weighted': 0.0,
        },
        'tolerant': {},
    }

    # init tolerant accumulators
    tolerant_batch_accs = {k: [] for k in tolerances}
    tolerant_correct = {k: 0 for k in tolerances}
    tolerant_total = {k: 0 for k in tolerances}

    strict_batch_accs = []
    strict_correct_total = 0
    strict_total_frames = 0

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

            strict_acc, correct, total_frames = compute_masked_accuracy(
                s_codes, t_codes, lengths, encoder_stride
            )
            strict_batch_accs.append(strict_acc)
            strict_correct_total += correct
            strict_total_frames += total_frames

            for k in tolerances:
                tol_acc, tol_correct, tol_total = compute_tolerant_correct(
                    s_codes, t_codes, lengths, k, encoder_stride
                )
                tolerant_batch_accs[k].append(tol_acc)
                tolerant_correct[k] += tol_correct
                tolerant_total[k] += tol_total

            split_metrics['num_batches'] += 1

            if progress_every and batch_idx % progress_every == 0:
                print(f"[{split_name}] {batch_idx} batches processed...")

    # finalize strict
    split_metrics['num_frames'] = strict_total_frames
    split_metrics['strict']['acc_batch_mean'] = float(sum(strict_batch_accs) / max(1, len(strict_batch_accs)))
    split_metrics['strict']['acc_frame_weighted'] = float(strict_correct_total / (strict_total_frames + 1e-8))

    # finalize tolerant
    for k in tolerances:
        split_metrics['tolerant'][f'k{k}'] = {
            'acc_batch_mean': float(sum(tolerant_batch_accs[k]) / max(1, len(tolerant_batch_accs[k]))),
            'acc_frame_weighted': float(tolerant_correct[k] / (tolerant_total[k] + 1e-8)),
        }

    return split_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda|cuda:N')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--encoder_stride', type=int, default=320)
    parser.add_argument('--tolerances', type=str, default='1,2,3')
    parser.add_argument('--progress_every', type=int, default=50)
    parser.add_argument('--output_dir', type=str,
                        default='exp_0112_intermediate/analysis/train_valid_gap_58a9b71')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load config
    config_path = run_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    batch_size = args.batch_size or int(config.get('batch_size', 8))
    device = select_device(args.device)

    # build model
    model = TeacherStudentIntermediate(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=config.get('lora_rank', 256),
        lora_alpha=config.get('lora_alpha', 512),
        lora_dropout=config.get('lora_dropout', 0.2),
        intermediate_indices=[3, 4, 6],
        device=device,
    )

    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # dataloaders
    train_loader = build_dataloader(TRAIN_CACHE, batch_size, args.num_workers, args.max_train_samples)
    val_loader = build_dataloader(VAL_CACHE, batch_size, args.num_workers, args.max_val_samples)

    tolerances = tuple(int(x) for x in args.tolerances.split(',') if x.strip())

    # eval
    print(f"Using device: {device}")

    train_metrics = evaluate_split(
        model, train_loader, device,
        encoder_stride=args.encoder_stride,
        tolerances=tolerances,
        progress_every=args.progress_every,
        split_name='train'
    )
    val_metrics = evaluate_split(
        model, val_loader, device,
        encoder_stride=args.encoder_stride,
        tolerances=tolerances,
        progress_every=args.progress_every,
        split_name='val'
    )

    summary = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': str(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch', None),
        'device': device,
        'batch_size': batch_size,
        'num_workers': args.num_workers,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'encoder_stride': args.encoder_stride,
        'tolerances': list(tolerances),
        'splits': {
            'train': train_metrics,
            'val': val_metrics,
        }
    }

    # write summary
    with open(out_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # sanity check vs history
    history_path = run_dir / 'history.json'
    sanity_lines = []
    sanity_lines.append('# Sanity Check: Offline Eval vs Training Log\n')
    sanity_lines.append(f"Run dir: {run_dir}\n")
    sanity_lines.append(f"Checkpoint: {checkpoint_path}\n")
    sanity_lines.append(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}\n")
    sanity_lines.append(f"Eval mode: model.eval() + torch.no_grad()\n")
    train_desc = "full dataset" if args.max_train_samples is None else f"max_samples={args.max_train_samples}"
    val_desc = "full dataset" if args.max_val_samples is None else f"max_samples={args.max_val_samples}"
    sanity_lines.append(f"Train loader: CurriculumDataset + shuffle=False + {train_desc}\n")
    sanity_lines.append(f"Val loader: CurriculumDataset + shuffle=False + {val_desc}\n")

    if history_path.exists():
        with open(history_path) as f:
            hist = json.load(f)
        ep = checkpoint.get('epoch', None)
        if ep is not None and ep >= 1 and ep <= len(hist['train_masked_acc']):
            train_log = hist['train_masked_acc'][ep - 1]
            val_log = hist['val_masked_acc'][ep - 1]
            sanity_lines.append('\n## Logged metrics at checkpoint epoch\n')
            sanity_lines.append(f"- train_masked_acc (log, batch-mean): {train_log:.6f}\n")
            sanity_lines.append(f"- val_masked_acc (log, batch-mean): {val_log:.6f}\n")

            train_off = train_metrics['strict']['acc_batch_mean']
            val_off = val_metrics['strict']['acc_batch_mean']
            sanity_lines.append('\n## Offline eval metrics (strict, batch-mean)\n')
            sanity_lines.append(f"- train strict acc_batch_mean: {train_off:.6f}\n")
            sanity_lines.append(f"- val strict acc_batch_mean: {val_off:.6f}\n")
            sanity_lines.append('\n## Differences (offline - log)\n')
            sanity_lines.append(f"- train diff: {train_off - train_log:+.6f}\n")
            sanity_lines.append(f"- val diff: {val_off - val_log:+.6f}\n")
        else:
            sanity_lines.append('\nNo matching epoch found in history.json.\n')
    else:
        sanity_lines.append('\nhistory.json not found; cannot compare.\n')

    with open(out_dir / 'sanity_check.md', 'w') as f:
        f.write(''.join(sanity_lines))


if __name__ == '__main__':
    main()
