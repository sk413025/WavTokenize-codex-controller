import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_0112_intermediate.train_v5 import IntermediateSupervisionLossV5
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy, create_length_mask
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum
from exp_1226.losses_diversity import CodeEntropyLoss


ENCODER_STRIDE = 320
NUM_CODES = 4096


class CodeDistKLLoss(torch.nn.Module):
    """
    KL(student || teacher) over code distribution.

    - student: soft assignment from student features to codebook (masked)
    - teacher: hard counts from teacher codes (masked)
    """

    def __init__(self, num_codes: int = 4096, temperature: float = 1.0):
        super().__init__()
        self.num_codes = num_codes
        self.temperature = temperature

    def forward(self, student_features, teacher_codes, codebook, lengths, encoder_stride=320):
        B, D, T = student_features.shape
        max_audio_len = T * encoder_stride
        mask = create_length_mask(lengths, max_audio_len, encoder_stride, device=student_features.device)

        # student soft assignment
        z = student_features.permute(0, 2, 1).reshape(-1, D)
        distances = torch.cdist(z, codebook)
        logits = -distances / self.temperature
        probs = F.softmax(logits, dim=-1)
        mask_flat = mask.reshape(-1).unsqueeze(-1)
        probs_masked = probs * mask_flat
        student_dist = probs_masked.sum(dim=0)
        student_dist = student_dist / (student_dist.sum() + 1e-8)

        # teacher hard distribution
        t_codes = teacher_codes
        if t_codes.dim() == 3:
            t_codes = t_codes[0]
        t_flat = t_codes.reshape(-1).long()
        mask_flat_1d = mask.reshape(-1)
        teacher_counts = torch.bincount(t_flat, weights=mask_flat_1d, minlength=self.num_codes)
        teacher_dist = teacher_counts / (teacher_counts.sum() + 1e-8)

        # KL(student || teacher)
        eps = 1e-8
        kl = (student_dist * (student_dist + eps).log() - student_dist * (teacher_dist + eps).log()).sum()
        return kl


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


def build_loader(cache_path, batch_size, num_workers, shuffle, max_samples=None):
    dataset = CurriculumDataset(cache_path, max_samples=max_samples, compute_snr=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    return loader


def counts_to_stats(counts, top_k=10):
    total = counts.sum()
    if total == 0:
        return {'total': 0, 'unique': 0, 'entropy': 0.0, 'top_k_mass': 0.0}
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


def evaluate_val(model, loader, device):
    model.eval()
    student_counts = np.zeros(NUM_CODES, dtype=np.int64)
    teacher_counts = np.zeros(NUM_CODES, dtype=np.int64)

    strict_batch_sum = 0.0
    strict_correct = 0.0
    strict_total = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
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
            max_audio_len = T * ENCODER_STRIDE
            mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=device)

            correct = (s_codes == t_codes).float() * mask
            correct_per_sample = correct.sum(dim=1)
            total_per_sample = mask.sum(dim=1)
            acc_per_sample = correct_per_sample / (total_per_sample + 1e-8)

            strict_batch_sum += acc_per_sample.sum().item()
            strict_correct += correct_per_sample.sum().item()
            strict_total += total_per_sample.sum().item()
            num_samples += B

            # counts
            s_flat = s_codes.reshape(-1).detach().cpu().numpy()
            t_flat = t_codes.reshape(-1).detach().cpu().numpy()
            student_counts += np.bincount(s_flat, minlength=NUM_CODES)
            teacher_counts += np.bincount(t_flat, minlength=NUM_CODES)

    strict = {
        'acc_batch_mean': strict_batch_sum / max(1, num_samples),
        'acc_frame_weighted': strict_correct / max(1e-8, strict_total),
        'num_samples': int(num_samples),
        'num_frames': int(strict_total),
    }

    student_stats = counts_to_stats(student_counts)
    teacher_stats = counts_to_stats(teacher_counts)
    kl = kl_div(student_counts, teacher_counts)

    return {
        'strict': strict,
        'student': student_stats,
        'teacher': teacher_stats,
        'kl_student_teacher': kl,
    }


def train_one_run(args, reg_weight, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)

    # load config for hyperparams
    cfg_path = Path(args.run_dir) / 'config.json'
    cfg = json.loads(cfg_path.read_text())

    device = select_device(args.device)
    print(f'Using device: {device}')

    # model
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=cfg.get('lora_rank', 256),
        lora_alpha=cfg.get('lora_alpha', 512),
        lora_dropout=cfg.get('lora_dropout', 0.2),
    )

    ckpt_path = args.checkpoint or str(Path(args.run_dir) / 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)

    # losses
    base_loss_fn = MaskedCombinedLossV2(
        feature_weight=cfg.get('feature_weight', 1.0),
        cosine_weight=cfg.get('cosine_weight', 0.0),
        triplet_weight=cfg.get('triplet_weight', 1.0),
        triplet_margin=cfg.get('triplet_margin', 0.2),
        ce_weight=cfg.get('ce_weight', 0.0),
        encoder_stride=ENCODER_STRIDE,
    )
    interm_loss_fn = IntermediateSupervisionLossV5(
        layer_weights={
            3: cfg.get('intermediate_L3_weight', 0.3),
            4: cfg.get('intermediate_L4_weight', 1.0),
            6: cfg.get('intermediate_L6_weight', 0.5),
        },
        target_scale=cfg.get('target_scale', 1.0),
    )
    entropy_loss_fn = CodeEntropyLoss(num_codes=NUM_CODES, temperature=args.reg_temperature)
    kl_loss_fn = CodeDistKLLoss(num_codes=NUM_CODES, temperature=args.reg_temperature)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get('lr', 1e-4),
        weight_decay=cfg.get('weight_decay', 0.1),
    )

    scaler = GradScaler(enabled=args.use_amp)

    train_loader = build_loader(TRAIN_CACHE, args.batch_size, args.num_workers, shuffle=True,
                                max_samples=args.max_train_samples)
    val_loader = build_loader(VAL_CACHE, args.batch_size, args.num_workers, shuffle=False,
                              max_samples=args.max_val_samples)

    # training loop (fixed optimizer steps)
    opt_steps = 0
    micro_steps = 0
    train_metrics = {
        'total_loss': 0.0,
        'feature_loss': 0.0,
        'triplet_loss': 0.0,
        'intermediate_loss': 0.0,
        'entropy_loss': 0.0,
        'masked_acc': 0.0,
        'num_steps': 0,
        'num_micro_steps': 0,
    }

    model.train()
    for epoch in range(1, args.num_epochs + 1):
        for batch in train_loader:
            if args.max_steps is not None and opt_steps >= args.max_steps:
                break

            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            lengths = batch['lengths'].to(device)

            if micro_steps % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            with autocast(enabled=args.use_amp):
                output = model(noisy, clean)
                base_loss, base_info = base_loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                )
                inter_loss, _ = interm_loss_fn(
                    output['student_intermediates'],
                    output['teacher_intermediates'],
                    layer_scale=1.0,
                )
                if args.reg_type == 'entropy':
                    reg_loss = entropy_loss_fn(
                        output['student_encoder_out'],
                        output['codebook'],
                        lengths=lengths,
                        encoder_stride=ENCODER_STRIDE,
                    )
                elif args.reg_type == 'kl':
                    reg_loss = kl_loss_fn(
                        output['student_encoder_out'],
                        output['teacher_codes'],
                        output['codebook'],
                        lengths=lengths,
                        encoder_stride=ENCODER_STRIDE,
                    )
                else:
                    raise ValueError(f"Unknown reg_type: {args.reg_type}")

                total_loss = base_loss + cfg.get('intermediate_weight', 0.5) * inter_loss + reg_weight * reg_loss

            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (micro_steps + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                opt_steps += 1

            # metrics
            s_codes = output['student_codes']
            t_codes = output['teacher_codes']
            if s_codes.dim() == 3:
                s_codes = s_codes[0]
            if t_codes.dim() == 3:
                t_codes = t_codes[0]
            masked_acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, ENCODER_STRIDE)

            train_metrics['total_loss'] += float(total_loss.item())
            train_metrics['feature_loss'] += float(base_info.get('feature_loss', 0.0))
            train_metrics['triplet_loss'] += float(base_info.get('triplet_loss', 0.0))
            train_metrics['intermediate_loss'] += float(inter_loss.item())
            train_metrics['entropy_loss'] += float(reg_loss.item())
            train_metrics['masked_acc'] += float(masked_acc)
            train_metrics['num_steps'] += 1
            train_metrics['num_micro_steps'] += 1

            micro_steps += 1

            if args.max_steps is not None and opt_steps >= args.max_steps:
                break

        if args.max_steps is not None and opt_steps >= args.max_steps:
            break

    # average metrics
    if train_metrics['num_steps'] > 0:
        for k in ['total_loss', 'feature_loss', 'triplet_loss', 'intermediate_loss', 'entropy_loss', 'masked_acc']:
            train_metrics[k] /= train_metrics['num_steps']

    # evaluation
    val_metrics = evaluate_val(model, val_loader, device)

    # save
    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint_init': ckpt_path,
        'reg_type': args.reg_type,
        'reg_weight': reg_weight,
        'reg_temperature': args.reg_temperature,
        'num_epochs': args.num_epochs,
        'max_steps': args.max_steps,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }

    (run_dir / 'metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--entropy_weights', type=str, default='0.0,0.005,0.01')
    parser.add_argument('--reg_type', type=str, default='entropy', choices=['entropy', 'kl'])
    parser.add_argument('--reg_temperature', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--output_root', type=str, default='epx_0123/train_valid_gap_58a9b71/ablation_anticollapse')
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    weights = [float(x) for x in args.entropy_weights.split(',')]

    summary = []
    for w in weights:
        run_dir = output_root / f'lambda_{w:.3f}'
        payload = train_one_run(args, w, run_dir)
        summary.append(payload)

    # write summary markdown
    lines = [
        '# Anti-collapse ablation summary',
        '',
        f'- timestamp: {datetime.now().isoformat(timespec="seconds")}',
        f'- run_dir: {args.run_dir}',
        f'- reg_type: {args.reg_type}',
        f'- reg_temperature: {args.reg_temperature}',
        f'- max_steps: {args.max_steps}',
        f'- num_epochs: {args.num_epochs}',
        f'- max_train_samples: {args.max_train_samples}',
        f'- max_val_samples: {args.max_val_samples}',
        '',
        '## Results',
        '| lambda | val_strict_fw | val_entropy | val_topk_mass | val_KL(stu||tea) |',
        '|---:|---:|---:|---:|---:|',
    ]

    for p in summary:
        val = p['val_metrics']
        lines.append(
            f"| {p['reg_weight']:.3f} | {val['strict']['acc_frame_weighted']:.6f} | "
            f"{val['student']['entropy']:.6f} | {val['student']['top_k_mass']:.6f} | {val['kl_student_teacher']:.6f} |"
        )

    out_md = Path('epx_0123/train_valid_gap_58a9b71/ablation_anticollapse_summary.md')
    out_md.write_text('\n'.join(lines) + '\n')

    # save summary json
    out_json = Path('epx_0123/train_valid_gap_58a9b71/ablation_anticollapse_summary.json')
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f'Wrote: {out_md}')
    print(f'Wrote: {out_json}')


if __name__ == '__main__':
    main()
