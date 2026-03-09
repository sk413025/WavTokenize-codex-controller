import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_0112_intermediate.train_v6 import IntermediateSupervisionLossV6
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1219.losses import MaskedCombinedLossV2, create_length_mask
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum

ENCODER_STRIDE = 320
NUM_CODES = 4096


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> str:
    if requested and requested != 'auto':
        return requested
    if torch.cuda.is_available():
        try:
            for idx in range(torch.cuda.device_count()):
                major, _ = torch.cuda.get_device_capability(idx)
                if major >= 6:
                    return f'cuda:{idx}'
        except Exception:
            pass
    return 'cpu'


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


@torch.no_grad()
def evaluate_split(model, loader, device):
    model.eval()
    student_counts = np.zeros(NUM_CODES, dtype=np.int64)
    teacher_counts = np.zeros(NUM_CODES, dtype=np.int64)
    strict_correct = 0.0
    strict_total = 0.0

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
        strict_correct += correct.sum().item()
        strict_total += mask.sum().item()

        s_flat = s_codes.reshape(-1).detach().cpu().numpy()
        t_flat = t_codes.reshape(-1).detach().cpu().numpy()
        student_counts += np.bincount(s_flat, minlength=NUM_CODES)
        teacher_counts += np.bincount(t_flat, minlength=NUM_CODES)

    strict_acc = strict_correct / (strict_total + 1e-8)
    student_stats = counts_to_stats(student_counts)
    teacher_stats = counts_to_stats(teacher_counts)
    kl = kl_div(student_counts, teacher_counts)
    return {
        'strict_acc_frame_weighted': float(strict_acc),
        'student': student_stats,
        'teacher': teacher_stats,
        'kl_student_teacher': kl,
        'num_frames': int(strict_total),
    }


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            return 'model_state_dict'
        if 'lora_state_dict' in ckpt:
            model.load_state_dict(ckpt['lora_state_dict'], strict=False)
            return 'lora_state_dict'
    model.load_state_dict(ckpt, strict=False)
    return 'raw_state_dict'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--tracin_scores', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/tracin_scores.csv')
    parser.add_argument('--tracin_meta', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/tracin_indices.json')
    parser.add_argument('--train_subset', type=int, default=2000)
    parser.add_argument('--drop_top_k', type=int, default=200)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--val_max_samples', type=int, default=500)
    parser.add_argument('--output_dir', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/counterfactual')
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    print(f'Using device: {device}')

    run_dir = Path(args.run_dir)
    checkpoint = args.checkpoint or str(run_dir / 'best_model.pt')

    # Load TracIn scores and pick top-K proponents (positive influence)
    scores = np.genfromtxt(args.tracin_scores, delimiter=',', skip_header=1, dtype=None, encoding=None)
    # Handle aggregate rows only
    train_scores = {}
    for row in scores:
        val_id, train_id, ckpt, loss_type, lr, score = row
        if val_id != 'VAL_AGG':
            continue
        train_scores[int(train_id)] = train_scores.get(int(train_id), 0.0) + float(score)

    # Use candidate list from TracIn meta
    meta = json.loads(Path(args.tracin_meta).read_text())
    train_candidates = meta['train_candidates']
    # Sort by influence desc and drop top-K
    sorted_by_score = sorted(train_candidates, key=lambda i: train_scores.get(i, 0.0), reverse=True)
    drop_set = set(sorted_by_score[:min(args.drop_top_k, len(sorted_by_score))])
    kept = [i for i in train_candidates if i not in drop_set]

    # If too many, subsample to train_subset
    rng = np.random.RandomState(args.seed)
    if len(kept) > args.train_subset:
        kept = rng.choice(kept, size=args.train_subset, replace=False).tolist()

    # Data
    train_dataset = CurriculumDataset(TRAIN_CACHE, compute_snr=False)
    val_dataset = CurriculumDataset(VAL_CACHE, max_samples=args.val_max_samples, compute_snr=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(kept),
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )

    # Model + losses
    config = json.loads((run_dir / 'config.json').read_text())
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=config.get('lora_rank', 256),
        lora_alpha=config.get('lora_alpha', 512),
        lora_dropout=config.get('lora_dropout', 0.2),
        intermediate_indices=[3, 4, 6],
    )
    ckpt_type = load_checkpoint(model, checkpoint)
    model.to(device)

    loss_fn = MaskedCombinedLossV2(
        feature_weight=config.get('feature_weight', 1.0),
        cosine_weight=config.get('cosine_weight', 0.0),
        triplet_weight=config.get('triplet_weight', 1.0),
        triplet_margin=config.get('triplet_margin', 0.2),
        ce_weight=config.get('ce_weight', 0.0),
        encoder_stride=ENCODER_STRIDE,
    )
    inter_loss_fn = IntermediateSupervisionLossV6(
        layer_weights={
            3: config.get('intermediate_L3_weight', 0.3),
            4: config.get('intermediate_L4_weight', 0.5),
            6: config.get('intermediate_L6_weight', 0.5),
        },
        target_scale=config.get('target_scale', 1.0),
    )
    intermediate_weight = config.get('intermediate_weight', 0.5)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=config.get('weight_decay', 0.1),
    )

    # Short training loop
    model.train()
    step = 0
    total_loss = 0.0
    for batch in train_loader:
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        if step % args.grad_accum == 0:
            optimizer.zero_grad()

        output = model(noisy, clean)
        final_loss, _ = loss_fn(
            student_features=output['student_encoder_out'],
            teacher_features=output['teacher_encoder_out'],
            teacher_codes=output['teacher_codes'],
            codebook=output['codebook'],
            lengths=lengths,
        )
        inter_loss, _ = inter_loss_fn(
            student_features=output['student_intermediates'],
            teacher_features=output['teacher_intermediates'],
        )
        loss = final_loss + intermediate_weight * inter_loss
        (loss / args.grad_accum).backward()

        if (step + 1) % args.grad_accum == 0:
            optimizer.step()

        total_loss += loss.item()
        step += 1
        if step >= args.steps:
            break

    # Eval
    train_metrics = evaluate_split(model, train_loader, device)
    val_metrics = evaluate_split(model, val_loader, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / 'summary.json'
    summary_md = out_dir / 'summary.md'

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'checkpoint_type': ckpt_type,
        'train_subset': len(kept),
        'drop_top_k': int(args.drop_top_k),
        'steps': int(args.steps),
        'batch_size': int(args.batch_size),
        'grad_accum': int(args.grad_accum),
        'lr': float(args.lr),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    baseline = json.loads(Path('exp_0125/tracin_token_collapse_589e6d/metrics_overview.json').read_text())
    lines = [
        "# Counterfactual short-run summary",
        f"- train subset: {len(kept)} (drop_top_k={args.drop_top_k})",
        f"- steps: {args.steps}, batch_size: {args.batch_size}, grad_accum: {args.grad_accum}",
        f"- val strict acc: {val_metrics['strict_acc_frame_weighted']:.6f} (baseline {baseline['splits']['val']['strict_acc_frame_weighted']:.6f})",
        f"- val entropy: {val_metrics['student']['entropy']:.3f} (baseline {baseline['splits']['val']['student']['entropy']:.3f})",
        f"- val top_k_mass: {val_metrics['student']['top_k_mass']:.3f} (baseline {baseline['splits']['val']['student']['top_k_mass']:.3f})",
        f"- val KL: {val_metrics['kl_student_teacher']:.3f} (baseline {baseline['splits']['val']['kl_student_teacher']:.3f})",
    ]
    summary_md.write_text("\n".join(lines))

    print(f'Wrote: {summary_json}')
    print(f'Wrote: {summary_md}')


if __name__ == '__main__':
    main()
