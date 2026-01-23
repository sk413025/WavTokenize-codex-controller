import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

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
def evaluate_split(model, loader, device, split_name, progress_every=200):
    model.eval()

    student_counts = np.zeros(NUM_CODES, dtype=np.int64)
    teacher_counts = np.zeros(NUM_CODES, dtype=np.int64)

    strict_correct = 0.0
    strict_total = 0.0

    num_batches = 0
    num_samples = 0

    for batch_idx, batch in enumerate(loader, start=1):
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

        num_batches += 1
        num_samples += B

        if progress_every and batch_idx % progress_every == 0:
            print(f'[{split_name}] batch {batch_idx}/{len(loader)}')

    strict_acc = strict_correct / (strict_total + 1e-8)

    student_stats = counts_to_stats(student_counts)
    teacher_stats = counts_to_stats(teacher_counts)
    kl = kl_div(student_counts, teacher_counts)

    return {
        'num_batches': num_batches,
        'num_samples': num_samples,
        'num_frames': int(strict_total),
        'strict_acc_frame_weighted': float(strict_acc),
        'student': student_stats,
        'teacher': teacher_stats,
        'kl_student_teacher': kl,
    }


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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress_every', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    checkpoint = args.checkpoint or str(run_dir / 'best_model.pt')

    device = select_device(args.device)
    print(f'Using device: {device}')

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

    train_metrics = evaluate_split(model, train_loader, device, 'train', progress_every=args.progress_every)
    val_metrics = evaluate_split(model, val_loader, device, 'val', progress_every=args.progress_every)

    out_dir = Path(__file__).resolve().parent
    out_json = out_dir / 'metrics_collapse_overview.json'

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'commit': '27e564a29c52a11cca5fb1290f694c6808a4007e',
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'device': device,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'splits': {
            'train': train_metrics,
            'val': val_metrics,
        }
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f'Wrote: {out_json}')


if __name__ == '__main__':
    main()
