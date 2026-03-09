import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
                major, _ = torch.cuda.get_device_capability(idx)
                if major >= 6:
                    return f'cuda:{idx}'
        except Exception:
            pass
    return 'cpu'


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item = dict(item)
        item['__index__'] = idx
        return item


def collate_with_indices(batch):
    indices = [item['__index__'] for item in batch]
    clean_batch = [{k: v for k, v in item.items() if k != '__index__'} for item in batch]
    out = collate_fn_curriculum(clean_batch)
    out['indices'] = torch.tensor(indices, dtype=torch.long)
    return out


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


def estimate_snr(noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> float:
    noisy = noisy_audio.squeeze()
    clean = clean_audio.squeeze()
    min_len = min(len(noisy), len(clean))
    noisy = noisy[:min_len]
    clean = clean[:min_len]
    signal_power = (clean ** 2).mean()
    noise = noisy - clean
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return 100.0
    snr = 10 * torch.log10(signal_power / noise_power + 1e-10)
    return float(snr.item())


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


@torch.no_grad()
def evaluate_split(model, loader, device, split_name, progress_every=200, collect_per_sample=False, dataset=None):
    model.eval()
    student_counts = np.zeros(NUM_CODES, dtype=np.int64)
    teacher_counts = np.zeros(NUM_CODES, dtype=np.int64)
    strict_correct = 0.0
    strict_total = 0.0
    num_batches = 0
    num_samples = 0
    per_sample = []

    for batch_idx, batch in enumerate(loader, start=1):
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)
        indices = batch.get('indices', None)

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

        if collect_per_sample and indices is not None:
            indices = indices.cpu().numpy().tolist()
            acc_per = (correct.sum(dim=1) / (mask.sum(dim=1) + 1e-8)).detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            s_codes_np = s_codes.detach().cpu().numpy()

            for i in range(B):
                idx = indices[i]
                valid = mask_np[i] > 0
                if valid.sum() == 0:
                    entropy = 0.0
                    top_k_mass = 0.0
                    unique = 0
                else:
                    codes = s_codes_np[i][valid]
                    counts = np.bincount(codes, minlength=NUM_CODES)
                    stats = counts_to_stats(counts)
                    entropy = stats['entropy']
                    top_k_mass = stats['top_k_mass']
                    unique = stats['unique']

                meta = {}
                if dataset is not None and idx < len(dataset.samples):
                    sample = dataset.samples[idx]
                    meta = {
                        'noisy_path': sample.get('noisy_path', None),
                        'clean_path': sample.get('clean_path', None),
                    }

                snr = estimate_snr(noisy[i].detach().cpu(), clean[i].detach().cpu())
                energy = float((noisy[i].detach().cpu() ** 2).mean().item())

                per_sample.append({
                    'index': int(idx),
                    'strict_acc': float(acc_per[i]),
                    'entropy': float(entropy),
                    'top_k_mass': float(top_k_mass),
                    'unique': int(unique),
                    'snr_db': float(snr),
                    'noisy_energy': float(energy),
                    **meta,
                })

        num_batches += 1
        num_samples += B

        if progress_every and batch_idx % progress_every == 0:
            print(f'[{split_name}] batch {batch_idx}/{len(loader)}')

    strict_acc = strict_correct / (strict_total + 1e-8)
    student_stats = counts_to_stats(student_counts)
    teacher_stats = counts_to_stats(teacher_counts)
    kl = kl_div(student_counts, teacher_counts)

    metrics = {
        'num_batches': num_batches,
        'num_samples': num_samples,
        'num_frames': int(strict_total),
        'strict_acc_frame_weighted': float(strict_acc),
        'student': student_stats,
        'teacher': teacher_stats,
        'kl_student_teacher': kl,
    }

    return metrics, per_sample


@torch.no_grad()
def compute_margin_stats(model, loader, device, split_name, progress_every=200):
    codebook = model.codebook.to(device)
    margins = []

    model.eval()
    for batch_idx, batch in enumerate(loader, start=1):
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        output = model(noisy, clean)
        s_out = output['student_encoder_out']  # (B, D, T)
        s_codes = output['student_codes']
        t_codes = output['teacher_codes']

        if s_codes.dim() == 3:
            s_codes = s_codes[0]
        if t_codes.dim() == 3:
            t_codes = t_codes[0]

        B, D, T = s_out.shape
        max_audio_len = T * ENCODER_STRIDE
        mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=device)

        z = s_out.permute(0, 2, 1).reshape(-1, D)
        dists = torch.cdist(z, codebook)
        d1, _ = torch.topk(dists, k=2, largest=False, dim=1)
        margin = (d1[:, 1] - d1[:, 0]).detach().cpu().numpy()

        mask_flat = mask.reshape(-1).detach().cpu().numpy()
        margin = margin[mask_flat > 0]
        margins.extend(margin.tolist())

        if progress_every and batch_idx % progress_every == 0:
            print(f'[{split_name}] margin batch {batch_idx}/{len(loader)}')

    margins = np.array(margins)
    if len(margins) == 0:
        return {'count': 0, 'mean': 0.0, 'std': 0.0, 'p10': 0.0, 'p50': 0.0, 'p90': 0.0}
    return {
        'count': int(len(margins)),
        'mean': float(np.mean(margins)),
        'std': float(np.std(margins)),
        'p10': float(np.percentile(margins, 10)),
        'p50': float(np.percentile(margins, 50)),
        'p90': float(np.percentile(margins, 90)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--max_margin_train_samples', type=int, default=2000)
    parser.add_argument('--max_margin_val_samples', type=int, default=500)
    parser.add_argument('--failure_top_n', type=int, default=100)
    parser.add_argument('--success_top_n', type=int, default=100)
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
        intermediate_indices=[3, 4, 6],
    )
    ckpt_type = load_checkpoint(model, checkpoint)
    model.to(device)

    train_dataset = CurriculumDataset(TRAIN_CACHE, max_samples=args.max_train_samples, compute_snr=False)
    val_dataset = CurriculumDataset(VAL_CACHE, max_samples=args.max_val_samples, compute_snr=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    val_loader = DataLoader(
        IndexedDataset(val_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_indices,
    )

    train_metrics, _ = evaluate_split(model, train_loader, device, 'train', progress_every=args.progress_every)
    val_metrics, val_per_sample = evaluate_split(
        model, val_loader, device, 'val', progress_every=args.progress_every,
        collect_per_sample=True, dataset=val_dataset
    )

    # collapse score: z(top_k_mass) - z(entropy)
    entropies = np.array([x['entropy'] for x in val_per_sample], dtype=np.float64)
    topk = np.array([x['top_k_mass'] for x in val_per_sample], dtype=np.float64)
    ent_z = (entropies - entropies.mean()) / (entropies.std() + 1e-12)
    topk_z = (topk - topk.mean()) / (topk.std() + 1e-12)
    collapse = topk_z - ent_z
    for i, score in enumerate(collapse):
        val_per_sample[i]['collapse_score'] = float(score)

    # failure / success selection
    n_fail = min(args.failure_top_n, len(val_per_sample))
    n_succ = min(args.success_top_n, len(val_per_sample))

    by_collapse = sorted(val_per_sample, key=lambda x: x['collapse_score'], reverse=True)
    by_acc = sorted(val_per_sample, key=lambda x: x['strict_acc'])
    failure_indices = {x['index'] for x in by_collapse[:n_fail]} | {x['index'] for x in by_acc[:n_fail]}

    # success: high acc + low collapse (collapse_score <= 0), and not failure
    acc_desc = sorted(val_per_sample, key=lambda x: x['strict_acc'], reverse=True)
    success = []
    for x in acc_desc:
        if x['index'] in failure_indices:
            continue
        if x['collapse_score'] <= 0:
            success.append(x)
        if len(success) >= n_succ:
            break
    if len(success) < n_succ:
        for x in acc_desc:
            if x['index'] in failure_indices:
                continue
            if x in success:
                continue
            success.append(x)
            if len(success) >= n_succ:
                break

    failure = [x for x in val_per_sample if x['index'] in failure_indices]

    # margin stats on subsets
    margin_train_dataset = CurriculumDataset(TRAIN_CACHE, max_samples=args.max_margin_train_samples, compute_snr=False)
    margin_val_dataset = CurriculumDataset(VAL_CACHE, max_samples=args.max_margin_val_samples, compute_snr=False)

    margin_train_loader = DataLoader(
        margin_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    margin_val_loader = DataLoader(
        margin_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )

    train_margin = compute_margin_stats(model, margin_train_loader, device, 'train', progress_every=args.progress_every)
    val_margin = compute_margin_stats(model, margin_val_loader, device, 'val', progress_every=args.progress_every)

    out_dir = Path(__file__).resolve().parent
    metrics_out = out_dir / 'metrics_overview.json'
    failure_out = out_dir / 'failure_set.json'

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'commit': '589e6d286cce5bb42a6f174b15eabc824c994a84',
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'checkpoint_type': ckpt_type,
        'device': device,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'margin_samples': {
            'train': int(len(margin_train_dataset)),
            'val': int(len(margin_val_dataset)),
        },
        'splits': {
            'train': train_metrics,
            'val': val_metrics,
        },
        'vq_margin': {
            'train': train_margin,
            'val': val_margin,
        }
    }
    metrics_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    failure_payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'seed': args.seed,
        'selection': {
            'failure_top_n': n_fail,
            'success_top_n': n_succ,
            'failure_rule': 'union(top collapse_score, bottom strict_acc)',
            'success_rule': 'top strict_acc with collapse_score<=0 (fallback to top acc)',
            'collapse_score_def': 'z(top_k_mass) - z(entropy)',
        },
        'failure_set': failure,
        'success_set': success,
    }
    failure_out.write_text(json.dumps(failure_payload, indent=2, ensure_ascii=False))

    print(f'Wrote: {metrics_out}')
    print(f'Wrote: {failure_out}')


if __name__ == '__main__':
    main()
