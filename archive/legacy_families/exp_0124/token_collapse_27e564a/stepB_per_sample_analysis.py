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
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, VAL_CACHE
from exp_1219.losses import create_length_mask
from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn

ENCODER_STRIDE = 320
NUM_CODES = 4096


class MetaAlignedDataset(AlignedNoisyCleanPairDataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = super().__getitem__(idx)
        # attach metadata if available
        for key in [
            'speaker_id', 'content_id', 'sentence_id', 'material',
            'noisy_path', 'clean_path', 'filename'
        ]:
            item[key] = sample.get(key, None)
        return item


def collate_with_meta(batch):
    base = aligned_collate_fn(batch)
    meta = {}
    for key in [
        'speaker_id', 'content_id', 'sentence_id', 'material',
        'noisy_path', 'clean_path', 'filename'
    ]:
        meta[key] = [item.get(key, None) for item in batch]
    base['meta'] = meta
    return base


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


def per_sample_stats(s_codes, t_codes, mask):
    # s_codes,t_codes: (B,T), mask: (B,T)
    B, T = s_codes.shape
    accs = []
    entropies = []
    topk_mass = []
    uniques = []

    for i in range(B):
        valid = mask[i] > 0
        if valid.sum().item() == 0:
            accs.append(0.0)
            entropies.append(0.0)
            topk_mass.append(0.0)
            uniques.append(0)
            continue

        s = s_codes[i][valid]
        t = t_codes[i][valid]
        correct = (s == t).float().mean().item()
        accs.append(correct)

        counts = torch.bincount(s, minlength=NUM_CODES).float()
        total = counts.sum().item()
        probs = counts / (total + 1e-12)
        nonzero = probs > 0
        entropy = -(probs[nonzero] * torch.log(probs[nonzero] + 1e-12)).sum().item()
        entropies.append(entropy)

        topk = torch.topk(probs, k=min(10, probs.numel())).values.sum().item()
        topk_mass.append(topk)

        uniques.append(int((counts > 0).sum().item()))

    return accs, entropies, topk_mass, uniques


def compute_correlations(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman


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

    dataset = MetaAlignedDataset(VAL_CACHE, max_samples=None)

    sample_strategy = 'full'
    if args.max_val_samples is not None and args.max_val_samples < len(dataset):
        sample_strategy = 'random_subset'
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(dataset), size=args.max_val_samples, replace=False)
        dataset = Subset(dataset, indices.tolist())

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta,
    )

    all_acc = []
    all_entropy = []
    all_topk = []
    all_unique = []
    all_meta = []

    model.eval()
    with torch.no_grad():
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

            accs, entropies, topk, uniques = per_sample_stats(s_codes, t_codes, mask)

            all_acc.extend(accs)
            all_entropy.extend(entropies)
            all_topk.extend(topk)
            all_unique.extend(uniques)

            meta = batch['meta']
            for i in range(B):
                all_meta.append({
                    'speaker_id': meta['speaker_id'][i],
                    'content_id': meta['content_id'][i],
                    'sentence_id': meta['sentence_id'][i],
                    'material': meta['material'][i],
                    'noisy_path': meta['noisy_path'][i],
                    'clean_path': meta['clean_path'][i],
                    'filename': meta['filename'][i],
                    'length': int(lengths[i].item()),
                })

            if batch_idx % 200 == 0:
                print(f'[val] batch {batch_idx}/{len(loader)}')

    # correlations
    pearson, spearman = compute_correlations(all_entropy, all_acc)

    # collapse score: z(topk) - z(entropy)
    topk_arr = np.asarray(all_topk)
    ent_arr = np.asarray(all_entropy)
    z_topk = (topk_arr - topk_arr.mean()) / (topk_arr.std() + 1e-8)
    z_ent = (ent_arr - ent_arr.mean()) / (ent_arr.std() + 1e-8)
    collapse_score = z_topk - z_ent

    # scatter plot
    out_dir = Path(__file__).resolve().parent
    out_json = out_dir / 'token_entropy_vs_acc_val.json'
    out_png = out_dir / 'token_entropy_vs_acc_val.png'
    out_case = out_dir / 'case_studies.md'

    plt.figure(figsize=(6, 4))
    plt.scatter(all_entropy, all_acc, s=6, alpha=0.5)
    plt.xlabel('Student token entropy (per-sample)')
    plt.ylabel('Strict acc (per-sample)')
    plt.title('Val entropy vs strict acc')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    # case studies: top-N collapse_score
    N = 20
    top_idx = np.argsort(-collapse_score)[:N]
    lines = [
        '# Case studies: most collapsed samples',
        '',
        f'- N: {N}',
        f'- collapse_score: z(top_k_mass) - z(entropy)',
        ''
    ]
    lines.append('| rank | acc | entropy | top_k_mass | unique | speaker_id | content_id | sentence_id | material | filename |')
    lines.append('|---:|---:|---:|---:|---:|---|---|---|---|---|')

    for rank, idx in enumerate(top_idx, start=1):
        meta = all_meta[idx]
        lines.append(
            f"| {rank} | {all_acc[idx]:.4f} | {all_entropy[idx]:.4f} | {all_topk[idx]:.4f} | {all_unique[idx]} | "
            f"{meta['speaker_id']} | {meta['content_id']} | {meta['sentence_id']} | {meta['material']} | {meta['filename']} |"
        )

    out_case.write_text('\n'.join(lines) + '\n')

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'seed': args.seed,
        'sample_strategy': sample_strategy,
        'max_val_samples': args.max_val_samples,
        'num_samples': len(all_acc),
        'pearson_entropy_acc': pearson,
        'spearman_entropy_acc': spearman,
        'collapse_score_def': 'z(top_k_mass) - z(entropy)',
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_png}')
    print(f'Wrote: {out_case}')


if __name__ == '__main__':
    main()
