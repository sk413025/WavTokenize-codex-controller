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
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, VAL_CACHE
from exp_1219.losses import create_length_mask
from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn

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


def estimate_snr(noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> float:
    noisy_audio = noisy_audio.squeeze()
    clean_audio = clean_audio.squeeze()
    min_len = min(len(noisy_audio), len(clean_audio))
    noisy_audio = noisy_audio[:min_len]
    clean_audio = clean_audio[:min_len]
    signal_power = (clean_audio ** 2).mean()
    noise = noisy_audio - clean_audio
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return 100.0
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return float(snr.item())


def energy_rms(audio: torch.Tensor) -> float:
    audio = audio.squeeze()
    return float(torch.sqrt((audio ** 2).mean() + 1e-12).item())


class MetaAlignedDataset(AlignedNoisyCleanPairDataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = super().__getitem__(idx)
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


def per_sample_stats(s_codes, t_codes, mask):
    B, T = s_codes.shape
    accs, entropies, topk_mass, uniques = [], [], [], []

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
        acc = (s == t).float().mean().item()
        accs.append(acc)

        counts = torch.bincount(s, minlength=NUM_CODES).float()
        total = counts.sum().item()
        probs = counts / (total + 1e-12)
        nz = probs > 0
        entropy = -(probs[nz] * torch.log(probs[nz] + 1e-12)).sum().item()
        entropies.append(entropy)

        topk = torch.topk(probs, k=min(10, probs.numel())).values.sum().item()
        topk_mass.append(topk)

        uniques.append(int((counts > 0).sum().item()))

    return accs, entropies, topk_mass, uniques


def aggregate_by_key(records, key):
    out = {}
    for r in records:
        k = r.get(key, None)
        if k is None:
            continue
        bucket = out.setdefault(str(k), {'count': 0, 'acc': [], 'entropy': [], 'topk': [], 'unique': [], 'collapse_score': []})
        bucket['count'] += 1
        bucket['acc'].append(r['acc'])
        bucket['entropy'].append(r['entropy'])
        bucket['topk'].append(r['topk'])
        bucket['unique'].append(r['unique'])
        bucket['collapse_score'].append(r['collapse_score'])

    # summarize
    summary = {}
    for k, v in out.items():
        summary[k] = {
            'count': v['count'],
            'acc_mean': float(np.mean(v['acc'])),
            'entropy_mean': float(np.mean(v['entropy'])),
            'topk_mean': float(np.mean(v['topk'])),
            'unique_mean': float(np.mean(v['unique'])),
            'collapse_score_mean': float(np.mean(v['collapse_score'])),
        }
    return summary


def aggregate_by_bins(records, key, bins):
    edges = np.array(bins, dtype=float)
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        bucket = [r for r in records if r[key] >= lo and r[key] < hi]
        if not bucket:
            out.append({'bin': [float(lo), float(hi)], 'count': 0})
            continue
        out.append({
            'bin': [float(lo), float(hi)],
            'count': len(bucket),
            'acc_mean': float(np.mean([r['acc'] for r in bucket])),
            'entropy_mean': float(np.mean([r['entropy'] for r in bucket])),
            'topk_mean': float(np.mean([r['topk'] for r in bucket])),
            'unique_mean': float(np.mean([r['unique'] for r in bucket])),
            'collapse_score_mean': float(np.mean([r['collapse_score'] for r in bucket])),
        })
    return out


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
    parser.add_argument('--snr_bin_width', type=float, default=2.0)
    parser.add_argument('--energy_bins', type=int, default=5)
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

    records = []

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

            accs, entropies, topk_mass, uniques = per_sample_stats(s_codes, t_codes, mask)

            # collapse score
            topk_arr = np.asarray(topk_mass)
            ent_arr = np.asarray(entropies)
            z_topk = (topk_arr - topk_arr.mean()) / (topk_arr.std() + 1e-8)
            z_ent = (ent_arr - ent_arr.mean()) / (ent_arr.std() + 1e-8)
            collapse_score = z_topk - z_ent

            meta = batch['meta']
            for i in range(B):
                snr = estimate_snr(noisy[i].detach().cpu(), clean[i].detach().cpu())
                rms = energy_rms(noisy[i].detach().cpu())
                records.append({
                    'acc': float(accs[i]),
                    'entropy': float(entropies[i]),
                    'topk': float(topk_mass[i]),
                    'unique': int(uniques[i]),
                    'collapse_score': float(collapse_score[i]),
                    'speaker_id': meta['speaker_id'][i],
                    'content_id': meta['content_id'][i],
                    'sentence_id': meta['sentence_id'][i],
                    'material': meta['material'][i],
                    'snr': float(snr),
                    'energy_rms': float(rms),
                })

            if batch_idx % 200 == 0:
                print(f'[val] batch {batch_idx}/{len(loader)}')

    # Aggregate by speaker
    collapse_by_speaker = aggregate_by_key(records, 'speaker_id')

    # SNR bins
    snr_vals = np.array([r['snr'] for r in records])
    snr_min = float(np.floor(snr_vals.min()))
    snr_max = float(np.ceil(snr_vals.max()))
    snr_bins = np.arange(snr_min, snr_max + args.snr_bin_width, args.snr_bin_width)
    collapse_by_snr = aggregate_by_bins(records, 'snr', snr_bins)

    # Energy bins (quantiles)
    energy_vals = np.array([r['energy_rms'] for r in records])
    q = np.linspace(0, 1, args.energy_bins + 1)
    energy_bins = np.quantile(energy_vals, q)
    collapse_by_energy = aggregate_by_bins(records, 'energy_rms', energy_bins)

    out_dir = Path(__file__).resolve().parent
    out_speaker = out_dir / 'collapse_by_speaker.json'
    out_snr = out_dir / 'collapse_by_snr.json'
    out_energy = out_dir / 'collapse_by_energy.json'

    meta = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'seed': args.seed,
        'sample_strategy': sample_strategy,
        'max_val_samples': args.max_val_samples,
        'snr_bin_width': args.snr_bin_width,
        'energy_bins_quantiles': [float(x) for x in energy_bins.tolist()],
        'num_samples': len(records),
    }

    out_speaker.write_text(json.dumps({'meta': meta, 'by_speaker': collapse_by_speaker}, indent=2, ensure_ascii=False))
    out_snr.write_text(json.dumps({'meta': meta, 'by_snr': collapse_by_snr}, indent=2, ensure_ascii=False))
    out_energy.write_text(json.dumps({'meta': meta, 'by_energy': collapse_by_energy}, indent=2, ensure_ascii=False))

    print(f'Wrote: {out_speaker}')
    print(f'Wrote: {out_snr}')
    print(f'Wrote: {out_energy}')


if __name__ == '__main__':
    main()
