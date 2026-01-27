import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def load_audio(path):
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        audio = audio[:, 0]
        return audio, sr
    except Exception:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        if wav.dim() > 1:
            wav = wav[0]
        return wav.numpy(), sr


def resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    try:
        import torchaudio
        import torch
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        wav = torch.from_numpy(audio).unsqueeze(0)
        return resampler(wav).squeeze(0).numpy()
    except Exception:
        from scipy.signal import resample_poly
        return resample_poly(audio, target_sr, orig_sr).astype('float32')


def compute_metrics(clean, test, sr):
    # align lengths
    min_len = min(len(clean), len(test))
    clean = clean[:min_len]
    test = test[:min_len]

    from pesq import pesq
    from pystoi import stoi

    pesq_score = pesq(sr, clean, test, 'wb')
    stoi_score = stoi(clean, test, sr, extended=False)
    return float(pesq_score), float(stoi_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613')
    parser.add_argument('--epoch', type=str, default='epoch_300')
    parser.add_argument('--splits', type=str, default='train,val')
    parser.add_argument('--target_sr', type=int, default=16000)
    parser.add_argument('--out_dir', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/audio_quality')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'epoch': args.epoch,
        'target_sr': args.target_sr,
        'splits': {},
    }

    for split in splits:
        split_dir = run_dir / 'audio_samples' / split / args.epoch
        if not split_dir.exists():
            continue

        records = []
        for clean_path in sorted(split_dir.glob('sample_*_clean.wav')):
            stem = clean_path.name.replace('_clean.wav', '')
            noisy_path = split_dir / f'{stem}_noisy.wav'
            recon_path = split_dir / f'{stem}_student_recon.wav'
            if not noisy_path.exists() or not recon_path.exists():
                continue

            clean, sr = load_audio(clean_path)
            noisy, sr2 = load_audio(noisy_path)
            recon, sr3 = load_audio(recon_path)

            if sr != sr2 or sr != sr3:
                # resample to common target
                clean = resample(clean, sr, args.target_sr)
                noisy = resample(noisy, sr2, args.target_sr)
                recon = resample(recon, sr3, args.target_sr)
                sr_use = args.target_sr
            else:
                # still resample to target for PESQ wb
                clean = resample(clean, sr, args.target_sr)
                noisy = resample(noisy, sr, args.target_sr)
                recon = resample(recon, sr, args.target_sr)
                sr_use = args.target_sr

            pesq_noisy, stoi_noisy = compute_metrics(clean, noisy, sr_use)
            pesq_recon, stoi_recon = compute_metrics(clean, recon, sr_use)

            records.append({
                'sample': stem,
                'pesq_noisy': pesq_noisy,
                'stoi_noisy': stoi_noisy,
                'pesq_recon': pesq_recon,
                'stoi_recon': stoi_recon,
            })

        if records:
            pesq_noisy = [r['pesq_noisy'] for r in records]
            pesq_recon = [r['pesq_recon'] for r in records]
            stoi_noisy = [r['stoi_noisy'] for r in records]
            stoi_recon = [r['stoi_recon'] for r in records]

            summary['splits'][split] = {
                'num_samples': len(records),
                'pesq_noisy_mean': float(np.mean(pesq_noisy)),
                'pesq_recon_mean': float(np.mean(pesq_recon)),
                'stoi_noisy_mean': float(np.mean(stoi_noisy)),
                'stoi_recon_mean': float(np.mean(stoi_recon)),
                'pesq_noisy_std': float(np.std(pesq_noisy)),
                'pesq_recon_std': float(np.std(pesq_recon)),
                'stoi_noisy_std': float(np.std(stoi_noisy)),
                'stoi_recon_std': float(np.std(stoi_recon)),
            }

        summary[f'{split}_records'] = records

    out_json = out_dir / 'pesq_stoi_summary.json'
    out_md = out_dir / 'pesq_stoi_summary.md'
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        "# PESQ/STOI Summary",
        f"- run_dir: {summary['run_dir']}",
        f"- epoch: {summary['epoch']}",
        f"- target_sr: {summary['target_sr']}",
    ]
    for split, stats in summary['splits'].items():
        lines.append(f"\n## {split}")
        lines.append(f"- samples: {stats['num_samples']}")
        lines.append(f"- PESQ noisy mean: {stats['pesq_noisy_mean']:.3f} (std {stats['pesq_noisy_std']:.3f})")
        lines.append(f"- PESQ recon mean: {stats['pesq_recon_mean']:.3f} (std {stats['pesq_recon_std']:.3f})")
        lines.append(f"- STOI noisy mean: {stats['stoi_noisy_mean']:.3f} (std {stats['stoi_noisy_std']:.3f})")
        lines.append(f"- STOI recon mean: {stats['stoi_recon_mean']:.3f} (std {stats['stoi_recon_std']:.3f})")
    out_md.write_text("\n".join(lines))

    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_md}')


if __name__ == '__main__':
    main()
