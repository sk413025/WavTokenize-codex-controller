import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_noise_type(path: str) -> str:
    if not path:
        return 'unknown'
    p = path.lower()
    if 'papercup' in p:
        return 'papercup'
    if 'plastic' in p:
        return 'plastic'
    if 'box' in p:
        return 'box'
    if 'clean' in p:
        return 'clean'
    return 'other'


def cohen_d(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def summarize_group(indices, meta, scores, label, top_k=100):
    rows = []
    for idx in indices:
        m = meta.get(str(idx)) or meta.get(idx) or {}
        rows.append({
            'index': idx,
            'score': scores.get(idx, 0.0),
            'snr_db': m.get('snr_db', None),
            'noisy_energy': m.get('noisy_energy', None),
            'noisy_path': m.get('noisy_path', None),
            'clean_path': m.get('clean_path', None),
            'noise_type': parse_noise_type(m.get('noisy_path', '')),
        })
    rows = sorted(rows, key=lambda x: x['score'], reverse=True)
    return rows[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_csv', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/tracin_scores.csv')
    parser.add_argument('--meta_json', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/tracin_indices.json')
    parser.add_argument('--out_dir', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d')
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)
    meta = json.loads(Path(args.meta_json).read_text())

    # aggregate scores across checkpoints
    score_by_train = scores.groupby('train_index')['score'].sum().to_dict()
    train_meta = meta['train_meta']

    train_indices = list(score_by_train.keys())
    all_snr = [train_meta[str(i)]['snr_db'] if str(i) in train_meta else train_meta[i]['snr_db'] for i in train_indices]
    all_energy = [train_meta[str(i)]['noisy_energy'] if str(i) in train_meta else train_meta[i]['noisy_energy'] for i in train_indices]

    # top/bottom
    sorted_indices = sorted(train_indices, key=lambda i: score_by_train.get(i, 0.0), reverse=True)
    top_pos = sorted_indices[:args.top_k]
    top_neg = sorted_indices[-args.top_k:]

    def group_stats(indices):
        snr = [train_meta[str(i)]['snr_db'] if str(i) in train_meta else train_meta[i]['snr_db'] for i in indices]
        energy = [train_meta[str(i)]['noisy_energy'] if str(i) in train_meta else train_meta[i]['noisy_energy'] for i in indices]
        noise_types = [parse_noise_type((train_meta[str(i)]['noisy_path'] if str(i) in train_meta else train_meta[i]['noisy_path']) or '') for i in indices]
        counts = {}
        for n in noise_types:
            counts[n] = counts.get(n, 0) + 1
        return {
            'count': len(indices),
            'snr_mean': float(np.mean(snr)),
            'snr_std': float(np.std(snr)),
            'energy_mean': float(np.mean(energy)),
            'energy_std': float(np.std(energy)),
            'noise_type_counts': counts,
            'snr_cohen_d_vs_all': cohen_d(snr, all_snr),
            'energy_cohen_d_vs_all': cohen_d(energy, all_energy),
        }

    proponents = {
        'summary': group_stats(top_pos),
        'top_examples': summarize_group(top_pos, train_meta, score_by_train, 'proponent', top_k=min(20, args.top_k)),
    }
    opponents = {
        'summary': group_stats(top_neg),
        'top_examples': summarize_group(top_neg[::-1], train_meta, score_by_train, 'opponent', top_k=min(20, args.top_k)),
    }

    out_dir = Path(args.out_dir)
    (out_dir / 'plots').mkdir(parents=True, exist_ok=True)

    (out_dir / 'proponents_profile.json').write_text(json.dumps(proponents, indent=2, ensure_ascii=False))
    (out_dir / 'opponents_profile.json').write_text(json.dumps(opponents, indent=2, ensure_ascii=False))

    # plots: influence vs snr / energy
    scores_list = [score_by_train[i] for i in train_indices]
    plt.figure(figsize=(6, 4))
    plt.scatter(all_snr, scores_list, s=8, alpha=0.6)
    plt.xlabel('SNR (dB)')
    plt.ylabel('TracIn score (aggregated)')
    plt.title('Influence vs SNR')
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'influence_vs_snr.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(all_energy, scores_list, s=8, alpha=0.6)
    plt.xlabel('Noisy energy')
    plt.ylabel('TracIn score (aggregated)')
    plt.title('Influence vs Energy')
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'influence_vs_energy.png')
    plt.close()

    print('Wrote proponents_profile.json, opponents_profile.json, plots')


if __name__ == '__main__':
    main()
