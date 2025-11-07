#!/usr/bin/env python3
"""
Summarize behavior analysis metrics per run into a compact epoch_summary.csv.

Inputs (expected precomputed by analyze_* scripts):
  - analysis/influence_breakdown/epoch_E/breakdown_epoch_E.csv
  - analysis/margins_topk/epoch_E/margins_bins_epoch_E.csv
  - analysis/logit_geometry/epoch_E/geometry_epoch_E.csv

Outputs:
  - analysis/epoch_summary.csv with columns per epoch:
      epoch,
      w2c_minus_c2w_zero_pp, w2c_minus_c2w_rand_pp,
      net_delta_acc_zero_pp, net_delta_acc_rand_pp,
      mid_delta_acc_zero_pp, mid_delta_acc_rand_pp,
      mid_coverage_zero_pct, mid_coverage_rand_pct,
      cos_mean_mid, dmargin_mean_mid,
      have_influence, have_margins, have_geometry

Default mid-margin range: [0.02, 0.20)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def summarize_epoch(base: Path, epoch: int, mid_left: float, mid_right: float) -> Dict[str, float]:
    out = {
        'epoch': epoch,
        'w2c_minus_c2w_zero_pp': None,
        'w2c_minus_c2w_rand_pp': None,
        'net_delta_acc_zero_pp': None,
        'net_delta_acc_rand_pp': None,
        'mid_delta_acc_zero_pp': None,
        'mid_delta_acc_rand_pp': None,
        'mid_coverage_zero_pct': None,
        'mid_coverage_rand_pct': None,
        'cos_mean_mid': None,
        'dmargin_mean_mid': None,
        'have_influence': 0,
        'have_margins': 0,
        'have_geometry': 0,
    }

    # Influence
    infl_path = base / 'analysis' / 'influence_breakdown' / f'epoch_{epoch}' / f'breakdown_epoch_{epoch}.csv'
    if infl_path.exists():
        rows = read_csv_rows(infl_path)
        by_cond = {r['condition']: r for r in rows if 'condition' in r}
        for cond, key_prefix in [('zero', 'zero'), ('random', 'rand')]:
            if cond in by_cond:
                r = by_cond[cond]
                c2w = safe_float(r.get('c2w_pct', '0'))
                w2c = safe_float(r.get('w2c_pct', '0'))
                net = safe_float(r.get('net_acc_delta_pct', '0'))
                out[f'w2c_minus_c2w_{key_prefix}_pp'] = w2c - c2w
                out[f'net_delta_acc_{key_prefix}_pp'] = net
        out['have_influence'] = 1

    # Margins/Top-k
    marg_path = base / 'analysis' / 'margins_topk' / f'epoch_{epoch}' / f'margins_bins_epoch_{epoch}.csv'
    if marg_path.exists():
        rows = read_csv_rows(marg_path)
        # Sum over bins
        totals = {'zero': 0, 'random': 0}
        mids = {
            'zero': {'count': 0, 'delta_acc_sum': 0.0},
            'random': {'count': 0, 'delta_acc_sum': 0.0},
        }
        for r in rows:
            cond = r.get('condition', '')
            if cond not in totals:
                continue
            cnt = int(float(r.get('count', '0')))
            l = safe_float(r.get('bin_left', '0'))
            rr = safe_float(r.get('bin_right', '0'))
            totals[cond] += cnt
            if l >= mid_left and rr <= mid_right:
                mids[cond]['count'] += cnt
                mids[cond]['delta_acc_sum'] += cnt * safe_float(r.get('delta_acc_pct', '0'))
        for cond, key_prefix in [('zero','zero'),('random','rand')]:
            total = totals[cond]
            midc = mids[cond]['count']
            if total > 0:
                out[f'mid_coverage_{key_prefix}_pct'] = 100.0 * midc / total
            if midc > 0:
                out[f'mid_delta_acc_{key_prefix}_pp'] = mids[cond]['delta_acc_sum'] / midc
        out['have_margins'] = 1

    # Geometry
    geo_path = base / 'analysis' / 'logit_geometry' / f'epoch_{epoch}' / f'geometry_epoch_{epoch}.csv'
    if geo_path.exists():
        rows = read_csv_rows(geo_path)
        cnt_sum = 0
        cos_sum = 0.0
        dmargin_sum = 0.0
        for r in rows:
            l = safe_float(r.get('bin_left', '0'))
            rr = safe_float(r.get('bin_right', '0'))
            cnt = int(float(r.get('count', '0')))
            if l >= mid_left and rr <= mid_right:
                cnt_sum += cnt
                cos_sum += cnt * safe_float(r.get('cos_mean', '0'))
                dmargin_sum += cnt * safe_float(r.get('dmargin_mean', '0'))
        if cnt_sum > 0:
            out['cos_mean_mid'] = cos_sum / cnt_sum
            out['dmargin_mean_mid'] = dmargin_sum / cnt_sum
        out['have_geometry'] = 1

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--mid_left', type=float, default=0.02)
    ap.add_argument('--mid_right', type=float, default=0.20)
    args = ap.parse_args()

    base = Path(args.results_dir)
    out_dir = base / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'epoch_summary.csv'

    fields = [
        'epoch',
        'w2c_minus_c2w_zero_pp','w2c_minus_c2w_rand_pp',
        'net_delta_acc_zero_pp','net_delta_acc_rand_pp',
        'mid_delta_acc_zero_pp','mid_delta_acc_rand_pp',
        'mid_coverage_zero_pct','mid_coverage_rand_pct',
        'cos_mean_mid','dmargin_mean_mid',
        'have_influence','have_margins','have_geometry'
    ]

    rows: List[Dict[str, float]] = []
    for e in args.epochs:
        rows.append(summarize_epoch(base, e, args.mid_left, args.mid_right))

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) if r.get(k) is not None else '' for k in fields})

    print(f"✓ Epoch summary saved: {out_csv}")


if __name__ == '__main__':
    main()

