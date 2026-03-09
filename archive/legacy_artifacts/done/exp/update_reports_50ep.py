#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


KEYS = [
    ('w2c_minus_c2w_zero_pp','W→C−C→W (zero) pp'),
    ('net_delta_acc_zero_pp','ΔAcc (zero) pp'),
    ('mid_delta_acc_zero_pp','Mid ΔAcc (zero) pp'),
    ('mid_coverage_zero_pct','Mid Coverage (zero) %'),
    ('cos_mean_mid','cos_mean (mid)'),
    ('dmargin_mean_mid','Δmargin (mid)'),
]


def read_summary(path: Path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    # keep only epochs 10..50 (if present)
    rows = [r for r in rows if int(r['epoch']) in (10,20,30,40,50)]
    rows.sort(key=lambda r: int(r['epoch']))
    return rows


def fmt(v):
    if v is None or v == '':
        return ''
    try:
        x = float(v)
        return f"{x:.3f}"
    except Exception:
        return str(v)


def table_for_run(rows):
    # Markdown table with epoch and key metrics
    headers = ['epoch'] + [lab for _, lab in KEYS]
    lines = ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---']*len(headers)) + '|']
    for r in rows:
        line = [r['epoch']] + [fmt(r.get(k,'')) for k,_ in KEYS]
        lines.append('|' + '|'.join(line) + '|')
    return '\n'.join(lines)


def comparison_table(rows_g, rows_d):
    # Assume both sets have same epoch set
    epochs = sorted({int(r['epoch']) for r in rows_g} | {int(r['epoch']) for r in rows_d})
    headers = ['epoch']
    for _, lab in KEYS:
        headers += [f"Gated {lab}", f"Deep {lab}"]
    lines = ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---']*len(headers)) + '|']
    by_e_g = {int(r['epoch']): r for r in rows_g}
    by_e_d = {int(r['epoch']): r for r in rows_d}
    for e in epochs:
        rg = by_e_g.get(e, {})
        rd = by_e_d.get(e, {})
        row = [str(e)]
        for k,_ in KEYS:
            row.append(fmt(rg.get(k,'')))
            row.append(fmt(rd.get(k,'')))
        lines.append('|' + '|'.join(row) + '|')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gated_dir', required=True)
    ap.add_argument('--deep_dir', required=True)
    ap.add_argument('--gated_md', default='CROSSATTN_GATED_50EP_20251106.md')
    ap.add_argument('--deep_md', default='CROSSATTN_DEEP_50EP_20251106.md')
    ap.add_argument('--cmp_md', default='CROSSATTN_GATED_DEEP_50EP_COMPARISON_20251106.md')
    args = ap.parse_args()

    g_sum = Path(args.gated_dir) / 'analysis' / 'epoch_summary.csv'
    d_sum = Path(args.deep_dir) / 'analysis' / 'epoch_summary.csv'
    rows_g = read_summary(g_sum) if g_sum.exists() else []
    rows_d = read_summary(d_sum) if d_sum.exists() else []

    # Build contents
    g_table = table_for_run(rows_g) if rows_g else '暫無數據（待分析完成）'
    d_table = table_for_run(rows_d) if rows_d else '暫無數據（待分析完成）'
    cmp_table = comparison_table(rows_g, rows_d) if rows_g and rows_d else '暫無數據（待分析完成）'

    # Write/replace contents of MDs (simple overwrite)
    Path(args.gated_md).write_text(
        '# CrossAttn Gated（50 epoch）分析摘要\n\n' + g_table + '\n', encoding='utf-8'
    )
    Path(args.deep_md).write_text(
        '# CrossAttn Deep（50 epoch）分析摘要\n\n' + d_table + '\n', encoding='utf-8'
    )
    Path(args.cmp_md).write_text(
        '# CrossAttn Gated‑50 vs Deep‑50 對齊摘要（10/20/30/40/50）\n\n' + cmp_table + '\n', encoding='utf-8'
    )

    print('✓ Updated MD reports for 50-epoch runs')


if __name__ == '__main__':
    main()

