#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Dict, List

EPOCHS_100 = [10,20,30,40,50,80,100]

def read_csv_rows(p: Path) -> List[Dict[str,str]]:
    if not p.exists():
        return []
    rows=[]
    with open(p,'r',newline='') as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def summarize_entropy(run_dir: Path, epochs=EPOCHS_100):
    out=[]
    for e in epochs:
        p = run_dir/ 'analysis' / 'attn_entropy' / f'epoch_{e}' / f'entropy_epoch_{e}.csv'
        rows = read_csv_rows(p)
        if rows:
            r = rows[0]
            out.append({'epoch':e,
                        'ent_mean': r.get('mean',''),
                        'ent_p90': r.get('p90',''),
                        'peaked_frac': r.get('peaked_frac_gt0.7','')})
    return out

def summarize_gate(run_dir: Path, epochs=EPOCHS_100):
    # Aggregate low gate fraction at low bin [0.0,0.02) and high gate at mid bins [0.02,0.2)
    out=[]
    for e in epochs:
        p = run_dir/ 'analysis' / 'gate_distribution' / f'epoch_{e}' / f'gate_stats_epoch_{e}.csv'
        rows = read_csv_rows(p)
        if not rows:
            continue
        low_frac = None
        mid_high = []
        mid_mean = []
        mid_p50 = []
        for r in rows:
            l = float(r['bin_left']); rr = float(r['bin_right'])
            if l==0.0 and rr<=0.02:
                low_frac = r.get('low_gate_frac','')
            if l>=0.02 and rr<=0.2:
                hf = r.get('high_gate_frac',''); gm=r.get('gate_mean',''); gp50=r.get('gate_p50','')
                if hf!='': mid_high.append(float(hf))
                if gm!='': mid_mean.append(float(gm))
                if gp50!='': mid_p50.append(float(gp50))
        def avg(v):
            return sum(v)/len(v) if v else ''
        out.append({'epoch':e,
                    'low_gate_frac_lowbin': low_frac if low_frac is not None else '',
                    'high_gate_frac_mid': avg(mid_high),
                    'gate_mean_mid': avg(mid_mean),
                    'gate_p50_mid': avg(mid_p50)})
    return out

def write_summary(run_dir: Path, ent_rows, gate_rows=None):
    out_dir = run_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'supplemental_summary_100ep.csv'
    # merge by epoch
    by_e = {r['epoch']: r for r in ent_rows}
    if gate_rows is not None:
        for r in gate_rows:
            e=r['epoch']; by_e.setdefault(e,{}).update(r)
    fields = ['epoch','ent_mean','ent_p90','peaked_frac','low_gate_frac_lowbin','high_gate_frac_mid','gate_mean_mid','gate_p50_mid']
    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in EPOCHS_100:
            row={'epoch':e}
            row.update(by_e.get(e,{}))
            w.writerow(row)
    print(f"✓ Supplemental summary saved: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gated_dir', required=True)
    ap.add_argument('--deep_dir', required=True)
    args=ap.parse_args()
    gated=Path(args.gated_dir); deep=Path(args.deep_dir)
    ent_g = summarize_entropy(gated)
    gate_g = summarize_gate(gated)
    write_summary(gated, ent_g, gate_g)
    ent_d = summarize_entropy(deep)
    write_summary(deep, ent_d, None)

if __name__=='__main__':
    main()

