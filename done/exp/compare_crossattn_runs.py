"""
Compare S=1 (old run) vs K=4 (new run) Cross-Attn analysis summaries.

Inputs:
  --s1_summary  path to old run summary.csv (S=1 degeneracy)
  --k4_summary  path to new run summary.csv (K=4 fixed)
  --out_csv     output csv path

Output CSV columns:
  epoch, run, attn_w_mean, attn_w_std, attn_w_min, attn_w_max,
  token_var_proxy, norm_enclast, norm_output_proj
"""

import argparse
import csv

def load_rows(path):
    import pandas as pd
    df = pd.read_csv(path)
    # Some runs may have different naming for token variance; unify to 'token_var_proxy'
    if 'attn_output_token_var_mean' in df.columns:
        df['token_var_proxy'] = df['attn_output_token_var_mean']
    else:
        df['token_var_proxy'] = 0.0
    cols = ['epoch','attn_w_mean','attn_w_std','attn_w_min','attn_w_max','token_var_proxy','norm_enclast','norm_output_proj']
    return df[cols].to_dict(orient='records')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s1_summary', required=True)
    ap.add_argument('--k4_summary', required=True)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    s1 = load_rows(args.s1_summary)
    k4 = load_rows(args.k4_summary)

    # Map by epoch for intersection
    s1_map = {int(r['epoch']): r for r in s1}
    k4_map = {int(r['epoch']): r for r in k4}
    epochs = sorted(set(s1_map.keys()) & set(k4_map.keys()))

    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','run','attn_w_mean','attn_w_std','attn_w_min','attn_w_max','token_var_proxy','norm_enclast','norm_output_proj'])
        for e in epochs:
            r1 = s1_map[e]
            r4 = k4_map[e]
            writer.writerow([e,'S1',r1['attn_w_mean'],r1['attn_w_std'],r1['attn_w_min'],r1['attn_w_max'],r1['token_var_proxy'],r1['norm_enclast'],r1['norm_output_proj']])
            writer.writerow([e,'K4',r4['attn_w_mean'],r4['attn_w_std'],r4['attn_w_min'],r4['attn_w_max'],r4['token_var_proxy'],r4['norm_enclast'],r4['norm_output_proj']])

if __name__ == '__main__':
    main()

