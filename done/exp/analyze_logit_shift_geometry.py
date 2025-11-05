"""
ΔLogits Geometry Analysis

目的:
- 量測 speaker 造成的 logits 差分 Δz 是否「朝向目標方向」。
- 驗證機轉 H4（方向性/位置不足）。

定義:
- 對同一批資料, 比較 Normal speaker 與 Zero speaker 的 logits。
- Δz = logits_normal - logits_zero  (B, T, V)
- 對每個 token:
  * 取 normal 條件下的 top-1 類別 c1 及 top-2 類別 c2。
  * 定義方向向量 d = e[target] - e[c2]（若 target==c2 則改用第三高）。
  * 餘弦相似度 cos = <Δz, d> / (||Δz||·||d||)。
  * 目標 margin 變化 Δmargin = [(z_t - z_c2)_normal - (z_t - z_c2)_zero]。

產出:
- results/.../analysis/logit_geometry/epoch_E/geometry_epoch_E.csv:
  每行包含: margin_bin, count, cos_mean, cos_w2c_mean, cos_c2w_mean, dmargin_mean, ...
- 圖: cos_hist_epoch_E.png（可選）

使用:
  python -u done/exp/analyze_logit_shift_geometry.py \
    --results_dir results/crossattn_k4_... \
    --cache_dir /home/.../data \
    --epochs 20 40 \
    --batch_size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import math

import torch
from torch.utils.data import DataLoader

from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn
from model_zeroshot_crossattn_deep import ZeroShotDenoisingTransformerCrossAttnDeep
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def load_model(results_dir: Path, epoch: int, device: torch.device):
    ckpt_path = results_dir / f'checkpoint_epoch_{epoch}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    codebook = state['codebook']
    keys = list(state.keys())
    if any(k.startswith('fusion0.') or k.startswith('layers.') for k in keys):
        model = ZeroShotDenoisingTransformerCrossAttnDeep(
            codebook=codebook,
            speaker_embed_dim=256,
            d_model=512,
            nhead=8,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            speaker_tokens=4,
            inject_layers=(1,3),
        ).to(device)
    else:
        model = ZeroShotDenoisingTransformerCrossAttn(
            codebook=codebook,
            speaker_embed_dim=256,
            d_model=512,
            nhead=8,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            speaker_tokens=4,
        ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def safe_top2(probs: torch.Tensor):
    # probs: (B,T,V)
    top2 = probs.topk(k=3, dim=-1)
    p = top2.values
    idx = top2.indices
    # 有時 target 會等於次高，保留第三備用
    return p, idx  # shapes: (B,T,3)


def analyze_epoch(results_dir: Path, cache_dir: Path, epoch: int, batch_size: int = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(results_dir, epoch, device)

    ds = ZeroShotAudioDatasetCached(str(cache_dir / 'val_cache.pt'))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=cached_collate_fn, num_workers=0)

    # margin bins
    bins = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    nb = len(bins) - 1
    def bindex(m):
        for i in range(nb):
            if bins[i] <= m < bins[i+1]:
                return i
        return nb-1

    # accumulators
    acc = {
        'count': [0]*nb,
        'cos_sum': [0.0]*nb,
        'cos_w2c_sum': [0.0]*nb,
        'cos_c2w_sum': [0.0]*nb,
        'cos_w2c_cnt': [0]*nb,
        'cos_c2w_cnt': [0]*nb,
        'dmargin_sum': [0.0]*nb,
    }

    with torch.no_grad():
        for batch in dl:
            noisy = batch['noisy_tokens'].to(device)
            clean = batch['clean_tokens'].to(device)
            spk = batch['speaker_embeddings'].to(device)
            mask = (clean != 0)

            # Normal & Zero logits
            logits_n = model(noisy, spk, return_logits=True)
            logits_z = model(noisy, torch.zeros_like(spk), return_logits=True)
            probs_n = torch.softmax(logits_n, dim=-1)
            probs_z = torch.softmax(logits_z, dim=-1)
            B,T,V = probs_n.shape

            # top-3 under normal
            p_top, idx_top = safe_top2(probs_n)  # (B,T,3)
            p1 = p_top[...,0]
            p2 = p_top[...,1]
            c1 = idx_top[...,0]
            c2 = idx_top[...,1]
            margin = (p1 - p2).clamp(0,1)

            # variant top-1 for correctness check
            pred_n = probs_n.argmax(dim=-1)
            pred_z = probs_z.argmax(dim=-1)

            # compute Δz and cos with d = e[target] - e[c2]
            delta = (logits_n - logits_z)  # (B,T,V)
            # gather vectors per token
            # unit vectors: we only need two components for dot product
            # dot = Δz[target] - Δz[c2]
            dz_target = delta.gather(-1, clean.unsqueeze(-1)).squeeze(-1)
            dz_c2 = delta.gather(-1, c2.unsqueeze(-1)).squeeze(-1)
            dot = dz_target - dz_c2  # (B,T)
            # ||d|| = sqrt(2)
            d_norm = math.sqrt(2.0)
            # ||Δz||_2 (across V)
            dz_norm = torch.linalg.vector_norm(delta, dim=-1) + 1e-9  # (B,T)
            cos = (dot / (dz_norm * d_norm))

            # margin change wrt c2: ( (z_t - z_c2)_n - (z_t - z_c2)_z )
            zt_n = logits_n.gather(-1, clean.unsqueeze(-1)).squeeze(-1)
            zc2_n = logits_n.gather(-1, c2.unsqueeze(-1)).squeeze(-1)
            zt_z = logits_z.gather(-1, clean.unsqueeze(-1)).squeeze(-1)
            zc2_z = logits_z.gather(-1, c2.unsqueeze(-1)).squeeze(-1)
            dmargin = (zt_n - zc2_n) - (zt_z - zc2_z)

            # categories for W→C / C→W w.r.t removal (zero)
            norm_correct = (pred_n == clean)
            zero_correct = (pred_z == clean)
            c2w = (norm_correct & (~zero_correct))
            w2c = ((~norm_correct) & zero_correct)

            # accumulate per bin
            for b in range(B):
                for t in range(T):
                    if not bool(mask[b,t].item()):
                        continue
                    bi = bindex(float(margin[b,t].item()))
                    acc['count'][bi] += 1
                    acc['cos_sum'][bi] += float(cos[b,t].item())
                    acc['dmargin_sum'][bi] += float(dmargin[b,t].item())
                    if bool(w2c[b,t].item()):
                        acc['cos_w2c_sum'][bi] += float(cos[b,t].item())
                        acc['cos_w2c_cnt'][bi] += 1
                    if bool(c2w[b,t].item()):
                        acc['cos_c2w_sum'][bi] += float(cos[b,t].item())
                        acc['cos_c2w_cnt'][bi] += 1

    # write CSV
    out_dir = results_dir / 'analysis' / 'logit_geometry' / f'epoch_{epoch}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'geometry_epoch_{epoch}.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_left','bin_right','count','cos_mean','cos_w2c_mean','cos_c2w_mean','dmargin_mean'])
        for i in range(nb):
            cnt = max(1, acc['count'][i])
            cos_mean = acc['cos_sum'][i]/cnt
            dmargin_mean = acc['dmargin_sum'][i]/cnt
            cw = acc['cos_w2c_sum'][i] / max(1, acc['cos_w2c_cnt'][i])
            cc = acc['cos_c2w_sum'][i] / max(1, acc['cos_c2w_cnt'][i])
            writer.writerow([bins[i], bins[i+1], acc['count'][i], f"{cos_mean:.6f}", f"{cw:.6f}", f"{cc:.6f}", f"{dmargin_mean:.6f}"])

    print(f"✓ Geometry saved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir = Path(args.cache_dir)
    for e in args.epochs:
        analyze_epoch(results_dir, cache_dir, e, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
