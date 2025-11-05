"""
Cross-Attention Checkpoint Analysis

目的:
  - 針對 EXP-20251105-CrossAttn 的每 10 個 epoch checkpoint，分析：
    1) Cross-Attention 權重與輸出（是否退化為常數偏置）
    2) Transformer 每層輸出分佈（mean/std）
    3) 關鍵參數群的權重範數與其跨 epoch 變化（學習是否停滯）

輸入:
  - results_dir: 包含 config.json 與 checkpoint_epoch_*.pth 的資料夾

輸出:
  - 在 results_dir/analysis/ 下寫出:
    - summary.csv: 每個 epoch 的關鍵統計
    - stats_epoch_XX.json: 該 epoch 的詳細統計

注意:
  - 若無資料集快取，分析會使用固定隨機種子的合成 batch（B=8, T=200）
  - 分析的核心為驗證 Cross-Attn 在單一 key 的設計是否導致注意力權重恆為 1
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn

from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn


def load_config(results_dir: Path) -> Dict[str, Any]:
    cfg_path = results_dir / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到 config.json: {cfg_path}")
    with open(cfg_path, 'r') as f:
        return json.load(f)


def list_checkpoints(results_dir: Path) -> List[Path]:
    # 依 epoch 排序
    ckpts = sorted(results_dir.glob('checkpoint_epoch_*.pth'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    return ckpts


@torch.no_grad()
def build_model_from_ckpt(ckpt_path: Path, device: torch.device,
                          d_model: int, nhead: int, num_layers: int,
                          dim_feedforward: int, dropout: float) -> ZeroShotDenoisingTransformerCrossAttn:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    codebook = state['codebook']  # buffer

    model = ZeroShotDenoisingTransformerCrossAttn(
        codebook=codebook,
        speaker_embed_dim=256,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_batch(device: torch.device, vocab_size: int = 4096,
                  B: int = 8, T: int = 200, speaker_dim: int = 256,
                  seed: int = 20251105) -> Dict[str, torch.Tensor]:
    """固定隨機種子以穩定分析結果。"""
    g = torch.Generator(device='cpu').manual_seed(seed)
    noisy_tokens = torch.randint(low=0, high=vocab_size, size=(B, T), generator=g)
    speaker_embeddings = torch.randn(B, speaker_dim, generator=g)
    return {
        'noisy_tokens': noisy_tokens.to(device),
        'speaker_embeddings': speaker_embeddings.to(device),
    }


@torch.no_grad()
def analyze_cross_attn(model: ZeroShotDenoisingTransformerCrossAttn,
                       batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    直接呼叫 cross_attn 模組以擷取 attn_output 和 attn_weights。
    注意: 使用單一 key (S=1) 的 MultiheadAttention，softmax 會退化為恆為 1。
    這裡驗證該退化是否存在，並衡量 attn_output 是否跨 token 相同（恆定向量）。
    """
    noisy_tokens = batch['noisy_tokens']
    speaker_embeddings = batch['speaker_embeddings']

    # Step 1: token embedding + pos enc
    token_emb = model.codebook[noisy_tokens]  # (B, T, D)
    token_emb = model.pos_encoding(token_emb)

    # Step 2: speaker proj
    speaker_emb = model.speaker_proj(speaker_embeddings)  # (B, D)
    speaker_kv = speaker_emb.unsqueeze(1)  # (B, 1, D)

    # Step 3: cross-attn via fusion module (K>1 aware)
    fused_emb, attn_weights = model.cross_attn_fusion(
        token_emb=token_emb,
        speaker_emb=speaker_emb,
    )  # attn_weights: (B, T, K)

    # 衡量 attn_weights 是否恆為 1
    aw = attn_weights
    aw_stats = {
        'shape': tuple(aw.shape),
        'min': float(aw.min().item()),
        'max': float(aw.max().item()),
        'mean': float(aw.mean().item()),
        'std': float(aw.std().item()),
    }

    # 衡量 attn_output 是否在序列維度上常數（不依賴 token）
    # token-wise 方差: 沿 T 維度的方差再取平均
    # 使用 attn_output 無法直接取得（已融合到 fused_emb），改用 attn_weights 的 token 變異作 proxy
    # 若需要 attn_output 的 token-variance，可改為復現 spk_tokens 並直接調用內部 cross_attn
    token_var = attn_weights.var(dim=1).mean().item()
    token_var_per_dim = token_var

    # 對比 token_emb 的序列變化幅度
    token_emb_token_var = token_emb.var(dim=1).mean().item()

    # fused 與原始 token_emb 的差值量級
    delta = fused_emb - token_emb
    delta_l2 = float(delta.norm(dim=-1).mean().item())

    return {
        'attn_weights': aw_stats,
        'attn_output_token_var_mean': float(token_var),
        'attn_output_token_var_mean_per_dim': float(token_var_per_dim),
        'token_emb_token_var_mean': float(token_emb_token_var),
        'delta_fused_minus_token_l2_mean': delta_l2,
    }


@torch.no_grad()
def analyze_encoder_layers(model: ZeroShotDenoisingTransformerCrossAttn,
                           batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    noisy_tokens = batch['noisy_tokens']
    speaker_embeddings = batch['speaker_embeddings']

    # 註冊 hooks 以擷取每層輸出
    layer_outputs: Dict[str, torch.Tensor] = {}

    def make_hook(layer_name: str):
        def hook(_, __, output):
            layer_outputs[layer_name] = output.detach()
        return hook

    handles = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        handles.append(layer.register_forward_hook(make_hook(f'encoder_layer_{i}')))

    # 前向
    _ = model(noisy_tokens, speaker_embeddings, return_logits=True)

    # 計算統計
    stats = {}
    for name, out in layer_outputs.items():
        stats[name] = {
            'shape': tuple(out.shape),
            'mean': float(out.mean().item()),
            'std': float(out.std().item()),
            'min': float(out.min().item()),
            'max': float(out.max().item()),
        }

    for h in handles:
        h.remove()

    return stats


def param_group_norms(model: ZeroShotDenoisingTransformerCrossAttn) -> Dict[str, float]:
    """彙整幾個關鍵參數群的 L2 範數。"""
    norms = {}

    def safe_norm(params: List[nn.Parameter]) -> float:
        if not params:
            return float('nan')
        with torch.no_grad():
            total = torch.zeros((), device=params[0].device)
            for p in params:
                total = total + (p.detach()**2).sum()
            return float(torch.sqrt(total).item())

    # speaker proj
    sp = [p for n, p in model.named_parameters() if n.startswith('speaker_proj')]
    norms['speaker_proj'] = safe_norm(sp)

    # cross-attn in_proj/out_proj
    ca_in = [p for n, p in model.named_parameters() if 'cross_attn_fusion.cross_attn.in_proj' in n]
    ca_out = [p for n, p in model.named_parameters() if 'cross_attn_fusion.cross_attn.out_proj' in n]
    norms['cross_attn_in_proj'] = safe_norm(ca_in)
    norms['cross_attn_out_proj'] = safe_norm(ca_out)

    # transformer encoder 層（首尾層作代表）
    # 注意: PyTorch 的 MultiheadAttention 使用 in_proj_weight 統合 QKV
    enc0 = [p for n, p in model.named_parameters() if n.startswith('transformer_encoder.layers.0')]
    enclast = [p for n, p in model.named_parameters() if n.startswith(f'transformer_encoder.layers.{len(model.transformer_encoder.layers)-1}')]
    norms['encoder_layer_0'] = safe_norm(enc0)
    norms['encoder_layer_last'] = safe_norm(enclast)

    # output proj
    out = [p for n, p in model.named_parameters() if n.startswith('output_proj')]
    norms['output_proj'] = safe_norm(out)

    return norms


def main():
    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='包含 config.json 與 checkpoint_epoch_*.pth 的資料夾')
    parser.add_argument('--use_real_cache', action='store_true',
                        help='若可取得指定 cache 位置，則用真實 batch 分析')
    parser.add_argument('--cache_dir', type=str, default='./data',
                        help='包含 val_cache.pt 的資料夾')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(results_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpts = list_checkpoints(results_dir)
    if not ckpts:
        raise FileNotFoundError(f"{results_dir} 中未找到 checkpoint_epoch_*.pth")

    # 準備 batch（預設使用合成資料，除非顯式要求 real cache 且可用）
    batch = None
    if args.use_real_cache:
        try:
            from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn
            val_cache_path = Path(args.cache_dir) / 'val_cache.pt'
            if val_cache_path.exists():
                from torch.utils.data import DataLoader
                ds = ZeroShotAudioDatasetCached(str(val_cache_path))
                dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=cached_collate_fn, num_workers=0)
                b = next(iter(dl))
                batch = {
                    'noisy_tokens': b['noisy_tokens'].to(device),
                    'speaker_embeddings': b['speaker_embeddings'].to(device),
                }
                print(f'使用真實 val_cache 批次進行分析: {val_cache_path}')
        except Exception as e:
            print(f"載入真實快取失敗，改用合成資料: {e}")

    if batch is None:
        batch = prepare_batch(device=device,
                              vocab_size=4096,
                              B=8,
                              T=200,
                              speaker_dim=256)
        print('使用合成固定批次進行分析')

    csv_rows = []

    for ckpt_path in ckpts:
        epoch = int(ckpt_path.stem.split('_')[-1])
        print(f"\n分析 checkpoint: epoch {epoch} ...")

        model = build_model_from_ckpt(
            ckpt_path=ckpt_path,
            device=device,
            d_model=cfg.get('d_model', 512),
            nhead=cfg.get('nhead', 8),
            num_layers=cfg.get('num_layers', 4),
            dim_feedforward=cfg.get('dim_feedforward', 2048),
            dropout=cfg.get('dropout', 0.1),
        )

        cross_attn_stats = analyze_cross_attn(model, batch)
        layer_stats = analyze_encoder_layers(model, batch)
        group_norms = param_group_norms(model)

        # 寫出詳細 JSON
        out_json = analysis_dir / f'stats_epoch_{epoch}.json'
        with open(out_json, 'w') as f:
            json.dump({
                'epoch': epoch,
                'cross_attn': cross_attn_stats,
                'encoder_layers': layer_stats,
                'param_group_norms': group_norms,
            }, f, indent=2)

        # 匯總到 CSV 行
        csv_rows.append({
            'epoch': epoch,
            'attn_w_mean': cross_attn_stats['attn_weights']['mean'],
            'attn_w_std': cross_attn_stats['attn_weights']['std'],
            'attn_w_min': cross_attn_stats['attn_weights']['min'],
            'attn_w_max': cross_attn_stats['attn_weights']['max'],
            'attn_output_token_var_mean': cross_attn_stats['attn_output_token_var_mean'],
            'token_emb_token_var_mean': cross_attn_stats['token_emb_token_var_mean'],
            'delta_fused_minus_token_l2_mean': cross_attn_stats['delta_fused_minus_token_l2_mean'],
            'norm_speaker_proj': group_norms['speaker_proj'],
            'norm_crossattn_in': group_norms['cross_attn_in_proj'],
            'norm_crossattn_out': group_norms['cross_attn_out_proj'],
            'norm_enc0': group_norms['encoder_layer_0'],
            'norm_enclast': group_norms['encoder_layer_last'],
            'norm_output_proj': group_norms['output_proj'],
        })

    # 依 epoch 排序並輸出 CSV
    csv_rows.sort(key=lambda r: r['epoch'])
    csv_path = analysis_dir / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n✓ 完成。輸出: {csv_path} 與 stats_epoch_XX.json")


if __name__ == '__main__':
    main()
