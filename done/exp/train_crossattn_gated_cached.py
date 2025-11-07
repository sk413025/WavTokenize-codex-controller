import argparse
import math
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model_zeroshot_crossattn_gated import ZeroShotDenoisingTransformerCrossAttnGated
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn
from decoder.pretrained import WavTokenizer


def setup_logger(output_dir):
    log_file = Path(output_dir) / 'training.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device, *, margin_cfg=None, dir_cfg=None):
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    total_dir_loss, total_dir_count = 0.0, 0
    for batch in dataloader:
        noisy = batch['noisy_tokens'].to(device)
        clean = batch['clean_tokens'].to(device)
        spk = batch['speaker_embeddings'].to(device)
        # forward with margin-aware gating if configured
        logits = model(noisy, spk, return_logits=True, labels=clean if margin_cfg is not None else None, margin_cfg=margin_cfg)
        B,T,V = logits.shape
        ce_loss = criterion(logits.view(B*T, V), clean.view(B*T))

        # Optional directional auxiliary loss
        dir_loss = torch.tensor(0.0, device=device)
        dir_count = 0
        if dir_cfg is not None and dir_cfg.get('weight', 0.0) > 0.0:
            # compute baseline logits with gate disabled (speaker off)
            zeros_gate = torch.zeros(B, T, 1, device=device)
            logits_off = model(noisy, spk, return_logits=True, g_override=zeros_gate)
            # log-probs
            logp_on = torch.log_softmax(logits, dim=-1)
            logp_off = torch.log_softmax(logits_off, dim=-1)
            # target indices
            tgt = clean.view(B, T, 1)
            lp_t_on = logp_on.gather(-1, tgt).squeeze(-1)
            lp_t_off = logp_off.gather(-1, tgt).squeeze(-1)
            # competitor indices from off logits (exclude target)
            probs_off = torch.softmax(logits_off, dim=-1)
            probs_off_excl = probs_off.clone()
            probs_off_excl.scatter_(-1, tgt, 0.0)
            c2_idx = probs_off_excl.argmax(dim=-1, keepdim=True)  # (B,T,1)
            lp_c2_on = logp_on.gather(-1, c2_idx).squeeze(-1)
            lp_c2_off = logp_off.gather(-1, c2_idx).squeeze(-1)
            # directional term
            dir_term = (lp_t_on - lp_c2_on) - (lp_t_off - lp_c2_off)
            # optionally restrict to mid-margin tokens under off case
            if dir_cfg.get('mid_only', True):
                # margin in probability under off logits
                top2 = torch.topk(torch.softmax(logits_off, dim=-1), k=2, dim=-1).values
                margin_off = top2[..., 0] - top2[..., 1]
                low_thr = float(dir_cfg.get('low_thr', 0.02))
                mid_thr = float(dir_cfg.get('mid_thr', 0.4))
                mask = (margin_off >= low_thr) & (margin_off < mid_thr)
                if mask.any():
                    dir_loss = -dir_term[mask].mean()
                    dir_count = int(mask.sum().item())
                else:
                    dir_loss = torch.tensor(0.0, device=device)
                    dir_count = 0
            else:
                dir_loss = -dir_term.mean()
                dir_count = B*T

        loss = ce_loss + (dir_cfg.get('weight', 0.0) * dir_loss if dir_cfg is not None else 0.0)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == clean).sum().item()
        total_tokens += B*T
        total_dir_loss += float(dir_loss.item()) if isinstance(dir_loss, torch.Tensor) else 0.0
        total_dir_count += dir_count
    out = {'loss': total_loss/len(dataloader), 'accuracy': total_correct/total_tokens*100}
    if dir_cfg is not None and dir_cfg.get('weight', 0.0) > 0.0:
        out['dir_loss'] = total_dir_loss/ max(1, len(dataloader))
        out['dir_count'] = total_dir_count
    return out


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, *, margin_cfg=None):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for batch in dataloader:
        noisy = batch['noisy_tokens'].to(device)
        clean = batch['clean_tokens'].to(device)
        spk = batch['speaker_embeddings'].to(device)
        logits = model(noisy, spk, return_logits=True, labels=None, margin_cfg=margin_cfg)
        B,T,V = logits.shape
        loss = criterion(logits.view(B*T, V), clean.view(B*T))
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == clean).sum().item()
        total_tokens += B*T
    return {'loss': total_loss/len(dataloader), 'accuracy': total_correct/total_tokens*100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', default='./data')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--num_layers', type=int, default=4)
    ap.add_argument('--dim_feedforward', type=int, default=2048)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--speaker_tokens', type=int, default=4)
    # Margin-aware gating schedule
    ap.add_argument('--margin_aware', action='store_true')
    ap.add_argument('--low_thr', type=float, default=0.02)
    ap.add_argument('--mid_thr', type=float, default=0.4)
    ap.add_argument('--mid_amp', type=float, default=1.5)
    ap.add_argument('--high_amp', type=float, default=0.5)
    # Directional auxiliary loss
    ap.add_argument('--dir_loss_weight', type=float, default=0.0)
    ap.add_argument('--dir_mid_only', action='store_true')
    # Gate initialization bias (sigmoid inverse)
    ap.add_argument('--gate_init', type=float, default=None, help='Initial gate value in (0,1); sets final gate bias via logit(init).')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_epochs', type=int, default=10)
    ap.add_argument('--learning_rate', type=float, default=1e-4)
    args = ap.parse_args()

    # env
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dirs
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/crossattn_k4_gate_{args.num_epochs}ep_{timestamp}'
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)
    logger.info('Cross-Attn Gated Experiment (small run)')

    # cache
    cache_dir = Path(args.cache_dir)
    train_ds = ZeroShotAudioDatasetCached(str(cache_dir/'train_cache.pt'))
    val_ds = ZeroShotAudioDatasetCached(str(cache_dir/'val_cache.pt'))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=cached_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=cached_collate_fn, pin_memory=True)

    # wavtokenizer to get codebook
    # Keep WavTokenizer on CPU to save GPU memory; extract codebook then free
    wavtokenizer = WavTokenizer.from_pretrained0802(
        'config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
        'models/wavtokenizer_large_speech_320_24k.ckpt'
    )
    wavtokenizer.eval()
    with torch.no_grad():
        codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook.detach().cpu()
    try:
        del wavtokenizer
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # model/opt
    model = ZeroShotDenoisingTransformerCrossAttnGated(
        codebook=codebook, speaker_embed_dim=256, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        speaker_tokens=args.speaker_tokens
    ).to(device)
    # Optional gate init
    if args.gate_init is not None:
        # gate seq: [LN, Linear, ReLU, Linear_out, Sigmoid]
        try:
            bias = math.log(args.gate_init/(1.0-args.gate_init))
        except Exception:
            bias = 0.0
        with torch.no_grad():
            model.cross_attn_fusion.gate[-2].bias.fill_(bias)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    # save config
    with open(outdir/'config.json','w') as f: json.dump(vars(args), f, indent=2)

    # Compose configs
    margin_cfg = None
    if args.margin_aware:
        margin_cfg = {'low_thr': args.low_thr, 'mid_thr': args.mid_thr, 'mid_amp': args.mid_amp, 'high_amp': args.high_amp}
    dir_cfg = None
    if args.dir_loss_weight > 0.0:
        dir_cfg = {'weight': args.dir_loss_weight, 'mid_only': args.dir_mid_only, 'low_thr': args.low_thr, 'mid_thr': args.mid_thr}

    best_val = 0.0
    for epoch in range(1, args.num_epochs+1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device, margin_cfg=margin_cfg, dir_cfg=dir_cfg)
        va = validate_epoch(model, val_loader, criterion, device, margin_cfg=margin_cfg)
        logger.info(f'Epoch {epoch}/{args.num_epochs}')
        if dir_cfg is not None:
            logger.info(f'  Train - Loss: {tr["loss"]:.4f}, CE+Dir, Acc: {tr["accuracy"]:.2f}% (dir_loss={tr.get("dir_loss",0.0):.4f}, dir_count={tr.get("dir_count",0)})')
        else:
            logger.info(f'  Train - Loss: {tr["loss"]:.4f}, Acc: {tr["accuracy"]:.2f}%')
        logger.info(f'  Val   - Loss: {va["loss"]:.4f}, Acc: {va["accuracy"]:.2f}%')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, outdir/f'checkpoint_epoch_{epoch}.pth')
        if va['accuracy'] > best_val:
            best_val = va['accuracy']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, outdir/'best_model.pth')
            logger.info(f'  ✓ 保存最佳模型 (Val Acc: {best_val:.2f}%)')

if __name__ == '__main__':
    main()
