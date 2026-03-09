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


def train_epoch(model, dataloader, optimizer, criterion, device, *, epoch_idx:int=1,
                margin_cfg=None, dir_cfg=None, qgate_cfg=None,
                gate_mode:str='quantile', dirgate_cfg=None, hard_cfg=None):
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    total_dir_loss, total_dir_count = 0.0, 0
    for batch in dataloader:
        noisy = batch['noisy_tokens'].to(device)
        clean = batch['clean_tokens'].to(device)
        spk = batch['speaker_embeddings'].to(device)
        B, T = noisy.shape
        valid_mask = (clean != 0)

        # Build gating override according to requested mode
        g_sched = None
        base_gate_for_reg = None
        zeros_gate = torch.zeros(B, T, 1, device=device)
        logits_off = None
        margin_off = None
        thr_low = None; thr_mid = None

        # Determine if we need off-logits and margin bins
        need_bins = (qgate_cfg is not None and qgate_cfg.get('enable', False)) \
                    or (hard_cfg is not None and hard_cfg.get('mid_off_warmup_epochs', 0) > 0) \
                    or (dir_cfg is not None and dir_cfg.get('mid_only', True)) \
                    or (dirgate_cfg is not None and dirgate_cfg.get('mid_off_warmup_epochs', 0) > 0)
        if need_bins:
            with torch.no_grad():
                logits_off = model(noisy, spk, return_logits=True, g_override=zeros_gate)
                top2_off = torch.topk(logits_off, k=2, dim=-1).values  # (B,T,2)
                margin_off = (top2_off[...,0] - top2_off[...,1])  # (B,T)
            mvec = margin_off[valid_mask]
            if mvec.numel() >= 8:
                p_low = float((qgate_cfg or {}).get('p_low', 0.2))
                p_mid = float((qgate_cfg or {}).get('p_mid', 0.6))
                thr_low = torch.quantile(mvec, p_low)
                thr_mid = torch.quantile(mvec, p_mid)
            else:
                thr_low = torch.tensor(0.02, device=device)
                thr_mid = torch.tensor(0.40, device=device)

        # Base gate (no grad) to serve as reference for scheduling/transforms
        with torch.no_grad():
            _, _, base_gate = model(noisy, spk, return_logits=True, return_attention=True)
            base_gate_for_reg = base_gate.detach()

        # Mode: quantile (default behavior)
        if gate_mode == 'quantile' and (qgate_cfg is not None and qgate_cfg.get('enable', False)):
            g_min = float(qgate_cfg.get('g_min', 0.1))
            g_max = float(qgate_cfg.get('g_max', 0.9))
            g_sched = base_gate.clamp(min=g_min, max=g_max)
        # Mode: direction-aware (gate乘上方向性因子)
        elif gate_mode == 'dir' and (dirgate_cfg is not None):
            with torch.no_grad():
                _log, attn_vec, token_vec, g_base2 = model(noisy, spk, return_logits=True, return_fusion=True)
            # competitor by off logits
            if logits_off is None:
                with torch.no_grad():
                    logits_off = model(noisy, spk, return_logits=True, g_override=zeros_gate)
            tgt = clean.view(B, T, 1)
            probs_off = torch.softmax(logits_off, dim=-1)
            probs_off_excl = probs_off.clone(); probs_off_excl.scatter_(-1, tgt, 0.0)
            c2_idx = probs_off_excl.argmax(dim=-1)  # (B,T)
            # delta direction
            W = model.output_proj.weight  # (V,D)
            delta = W.index_select(0, clean.view(-1)).view(B, T, -1) - W.index_select(0, c2_idx.view(-1)).view(B, T, -1)
            attn_unit = attn_vec / (attn_vec.norm(dim=-1, keepdim=True) + 1e-8)
            delta_unit = delta / (delta.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sim = (attn_unit * delta_unit).sum(dim=-1)  # (B,T)
            tau = float(dirgate_cfg.get('tau', 0.0))
            k = float(dirgate_cfg.get('k', 5.0))
            g_min = float(dirgate_cfg.get('g_min', 0.1))
            g_max = float(dirgate_cfg.get('g_max', 0.9))
            g_dir = torch.sigmoid(k * (cos_sim - tau))
            g_new = (g_base2.squeeze(-1) * g_dir).clamp(min=g_min, max=g_max).unsqueeze(-1)
            # optional mid-off warmup
            warm_mid_off = int(dirgate_cfg.get('mid_off_warmup_epochs', 0))
            if warm_mid_off > 0 and margin_off is not None and epoch_idx <= warm_mid_off:
                mid_mask = (margin_off >= thr_low) & (margin_off < thr_mid) & valid_mask
                if mid_mask.any():
                    g_new[mid_mask] = 0.0
            g_sched = g_new
        # Mode: hard-binary gate with anneal and mid-off warmup
        elif gate_mode == 'hard' and (hard_cfg is not None):
            gb = base_gate_for_reg  # (B,T,1)
            thr = float(hard_cfg.get('thr', 0.5))
            t_start = float(hard_cfg.get('temp_start', 1.0))
            t_end = float(hard_cfg.get('temp_end', 0.1))
            t_epochs = int(hard_cfg.get('temp_anneal_epochs', 50))
            frac = min(1.0, max(0.0, (epoch_idx-1)/max(1, t_epochs)))
            temp = t_start * (t_end / max(1e-6, t_start)) ** frac
            g_soft = torch.sigmoid((gb - thr) / max(1e-6, temp))
            g_hard = (g_soft > 0.5).float()
            g_st = g_hard + (g_soft - g_soft.detach())
            g_min = float(hard_cfg.get('g_min', 0.1)); g_max = float(hard_cfg.get('g_max', 0.9))
            g_bin = g_min * (1.0 - g_st) + g_max * g_st
            warm_mid_off = int(hard_cfg.get('mid_off_warmup_epochs', 0))
            if warm_mid_off > 0 and margin_off is not None and epoch_idx <= warm_mid_off:
                mid_mask = (margin_off >= thr_low) & (margin_off < thr_mid) & valid_mask
                if mid_mask.any():
                    g_bin[mid_mask] = 0.0
            # Per-bin clamp: mid to [mid_min, mid_max], hi to <= hi_max
            mid_min = hard_cfg.get('mid_min', None)
            mid_max = hard_cfg.get('mid_max', None)
            hi_max  = hard_cfg.get('hi_max', None)
            if margin_off is not None and (mid_min is not None or mid_max is not None or hi_max is not None):
                mid_mask2 = (margin_off >= thr_low) & (margin_off < thr_mid) & valid_mask
                hi_mask2  = (margin_off >= thr_mid) & valid_mask
                if mid_mask2.any() and (mid_min is not None or mid_max is not None):
                    lo = float(mid_min) if mid_min is not None else 0.0
                    hi = float(mid_max) if mid_max is not None else 1.0
                    g_bin[mid_mask2] = g_bin[mid_mask2].clamp(min=lo, max=hi)
                if hi_mask2.any() and (hi_max is not None):
                    g_bin[hi_mask2] = g_bin[hi_mask2].clamp(max=float(hi_max))
            g_sched = g_bin

        # forward (use g_sched if exists)
        logits = model(noisy, spk, return_logits=True, g_override=g_sched)
        B,T,V = logits.shape
        ce_loss = criterion(logits.view(B*T, V), clean.view(B*T))

        # Optional directional auxiliary loss
        dir_loss = torch.tensor(0.0, device=device)
        dir_count = 0
        if dir_cfg is not None and dir_cfg.get('weight', 0.0) > 0.0:
            # compute baseline logits with gate disabled (speaker off)
            if logits_off is None:
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
                if margin_off is None:
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

            # competitor constraint to suppress on-competitor rise
            comp_w = float(dir_cfg.get('comp_weight', 0.0))
            comp_eps = float(dir_cfg.get('comp_eps', 0.0))
            if comp_w > 0.0:
                z_on_c = logits.gather(-1, c2_idx).squeeze(-1)
                z_off_c = logits_off.gather(-1, c2_idx).squeeze(-1)
                rise = (z_on_c - z_off_c) - comp_eps
                comp_term = torch.relu(rise).pow(2)
                if dir_cfg.get('mid_only', True):
                    mask = (margin_off >= low_thr) & (margin_off < mid_thr)
                    if mask.any():
                        comp_term = comp_term[mask]
                dir_loss = dir_loss + comp_w * comp_term.mean()

        # Optional: gate bin-wise regularization and temporal smoothing
        gate_reg_loss = torch.tensor(0.0, device=device)
        smooth_loss = torch.tensor(0.0, device=device)
        if gate_mode == 'quantile' and (qgate_cfg is not None and qgate_cfg.get('enable', False)):
            # Recompute base gate for current batch (if not already)
            if base_gate_for_reg is None:
                with torch.no_grad():
                    _, _, base_gate = model(noisy, spk, return_logits=True, return_attention=True)
                    base_gate_for_reg = base_gate.detach()

            # recompute thresholds if not in scope (fallback)
            if 'thr_low' not in locals():
                zeros_gate = torch.zeros(B, T, 1, device=device)
                with torch.no_grad():
                    logits_off = model(noisy, spk, return_logits=True, g_override=zeros_gate)
                    top2_off = torch.topk(logits_off, k=2, dim=-1).values
                    margin_off = (top2_off[...,0] - top2_off[...,1])
                mvec = margin_off[valid_mask]
                thr_low = torch.quantile(mvec, qgate_cfg.get('p_low', 0.2)) if mvec.numel()>=8 else torch.tensor(0.02, device=device)
                thr_mid = torch.quantile(mvec, qgate_cfg.get('p_mid', 0.6)) if mvec.numel()>=8 else torch.tensor(0.40, device=device)

            # masks
            low_mask = (margin_off < thr_low) & valid_mask
            mid_mask = (margin_off >= thr_low) & (margin_off < thr_mid) & valid_mask
            hi_mask  = (margin_off >= thr_mid) & valid_mask

            # Bin-wise L2 towards targets (after warmup epochs)
            warm = int(qgate_cfg.get('reg_warmup_epochs', 20))
            if epoch_idx > warm:
                t_low = float(qgate_cfg.get('t_low', 0.2))
                t_mid = float(qgate_cfg.get('t_mid', 0.3))
                t_hi  = float(qgate_cfg.get('t_hi', 0.8))
                w_low = float(qgate_cfg.get('w_low', 0.01))
                w_mid = float(qgate_cfg.get('w_mid', 0.01))
                w_hi  = float(qgate_cfg.get('w_hi', 0.01))
                g = base_gate_for_reg.squeeze(-1)
                if low_mask.any():
                    gate_reg_loss = gate_reg_loss + w_low * ((g[low_mask] - t_low) ** 2).mean()
                if mid_mask.any():
                    gate_reg_loss = gate_reg_loss + w_mid * ((g[mid_mask] - t_mid) ** 2).mean()
                if hi_mask.any():
                    gate_reg_loss = gate_reg_loss + w_hi * ((g[hi_mask] - t_hi) ** 2).mean()

            # Temporal smoothing on scheduled gate (or base if no sched)
            gs = g_sched if g_sched is not None else base_gate_for_reg
            w_smooth = float(qgate_cfg.get('smooth_w', 0.0))
            if w_smooth > 0.0 and T > 1:
                diffs = (gs[:,1:,:] - gs[:,:-1,:])
                # only count valid positions (both t and t-1 valid)
                vm = valid_mask[:,1:] & valid_mask[:,:-1]
                if vm.any():
                    smooth_loss = w_smooth * (diffs[vm].pow(2).mean())

        loss = ce_loss \
               + ((dir_cfg.get('weight', 0.0) * dir_loss) if dir_cfg is not None else 0.0) \
               + gate_reg_loss + smooth_loss
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
def validate_epoch(model, dataloader, criterion, device, *, margin_cfg=None,
                   gate_mode: str = 'quantile', qgate_cfg=None, dirgate_cfg=None, hard_cfg=None,
                   epoch_idx: int = 0):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for batch in dataloader:
        noisy = batch['noisy_tokens'].to(device)
        clean = batch['clean_tokens'].to(device)
        spk = batch['speaker_embeddings'].to(device)
        B, T = noisy.shape
        # Optional eval-time gate override
        g_sched = None
        if gate_mode in ('quantile','dir','hard'):
            zeros_gate = torch.zeros(B, T, 1, device=device)
            logits_off = None; margin_off = None; thr_low=None; thr_mid=None
            need_bins = (qgate_cfg is not None and qgate_cfg.get('enable', False)) \
                        or (hard_cfg is not None and hard_cfg.get('mid_off_warmup_epochs', 0) > 0) \
                        or (dirgate_cfg is not None and dirgate_cfg.get('mid_off_warmup_epochs', 0) > 0)
            if need_bins:
                logits_off = model(noisy, spk, return_logits=True, g_override=zeros_gate)
                top2_off = torch.topk(logits_off, k=2, dim=-1).values
                margin_off = (top2_off[...,0] - top2_off[...,1])
                valid_mask = (clean != 0)
                mvec = margin_off[valid_mask]
                if mvec.numel() >= 8:
                    p_low = float((qgate_cfg or {}).get('p_low', (dirgate_cfg or {}).get('p_low', 0.2)))
                    p_mid = float((qgate_cfg or {}).get('p_mid', (dirgate_cfg or {}).get('p_mid', 0.6)))
                    thr_low = torch.quantile(mvec, p_low)
                    thr_mid = torch.quantile(mvec, p_mid)
                else:
                    thr_low = torch.tensor(0.02, device=device)
                    thr_mid = torch.tensor(0.40, device=device)
            # base gate (no grad)
            _, _, base_gate = model(noisy, spk, return_logits=True, return_attention=True)
            if gate_mode == 'quantile' and (qgate_cfg is not None and qgate_cfg.get('enable', False)):
                g_min = float(qgate_cfg.get('g_min', 0.1)); g_max = float(qgate_cfg.get('g_max', 0.9))
                g_sched = base_gate.clamp(min=g_min, max=g_max)
            elif gate_mode == 'dir' and (dirgate_cfg is not None):
                logits_tmp, attn_vec, token_vec, g_base2 = model(noisy, spk, return_logits=True, return_fusion=True)
                if logits_off is None:
                    logits_off = model(noisy, spk, return_logits=True, g_override=zeros_gate)
                tgt = clean.view(B, T, 1)
                probs_off = torch.softmax(logits_off, dim=-1)
                probs_off_excl = probs_off.clone(); probs_off_excl.scatter_(-1, tgt, 0.0)
                c2_idx = probs_off_excl.argmax(dim=-1)
                W = model.output_proj.weight
                delta = W.index_select(0, clean.view(-1)).view(B, T, -1) - W.index_select(0, c2_idx.view(-1)).view(B, T, -1)
                attn_unit = attn_vec / (attn_vec.norm(dim=-1, keepdim=True) + 1e-8)
                delta_unit = delta / (delta.norm(dim=-1, keepdim=True) + 1e-8)
                cos_sim = (attn_unit * delta_unit).sum(dim=-1)
                tau = float(dirgate_cfg.get('tau', 0.0)); k = float(dirgate_cfg.get('k', 5.0))
                g_min = float(dirgate_cfg.get('g_min', 0.1)); g_max = float(dirgate_cfg.get('g_max', 0.9))
                g_dir = torch.sigmoid(k * (cos_sim - tau))
                g_new = (g_base2.squeeze(-1) * g_dir).clamp(min=g_min, max=g_max).unsqueeze(-1)
                warm_mid_off = int(dirgate_cfg.get('mid_off_warmup_epochs', 0))
                if warm_mid_off > 0 and margin_off is not None and epoch_idx <= warm_mid_off:
                    mid_mask = (margin_off >= thr_low) & (margin_off < thr_mid) & (clean != 0)
                    if mid_mask.any():
                        g_new[mid_mask] = 0.0
                g_sched = g_new
            elif gate_mode == 'hard' and (hard_cfg is not None):
                gb = base_gate
                thr = float(hard_cfg.get('thr', 0.5))
                t_start = float(hard_cfg.get('temp_start', 1.0)); t_end = float(hard_cfg.get('temp_end', 0.1))
                t_epochs = int(hard_cfg.get('temp_anneal_epochs', 50))
                frac = min(1.0, max(0.0, (max(1, epoch_idx)-1)/max(1, t_epochs)))
                temp = t_start * (t_end / max(1e-6, t_start)) ** frac
                g_soft = torch.sigmoid((gb - thr) / max(1e-6, temp))
                g_hard = (g_soft > 0.5).float(); g_st = g_hard + (g_soft - g_soft.detach())
                g_min = float(hard_cfg.get('g_min', 0.1)); g_max = float(hard_cfg.get('g_max', 0.9))
                g_bin = g_min * (1.0 - g_st) + g_max * g_st
                warm_mid_off = int(hard_cfg.get('mid_off_warmup_epochs', 0))
                if warm_mid_off > 0 and margin_off is not None and epoch_idx <= warm_mid_off:
                    mid_mask = (margin_off >= thr_low) & (margin_off < thr_mid) & (clean != 0)
                    if mid_mask.any():
                        g_bin[mid_mask] = 0.0
                # Per-bin clamp in eval
                mid_min = hard_cfg.get('mid_min', None)
                mid_max = hard_cfg.get('mid_max', None)
                hi_max  = hard_cfg.get('hi_max', None)
                if margin_off is not None and (mid_min is not None or mid_max is not None or hi_max is not None):
                    mid_mask2 = (margin_off >= thr_low) & (margin_off < thr_mid) & (clean != 0)
                    hi_mask2  = (margin_off >= thr_mid) & (clean != 0)
                    if mid_mask2.any() and (mid_min is not None or mid_max is not None):
                        lo = float(mid_min) if mid_min is not None else 0.0
                        hi = float(mid_max) if mid_max is not None else 1.0
                        g_bin[mid_mask2] = g_bin[mid_mask2].clamp(min=lo, max=hi)
                    if hi_mask2.any() and (hi_max is not None):
                        g_bin[hi_mask2] = g_bin[hi_mask2].clamp(max=float(hi_max))
                g_sched = g_bin

        logits = model(noisy, spk, return_logits=True, labels=None, margin_cfg=margin_cfg, g_override=g_sched)
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
    ap.add_argument('--dir_comp_weight', type=float, default=0.0, help='weight of competitor rise penalty')
    ap.add_argument('--dir_comp_eps', type=float, default=0.0, help='tolerance for competitor rise (logit)')
    # Gating mode
    ap.add_argument('--gate_mode', type=str, default='quantile', choices=['quantile','dir','hard'])
    # Direction-aware gate params
    ap.add_argument('--dir_gate_tau', type=float, default=0.0)
    ap.add_argument('--dir_gate_k', type=float, default=5.0)
    ap.add_argument('--dir_gate_mid_off_warmup_epochs', type=int, default=0)
    # Hard-binary gate params
    ap.add_argument('--hard_thr', type=float, default=0.5)
    ap.add_argument('--hard_temp_start', type=float, default=1.0)
    ap.add_argument('--hard_temp_end', type=float, default=0.1)
    ap.add_argument('--hard_temp_anneal_epochs', type=int, default=50)
    ap.add_argument('--hard_mid_off_warmup_epochs', type=int, default=0)
    # Per-bin clamp (hard mode)
    ap.add_argument('--gate_mid_min', type=float, default=None)
    ap.add_argument('--gate_mid_max', type=float, default=None)
    ap.add_argument('--gate_hi_max', type=float, default=None)
    # Gate initialization bias (sigmoid inverse)
    ap.add_argument('--gate_init', type=float, default=None, help='Initial gate value in (0,1); sets final gate bias via logit(init).')
    # Quantile-gate schedule & regularization
    ap.add_argument('--qgate_enable', action='store_true')
    ap.add_argument('--margin_p_low', type=float, default=0.2)
    ap.add_argument('--margin_p_mid', type=float, default=0.6)
    ap.add_argument('--gate_min', type=float, default=0.1)
    ap.add_argument('--gate_max', type=float, default=0.9)
    ap.add_argument('--gate_reg_warmup_epochs', type=int, default=20)
    ap.add_argument('--gate_reg_w_low', type=float, default=0.01)
    ap.add_argument('--gate_reg_w_mid', type=float, default=0.01)
    ap.add_argument('--gate_reg_w_hi', type=float, default=0.01)
    ap.add_argument('--gate_reg_t_low', type=float, default=0.2)
    ap.add_argument('--gate_reg_t_mid', type=float, default=0.3)
    ap.add_argument('--gate_reg_t_hi', type=float, default=0.8)
    ap.add_argument('--gate_smooth_weight', type=float, default=0.0)
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
        dir_cfg = {
            'weight': args.dir_loss_weight,
            'mid_only': args.dir_mid_only,
            'low_thr': args.low_thr,
            'mid_thr': args.mid_thr,
            'comp_weight': args.dir_comp_weight,
            'comp_eps': args.dir_comp_eps,
        }

    # Compose configs
    margin_cfg = None
    if args.margin_aware:
        margin_cfg = {'low_thr': args.low_thr, 'mid_thr': args.mid_thr, 'mid_amp': args.mid_amp, 'high_amp': args.high_amp}
    dir_cfg = None
    if args.dir_loss_weight > 0.0:
        dir_cfg = {
            'weight': args.dir_loss_weight,
            'mid_only': args.dir_mid_only,
            'low_thr': args.low_thr,
            'mid_thr': args.mid_thr,
            'comp_weight': args.dir_comp_weight,
            'comp_eps': args.dir_comp_eps,
        }
    qgate_cfg = None
    if args.qgate_enable:
        qgate_cfg = {
            'enable': True,
            'p_low': args.margin_p_low,
            'p_mid': args.margin_p_mid,
            'g_min': args.gate_min,
            'g_max': args.gate_max,
            'reg_warmup_epochs': args.gate_reg_warmup_epochs,
            'w_low': args.gate_reg_w_low,
            'w_mid': args.gate_reg_w_mid,
            'w_hi': args.gate_reg_w_hi,
            't_low': args.gate_reg_t_low,
            't_mid': args.gate_reg_t_mid,
            't_hi': args.gate_reg_t_hi,
            'smooth_w': args.gate_smooth_weight,
        }
    dirgate_cfg = None
    if args.gate_mode == 'dir':
        dirgate_cfg = {
            'tau': args.dir_gate_tau,
            'k': args.dir_gate_k,
            'g_min': args.gate_min,
            'g_max': args.gate_max,
            'mid_off_warmup_epochs': args.dir_gate_mid_off_warmup_epochs,
            'p_low': args.margin_p_low,
            'p_mid': args.margin_p_mid,
        }
    hard_cfg = None
    if args.gate_mode == 'hard':
        hard_cfg = {
            'thr': args.hard_thr,
            'temp_start': args.hard_temp_start,
            'temp_end': args.hard_temp_end,
            'temp_anneal_epochs': args.hard_temp_anneal_epochs,
            'mid_off_warmup_epochs': args.hard_mid_off_warmup_epochs,
            'g_min': args.gate_min,
            'g_max': args.gate_max,
            'p_low': args.margin_p_low,
            'p_mid': args.margin_p_mid,
            'mid_min': args.gate_mid_min,
            'mid_max': args.gate_mid_max,
            'hi_max': args.gate_hi_max,
        }

    best_val = 0.0
    for epoch in range(1, args.num_epochs+1):
        tr = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch_idx=epoch, margin_cfg=margin_cfg, dir_cfg=dir_cfg, qgate_cfg=qgate_cfg,
            gate_mode=args.gate_mode, dirgate_cfg=dirgate_cfg, hard_cfg=hard_cfg
        )
        va = validate_epoch(
            model, val_loader, criterion, device, margin_cfg=margin_cfg,
            gate_mode=args.gate_mode, qgate_cfg=qgate_cfg, dirgate_cfg=dirgate_cfg, hard_cfg=hard_cfg, epoch_idx=epoch
        )
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
