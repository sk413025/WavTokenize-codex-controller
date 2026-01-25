import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, "/home/sbplab/ruizi/WavTokenizer-main")
sys.path.insert(0, "/home/sbplab/ruizi/WavTokenize-self-supervised")

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_0112_intermediate.train_v5 import IntermediateSupervisionLossV5
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy, create_length_mask
from exp_1212.data_aligned import AlignedNoisyCleanPairDataset
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum


ENCODER_STRIDE = 320
NUM_CODES = 4096


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def ensure_frozen(model):
    model.teacher.eval()
    model.teacher.feature_extractor.encodec.quantizer.eval()
    model.student.feature_extractor.encodec.quantizer.eval()


def _hash_indices(indices):
    payload = ",".join(str(i) for i in indices).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def build_subset_loader(cache_path, batch_size, num_workers, shuffle, max_samples, seed):
    dataset = CurriculumDataset(cache_path, max_samples=None, compute_snr=False)
    if max_samples is not None and max_samples < len(dataset):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(dataset), size=max_samples, replace=False).tolist()
        subset = Subset(dataset, indices)
    else:
        indices = list(range(len(dataset)))
        subset = dataset

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    return loader, indices


def counts_to_stats(counts, top_k=10):
    total = counts.sum()
    if total == 0:
        return {"total": 0, "unique": 0, "entropy": 0.0, "top_k_mass": 0.0}
    probs = counts / total
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0] + 1e-12)).sum())
    top_k_mass = float(np.sort(counts)[-top_k:].sum() / total)
    unique = int((counts > 0).sum())
    return {
        "total": int(total),
        "unique": unique,
        "entropy": entropy,
        "top_k_mass": top_k_mass,
    }


def kl_div(p_counts, q_counts):
    p = p_counts / (p_counts.sum() + 1e-12)
    q = q_counts / (q_counts.sum() + 1e-12)
    p = p + 1e-12
    q = q + 1e-12
    return float((p * np.log(p / q)).sum())


def mix_with_snr(clean, noise, lengths, snr_db):
    # clean/noise: (B, T)
    B, T = clean.shape
    device = clean.device
    lengths = lengths.to(device)
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.float()

    sig_power = (clean ** 2 * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    noise_power = (noise ** 2 * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    snr_linear = 10 ** (snr_db / 10.0)
    scale = torch.sqrt(sig_power / (snr_linear * noise_power + 1e-8))
    scale = scale.unsqueeze(1)
    return clean + noise * scale


def build_noisy_view(clean, noisy, lengths, snr_list, rng):
    # Use shuffled noise from batch to create a second noisy view
    B = clean.shape[0]
    perm = torch.randperm(B, device=clean.device)
    noise = (noisy - clean)[perm]
    snr_choices = np.array(snr_list)
    snr_db = torch.tensor(
        snr_choices[rng.randint(0, len(snr_choices), size=B)],
        device=clean.device,
        dtype=clean.dtype,
    )
    return mix_with_snr(clean, noise, lengths, snr_db)


def soft_assign(student_features, codebook, temperature=1.0):
    B, D, T = student_features.shape
    z = student_features.permute(0, 2, 1).reshape(-1, D)
    dists = torch.cdist(z, codebook)
    logits = -dists / temperature
    probs = F.softmax(logits, dim=-1)
    return probs


def masked_symmetric_kl(p, q, mask_flat):
    eps = 1e-8
    p = p + eps
    q = q + eps
    kl_pq = (p * (p.log() - q.log())).sum(dim=1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=1)
    kl = 0.5 * (kl_pq + kl_qp)
    return (kl * mask_flat).sum() / (mask_flat.sum() + 1e-8)


def _best_shift_per_sample(codes_a, codes_b, mask, max_shift):
    """Find best global shift in [-max_shift, max_shift] per sample."""
    B, T = codes_a.shape
    shifts = []
    for b in range(B):
        best_s = 0
        best_acc = -1.0
        for s in range(-max_shift, max_shift + 1):
            if s > 0:
                a = codes_a[b, s:]
                bcodes = codes_b[b, :-s]
                m = mask[b, s:]
            elif s < 0:
                a = codes_a[b, :s]
                bcodes = codes_b[b, -s:]
                m = mask[b, :s]
            else:
                a = codes_a[b]
                bcodes = codes_b[b]
                m = mask[b]

            if a.numel() == 0:
                continue
            correct = (a == bcodes).float() * m
            total = m.sum()
            acc = (correct.sum() / (total + 1e-8)).item()
            if acc > best_acc:
                best_acc = acc
                best_s = s
        shifts.append(best_s)
    return shifts


def _shift_tensor_2d(x, shift):
    """Shift (T, C) tensor with zero padding."""
    if shift == 0:
        return x
    out = torch.zeros_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
    else:
        out[:shift] = x[-shift:]
    return out


def _shift_mask_1d(mask, shift):
    if shift == 0:
        return mask
    out = torch.zeros_like(mask)
    if shift > 0:
        out[shift:] = mask[:-shift]
    else:
        out[:shift] = mask[-shift:]
    return out


def evaluate_split(model, loader, device):
    model.eval()
    ensure_frozen(model)

    student_counts = np.zeros(NUM_CODES, dtype=np.int64)
    teacher_counts = np.zeros(NUM_CODES, dtype=np.int64)
    strict_batch_sum = 0.0
    strict_correct = 0.0
    strict_total = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy_audio"].to(device)
            clean = batch["clean_audio"].to(device)
            lengths = batch["lengths"].to(device)

            output = model(noisy, clean)
            s_codes = output["student_codes"]
            t_codes = output["teacher_codes"]
            if s_codes.dim() == 3:
                s_codes = s_codes[0]
            if t_codes.dim() == 3:
                t_codes = t_codes[0]

            B, T = s_codes.shape
            max_audio_len = T * ENCODER_STRIDE
            mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=device)

            correct = (s_codes == t_codes).float() * mask
            correct_per_sample = correct.sum(dim=1)
            total_per_sample = mask.sum(dim=1)
            acc_per_sample = correct_per_sample / (total_per_sample + 1e-8)

            strict_batch_sum += acc_per_sample.sum().item()
            strict_correct += correct_per_sample.sum().item()
            strict_total += total_per_sample.sum().item()
            num_samples += B

            s_flat = s_codes.reshape(-1).detach().cpu().numpy()
            t_flat = t_codes.reshape(-1).detach().cpu().numpy()
            student_counts += np.bincount(s_flat, minlength=NUM_CODES)
            teacher_counts += np.bincount(t_flat, minlength=NUM_CODES)

    strict = {
        "acc_batch_mean": strict_batch_sum / max(1, num_samples),
        "acc_frame_weighted": strict_correct / max(1e-8, strict_total),
        "num_samples": int(num_samples),
        "num_frames": int(strict_total),
    }

    student_stats = counts_to_stats(student_counts)
    teacher_stats = counts_to_stats(teacher_counts)
    kl = kl_div(student_counts, teacher_counts)

    return {
        "strict": strict,
        "student": student_stats,
        "teacher": teacher_stats,
        "kl_student_teacher": kl,
    }


def compute_margin_stats(model, loader, device):
    model.eval()
    ensure_frozen(model)

    codebook = model.codebook.to(device)
    margins = []

    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy_audio"].to(device)
            clean = batch["clean_audio"].to(device)
            lengths = batch["lengths"].to(device)

            output = model(noisy, clean)
            s_out = output["student_encoder_out"]
            s_codes = output["student_codes"]
            t_codes = output["teacher_codes"]
            if s_codes.dim() == 3:
                s_codes = s_codes[0]
            if t_codes.dim() == 3:
                t_codes = t_codes[0]

            B, D, T = s_out.shape
            max_audio_len = T * ENCODER_STRIDE
            mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=device)

            z = s_out.permute(0, 2, 1).reshape(-1, D)
            dists = torch.cdist(z, codebook)
            d1, _ = torch.topk(dists, k=2, largest=False, dim=1)
            margin = (d1[:, 1] - d1[:, 0]).detach().cpu().numpy()

            mask_flat = mask.reshape(-1).detach().cpu().numpy()
            margin = margin[mask_flat > 0]
            margins.extend(margin.tolist())

    margins = np.array(margins)
    if len(margins) == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "count": int(len(margins)),
        "mean": float(np.mean(margins)),
        "p50": float(np.percentile(margins, 50)),
        "p90": float(np.percentile(margins, 90)),
    }


def token_change_rate(codes_a, codes_b, lengths):
    if codes_a.dim() == 3:
        codes_a = codes_a[0]
    if codes_b.dim() == 3:
        codes_b = codes_b[0]
    T = min(codes_a.shape[1], codes_b.shape[1])
    codes_a = codes_a[:, :T]
    codes_b = codes_b[:, :T]
    max_audio_len = T * ENCODER_STRIDE
    mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=codes_a.device)
    diff = (codes_a != codes_b).float() * mask
    return diff.sum().item() / (mask.sum().item() + 1e-8)


def evaluate_noise_sensitivity(model, dataset, device, candidate_indices, num_pairs, snr_list, seed):
    rng = np.random.RandomState(seed)
    if candidate_indices is None:
        candidate_indices = list(range(len(dataset)))
    clean_indices = rng.choice(candidate_indices, size=min(num_pairs, len(candidate_indices)), replace=False)
    noise_indices = rng.choice(len(dataset), size=len(clean_indices) * 2, replace=True)
    snr_list = [float(x) for x in snr_list]

    change_rates = []
    model.eval()
    ensure_frozen(model)

    with torch.no_grad():
        for i, idx in enumerate(clean_indices):
            clean_item = dataset[idx]
            clean_audio = clean_item["clean_audio"]

            noise_item_a = dataset[noise_indices[i * 2]]
            noise_item_b = dataset[noise_indices[i * 2 + 1]]
            noise_a = noise_item_a["noisy_audio"] - noise_item_a["clean_audio"]
            noise_b = noise_item_b["noisy_audio"] - noise_item_b["clean_audio"]

            # align lengths across clean/noise_a/noise_b for fair comparison
            min_len = min(clean_audio.numel(), noise_a.numel(), noise_b.numel())
            clean_audio = clean_audio[:min_len]
            noise_a = noise_a[:min_len]
            noise_b = noise_b[:min_len]
            length = torch.tensor([min_len], device=device)

            snr_a = snr_list[rng.randint(0, len(snr_list))]
            snr_b = snr_list[rng.randint(0, len(snr_list))]

            # Move tensors to device before mixing
            clean_audio_dev = clean_audio.unsqueeze(0).to(device)
            noise_a_dev = noise_a.unsqueeze(0).to(device)
            noise_b_dev = noise_b.unsqueeze(0).to(device)

            mixed_a = mix_with_snr(clean_audio_dev, noise_a_dev, length, torch.tensor([snr_a], device=device))
            mixed_b = mix_with_snr(clean_audio_dev, noise_b_dev, length, torch.tensor([snr_b], device=device))

            out_a = model(mixed_a, clean_audio_dev)
            out_b = model(mixed_b, clean_audio_dev)

            rate = token_change_rate(out_a["student_codes"], out_b["student_codes"], length)
            change_rates.append(rate)

    return {
        "num_pairs": int(len(change_rates)),
        "token_change_rate_mean": float(np.mean(change_rates)) if change_rates else 0.0,
        "token_change_rate_std": float(np.std(change_rates)) if change_rates else 0.0,
        "snr_list": snr_list,
        "seed": int(seed),
    }


def train_one_run(args, lambda_invar, run_dir, train_indices, val_indices):
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.run_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())

    device = select_device(args.device)
    print(f"Using device: {device}")

    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=cfg.get("lora_rank", 256),
        lora_alpha=cfg.get("lora_alpha", 512),
        lora_dropout=cfg.get("lora_dropout", 0.2),
    )

    ckpt_path = args.checkpoint or str(Path(args.run_dir) / "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    base_loss_fn = MaskedCombinedLossV2(
        feature_weight=cfg.get("feature_weight", 1.0),
        cosine_weight=cfg.get("cosine_weight", 0.0),
        triplet_weight=cfg.get("triplet_weight", 1.0),
        triplet_margin=cfg.get("triplet_margin", 0.2),
        ce_weight=cfg.get("ce_weight", 0.0),
        encoder_stride=ENCODER_STRIDE,
    )
    interm_loss_fn = IntermediateSupervisionLossV5(
        layer_weights={
            3: cfg.get("intermediate_L3_weight", 0.3),
            4: cfg.get("intermediate_L4_weight", 1.0),
            6: cfg.get("intermediate_L6_weight", 0.5),
        },
        target_scale=cfg.get("target_scale", 1.0),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.1),
    )
    scaler = GradScaler(enabled=args.use_amp)

    # loaders (fixed subsets already selected)
    train_dataset = CurriculumDataset(TRAIN_CACHE, max_samples=None, compute_snr=False)
    val_dataset = CurriculumDataset(VAL_CACHE, max_samples=None, compute_snr=False)
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )

    opt_steps = 0
    micro_steps = 0
    train_metrics = {
        "total_loss": 0.0,
        "anchor_loss": 0.0,
        "invar_loss": 0.0,
        "masked_acc": 0.0,
        "num_steps": 0,
        "num_micro_steps": 0,
    }

    rng = np.random.RandomState(args.noise_seed)
    model.train()
    ensure_frozen(model)

    for epoch in range(1, args.num_epochs + 1):
        for batch in train_loader:
            if args.max_steps is not None and opt_steps >= args.max_steps:
                break

            noisy = batch["noisy_audio"].to(device)
            clean = batch["clean_audio"].to(device)
            lengths = batch["lengths"].to(device)

            noisy2 = build_noisy_view(clean, noisy, lengths, args.snr_list, rng).to(device)

            if micro_steps % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            with autocast(enabled=args.use_amp):
                out1 = model(noisy, clean)
                out2 = model(noisy2, clean)

                base1, _ = base_loss_fn(
                    student_features=out1["student_encoder_out"],
                    teacher_features=out1["teacher_encoder_out"],
                    teacher_codes=out1["teacher_codes"],
                    codebook=out1["codebook"],
                    lengths=lengths,
                )
                base2, _ = base_loss_fn(
                    student_features=out2["student_encoder_out"],
                    teacher_features=out2["teacher_encoder_out"],
                    teacher_codes=out2["teacher_codes"],
                    codebook=out2["codebook"],
                    lengths=lengths,
                )
                inter1, _ = interm_loss_fn(
                    out1["student_intermediates"],
                    out1["teacher_intermediates"],
                    layer_scale=1.0,
                )
                inter2, _ = interm_loss_fn(
                    out2["student_intermediates"],
                    out2["teacher_intermediates"],
                    layer_scale=1.0,
                )
                base_loss = 0.5 * (base1 + base2)
                inter_loss = 0.5 * (inter1 + inter2)
                anchor_loss = base_loss + cfg.get("intermediate_weight", 0.5) * inter_loss

                codebook = out1["codebook"]
                probs1 = soft_assign(out1["student_encoder_out"], codebook, temperature=args.invar_temperature)
                probs2 = soft_assign(out2["student_encoder_out"], codebook, temperature=args.invar_temperature)

                B, D, T = out1["student_encoder_out"].shape
                max_audio_len = T * ENCODER_STRIDE
                mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=device)
                mask_flat = mask.reshape(-1)

                if args.global_shift_k > 0:
                    s1 = out1["student_codes"][0] if out1["student_codes"].dim() == 3 else out1["student_codes"]
                    s2 = out2["student_codes"][0] if out2["student_codes"].dim() == 3 else out2["student_codes"]
                    shifts = _best_shift_per_sample(s1, s2, mask, args.global_shift_k)

                    probs2_shifted = []
                    mask_overlap = []
                    for b in range(s1.shape[0]):
                        p2 = probs2.view(s1.shape[0], -1, probs2.shape[-1])[b]
                        p2s = _shift_tensor_2d(p2, shifts[b])
                        probs2_shifted.append(p2s)

                        m = mask[b]
                        ms = _shift_mask_1d(m, shifts[b])
                        mask_overlap.append(m * ms)

                    probs2 = torch.stack(probs2_shifted, dim=0).reshape(-1, probs2.shape[-1])
                    mask_flat = torch.stack(mask_overlap, dim=0).reshape(-1)

                invar_loss = masked_symmetric_kl(probs1, probs2, mask_flat)
                total_loss = anchor_loss + lambda_invar * invar_loss

            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            # Compute accuracy before logging
            s_codes = out1["student_codes"]
            t_codes = out1["teacher_codes"]
            if s_codes.dim() == 3:
                s_codes = s_codes[0]
            if t_codes.dim() == 3:
                t_codes = t_codes[0]
            masked_acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, ENCODER_STRIDE)

            if (micro_steps + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                opt_steps += 1
                if args.log_every and opt_steps % args.log_every == 0:
                    print(f"[step {opt_steps}] total_loss={total_loss.item():.4f} "
                          f"anchor={anchor_loss.item():.4f} invar={invar_loss.item():.4f} "
                          f"acc={masked_acc:.4f}")

            train_metrics["total_loss"] += float(total_loss.item())
            train_metrics["anchor_loss"] += float(anchor_loss.item())
            train_metrics["invar_loss"] += float(invar_loss.item())
            train_metrics["masked_acc"] += float(masked_acc)
            train_metrics["num_steps"] += 1
            train_metrics["num_micro_steps"] += 1

            micro_steps += 1
            if args.max_steps is not None and opt_steps >= args.max_steps:
                break

        if args.max_steps is not None and opt_steps >= args.max_steps:
            break

    if train_metrics["num_steps"] > 0:
        for k in ["total_loss", "anchor_loss", "invar_loss", "masked_acc"]:
            train_metrics[k] /= train_metrics["num_steps"]

    # evaluation
    train_eval = evaluate_split(model, train_loader, device)
    val_eval = evaluate_split(model, val_loader, device)

    train_margin = compute_margin_stats(model, train_loader, device)
    val_margin = compute_margin_stats(model, val_loader, device)

    noise_dataset = AlignedNoisyCleanPairDataset(VAL_CACHE, max_samples=None)
    noise_eval = evaluate_noise_sensitivity(
        model,
        noise_dataset,
        device,
        val_indices,
        args.num_pairs,
        args.snr_list,
        args.noise_seed + 1,
    )

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "checkpoint_init": ckpt_path,
        "lambda_invar": lambda_invar,
        "invar_temperature": args.invar_temperature,
        "snr_list": args.snr_list,
        "num_pairs": args.num_pairs,
        "train_subset_size": len(train_indices),
        "val_subset_size": len(val_indices),
        "train_subset_hash": _hash_indices(train_indices),
        "val_subset_hash": _hash_indices(val_indices),
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "use_amp": args.use_amp,
        "train_metrics": train_metrics,
        "train_eval": train_eval,
        "val_eval": val_eval,
        "train_margin": train_margin,
        "val_margin": val_margin,
        "noise_sensitivity": noise_eval,
    }

    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str,
                        default="exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_root", type=str,
                        default="exp_0124/token_collapse_27e564a/invariance_short_run/runs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_val_samples", type=int, default=500)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--val_seed", type=int, default=43)
    parser.add_argument("--noise_seed", type=int, default=44)
    parser.add_argument("--lambdas", type=str, default="0.0")
    parser.add_argument("--invar_temperature", type=float, default=1.0)
    parser.add_argument("--snr_list", type=str, default="0,5,10")
    parser.add_argument("--num_pairs", type=int, default=30)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--global_shift_k", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.train_seed)
    args.snr_list = [float(x) for x in args.snr_list.split(",")]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _, train_indices = build_subset_loader(
        TRAIN_CACHE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_samples=args.max_train_samples,
        seed=args.train_seed,
    )
    _, val_indices = build_subset_loader(
        VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        max_samples=args.max_val_samples,
        seed=args.val_seed,
    )

    # Save indices for reproducibility
    (output_root / "train_indices.json").write_text(json.dumps(train_indices, indent=2))
    (output_root / "val_indices.json").write_text(json.dumps(val_indices, indent=2))

    lambdas = [float(x) for x in args.lambdas.split(",")]
    summary = []
    for lam in lambdas:
        run_dir = output_root / f"lambda_{lam:.3f}"
        payload = train_one_run(args, lam, run_dir, train_indices, val_indices)
        summary.append(payload)

    # write summary
    summary_path = output_root.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        "# Invariance short-run summary",
        "",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- run_dir: {args.run_dir}",
        f"- max_steps: {args.max_steps}",
        f"- train_samples: {args.max_train_samples}",
        f"- val_samples: {args.max_val_samples}",
        "",
        "## Results",
        "| lambda | val_strict_fw | val_entropy | val_topk_mass | val_KL | token_change_rate | val_margin_p50 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in summary:
        val = p["val_eval"]
        lines.append(
            f"| {p['lambda_invar']:.3f} | {val['strict']['acc_frame_weighted']:.6f} | "
            f"{val['student']['entropy']:.6f} | {val['student']['top_k_mass']:.6f} | "
            f"{val['kl_student_teacher']:.6f} | {p['noise_sensitivity']['token_change_rate_mean']:.6f} | "
            f"{p['val_margin']['p50']:.6f} |"
        )

    (output_root.parent / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {output_root.parent / 'summary.md'}")


if __name__ == "__main__":
    main()
