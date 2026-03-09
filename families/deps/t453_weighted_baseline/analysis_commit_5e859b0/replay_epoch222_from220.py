#!/usr/bin/env python3
"""
Replay training from epoch 220 checkpoint to produce epoch 222 supplemental checkpoint.

Important:
- This is a supplemental recovery path and does NOT overwrite original run artifacts.
- Output files are explicitly labeled with replay_from220 metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp_0216"))
sys.path.insert(0, "/home/sbplab/ruizi/WavTokenizer-main")
sys.path.insert(0, "/home/sbplab/ruizi/WavTokenize-self-supervised")

from families.compat_legacy.intermediate_stack.train_v6 import IntermediateSupervisionLossV6
from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import TeacherStudentSingleVQ
from families.deps.encoder_aug.data_augmented import create_augmented_curriculum_dataloaders
from families.deps.encoder_aug.train_augmented import (
    evaluate_single_vq,
    get_lora_vq_state_dict,
    set_seed,
    train_epoch,
)
from families.deps.wavtokenizer_core.config import TRAIN_CACHE, VAL_CACHE, WAVTOK_CKPT, WAVTOK_CONFIG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_run", type=str, default="families/deps/encoder_aug/runs/augmented_long_20260216")
    p.add_argument("--start_epoch", type=int, default=220)
    p.add_argument("--target_epoch", type=int, default=222)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default="families/deps/t453_weighted_baseline/analysis_commit_5e859b0/replay_from220_to222")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def make_scheduler(optimizer, cfg):
    total_epochs = int(cfg["epochs"])
    warmup_epochs = int(cfg["warmup_epochs"])
    lr = float(cfg["learning_rate"])
    min_lr = float(cfg["min_lr"])

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def simulate_sampler_to_epoch_end(curriculum_sampler, upto_epoch: int) -> None:
    # Match original loop behavior:
    # if epoch > 1 and (epoch - 1) % 10 == 0: advance_phase()
    for epoch in range(1, upto_epoch + 1):
        if epoch > 1 and (epoch - 1) % 10 == 0:
            curriculum_sampler.advance_phase()


def main():
    args = parse_args()
    source_run = Path(args.source_run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    cfg = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    phase_increment = (cfg["curriculum_end"] - cfg["curriculum_start"]) / max(1, cfg["curriculum_epochs"] / 10)
    train_loader, val_loader, curriculum_sampler = create_augmented_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=int(cfg["batch_size"]),
        num_workers=args.num_workers,
        compute_snr=False,
        initial_phase=float(cfg["curriculum_start"]),
        phase_increment=phase_increment,
        snr_remix_prob=float(cfg["snr_remix_prob"]),
        snr_remix_range=(float(cfg["snr_remix_min"]), float(cfg["snr_remix_max"])),
        random_gain_prob=float(cfg["random_gain_prob"]),
        random_gain_db=float(cfg["random_gain_db"]),
        random_crop_prob=float(cfg["random_crop_prob"]),
        random_crop_min_ratio=float(cfg["random_crop_min_ratio"]),
        time_stretch_prob=float(cfg["time_stretch_prob"]),
        time_stretch_range=(float(cfg["time_stretch_min"]), float(cfg["time_stretch_max"])),
    )

    simulate_sampler_to_epoch_end(curriculum_sampler, args.start_epoch)
    print(f"sampler phase after epoch {args.start_epoch}: {curriculum_sampler.current_phase:.4f}")

    model = TeacherStudentSingleVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        intermediate_indices=[3, 4, 6],
        device=device,
        vq_ema_decay=float(cfg["vq_ema_decay"]),
        vq_ema_threshold=int(cfg["vq_ema_threshold"]),
        vq_ema_usage_penalty=float(cfg["vq_ema_usage_penalty"]),
    )

    inter_loss_fn = IntermediateSupervisionLossV6(
        layer_weights={
            3: float(cfg["intermediate_L3_weight"]),
            4: float(cfg["intermediate_L4_weight"]),
            6: float(cfg["intermediate_L6_weight"]),
        }
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=bool(cfg.get("use_amp", True)))

    src_ckpt_path = source_run / "checkpoints" / f"checkpoint_epoch{args.start_epoch:03d}.pt"
    if not src_ckpt_path.exists():
        raise FileNotFoundError(f"missing source checkpoint: {src_ckpt_path}")
    src_ckpt = torch.load(src_ckpt_path, map_location="cpu")

    # restore model
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in src_ckpt["lora_state"]:
                param.copy_(src_ckpt["lora_state"][name].to(param.device, dtype=param.dtype))
    model.vq.load_state_dict(src_ckpt["vq_state_dict"])

    # restore optimizer/scheduler
    optimizer.load_state_dict(src_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in src_ckpt:
        scheduler.load_state_dict(src_ckpt["scheduler_state_dict"])

    history = []
    replay_cfg = dict(cfg)
    replay_cfg["output_dir"] = str(output_dir)

    for epoch in range(args.start_epoch + 1, args.target_epoch + 1):
        if epoch > 1 and (epoch - 1) % 10 == 0:
            curriculum_sampler.advance_phase()

        t0 = time.time()
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            inter_loss_fn=inter_loss_fn,
            device=device,
            epoch=epoch,
            config=replay_cfg,
            scaler=scaler,
            curriculum_sampler=curriculum_sampler,
        )
        scheduler.step()
        val_metrics = evaluate_single_vq(
            model=model,
            dataloader=val_loader,
            inter_loss_fn=inter_loss_fn,
            device=device,
            config=replay_cfg,
            max_batches=int(cfg.get("eval_max_batches", 30)),
        )
        dt = time.time() - t0
        print(
            f"epoch={epoch} done in {dt:.1f}s | "
            f"train_loss={train_metrics['total_loss']:.4f} "
            f"val_mse={val_metrics['feature_mse']:.6f} "
            f"val_total={val_metrics['val_total_loss']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            **get_lora_vq_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": val_metrics,
            "config": replay_cfg,
            "replay_meta": {
                "mode": "replay_from_checkpoint",
                "source_run": str(source_run),
                "source_checkpoint": str(src_ckpt_path),
                "start_epoch": args.start_epoch,
                "target_epoch": args.target_epoch,
                "note": "supplemental replay, not original run artifact",
            },
        }
        out_ckpt = output_dir / "checkpoints" / f"checkpoint_epoch{epoch:03d}_replay.pt"
        torch.save(ckpt, out_ckpt)

        history.append(
            {
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "checkpoint": str(out_ckpt),
                "elapsed_sec": dt,
            }
        )
        (output_dir / "replay_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "source_run": str(source_run),
        "source_checkpoint": str(src_ckpt_path),
        "start_epoch": args.start_epoch,
        "target_epoch": args.target_epoch,
        "history_file": str(output_dir / "replay_history.json"),
        "final_checkpoint": str(output_dir / "checkpoints" / f"checkpoint_epoch{args.target_epoch:03d}_replay.pt"),
        "note": "supplemental replay artifact for acceptance gap closure",
    }
    (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()

