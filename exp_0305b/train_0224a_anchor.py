#!/usr/bin/env python3
"""
exp_0305b: 0224a-baseline + Layer Anchor Regularization

目標:
1) 以 exp_0224a best_model.pt 為初始化基線（warm start，繼承降噪已學的能力）
2) 在續訓時對指定層加入「不偏離原始 WavTokenizer」約束
   → anchor 目標 = model.teacher（原始 pretrained encoder，完全 frozen）
   → 確保 student encoder 輸出空間不偏離太遠，保護 frozen decoder 正常解碼清晰語音
3) 比較不同錨定層策略:
   - tail_lock: 錨定 L16, L17（靠近 decoder 介面）
   - front_lock: 錨定 L0, L1（靠近輸入，保護基礎特徵）
   - front_tail_lock: 錨定 L0, L1, L16, L17（雙端錨定）

Loss:
    L_total = L_wav + L_stft + L_mel + lambda_anchor * L_anchor

其中:
    L_anchor = mean_i( w_i * MSE(h_i_student, h_i_original_wavtokenizer) )
"""

import argparse
import atexit
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0216.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum
from exp_0224.models_no_vq import TeacherStudentNoVQ
from exp_0224.train_no_vq import MultiResolutionSTFTLoss, MelReconstructionLoss
from encoder.modules.conv import SConv1d
from encoder.modules.seanet import SEANetResnetBlock


EXP0224A_BEST = Path(
    "/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0224/runs/no_vq_epoch_20260223_055458/best_model.pt"
)
SAMPLE_RATE = 24000


class _TeeIO:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def setup_logging(output_dir: Path):
    log_path = output_dir / "train.log"
    log_f = open(log_path, "a", buffering=1, encoding="utf-8", errors="ignore")
    atexit.register(lambda: log_f.close())
    sys.stdout = _TeeIO(sys.stdout, log_f)
    sys.stderr = _TeeIO(sys.stderr, log_f)
    return log_path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] {seed}")


def cuda_preinit(device: torch.device, retries: int = 10, sleep_s: float = 2.0):
    if device.type != "cuda":
        return
    for attempt in range(retries):
        try:
            torch.zeros(1, device=device)
            print(f"CUDA pre-init OK (attempt {attempt + 1})")
            return
        except RuntimeError as e:
            print(f"CUDA pre-init failed {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(sleep_s)
    raise RuntimeError("CUDA pre-init failed")


def make_loaders(batch_size: int, num_workers: int, smoke: bool):
    if smoke:
        ds = AugmentedCurriculumDataset(
            VAL_CACHE, augment=False, filter_clean_to_clean=True, compute_snr=False
        )
        idx = list(range(min(20, len(ds))))
        sub = torch.utils.data.Subset(ds, idx)
        train_loader = DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_curriculum)
        val_loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn_curriculum)
        return train_loader, val_loader

    train_ds = AugmentedCurriculumDataset(
        TRAIN_CACHE, augment=True, filter_clean_to_clean=True, compute_snr=False,
        snr_remix_prob=0.5, snr_remix_range=(-5.0, 25.0),
        random_gain_prob=0.3, random_gain_db=3.0,
        random_crop_prob=0.3, random_crop_min_ratio=0.7,
        time_stretch_prob=0.2, time_stretch_range=(0.95, 1.05),
    )
    val_ds = AugmentedCurriculumDataset(
        VAL_CACHE, augment=False, filter_clean_to_clean=True, compute_snr=False
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_curriculum)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_curriculum)
    return train_loader, val_loader


def build_conv18_modules(encoder: torch.nn.Module) -> Dict[int, torch.nn.Module]:
    modules: Dict[int, torch.nn.Module] = {}
    li = 0
    for t_i, m in enumerate(encoder.model):
        if isinstance(m, SConv1d):
            modules[li] = m
            li += 1
        elif isinstance(m, SEANetResnetBlock):
            c1 = m.block[1]
            c2 = m.block[3]
            sc = m.shortcut
            assert isinstance(c1, SConv1d)
            assert isinstance(c2, SConv1d)
            modules[li] = c1
            li += 1
            modules[li] = c2
            li += 1
            if isinstance(sc, SConv1d):
                modules[li] = sc
                li += 1
    if len(modules) != 18:
        raise RuntimeError(f"Expected 18 conv-like modules, got {len(modules)}")
    return modules


class LayerHookBank:
    def __init__(self, layer_modules: Dict[int, torch.nn.Module], layer_ids: List[int]):
        self.layer_ids = layer_ids
        self.cache: Dict[int, torch.Tensor] = {}
        self.handles = []
        for li in layer_ids:
            module = layer_modules[li]
            self.handles.append(module.register_forward_hook(self._hook_factory(li)))

    def _hook_factory(self, li: int):
        def _hook(_m, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            self.cache[li] = out
        return _hook

    def clear(self):
        self.cache.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def parse_layer_ids(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if v < 0 or v > 17:
            raise ValueError(f"Layer id out of range 0..17: {v}")
        out.append(v)
    if not out:
        raise ValueError("anchor layers cannot be empty")
    return sorted(set(out))


def parse_layer_weights(s: str, layer_ids: List[int]) -> Dict[int, float]:
    if s.strip() == "":
        return {li: 1.0 for li in layer_ids}
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) != len(layer_ids):
        raise ValueError(f"anchor_weights count ({len(vals)}) must equal anchor_layers count ({len(layer_ids)})")
    return {li: w for li, w in zip(layer_ids, vals)}


def preset_to_layers(preset: str) -> List[int]:
    if preset == "tail_lock":
        return [16, 17]
    if preset == "front_lock":
        return [0, 1]
    if preset == "front_tail_lock":
        return [0, 1, 16, 17]
    raise ValueError(f"Unknown preset: {preset}")


def load_0224a_weights(model: TeacherStudentNoVQ, ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise RuntimeError("0224a checkpoint missing model_state_dict")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[0224a load] missing={len(missing)} unexpected={len(unexpected)}")
    return ckpt


def compute_anchor_loss(
    student_cache: Dict[int, torch.Tensor],
    anchor_cache: Dict[int, torch.Tensor],
    layer_weights: Dict[int, float],
) -> torch.Tensor:
    loss = 0.0
    denom = 0.0
    for li, w in layer_weights.items():
        s = student_cache[li]
        a = anchor_cache[li].detach()
        l = F.mse_loss(s, a)
        loss = loss + w * l
        denom += w
    if denom <= 0:
        return torch.tensor(0.0, device=next(iter(student_cache.values())).device)
    return loss / denom


def train_epoch(
    model: TeacherStudentNoVQ,
    anchor_encoder: torch.nn.Module,
    student_hooks: LayerHookBank,
    anchor_hooks: LayerHookBank,
    layer_weights: Dict[int, float],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
    mr_stft: MultiResolutionSTFTLoss,
    mel_fn: MelReconstructionLoss,
    epoch: int,
) -> Dict:
    model.train()
    model.teacher.backbone.eval()
    model.teacher.head.eval()
    model.student.train()
    anchor_encoder.eval()

    accum = cfg["grad_accum"]
    lambda_wav = cfg["lambda_wav"]
    lambda_stft = cfg["lambda_stft"]
    lambda_mel = cfg["lambda_mel"]
    lambda_anchor = cfg["lambda_anchor"]

    sums = {
        "total_loss": 0.0,
        "wav_mse": 0.0,
        "stft_sc": 0.0,
        "stft_mag": 0.0,
        "mel_loss": 0.0,
        "anchor_loss": 0.0,
    }
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [exp_0305b]")
    for bi, batch in enumerate(pbar):
        noisy = batch["noisy_audio"].to(device)
        clean = batch["clean_audio"].to(device)
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)

        if bi % accum == 0:
            optimizer.zero_grad()

        # Clear hook caches
        student_hooks.clear()
        anchor_hooks.clear()

        with torch.no_grad():
            _ = anchor_encoder(noisy)

        with autocast(device_type=device.type, enabled=cfg["use_amp"]):
            out = model.forward_wav(clean, noisy)
            recon = out["recon_wav"]
            T = min(recon.shape[-1], clean.shape[-1])
            recon_t = recon[..., :T]
            clean_t = clean[..., :T]

            wav_mse = F.mse_loss(recon_t, clean_t)
            sc, mag = mr_stft(recon_t, clean_t)
            mel = mel_fn(recon_t, clean_t)
            anchor = compute_anchor_loss(student_hooks.cache, anchor_hooks.cache, layer_weights)

            total = (
                lambda_wav * wav_mse
                + lambda_stft * (sc + mag)
                + lambda_mel * mel
                + lambda_anchor * anchor
            ) / accum

        if scaler is not None:
            scaler.scale(total).backward()
            if (bi + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
        else:
            total.backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
                optimizer.step()

        sums["total_loss"] += total.item() * accum
        sums["wav_mse"] += wav_mse.item()
        sums["stft_sc"] += sc.item()
        sums["stft_mag"] += mag.item()
        sums["mel_loss"] += mel.item()
        sums["anchor_loss"] += anchor.item()
        n += 1

        pbar.set_postfix(
            total=f"{(total.item()*accum):.4f}",
            wav=f"{wav_mse.item():.5f}",
            anchor=f"{anchor.item():.5f}",
        )

    if n > 0:
        for k in sums:
            sums[k] /= n
    return sums


@torch.no_grad()
def evaluate(
    model: TeacherStudentNoVQ,
    anchor_encoder: torch.nn.Module,
    student_hooks: LayerHookBank,
    anchor_hooks: LayerHookBank,
    layer_weights: Dict[int, float],
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    mr_stft: MultiResolutionSTFTLoss,
    mel_fn: MelReconstructionLoss,
    max_batches: int,
) -> Dict:
    model.eval()
    anchor_encoder.eval()

    vals = {
        "val_wav_mse": [],
        "val_noisy_mse": [],
        "val_stft_sc": [],
        "val_mel": [],
        "val_anchor": [],
    }
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        noisy = batch["noisy_audio"].to(device)
        clean = batch["clean_audio"].to(device)
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)

        student_hooks.clear()
        anchor_hooks.clear()
        _ = anchor_encoder(noisy)
        out = model.forward_wav(clean, noisy)
        recon = out["recon_wav"]
        T = min(recon.shape[-1], clean.shape[-1], noisy.shape[-1])
        r = recon[..., :T]
        c = clean[..., :T]
        n = noisy[..., :T]

        vals["val_wav_mse"].append(F.mse_loss(r, c).item())
        vals["val_noisy_mse"].append(F.mse_loss(n, c).item())
        sc, _ = mr_stft(r, c)
        vals["val_stft_sc"].append(sc.item())
        vals["val_mel"].append(mel_fn(r, c).item())
        vals["val_anchor"].append(compute_anchor_loss(student_hooks.cache, anchor_hooks.cache, layer_weights).item())

    return {k: float(np.mean(v)) if v else float("nan") for k, v in vals.items()}


def plot_curves(history: dict, output_dir: Path, epoch: int, preset: str):
    """繪製訓練曲線並存檔，包含 loss、anchor loss、MSE improvement。

    Args:
        history: 訓練歷史字典，含各 metric 的 list。
        output_dir: 輸出目錄路徑。
        epoch: 當前 epoch 編號（用於檔名）。
        preset: anchor 策略名稱（用於標題）。
    """
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep = list(range(1, len(history.get('train_total_loss', [])) + 1))
    if not ep:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'exp_0305b [{preset}] Training Curves — Epoch {epoch}', fontsize=13)

    def _plot(ax, keys, title, log=False):
        for key, label, color in keys:
            vals = history.get(key, [])
            if vals:
                ax.plot(ep[:len(vals)], vals, color=color, label=label, alpha=0.85)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(fontsize=8)
        if log:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], [('train_total_loss', 'Train total', 'steelblue')], 'Total Loss (train)', log=True)
    _plot(axes[0, 1], [('train_wav_mse', 'Train MSE', 'blue'), ('val_wav_mse', 'Val MSE', 'red'), ('val_noisy_mse', 'Noisy MSE', 'orange')], 'Wav MSE')
    _plot(axes[0, 2], [('train_anchor', 'Train anchor', 'purple'), ('val_anchor', 'Val anchor', 'violet')], 'Anchor Loss')
    _plot(axes[1, 0], [('lr', 'LR', 'gray')], 'Learning Rate', log=True)

    # MSE improvement ratio
    ax = axes[1, 1]
    val_mse = history.get('val_wav_mse', [])
    noisy_mse = history.get('val_noisy_mse', [])
    n_pts = min(len(val_mse), len(noisy_mse))
    if n_pts > 0:
        impr = [(noisy_mse[i] - val_mse[i]) / (noisy_mse[i] + 1e-9) for i in range(n_pts)]
        ax.plot(range(1, n_pts + 1), impr, 'g-', label='MSE improvement', alpha=0.85)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Val MSE Improvement Ratio'); ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Anchor baseline reference
    ax = axes[1, 2]
    ax.text(0.5, 0.5,
            f'Anchor target: original WavTokenizer\n(NOT exp_0224a)\n\npreset: {preset}\n\nbaseline val_mse: 0.0233',
            transform=ax.transAxes, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')

    plt.tight_layout()
    fname = output_dir / f'exp0305b_{preset}_epoch{epoch:03d}_{date_str}_plot_curves.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Plot] 已儲存 loss 曲線圖: {fname.name}')


@torch.no_grad()
def save_audio_samples(
    model: TeacherStudentNoVQ,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    n_samples: int = 4,
    sample_rate: int = SAMPLE_RATE,
):
    """存儲 val 音檔樣本：noisy / clean / recon 各一份，方便聽感評估。

    Args:
        model: 當前 student-teacher 模型。
        val_loader: 驗證 DataLoader。
        device: 運算裝置。
        output_dir: 輸出目錄。
        epoch: 當前 epoch 編號（用於檔名）。
        n_samples: 儲存幾筆樣本。
        sample_rate: 音訊取樣率。
    """
    audio_dir = output_dir / 'audio_samples' / f'epoch{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in val_loader:
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)
        out = model.forward_wav(clean, noisy)
        recon = out['recon_wav']
        for i in range(min(noisy.shape[0], n_samples - saved)):
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_noisy.wav'), noisy[i].cpu(), sample_rate)
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_clean.wav'), clean[i].cpu(), sample_rate)
            r = recon[i].cpu()
            if r.dim() == 1:
                r = r.unsqueeze(0)
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_recon.wav'), r, sample_rate)
        saved += min(noisy.shape[0], n_samples - saved)
        if saved >= n_samples:
            break
    print(f'  [Audio] 已儲存 {saved} 筆 val 音檔至 {audio_dir}')


def parse_args():
    p = argparse.ArgumentParser(description="exp_0305b: 0224a baseline + layer anchor")
    p.add_argument("--mode", default="smoke", choices=["smoke", "epoch"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_amp", dest="use_amp", action="store_true")
    p.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.set_defaults(use_amp=True)
    p.add_argument("--num_workers", type=int, default=2)

    # Loss weights
    p.add_argument("--lambda_wav", type=float, default=1.0)
    p.add_argument("--lambda_stft", type=float, default=1.0)
    p.add_argument("--lambda_mel", type=float, default=45.0)
    p.add_argument("--lambda_anchor", type=float, default=3.0,
                   help="Anchor regularization weight.")

    # Anchor setting
    p.add_argument(
        "--preset",
        default="tail_lock",
        choices=["tail_lock", "front_lock", "front_tail_lock", "custom"],
    )
    p.add_argument("--anchor_layers", default="16,17",
                   help="Comma-separated conv18 layer ids; only used when preset=custom.")
    p.add_argument("--anchor_weights", default="",
                   help="Comma-separated per-layer weights, same length as anchor_layers. Empty => all 1.0")

    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--baseline_ckpt", type=str, default=str(EXP0224A_BEST))
    p.add_argument("--eval_max_batches", type=int, default=30)
    p.add_argument("--save_every", type=int, default=10,
                   help="每隔多少 epoch 存儲 checkpoint 和音檔")
    return p.parse_args()


def main():
    args = parse_args()
    smoke = args.mode == "smoke"
    if smoke:
        args.epochs = min(args.epochs, 5)
        args.eval_max_batches = min(args.eval_max_batches, 5)

    if args.preset == "custom":
        layer_ids = parse_layer_ids(args.anchor_layers)
    else:
        layer_ids = preset_to_layers(args.preset)
    layer_weights = parse_layer_weights(args.anchor_weights, layer_ids)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" and args.use_amp:
        print("[Info] AMP disabled because device is not CUDA")
        args.use_amp = False
    cuda_preinit(device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / "runs" / f"exp0305b_{args.preset}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    cfg = vars(args).copy()
    cfg["timestamp"] = ts
    cfg["anchor_layer_ids"] = layer_ids
    cfg["anchor_layer_weights"] = layer_weights
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("exp_0305b: 0224a baseline + layer anchor regularization")
    print(f"mode={args.mode}, epochs={args.epochs}, device={device}")
    print(f"preset={args.preset}, anchor_layers={layer_ids}, anchor_weights={layer_weights}")
    print(f"lambda_anchor={args.lambda_anchor}")
    print(f"baseline_ckpt={args.baseline_ckpt}")
    print(f"output={out_dir}")
    print("=" * 70)

    # Data
    train_loader, val_loader = make_loaders(
        batch_size=(4 if smoke else args.batch_size),
        num_workers=(0 if smoke else args.num_workers),
        smoke=smoke,
    )

    # Model (same as exp_0224a)
    model = TeacherStudentNoVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6],
        device=str(device),
    ).to(device)

    # Load 0224a baseline weights
    load_0224a_weights(model, Path(args.baseline_ckpt))

    # Anchor encoder = 原始 WavTokenizer teacher encoder（已 frozen）
    # 目的：約束 student encoder 不偏離官方預訓練權重 → 保護 frozen decoder 解碼出清晰語音
    # teacher 在 TeacherStudentNoVQ.__init__ 中已整體 frozen，直接複用即可
    anchor_encoder = model.teacher.feature_extractor.encodec.encoder.eval()
    # 雙重確認無梯度
    for p in anchor_encoder.parameters():
        p.requires_grad_(False)
    print("[Anchor] 錨定對象：原始 WavTokenizer teacher encoder（非 exp_0224a LoRA 版本）")

    student_enc = model.student.feature_extractor.encodec.encoder
    student_modules = build_conv18_modules(student_enc)
    anchor_modules = build_conv18_modules(anchor_encoder)
    student_hooks = LayerHookBank(student_modules, layer_ids)
    anchor_hooks = LayerHookBank(anchor_modules, layer_ids)

    mr_stft = MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[512, 256, 128],
        win_sizes=[2048, 1024, 512],
    ).to(device)
    mel_fn = MelReconstructionLoss(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        progress = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.learning_rate, 0.5 * (1 + math.cos(math.pi * progress)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler(device=device.type, enabled=args.use_amp)

    history = {
        "train_total_loss": [], "train_wav_mse": [], "train_anchor": [],
        "val_wav_mse": [], "val_noisy_mse": [], "val_anchor": [], "lr": []
    }
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(
            model, anchor_encoder, student_hooks, anchor_hooks, layer_weights,
            train_loader, opt, scaler, device, cfg, mr_stft, mel_fn, ep
        )
        va = evaluate(
            model, anchor_encoder, student_hooks, anchor_hooks, layer_weights,
            val_loader, device, cfg, mr_stft, mel_fn, args.eval_max_batches
        )
        sch.step()
        lr = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_total_loss"].append(tr["total_loss"])
        history["train_wav_mse"].append(tr["wav_mse"])
        history["train_anchor"].append(tr["anchor_loss"])
        history["val_wav_mse"].append(va["val_wav_mse"])
        history["val_noisy_mse"].append(va["val_noisy_mse"])
        history["val_anchor"].append(va["val_anchor"])
        history["lr"].append(lr)

        improve = (va["val_noisy_mse"] - va["val_wav_mse"]) / (va["val_noisy_mse"] + 1e-9) * 100.0
        print(
            f"Epoch {ep}/{args.epochs} ({elapsed:.1f}s) "
            f"train_total={tr['total_loss']:.4f} train_anchor={tr['anchor_loss']:.5f} "
            f"val_mse={va['val_wav_mse']:.5f} noisy={va['val_noisy_mse']:.5f} "
            f"improve=+{improve:.2f}% val_anchor={va['val_anchor']:.5f} lr={lr:.3e}"
        )

        if va["val_wav_mse"] < best_val:
            best_val = va["val_wav_mse"]
            ckpt = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sch.state_dict(),
                "metrics": va,
                "config": cfg,
            }
            torch.save(ckpt, out_dir / "best_model.pt")
            print(f"  New best val_wav_mse={best_val:.6f}")

        if ep % args.save_every == 0:
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sch.state_dict(),
                "metrics": va,
                "config": cfg,
            }, out_dir / f"checkpoint_epoch{ep:03d}.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # Loss 曲線圖：每 25 epoch 或 smoke 模式結束時
        if ep % 25 == 0 or (smoke and ep == args.epochs):
            plot_curves(history, out_dir, ep, args.preset)

        # 音檔樣本：每 10 epoch 或 smoke 模式結束時
        if ep % args.save_every == 0 or (smoke and ep == args.epochs):
            save_audio_samples(model, val_loader, device, out_dir, ep)

    student_hooks.close()
    anchor_hooks.close()
    # 訓練結束後存最終 loss 圖
    plot_curves(history, out_dir, args.epochs, args.preset)
    print(f"Training complete. Best val_wav_mse={best_val:.6f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
