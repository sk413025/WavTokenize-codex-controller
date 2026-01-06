"""
Exp E: 分階段訓練 (Progressive Training)

核心概念：
- 階段 1: 只訓練淺層 (L0-L4)，讓它先學會處理噪音
- 階段 2: 解凍中層 (L5-L8)，讓噪音敏感層開始學習
- 階段 3: 解凍深層 (L9-L17)，讓深層適應新的輸入分布

這樣可以確保：
1. 淺層先學會去噪，而不是讓深層硬記
2. 深層只需要「適應」而非「學習去噪」
3. 梯度不會一開始就集中在深層

配置：
- 階段 1: 100 epochs, L0-L4 only
- 階段 2: 100 epochs, L0-L8
- 階段 3: 100 epochs, L0-L17 (全部)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from peft import LoraConfig, get_peft_model

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX, TRAIN_CACHE, VAL_CACHE
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy
from exp_1226.data_curriculum import create_curriculum_dataloaders
from decoder.pretrained import WavTokenizer
from exp_1201.wavtok_lora_patch import apply_lora_patch

apply_lora_patch()


# ============================================================
# Layer Groups 定義
# ============================================================

# 階段 1: 淺層 (L0-L4)
PHASE1_LAYERS = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",           # L0
    "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",   # L1
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",   # L2
    "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",  # L3
    "feature_extractor.encodec.encoder.model.3.conv.conv",           # L4
]

# 階段 2: 淺層 + 中層 (L0-L8)
PHASE2_LAYERS = PHASE1_LAYERS + [
    "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",   # L5
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",   # L6
    "feature_extractor.encodec.encoder.model.4.shortcut.conv.conv",  # L7
    "feature_extractor.encodec.encoder.model.6.conv.conv",           # L8
]

# 階段 3: 全部 (L0-L17)
PHASE3_LAYERS = PHASE2_LAYERS + [
    "feature_extractor.encodec.encoder.model.7.block.1.conv.conv",   # L9
    "feature_extractor.encodec.encoder.model.7.block.3.conv.conv",   # L10
    "feature_extractor.encodec.encoder.model.7.shortcut.conv.conv",  # L11
    "feature_extractor.encodec.encoder.model.9.conv.conv",           # L12
    "feature_extractor.encodec.encoder.model.10.block.1.conv.conv",  # L13
    "feature_extractor.encodec.encoder.model.10.block.3.conv.conv",  # L14
    "feature_extractor.encodec.encoder.model.10.shortcut.conv.conv", # L15
    "feature_extractor.encodec.encoder.model.12.conv.conv",          # L16
    "feature_extractor.encodec.encoder.model.15.conv.conv",          # L17
]

PHASE_CONFIGS = {
    1: {'layers': PHASE1_LAYERS, 'name': 'shallow (L0-L4)', 'count': 5},
    2: {'layers': PHASE2_LAYERS, 'name': 'shallow+mid (L0-L8)', 'count': 9},
    3: {'layers': PHASE3_LAYERS, 'name': 'all (L0-L17)', 'count': 18},
}


class ProgressiveLoRAModel(nn.Module):
    """
    支援分階段解凍的 LoRA 模型
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        initial_phase: int = 1,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        lora_dropout: float = 0.2,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.current_phase = initial_phase
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.wavtok_config = wavtok_config
        self.wavtok_ckpt = wavtok_ckpt

        # Teacher: 完全凍結
        print("=" * 60)
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)
        self._freeze_quantizer(self.teacher, "Teacher")

        # Student: 初始化為階段 1
        print("=" * 60)
        print(f"Loading Student with Phase {initial_phase} LoRA...")
        self._init_student_phase(initial_phase)

        # Codebook
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        print(f"Codebook shape: {self.codebook.shape}")
        print("=" * 60)

    def _init_student_phase(self, phase: int):
        """初始化或重新配置 Student 的 LoRA"""
        phase_config = PHASE_CONFIGS[phase]
        target_modules = phase_config['layers']

        print(f"  Phase {phase}: {phase_config['name']}")
        print(f"  Target layers: {phase_config['count']} / 18")

        # 重新載入 Student
        self.student = WavTokenizer.from_pretrained0802(self.wavtok_config, self.wavtok_ckpt)

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
        )

        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()
        self.student = self.student.to(self.device)
        self._freeze_quantizer(self.student, "Student")
        self._initial_student_codebook = self._get_student_codebook().clone()

    def advance_phase(self, prev_state_dict=None):
        """進入下一個階段，保留已學習的權重"""
        if self.current_phase >= 3:
            print("Already at final phase (3)")
            return False

        new_phase = self.current_phase + 1
        print("\n" + "=" * 60)
        print(f"ADVANCING TO PHASE {new_phase}")
        print("=" * 60)

        # 保存當前 LoRA 權重
        if prev_state_dict is None:
            prev_state_dict = {}
            for name, param in self.student.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    prev_state_dict[name] = param.data.clone()

        # 初始化新階段
        self._init_student_phase(new_phase)

        # 恢復已學習的權重
        restored = 0
        for name, param in self.student.named_parameters():
            if name in prev_state_dict:
                param.data.copy_(prev_state_dict[name])
                restored += 1

        print(f"  Restored {restored} LoRA parameters from previous phase")
        self.current_phase = new_phase
        return True

    def _freeze_quantizer(self, model, name: str):
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen")

    def _get_codebook(self) -> torch.Tensor:
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        return quantizer.vq.layers[0].codebook.detach().clone()

    def _get_teacher_codebook(self) -> torch.Tensor:
        return self.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def _get_student_codebook(self) -> torch.Tensor:
        return self.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()
        self.student.feature_extractor.encodec.quantizer.eval()
        return self

    def forward(self, noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> dict:
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward
        self.teacher.eval()
        with torch.no_grad():
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward
        student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)
        self.student.feature_extractor.encodec.quantizer.eval()
        with torch.no_grad():
            student_vq = self.student.feature_extractor.encodec.quantizer(
                student_encoder_out, frame_rate=75, bandwidth=0.075
            )
            student_codes = student_vq.codes

        return {
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
        }


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                distance_matrix, encoder_stride=320, scaler=None, use_amp=True,
                grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'masked_acc': 0, 'distance_loss': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                output = model(noisy_audio, clean_audio)
                loss, loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                )

            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(noisy_audio, clean_audio)
            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
            )

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc

        with torch.no_grad():
            s_flat = s_codes.reshape(-1).long()
            t_flat = t_codes.reshape(-1).long()
            dist = distance_matrix[s_flat, t_flat].mean().item()
            metrics['distance_loss'] += dist

        n_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'm_acc': f"{masked_acc*100:.2f}%"})

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, distance_matrix,
             encoder_stride=320, use_amp=True):
    model.eval()

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'masked_acc': 0, 'distance_loss': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)
            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
            )

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc

        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        dist = distance_matrix[s_flat, t_flat].mean().item()
        metrics['distance_loss'] += dist

        n_batches += 1

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=2, split='val'):
    """保存音檔樣本"""
    model.eval()
    audio_dir = exp_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    data_iter = iter(dataloader)
    torch.cuda.empty_cache()

    for i in range(min(num_samples, len(dataloader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        noisy_audio = batch['noisy_audio'][:1].to(device)
        clean_audio = batch['clean_audio'][:1].to(device)

        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)

        import torchaudio
        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'), noisy_audio.cpu(), sample_rate)
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'), clean_audio.cpu(), sample_rate)

        try:
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_student_recon.wav'), student_recon.cpu(), sample_rate)
            del student_features, student_recon
            torch.cuda.empty_cache()

            teacher_features, _, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(device))
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_teacher_recon.wav'), teacher_recon.cpu(), sample_rate)
            del teacher_features, teacher_recon
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM when saving audio sample {i+1}, skipping")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples")


def plot_metrics(history, exp_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Total Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    # 標記階段轉換
    for phase_epoch in history.get('phase_transitions', []):
        ax.axvline(x=phase_epoch, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    # Masked Accuracy
    ax = axes[0, 1]
    ax.plot([x * 100 for x in history['train_masked_acc']], label='Train')
    ax.plot([x * 100 for x in history['val_masked_acc']], label='Val')
    for phase_epoch in history.get('phase_transitions', []):
        ax.axvline(x=phase_epoch, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Masked Token Accuracy')
    ax.legend()
    ax.grid(True)

    # Current Phase
    ax = axes[0, 2]
    ax.plot(history['phase'], color='purple', marker='.')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Phase')
    ax.set_title('Training Phase')
    ax.set_ylim(0.5, 3.5)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Phase 1\n(L0-L4)', 'Phase 2\n(L0-L8)', 'Phase 3\n(L0-L17)'])
    ax.grid(True)

    # Feature Loss
    ax = axes[1, 0]
    ax.plot(history['train_feature_loss'], label='Train')
    ax.plot(history['val_feature_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature Loss')
    ax.legend()
    ax.grid(True)

    # Distance
    ax = axes[1, 1]
    ax.plot(history['train_dist'], label='Train')
    ax.plot(history['val_dist'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Average Distance')
    ax.legend()
    ax.grid(True)

    # Learning Rate
    ax = axes[1, 2]
    ax.plot(history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Exp E: Progressive Training')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp_e_progressive')
    parser.add_argument('--output_dir', type=str, default=None)

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)

    # Loss weights
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)

    # Progressive Training 參數
    parser.add_argument('--phase1_epochs', type=int, default=100,
                        help='Phase 1 (L0-L4) 訓練 epochs')
    parser.add_argument('--phase2_epochs', type=int, default=100,
                        help='Phase 2 (L0-L8) 訓練 epochs')
    parser.add_argument('--phase3_epochs', type=int, default=100,
                        help='Phase 3 (L0-L17) 訓練 epochs')

    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--encoder_stride', type=int, default=320)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(__file__).parent / 'runs' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = args.phase1_epochs + args.phase2_epochs + args.phase3_epochs

    config = vars(args)
    config['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config['total_epochs'] = total_epochs
    config['experiment_type'] = 'Exp E: Progressive Training'
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"Exp E: Progressive Training")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)
    print(f"Training Schedule:")
    print(f"  Phase 1 (L0-L4):   {args.phase1_epochs} epochs")
    print(f"  Phase 2 (L0-L8):   {args.phase2_epochs} epochs")
    print(f"  Phase 3 (L0-L17):  {args.phase3_epochs} epochs")
    print(f"  Total:             {total_epochs} epochs")
    print("=" * 60)

    # Load data
    print("\n載入資料...")
    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        curriculum_mode='curriculum',
        initial_phase=1.0,  # 使用全部資料
        compute_snr=False,  # Progressive training 不需要 SNR 分類
    )

    # Load distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)

    # Create model
    print("\n創建模型...")
    model = ProgressiveLoRAModel(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        initial_phase=1,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )

    # Loss Function
    loss_fn = MaskedCombinedLossV2(
        feature_weight=args.feature_weight,
        cosine_weight=0.0,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        ce_weight=0.0,
        encoder_stride=args.encoder_stride,
    )

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_masked_acc': [], 'val_masked_acc': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_dist': [], 'val_dist': [],
        'phase': [],
        'phase_transitions': [],
        'lr': [],
    }

    best_val_acc = 0
    best_epoch = 0
    global_epoch = 0

    # Phase-wise training
    phase_epochs = [
        (1, args.phase1_epochs),
        (2, args.phase2_epochs),
        (3, args.phase3_epochs),
    ]

    for phase, num_epochs in phase_epochs:
        print("\n" + "=" * 60)
        print(f"PHASE {phase}: {PHASE_CONFIGS[phase]['name']}")
        print(f"Training for {num_epochs} epochs")
        print("=" * 60)

        if phase > 1:
            model.advance_phase()
            history['phase_transitions'].append(global_epoch)

        # 重新創建 optimizer（因為 LoRA 參數變了）
        optimizer = torch.optim.AdamW(
            get_trainable_params(model),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Scheduler for this phase
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=min(args.warmup_epochs, num_epochs // 2)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs - args.warmup_epochs, eta_min=args.lr * 0.01
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[min(args.warmup_epochs, num_epochs // 2)]
        )

        scaler = GradScaler() if args.use_amp else None

        for epoch in range(1, num_epochs + 1):
            global_epoch += 1

            print(f"\n[Phase {phase}] Epoch {epoch}/{num_epochs} (Global: {global_epoch})")

            train_metrics = train_epoch(
                model, train_loader, optimizer, loss_fn, device, global_epoch,
                distance_matrix, args.encoder_stride, scaler, args.use_amp,
                args.grad_clip, args.gradient_accumulation_steps
            )

            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            scheduler.step()

            val_metrics = validate(
                model, val_loader, loss_fn, device,
                distance_matrix, args.encoder_stride, args.use_amp
            )

            # Update history
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['train_masked_acc'].append(train_metrics['masked_acc'])
            history['val_masked_acc'].append(val_metrics['masked_acc'])
            history['train_feature_loss'].append(train_metrics['feature_loss'])
            history['val_feature_loss'].append(val_metrics['feature_loss'])
            history['train_dist'].append(train_metrics['distance_loss'])
            history['val_dist'].append(val_metrics['distance_loss'])
            history['phase'].append(phase)

            train_val_gap = train_metrics['masked_acc'] - val_metrics['masked_acc']

            print(f"  Train: Loss={train_metrics['total_loss']:.4f}, "
                  f"Acc={train_metrics['masked_acc']*100:.2f}%")
            print(f"  Val:   Loss={val_metrics['total_loss']:.4f}, "
                  f"Acc={val_metrics['masked_acc']*100:.2f}%")
            print(f"  Gap:   {train_val_gap*100:.2f}%")

            if val_metrics['masked_acc'] > best_val_acc:
                best_val_acc = val_metrics['masked_acc']
                best_epoch = global_epoch
                torch.save({
                    'epoch': global_epoch,
                    'phase': phase,
                    'model_state_dict': model.state_dict(),
                    'val_masked_acc': val_metrics['masked_acc'],
                    'config': config,
                }, exp_dir / 'best_model.pt')
                print(f"  ★ New best! Val Acc: {best_val_acc*100:.2f}%")

            # 每個 phase 結束時保存
            if epoch == num_epochs:
                torch.save({
                    'epoch': global_epoch,
                    'phase': phase,
                    'model_state_dict': model.state_dict(),
                    'val_masked_acc': val_metrics['masked_acc'],
                }, exp_dir / f'phase{phase}_final.pt')

            # 保存音檔樣本（每 50 epochs 或每個 phase 開始/結束）
            if epoch == 1 or epoch == num_epochs or epoch % 50 == 0:
                save_audio_samples(model, train_loader, device, exp_dir, global_epoch, num_samples=2, split='train')
                save_audio_samples(model, val_loader, device, exp_dir, global_epoch, num_samples=2, split='val')

            plot_metrics(history, exp_dir)
            with open(exp_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("訓練完成!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"Results saved to: {exp_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
