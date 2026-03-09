"""
exp_0224: No-VQ Encoder LoRA 模型

架構：
    Teacher (Frozen):
        Clean Audio → Encoder → teacher_encoder_out

    Student (Trainable, Encoder LoRA):
        Noisy Audio → LoRA Encoder → student_encoder_out (連續)

    Decoder (Frozen, WavTokenizer pretrained):
        student_encoder_out → backbone → head → recon_wav

Loss: MSE + MR-STFT + Mel Loss (student_recon_wav, clean_wav)

科學目標：
    跳過 VQ 量化瓶頸，直接訓練 Encoder LoRA 讓連續 feature
    可以被 frozen decoder 正確解碼成 clean wav。

    對比 exp_0223（Decoder LoRA，有 VQ）：
      - exp_0223: Encoder+VQ frozen → Decoder LoRA trainable
      - exp_0224: Encoder LoRA trainable → VQ 跳過 → Decoder frozen

關鍵設計：
    - decoder 完全 frozen（不加 LoRA），使用原始 WavTokenizer weights
    - encoder LoRA 繼承 exp_0217 checkpoint（或從 exp_0216 baseline）
    - decode() 直接呼叫 backbone+head，繞過 @inference_mode
    - student_encoder_out [B, 512, T] 直接餵 decoder（與 VQ quantized 同維度）
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from families.deps.encoder_vq_core.models_single_vq import TeacherStudentSingleVQ


class TeacherStudentNoVQ(TeacherStudentSingleVQ):
    """No-VQ 模型：訓練 Encoder LoRA，跳過 VQ，Decoder 完全 Frozen

    繼承 TeacherStudentSingleVQ，主要差異：
    1. VQ 完全不使用（不量化，直接傳 continuous feature）
    2. Decoder 凍結（使用原始 pretrained weights，不加 LoRA）
    3. 只有 Encoder LoRA 可訓練

    Args:
        wavtok_config: WavTokenizer config yaml 路徑
        wavtok_ckpt: WavTokenizer checkpoint 路徑
        lora_rank: Encoder LoRA rank
        lora_alpha: Encoder LoRA alpha
        device: 計算裝置
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        intermediate_indices: List[int] = [3, 4, 6],
        device: str = 'cuda',
        vq_ema_decay: float = 0.99,
        vq_ema_eps: float = 1e-5,
        vq_ema_threshold: int = 2,
        vq_ema_usage_penalty: float = 0.0,
    ):
        super().__init__(
            wavtok_config=wavtok_config,
            wavtok_ckpt=wavtok_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            intermediate_indices=intermediate_indices,
            device=device,
            vq_ema_decay=vq_ema_decay,
            vq_ema_eps=vq_ema_eps,
            vq_ema_threshold=vq_ema_threshold,
            vq_ema_usage_penalty=vq_ema_usage_penalty,
        )

        # 凍結 VQ 和 teacher（student encoder LoRA 保持可訓練）
        for p in self.vq.parameters():
            p.requires_grad_(False)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # 統計可訓練參數
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"TeacherStudentNoVQ Configuration:")
        print(f"  Encoder LoRA rank: {lora_rank}, alpha: {lora_alpha}")
        print(f"  VQ: FROZEN (unused in forward_wav)")
        print(f"  Decoder: FROZEN (original pretrained weights)")
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"{'='*60}\n")

    def decode_continuous(self, features: torch.Tensor) -> torch.Tensor:
        """直接用連續 feature 解碼，繞過 @inference_mode

        Args:
            features: Encoder 輸出的連續 feature [B, 512, T]

        Returns:
            重建音頻 [B, 1, T_wav]
        """
        bandwidth_id = torch.tensor([0], device=features.device)
        x = self.teacher.backbone(features, bandwidth_id=bandwidth_id)
        audio = self.teacher.head(x)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio

    def forward_wav(self, clean_audio: torch.Tensor,
                    noisy_audio: torch.Tensor) -> dict:
        """Forward pass，No-VQ 版本

        Encoder LoRA 在梯度模式下執行（可訓練），
        Decoder 在 no_grad 下執行（frozen，但需要繞過 @inference_mode）。

        Args:
            clean_audio: 乾淨音訊 [B, 1, T]（loss target）
            noisy_audio: 帶噪音訊 [B, 1, T]（encoder 輸入）

        Returns:
            dict:
                - recon_wav: decoder 輸出 [B, 1, T_wav]
                - student_encoder_out: encoder 連續輸出 [B, 512, T_frame]
                - teacher_encoder_out: teacher encoder 輸出（用於 debug）
        """
        # Teacher encoder（完全 frozen，用於 debug/比較）
        with torch.no_grad():
            teacher_encoder_out, _ = self.teacher_extractor(clean_audio)

        # Student encoder LoRA（可訓練，有梯度）
        student_encoder_out, _ = self.student_extractor(noisy_audio)

        # 直接用連續 feature 解碼（跳過 VQ）
        # decode_continuous 呼叫 backbone+head，需要 gradient 流過 encoder
        recon_wav = self.decode_continuous(student_encoder_out)

        return {
            'recon_wav': recon_wav,
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
        }

    def load_encoder_checkpoint(self, ckpt_path: str, strict: bool = False):
        """從 exp_0217 checkpoint 載入 encoder LoRA 參數

        Args:
            ckpt_path: checkpoint 路徑
            strict: 是否嚴格匹配 keys
        """
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

        if 'model_state_dict' in ckpt:
            full_state = ckpt['model_state_dict']
            student_state = {
                k.replace('student.', '', 1): v
                for k, v in full_state.items()
                if k.startswith('student.')
            }
            missing, unexpected = self.student.load_state_dict(
                student_state, strict=strict
            )
            if missing:
                print(f"  Student: {len(missing)} missing keys")
            if unexpected:
                print(f"  Student: {len(unexpected)} unexpected keys")

        elif 'lora_state' in ckpt:
            self.student.load_state_dict(ckpt['lora_state'], strict=False)
            print("  Loaded encoder from lora_state")

        else:
            raise ValueError(f"Unknown checkpoint format, keys: {list(ckpt.keys())}")

        print(f"  Checkpoint loaded: {ckpt_path}")
        if 'epoch' in ckpt:
            print(f"  Source epoch: {ckpt['epoch']}")
        if 'metrics' in ckpt:
            m = ckpt['metrics']
            if isinstance(m, dict):
                mse = m.get('feature_mse', m.get('val_total_loss', 'N/A'))
                print(f"  Source val MSE: {mse}")
