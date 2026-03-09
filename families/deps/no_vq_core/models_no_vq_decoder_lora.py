"""
exp_0224b: No-VQ + Decoder LoRA 模型

架構：
    Teacher (Frozen):
        Clean Audio → Encoder → teacher_encoder_out

    Student Encoder (Frozen, 繼承 exp_0217):
        Noisy Audio → LoRA Encoder → student_encoder_out (連續 [B, 512, T])

    Decoder (LoRA Trainable):
        student_encoder_out → [backbone with LoRA on pwconv1/pwconv2] → head → recon_wav

Loss: MSE + MR-STFT + Mel Loss (recon_wav, clean_wav)

科學目標：
    同時跳過 VQ 量化瓶頸，並讓 Decoder LoRA 學習從連續 feature 還原 clean wav。

    對比矩陣：
      exp_0223 v1/v2: Encoder+VQ frozen, Decoder LoRA → 改善 decoder 適應 VQ tokens
      exp_0224a:      Encoder LoRA 訓練, VQ 跳過, Decoder frozen → 學習對齊 decoder input space
      exp_0224b:      Encoder frozen, VQ 跳過, Decoder LoRA → 讓 decoder 學習連續 feature 解碼
      (未來) exp_0225: Encoder LoRA + VQ 跳過 + Decoder LoRA → 全端對端

關鍵設計：
    - Encoder LoRA 完全 frozen（繼承 exp_0217 weights）
    - VQ 完全不使用
    - Decoder backbone pwconv1/pwconv2 加 LoRA (rank=32)
    - decode_continuous() 繞過 @inference_mode，接受連續 feature 輸入
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from peft import LoraConfig, get_peft_model
from families.deps.encoder_vq_core.models_single_vq import TeacherStudentSingleVQ


class TeacherStudentNoVQDecoderLoRA(TeacherStudentSingleVQ):
    """No-VQ + Decoder LoRA 模型

    繼承 TeacherStudentSingleVQ，主要差異：
    1. 凍結所有繼承的參數（encoder LoRA、VQ、teacher）
    2. VQ 完全不使用（forward_wav 中跳過）
    3. Decoder backbone pwconv1/pwconv2 加 LoRA，可訓練
    4. decode_continuous() 接受連續 feature，繞過 @inference_mode

    Args:
        wavtok_config: WavTokenizer config yaml 路徑
        wavtok_ckpt: WavTokenizer checkpoint 路徑
        lora_rank: Encoder LoRA rank（用於載入 checkpoint）
        lora_alpha: Encoder LoRA alpha
        device: 計算裝置
        decoder_lora_rank: Decoder LoRA rank，預設 32
        decoder_lora_alpha: Decoder LoRA alpha，預設 64
        decoder_lora_dropout: Decoder LoRA dropout，預設 0.1
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
        decoder_lora_rank: int = 32,
        decoder_lora_alpha: int = 64,
        decoder_lora_dropout: float = 0.1,
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

        # Step 1: 凍結所有繼承的參數（encoder LoRA + VQ + teacher）
        for p in self.parameters():
            p.requires_grad_(False)

        # Step 2: 對 backbone 的 ConvNeXt pwconv1/pwconv2 加 LoRA
        decoder_lora_config = LoraConfig(
            r=decoder_lora_rank,
            lora_alpha=decoder_lora_alpha,
            target_modules=["pwconv1", "pwconv2"],
            lora_dropout=decoder_lora_dropout,
            bias="none",
        )
        self.teacher.backbone = get_peft_model(
            self.teacher.backbone, decoder_lora_config
        )

        self.decoder_lora_rank = decoder_lora_rank
        self.decoder_lora_alpha = decoder_lora_alpha

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"TeacherStudentNoVQDecoderLoRA Configuration:")
        print(f"  Encoder LoRA: FROZEN (from exp_0217)")
        print(f"  VQ: SKIPPED (no quantization)")
        print(f"  Decoder LoRA rank: {decoder_lora_rank}, alpha: {decoder_lora_alpha}")
        print(f"  Target modules: pwconv1, pwconv2 (ConvNeXtBlock × 12)")
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"{'='*60}\n")

    def decode_continuous(self, features: torch.Tensor) -> torch.Tensor:
        """用連續 feature 解碼，繞過 @inference_mode，讓 gradient 流過 decoder LoRA

        Args:
            features: 連續 feature [B, 512, T]（直接來自 encoder，未量化）

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
        """Forward pass：Encoder frozen，VQ 跳過，Decoder LoRA 可訓練

        Args:
            clean_audio: 乾淨音訊 [B, 1, T]（loss target）
            noisy_audio: 帶噪音訊 [B, 1, T]（encoder 輸入）

        Returns:
            dict:
                - recon_wav: decoder 輸出 [B, 1, T_wav]
                - student_encoder_out: encoder 連續輸出 [B, 512, T_frame]
                - teacher_encoder_out: teacher encoder 輸出（debug 用）
        """
        with torch.no_grad():
            # Teacher encoder（debug/比較用）
            teacher_encoder_out, _ = self.teacher_extractor(clean_audio)
            # Student encoder frozen
            student_encoder_out, _ = self.student_extractor(noisy_audio)

        # Decoder LoRA 需要 gradient，從 student_encoder_out 解碼
        # student_encoder_out 是 no_grad tensor，但 decode_continuous
        # 的計算圖建立在 decoder LoRA 參數上
        student_features = student_encoder_out.detach()
        recon_wav = self.decode_continuous(student_features)

        return {
            'recon_wav': recon_wav,
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
        }

    def get_decoder_lora_state_dict(self) -> dict:
        """提取 decoder LoRA 參數（用於 checkpoint）"""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad and 'lora_' in name
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
