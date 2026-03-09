"""
exp_0226: No-VQ End-to-End LoRA 模型

架構：
    Encoder LoRA (Trainable, 初始化自 exp_0224a best_model.pt):
        Noisy Audio → LoRA Encoder → student_encoder_out [B, 512, T]

    VQ: 完全跳過

    Decoder LoRA (Trainable, 從 WavTokenizer pretrain 初始化):
        student_encoder_out → [backbone LoRA pwconv1/pwconv2] → head → recon_wav

Loss: MSE + MR-STFT + Mel (recon_wav, clean_wav)

設計動機：
    exp_0224a (Encoder LoRA only):
        - Encoder 學習把 LDV feature 對齊到 decoder 期望的分佈
        - Decoder frozen → 受限於預訓練分佈，無機械音但有天花板
        - PESQ ≈ 1.586

    exp_0224b (Decoder LoRA only):
        - Encoder frozen → Decoder 被迫適應不熟悉的 LDV feature
        - PESQ ≈ 1.868，但有機械音（phase artifact）
        - LoRA rank=32 能力有限，phase 學不乾淨

    exp_0226 (E2E, 兩者同時):
        - Encoder + Decoder 協同學習，找到對 LDV 任務最優的中間表示
        - Encoder 不需要強迫對齊 VQ 分佈
        - Decoder 從 encoder 習慣的 feature 學習解碼
        - 理論上是目前架構能達到的音質上限

Checkpoint 載入策略：
    - Encoder LoRA: 從 exp_0224a best_model.pt 的 model_state_dict 載入
    - Decoder LoRA: 隨機初始化（WavTokenizer backbone frozen weights + 新的 LoRA）
    - 不載入 exp_0224b decoder 權重（避免引入 phase artifact 的偏差）
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
from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import TeacherStudentSingleVQ


class TeacherStudentNoVQE2E(TeacherStudentSingleVQ):
    """No-VQ 端對端 LoRA 模型：Encoder LoRA + Decoder LoRA 同時訓練

    繼承 TeacherStudentSingleVQ，主要差異：
    1. 凍結所有繼承的參數（Teacher encoder, VQ, Teacher model）
    2. Student encoder LoRA 保持可訓練
    3. Decoder backbone pwconv1/pwconv2 加 LoRA，也可訓練
    4. forward_wav: encoder → feature → decoder，全程梯度流通

    Args:
        wavtok_config: WavTokenizer config yaml 路徑
        wavtok_ckpt: WavTokenizer checkpoint 路徑
        lora_rank: Encoder LoRA rank，預設 64
        lora_alpha: Encoder LoRA alpha，預設 128
        device: 計算裝置
        decoder_lora_rank: Decoder LoRA rank，預設 32
        decoder_lora_alpha: Decoder LoRA alpha，預設 64
        decoder_lora_dropout: Decoder LoRA dropout，預設 0.0（E2E 不用 dropout 避免梯度噪音）
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
        decoder_lora_dropout: float = 0.0,
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

        # Step 1: 凍結所有繼承的參數（teacher encoder, VQ, teacher model）
        for p in self.parameters():
            p.requires_grad_(False)

        # Step 2: 解凍 student encoder LoRA（已由父類 peft 包裝）
        for name, p in self.student.named_parameters():
            if 'lora_' in name:
                p.requires_grad_(True)

        # Step 3: 對 decoder backbone 的 ConvNeXt pwconv1/pwconv2 加 LoRA
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

        enc_trainable = sum(
            p.numel() for n, p in self.student.named_parameters()
            if p.requires_grad
        )
        dec_trainable = sum(
            p.numel() for n, p in self.teacher.backbone.named_parameters()
            if p.requires_grad
        )
        total = sum(p.numel() for p in self.parameters())

        print(f"\n{'='*60}")
        print(f"TeacherStudentNoVQE2E Configuration:")
        print(f"  Encoder LoRA rank: {lora_rank}, alpha: {lora_alpha}")
        print(f"  Decoder LoRA rank: {decoder_lora_rank}, alpha: {decoder_lora_alpha}")
        print(f"  Decoder target modules: pwconv1, pwconv2 (ConvNeXtBlock × 12)")
        print(f"  VQ: SKIPPED")
        print(f"  Trainable - Encoder LoRA: {enc_trainable:,}")
        print(f"  Trainable - Decoder LoRA: {dec_trainable:,}")
        print(f"  Trainable - Total: {enc_trainable + dec_trainable:,} / {total:,} "
              f"({100*(enc_trainable+dec_trainable)/total:.2f}%)")
        print(f"{'='*60}\n")

    def decode_continuous(self, features: torch.Tensor) -> torch.Tensor:
        """用連續 feature 解碼，gradient 流過 decoder LoRA

        Args:
            features: [B, 512, T]（encoder 輸出，未量化）

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
        """E2E Forward: Encoder LoRA + Decoder LoRA 全程梯度流通

        Args:
            clean_audio: 乾淨音訊 [B, 1, T]（loss target，teacher encoder 用）
            noisy_audio: 帶噪音訊 [B, 1, T]（student encoder 輸入）

        Returns:
            dict:
                - recon_wav: decoder 輸出 [B, 1, T_wav]
                - student_encoder_out: encoder 連續輸出 [B, 512, T_frame]
                - teacher_encoder_out: teacher encoder 輸出（debug 用，no_grad）
        """
        # Teacher encoder 只用於 debug/監控，不參與訓練
        with torch.no_grad():
            teacher_encoder_out, _ = self.teacher_extractor(clean_audio)

        # Student encoder LoRA：梯度流通（E2E 關鍵）
        student_encoder_out, _ = self.student_extractor(noisy_audio)

        # Decoder LoRA：直接從 student feature 解碼，梯度流通
        # 注意：不 detach()，讓梯度同時更新 encoder 和 decoder
        recon_wav = self.decode_continuous(student_encoder_out)

        return {
            'recon_wav': recon_wav,
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
        }

    def load_encoder_checkpoint(self, ckpt_path: str, strict: bool = False):
        """從 exp_0224a checkpoint 載入 encoder LoRA 參數

        Args:
            ckpt_path: exp_0224a best_model.pt 路徑
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
                print(f"  Encoder: {len(missing)} missing keys (expected for non-LoRA)")
            if unexpected:
                print(f"  Encoder: {len(unexpected)} unexpected keys")
            print(f"  Encoder LoRA loaded from model_state_dict: {ckpt_path}")

        elif 'encoder_lora_state' in ckpt:
            self.student.load_state_dict(ckpt['encoder_lora_state'], strict=False)
            print(f"  Encoder LoRA loaded from encoder_lora_state: {ckpt_path}")

        else:
            raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

        if 'epoch' in ckpt:
            print(f"  Source epoch: {ckpt['epoch']}")
        if 'metrics' in ckpt:
            m = ckpt['metrics']
            if isinstance(m, dict):
                mse = m.get('val_wav_mse', m.get('feature_mse', 'N/A'))
                print(f"  Source val MSE: {mse}")

    def get_trainable_state_dict(self) -> dict:
        """提取所有可訓練參數（encoder LoRA + decoder LoRA）用於 checkpoint"""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
