"""
exp_0223: Decoder LoRA Fine-tune 模型

架構：
    Teacher (Frozen):
        Clean Audio → Encoder → Teacher VQ → teacher_quantized
    Student (Frozen, from exp_0217):
        Noisy Audio → LoRA Encoder → Student VQ → student_quantized
    Decoder (LoRA, Trainable):
        student_quantized → [backbone with LoRA on pwconv1/pwconv2] → head → recon_wav

Loss: MSE(recon_wav, clean_wav)  — 端對端 wav-domain 監督

科學目標：
    解凍 WavTokenizer decoder 的 backbone ConvNeXt 層（pwconv1/pwconv2 加 LoRA），
    讓 decoder 學習從 student VQ tokens 還原更乾淨的音訊，突破 frozen decoder 的
    PESQ ceiling (1.79)。

關鍵設計：
    - student encoder + VQ 完全凍結（繼承 exp_0217 weights）
    - decoder backbone 的 12×2=24 個 Linear 層加 LoRA (rank=32)
    - WavTokenizer.decode() 有 @inference_mode decorator → 直接呼叫 backbone+head 繞過
    - PEFT 對 Linear 有原生支援，不需要 apply_lora_patch（那是給 Conv1d 的）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from peft import LoraConfig, get_peft_model
from families.deps.encoder_vq_core.models_single_vq import TeacherStudentSingleVQ


class TeacherStudentDecoderLoRA(TeacherStudentSingleVQ):
    """Decoder LoRA 模型：凍結 Encoder+VQ，解鎖 Decoder ConvNeXt pwconv1/pwconv2

    繼承 TeacherStudentSingleVQ，新增：
    1. 凍結所有繼承的參數（encoder LoRA、VQ）
    2. 對 teacher.backbone 的 ConvNeXt Block 的 pwconv1/pwconv2 加 LoRA
    3. 覆寫 decode()，繞過 @inference_mode 讓 gradient 可流過 decoder

    Args:
        wavtok_config: WavTokenizer config yaml 路徑
        wavtok_ckpt: WavTokenizer checkpoint 路徑
        lora_rank: Student encoder LoRA rank（用於載入 checkpoint，不影響 trainability）
        lora_alpha: Student encoder LoRA alpha
        device: 計算裝置
        decoder_lora_rank: Decoder ConvNeXt LoRA rank，預設 32
        decoder_lora_alpha: Decoder ConvNeXt LoRA alpha，預設 64
        decoder_lora_dropout: LoRA dropout，預設 0.1
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
        # 先建立父類（含 encoder LoRA + VQ）
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
        # VocosBackbone 包含 12 個 ConvNeXtBlock，每個有：
        #   pwconv1: Linear(768 → 2304)
        #   pwconv2: Linear(2304 → 768)
        # 共 24 個 Linear 層，PEFT 原生支援 Linear LoRA
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

        # 儲存 decoder LoRA 設定
        self.decoder_lora_rank = decoder_lora_rank
        self.decoder_lora_alpha = decoder_lora_alpha

        # 統計 trainable 參數
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"TeacherStudentDecoderLoRA Configuration:")
        print(f"  Decoder LoRA rank: {decoder_lora_rank}")
        print(f"  Decoder LoRA alpha: {decoder_lora_alpha}")
        print(f"  Decoder LoRA dropout: {decoder_lora_dropout}")
        print(f"  Target modules: pwconv1, pwconv2 (ConvNeXtBlock × 12)")
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"  Encoder + VQ: FROZEN (from exp_0217 checkpoint)")
        print(f"{'='*60}\n")

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """覆寫 decode，繞過 WavTokenizer.decode() 的 @inference_mode decorator

        WavTokenizer.decode() 有 @torch.inference_mode() 裝飾器，在 inference_mode
        下所有計算都不追蹤梯度。我們直接呼叫 backbone → head 讓梯度流過 decoder LoRA。

        Args:
            quantized: Student VQ 量化輸出 [B, dim, T]

        Returns:
            重建的音頻 [B, 1, audio_length]
        """
        bandwidth_id = torch.tensor([0], device=quantized.device)
        # 直接呼叫 backbone（已有 LoRA，可接收 gradient）
        x = self.teacher.backbone(quantized, bandwidth_id=bandwidth_id)
        # head 也不用 no_grad（讓 gradient 流過）
        audio = self.teacher.head(x)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio

    def forward_wav(self, clean_audio: torch.Tensor,
                    noisy_audio: torch.Tensor) -> dict:
        """Forward pass，返回 recon_wav 用於 wav-domain loss

        encoder + VQ 全部在 torch.no_grad() 中執行（凍結），
        只有 decode() 需要梯度。

        Args:
            clean_audio: 乾淨音訊 [B, 1, T]（作為 loss target）
            noisy_audio: 帶噪音訊 [B, 1, T]（作為 encoder 輸入）

        Returns:
            dict 包含:
                - recon_wav: decoder 輸出 [B, 1, T_wav]
                - student_quantized: student VQ 輸出 [B, dim, T_frame]
                - teacher_encoder_out: teacher encoder 輸出（用於 debug）
        """
        with torch.no_grad():
            # 父類 forward 包含 teacher + student encoder + VQ，全部凍結
            out = super().forward(clean_audio, noisy_audio)

        # decode 需要梯度（decoder LoRA）
        # student_quantized 是 no_grad tensor，但 decode 的計算圖會建立在 backbone 上
        student_quantized = out['student_quantized'].detach()
        recon_wav = self.decode(student_quantized)

        return {
            'recon_wav': recon_wav,
            'student_quantized': student_quantized,
            'teacher_encoder_out': out['teacher_encoder_out'],
        }

    def get_decoder_lora_state_dict(self) -> dict:
        """只提取 decoder LoRA 參數（用於 checkpoint）"""
        lora_state = {}
        for name, param in self.named_parameters():
            if param.requires_grad and 'lora_' in name:
                lora_state[name] = param.data.clone()
        return lora_state

    def load_encoder_vq_checkpoint(self, ckpt_path: str, strict: bool = False):
        """從 exp_0217 checkpoint 載入 encoder + VQ 參數

        Args:
            ckpt_path: checkpoint 路徑 (best_model.pt)
            strict: 是否嚴格匹配 keys
        """
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

        if 'model_state_dict' in ckpt:
            # best_model.pt 格式（exp_0217）
            # 從 full state dict 提取 encoder（student）和 VQ 部分
            full_state = ckpt['model_state_dict']
            student_state = {
                k.replace('student.', '', 1): v
                for k, v in full_state.items()
                if k.startswith('student.')
            }
            vq_state = {
                k.replace('vq.', '', 1): v
                for k, v in full_state.items()
                if k.startswith('vq.')
            }
            # Load student (encoder LoRA)
            missing, unexpected = self.student.load_state_dict(
                student_state, strict=strict
            )
            if missing:
                print(f"  Student: {len(missing)} missing keys")
            if unexpected:
                print(f"  Student: {len(unexpected)} unexpected keys")

            # Load VQ
            if vq_state:
                self.vq.load_state_dict(vq_state)
                print("  VQ: loaded from model_state_dict")
            elif 'vq_state_dict' in ckpt:
                self.vq.load_state_dict(ckpt['vq_state_dict'])
                print("  VQ: loaded from vq_state_dict")

        elif 'lora_state' in ckpt:
            # checkpoint_epochXXX.pt 格式
            self.student.load_state_dict(ckpt['lora_state'], strict=False)
            self.vq.load_state_dict(ckpt['vq_state_dict'])
            print("  Loaded from lora_state + vq_state_dict")

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
