"""
exp_0206 Plan Original: Single VQ 4096 + EMA Update 模型

核心概念:
- 使用單層 VQ (K=4096) 替代 RVQ (4×2048)
- 從 WavTokenizer 預訓練 codebook 初始化 (warm start)
- EMA update + dead-code reset (避免 frozen 導致的 collapse)

科學問題:
1. 預訓練 codebook + EMA 能否避免 token collapse？
2. Warm start (預訓練) vs Cold start (隨機初始化) 哪個更好？
3. 單層 vs 多層 VQ 的必要性是什麼？

架構:
    Teacher (Frozen):
        Clean Audio → Encoder → t_e [B,128,T] → (Frozen VQ)
                        ↓ (L3, L4, L6)
                      t_inter

    Student (LoRA):
        Noisy Audio → Encoder → z_e [B,128,T]
                        ↓ (L3, L4, L6)     ↓
                      s_inter        ┌─────────────┐
                                     │  Single VQ  │
                                     │  + EMA      │
                                     │  K=4096     │
                                     └─────────────┘
                                           ↓
                                      z_q [B,128,T]

    Loss = L_quant + L_commit + L_inter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from exp_1201.wavtok_lora_patch import apply_lora_patch
from exp_0112_intermediate.models import (
    TeacherStudentIntermediate,
    IntermediateExtractor,
    ALL_18_LAYERS,
    INTERMEDIATE_SUPERVISION_POINTS,
)

apply_lora_patch()


class SingleVQWithEMA(nn.Module):
    """單層 Vector Quantizer + EMA 更新機制

    從 WavTokenizer 預訓練 codebook 初始化，使用 EMA 更新
    而非凍結。搭配 dead-code reset 避免 token collapse。

    Args:
        codebook_size: Codebook 大小
        dim: 向量維度
        pretrained_codebook: 預訓練的 codebook 權重 (codebook_size × dim)
        ema_decay: EMA 衰減率
        ema_eps: EMA epsilon (Laplace smoothing)
        ema_dead_code_threshold: Dead-code reset 門檻 (0 表示停用)
        ema_usage_penalty: 使用頻率懲罰 (用於降低 top-k mass)
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        dim: int = 128,
        pretrained_codebook: torch.Tensor = None,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        ema_dead_code_threshold: int = 2,
        ema_usage_penalty: float = 0.0,
    ):
        """初始化 Single VQ + EMA

        Args:
            codebook_size: Codebook 大小，預設 4096
            dim: 向量維度，預設 128
            pretrained_codebook: 預訓練 codebook 權重張量
            ema_decay: EMA 衰減率，預設 0.99
            ema_eps: EMA 平滑項，預設 1e-5
            ema_dead_code_threshold: Dead-code 門檻，預設 2
            ema_usage_penalty: 使用頻率懲罰係數，預設 0.0
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.ema_dead_code_threshold = int(ema_dead_code_threshold)
        self.ema_usage_penalty = float(ema_usage_penalty)

        # Codebook (nn.Embedding)
        self.codebook = nn.Embedding(codebook_size, dim)

        # 從預訓練 codebook 初始化 (warm start)
        if pretrained_codebook is not None:
            assert pretrained_codebook.shape == (codebook_size, dim), \
                f"預期 shape ({codebook_size}, {dim})，得到 {pretrained_codebook.shape}"
            self.codebook.weight.data = pretrained_codebook.clone()
        else:
            # Fallback: uniform random
            self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # EMA 模式：停用梯度
        self.codebook.weight.requires_grad_(False)

        # EMA buffers
        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(codebook_size),
            persistent=True,
        )
        self.register_buffer(
            "ema_embed_avg",
            torch.zeros(codebook_size, dim),
            persistent=True,
        )

        # 初始化 EMA buffers（如果有預訓練 codebook）
        if pretrained_codebook is not None:
            # 初始化為均勻分佈假設
            self.ema_cluster_size.fill_(1.0)
            self.ema_embed_avg.copy_(pretrained_codebook.clone())

    @torch.no_grad()
    def _ema_update(
        self,
        z_flat: torch.Tensor,
        indices: torch.Tensor,
    ):
        """EMA codebook 更新 + dead-code reset

        Args:
            z_flat: 展平的輸入張量 [N, dim]，已 detach
            indices: 最近鄰索引 [N]，long 類型
        """
        z_flat = z_flat.float()  # 保持 EMA 在 fp32
        device = z_flat.device
        K = self.codebook_size

        # 計算使用次數
        counts = torch.bincount(indices, minlength=K).float()

        # 計算嵌入向量總和
        embed_sum = torch.zeros(K, self.dim, device=device)
        embed_sum.index_add_(0, indices, z_flat)

        # EMA 更新
        self.ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=(1.0 - self.ema_decay))
        self.ema_embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=(1.0 - self.ema_decay))

        # Laplace smoothing 正規化
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.ema_eps) / (n + K * self.ema_eps) * n
        )
        embed = self.ema_embed_avg / cluster_size.unsqueeze(1).clamp(min=1e-12)
        self.codebook.weight.data.copy_(embed)

        # Dead-code reset
        if self.ema_dead_code_threshold > 0:
            dead = self.ema_cluster_size < float(self.ema_dead_code_threshold)
            if dead.any() and z_flat.numel() > 0:
                dead_idx = dead.nonzero(as_tuple=False).squeeze(1)
                num_dead = int(dead_idx.numel())
                rand = torch.randint(0, z_flat.shape[0], (num_dead,), device=device)
                sampled = z_flat[rand]

                self.codebook.weight.data[dead_idx] = sampled
                self.ema_cluster_size[dead_idx] = 1.0
                self.ema_embed_avg[dead_idx] = sampled

    def forward(
        self,
        z: torch.Tensor,
        frame_rate: int = 75,
        bandwidth: float = 0.075,
    ) -> Dict:
        """Single VQ forward pass + EMA 更新

        Args:
            z: 輸入張量 [B, dim, T]
            frame_rate: 幀率，預設 75
            bandwidth: 帶寬，預設 0.075

        Returns:
            字典包含:
                quantized: 量化後的向量 [B, dim, T]
                codes: token 索引 [1, B, 1, T]（相容 WavTokenizer 格式）
                loss_commit: encoder commitment loss
                loss_codebook: 0（EMA 模式無梯度）
                commitment_loss: 向後相容別名
                bandwidth: 帶寬張量
        """
        B, dim, T = z.shape

        # 轉置: [B, dim, T] → [B, T, dim]
        z = z.transpose(1, 2)
        z_flat = z.reshape(-1, dim)  # [B*T, dim]

        # 計算距離（memory-efficient L2 距離）
        x2 = (z_flat ** 2).sum(dim=1, keepdim=True)   # [B*T, 1]
        y2 = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)  # [1, K]
        distances = x2 + y2 - 2.0 * (z_flat @ self.codebook.weight.t())

        # 可選：使用頻率懲罰
        if self.ema_usage_penalty > 0.0:
            usage_penalty = torch.log(self.ema_cluster_size.clamp(min=1.0))
            usage_penalty = usage_penalty * self.ema_usage_penalty
            distances = distances + usage_penalty.to(distances.dtype).unsqueeze(0)

        # 找最近的 code
        indices = torch.argmin(distances, dim=1)  # [B*T]

        # 獲取量化向量
        q = self.codebook(indices)   # [B*T, dim]
        q = q.reshape(B, T, dim)     # [B, T, dim]

        # Commitment loss (encoder → quantized)
        loss_commit = F.mse_loss(z, q.detach())

        # EMA 更新（僅在訓練時）
        if self.training:
            self._ema_update(z_flat.detach(), indices)

        # Straight-through estimator
        q = z + (q - z).detach()

        # 轉回原始格式: [B, T, dim] → [B, dim, T]
        z_q = q.transpose(1, 2)

        # 格式化 codes: [B*T] → [1, B, 1, T]（相容 baseline/WavTokenizer 格式）
        codes = indices.reshape(B, T).unsqueeze(0).unsqueeze(2)

        return {
            'quantized': z_q,
            'codes': codes,
            'loss_commit': loss_commit,
            'loss_codebook': torch.tensor(0.0, device=z.device),  # EMA 模式
            'commitment_loss': loss_commit,  # 向後相容
            'bandwidth': torch.tensor([bandwidth], device=z.device),
        }

    def get_codebook_usage(self, codes: torch.Tensor) -> Dict:
        """分析 codebook 使用情況

        Args:
            codes: token 索引 [B, T] 或 [1, B, 1, T]

        Returns:
            字典包含:
                usage_count: 每個 code 的使用次數 [K]
                n_used: 被使用的 code 數量
                entropy: 使用分佈的熵 (bits)
        """
        if codes.dim() == 4:
            codes = codes[0, :, 0, :]  # [B, T]

        codes_flat = codes.flatten()
        usage_count = torch.bincount(codes_flat, minlength=self.codebook_size)

        n_used = (usage_count > 0).sum().item()

        # Entropy (bits)
        probs = usage_count.float() / usage_count.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log2()).sum().item()

        return {
            'usage_count': usage_count,
            'n_used': n_used,
            'entropy': entropy,
        }


class TeacherStudentSingleVQ(TeacherStudentIntermediate):
    """使用 Single VQ + EMA 的 Teacher-Student 模型

    繼承 TeacherStudentIntermediate（baseline），
    將凍結的 quantizer 替換為 SingleVQWithEMA。

    Args:
        wavtok_config: WavTokenizer 配置檔路徑
        wavtok_ckpt: WavTokenizer checkpoint 路徑
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        intermediate_indices: 中間層監督索引列表
        device: 計算裝置
        vq_ema_decay: EMA 衰減率
        vq_ema_eps: EMA epsilon
        vq_ema_threshold: Dead-code reset 門檻
        vq_ema_usage_penalty: 使用頻率懲罰
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        intermediate_indices: List[int] = [3, 4, 6],
        device: str = 'cuda',
        vq_ema_decay: float = 0.99,
        vq_ema_eps: float = 1e-5,
        vq_ema_threshold: int = 2,
        vq_ema_usage_penalty: float = 0.0,
    ):
        """初始化 TeacherStudentSingleVQ 模型

        Args:
            wavtok_config: WavTokenizer 配置檔路徑
            wavtok_ckpt: WavTokenizer checkpoint 路徑
            lora_rank: LoRA rank，預設 256
            lora_alpha: LoRA alpha，預設 512
            intermediate_indices: 中間層監督索引，預設 [3, 4, 6]
            device: 計算裝置，預設 'cuda'
            vq_ema_decay: EMA 衰減率，預設 0.99
            vq_ema_eps: EMA epsilon，預設 1e-5
            vq_ema_threshold: Dead-code 門檻，預設 2
            vq_ema_usage_penalty: 使用頻率懲罰，預設 0.0
        """
        # 先初始化父類（TeacherStudentIntermediate）
        super().__init__(
            wavtok_config=wavtok_config,
            wavtok_ckpt=wavtok_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            intermediate_indices=intermediate_indices,
            device=device,
        )

        # 提取預訓練 codebook
        original_quantizer = self.student.feature_extractor.encodec.quantizer
        pretrained_codebook = original_quantizer.vq.layers[0].codebook.detach().clone()
        codebook_size, quantizer_dim = pretrained_codebook.shape

        # 創建 SingleVQWithEMA（使用預訓練 codebook 初始化）
        self.vq = SingleVQWithEMA(
            codebook_size=codebook_size,
            dim=quantizer_dim,
            pretrained_codebook=pretrained_codebook,
            ema_decay=vq_ema_decay,
            ema_eps=vq_ema_eps,
            ema_dead_code_threshold=vq_ema_threshold,
            ema_usage_penalty=vq_ema_usage_penalty,
        ).to(device)

        # 保存配置
        self.vq_codebook_size = codebook_size

        print(f"\n{'='*60}")
        print(f"SingleVQ + EMA Configuration:")
        print(f"  Codebook size: {codebook_size}")
        print(f"  Dimension: {quantizer_dim}")
        print(f"  Initialization: Pretrained (warm start)")
        print(f"  EMA decay: {vq_ema_decay}")
        print(f"  EMA eps: {vq_ema_eps}")
        print(f"  Dead-code threshold: {vq_ema_threshold}")
        print(f"  Usage penalty: {vq_ema_usage_penalty}")
        print(f"{'='*60}\n")

    def forward(self, clean_audio: torch.Tensor, noisy_audio: torch.Tensor) -> dict:
        """Forward pass（使用 SingleVQWithEMA 替代凍結 quantizer）

        Args:
            clean_audio: 乾淨音訊 [B, 1, T]
            noisy_audio: 帶噪音訊 [B, 1, T]

        Returns:
            字典包含 teacher/student 編碼器輸出、中間層特徵、VQ 結果
        """
        # Teacher forward（使用原始 quantizer，凍結）
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            teacher_encoder_out, teacher_intermediates = self.teacher_extractor(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward（使用 SingleVQWithEMA）
        student_encoder_out, student_intermediates = self.student_extractor(noisy_audio)

        # 使用 SingleVQWithEMA 量化
        student_vq = self.vq(student_encoder_out, frame_rate=75, bandwidth=0.075)
        student_codes = student_vq['codes']

        return {
            'teacher_codes': teacher_codes,
            'student_codes': student_codes,
            'teacher_intermediates': teacher_intermediates,
            'student_intermediates': student_intermediates,
            'teacher_encoder_out': teacher_encoder_out,
            'student_encoder_out': student_encoder_out,
            'student_quantized': student_vq['quantized'],
            'vq_loss_commit': student_vq['loss_commit'],
            'vq_loss_codebook': student_vq['loss_codebook'],
            'vq_commitment_loss': student_vq['commitment_loss'],
        }

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """使用 teacher decoder 重建音頻

        Args:
            quantized: 量化後的向量 [B, dim, T]

        Returns:
            重建的音頻 [B, 1, audio_length]
        """
        with torch.no_grad():
            bandwidth_id = torch.tensor([0], device=quantized.device)
            audio = self.teacher.decode(quantized, bandwidth_id=bandwidth_id)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
        return audio

    def get_vq_usage(self, codes: torch.Tensor) -> Dict:
        """分析 VQ 使用情況

        Args:
            codes: token 索引 [1, B, 1, T] 或 [B, T]

        Returns:
            字典包含 usage_count, n_used, entropy
        """
        return self.vq.get_codebook_usage(codes)
