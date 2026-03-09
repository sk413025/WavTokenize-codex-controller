"""
exp_0128 Phase 3: Residual Vector Quantization (RVQ) 模型

核心改進:
- 使用多層 VQ 替代單層 VQ
- 每層量化殘差，強制多樣化
- 基於 exp_0112_intermediate 架構

RVQ 原理:
    z → q1 → residual1 → q2 → residual2 → q3 → ...
    最終: z_q = q1 + q2 + q3 + ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from exp_1201.wavtok_lora_patch import apply_lora_patch
from exp_0112_intermediate.models import (
    TeacherStudentIntermediate,
    IntermediateExtractor,
    ALL_18_LAYERS,
    INTERMEDIATE_SUPERVISION_POINTS
)

apply_lora_patch()


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization (RVQ)

    多層殘差量化器，每層量化前一層的殘差

    Args:
        n_layers: VQ 層數
        codebook_size: 每層 codebook 大小
        dim: 向量維度
        update_mode: "grad" (codebook loss + optimizer) or "ema" (EMA update + dead-code reset)
        ema_decay: EMA decay (only for update_mode="ema")
        ema_eps: EMA epsilon (only for update_mode="ema")
        ema_dead_code_threshold: dead-code threshold (0 disables; only for update_mode="ema")
        ema_usage_penalty: penalize frequently-used codes via EMA cluster_size (only for update_mode="ema")
    """

    def __init__(
        self,
        n_layers: int = 4,
        codebook_size: int = 1024,
        dim: int = 128,
        update_mode: str = "grad",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        ema_dead_code_threshold: int = 0,
        ema_usage_penalty: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self.update_mode = update_mode
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.ema_dead_code_threshold = int(ema_dead_code_threshold)
        self.ema_usage_penalty = float(ema_usage_penalty)

        # 每層獨立的 codebook
        # 使用 nn.Embedding 實現 codebook
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim)
            for _ in range(n_layers)
        ])

        # 初始化 codebooks (uniform distribution)
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        if self.update_mode not in {"grad", "ema"}:
            raise ValueError(f"Invalid update_mode={self.update_mode!r}. Must be one of: 'grad', 'ema'")

        # EMA buffers (mode B)
        if self.update_mode == "ema":
            # codebooks should not be updated by optimizer when using EMA
            for codebook in self.codebooks:
                codebook.weight.requires_grad_(False)

            self.register_buffer(
                "ema_cluster_size",
                torch.zeros(n_layers, codebook_size),
                persistent=True,
            )
            self.register_buffer(
                "ema_embed_avg",
                torch.zeros(n_layers, codebook_size, dim),
                persistent=True,
            )

    @torch.no_grad()
    def _ema_update_layer(
        self,
        layer_idx: int,
        residual_flat: torch.Tensor,  # [N, dim], detached
        indices: torch.Tensor,        # [N], long
    ) -> None:
        """EMA codebook update + (optional) dead-code reset for a single layer."""
        # AMP can produce fp16 residuals; keep EMA stats in fp32 for stability and to avoid dtype mismatch.
        residual_flat = residual_flat.float()
        device = residual_flat.device
        K = self.codebook_size

        counts = torch.bincount(indices, minlength=K).float()  # [K]

        embed_sum = torch.zeros(K, self.dim, device=device)
        embed_sum.index_add_(0, indices, residual_flat)

        self.ema_cluster_size[layer_idx].mul_(self.ema_decay).add_(counts, alpha=(1.0 - self.ema_decay))
        self.ema_embed_avg[layer_idx].mul_(self.ema_decay).add_(embed_sum, alpha=(1.0 - self.ema_decay))

        n = self.ema_cluster_size[layer_idx].sum()
        # Laplace smoothing (as in VQ-VAE / common EMA codebooks)
        cluster_size = (self.ema_cluster_size[layer_idx] + self.ema_eps) / (n + K * self.ema_eps) * n
        embed = self.ema_embed_avg[layer_idx] / cluster_size.unsqueeze(1).clamp(min=1e-12)
        self.codebooks[layer_idx].weight.data.copy_(embed)

        # dead-code reset
        if self.ema_dead_code_threshold > 0:
            dead = self.ema_cluster_size[layer_idx] < float(self.ema_dead_code_threshold)
            if dead.any() and residual_flat.numel() > 0:
                dead_idx = dead.nonzero(as_tuple=False).squeeze(1)
                num_dead = int(dead_idx.numel())
                rand = torch.randint(0, residual_flat.shape[0], (num_dead,), device=device)
                sampled = residual_flat[rand]

                self.codebooks[layer_idx].weight.data[dead_idx] = sampled
                # Keep EMA buffers consistent so the next normalization preserves sampled vectors
                self.ema_cluster_size[layer_idx, dead_idx] = 1.0
                self.ema_embed_avg[layer_idx, dead_idx] = sampled

    def forward(
        self,
        z: torch.Tensor,
        frame_rate: int = 75,
        bandwidth: float = 0.075
    ) -> Dict:
        """
        RVQ forward pass

        Args:
            z: Input tensor [batch, dim, time]
            frame_rate: 幀率 (與原始 quantizer 一致)
            bandwidth: 帶寬 (與原始 quantizer 一致)

        Returns:
            Dict containing:
                - quantized: 量化後的向量 [batch, dim, time]
                - codes: 所有層的 codes [n_layers, batch, time]
                - loss_commit: Encoder commitment (Σ_i mse(r_i, q_i.detach()))
                - loss_codebook: Codebook loss (Σ_i mse(r_i.detach(), q_i)) (only for update_mode="grad")
                - commitment_loss: Backward-compatible alias (= loss_commit + loss_codebook)
        """
        batch_size, dim, time = z.shape

        # 轉置: [batch, dim, time] → [batch, time, dim]
        z = z.transpose(1, 2)

        z_q = torch.zeros_like(z)
        residual = z
        all_codes = []
        # Keep losses as tensors for consistent logging (.item()) across update modes.
        loss_commit = z.new_zeros(())
        loss_codebook = z.new_zeros(())

        for layer_idx, codebook in enumerate(self.codebooks):
            # 計算到 codebook 的距離
            # residual: [batch, time, dim]
            # codebook.weight: [codebook_size, dim]

            # 展平時間維度
            residual_flat = residual.reshape(-1, dim)  # [batch*time, dim]

            # 計算距離 (memory-efficient; avoid torch.cdist O(N*K*D) expansion)
            # We only need argmin, so squared L2 distance is sufficient:
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
            #
            # residual_flat: [N, dim], codebook.weight: [K, dim]
            # distances: [N, K]
            x2 = (residual_flat ** 2).sum(dim=1, keepdim=True)  # [N, 1]
            y2 = (codebook.weight ** 2).sum(dim=1).unsqueeze(0)  # [1, K]
            distances = x2 + y2 - 2.0 * (residual_flat @ codebook.weight.t())

            # Optional: usage-aware penalty (EMA mode only).
            # Idea: penalize frequently-used codes (large EMA cluster_size) so top-k mass doesn't drift up.
            if self.update_mode == "ema" and self.ema_usage_penalty > 0.0:
                # clamp(min=1) keeps penalty >= 0 and avoids -inf for never-used codes.
                usage_penalty = torch.log(self.ema_cluster_size[layer_idx].clamp(min=1.0))
                usage_penalty = usage_penalty * self.ema_usage_penalty
                distances = distances + usage_penalty.to(distances.dtype).unsqueeze(0)

            # 找最近的 code
            indices = torch.argmin(distances, dim=1)  # [batch*time]

            # 獲取量化向量
            q = codebook(indices)  # [batch*time, dim]
            q = q.reshape(batch_size, time, dim)  # [batch, time, dim]

            # Phase 3-2 losses
            # Encoder commitment (updates encoder; codebook is detached)
            loss_commit = loss_commit + F.mse_loss(residual, q.detach())

            if self.update_mode == "grad":
                # Codebook loss (updates codebook; residual is detached)
                loss_codebook = loss_codebook + F.mse_loss(residual.detach(), q)
            else:
                # EMA update (updates codebook buffers/weights, not via optimizer)
                # Only update EMA during training; evaluation should not mutate the codebooks.
                if self.training:
                    self._ema_update_layer(layer_idx, residual_flat.detach(), indices)

            # Straight-through estimator
            q = residual + (q - residual).detach()

            # 累積量化結果
            z_q = z_q + q

            # 計算新的殘差
            residual = residual - q.detach()

            # 保存 codes
            codes = indices.reshape(batch_size, time)  # [batch, time]
            all_codes.append(codes)

        # 轉回原始格式: [batch, time, dim] → [batch, dim, time]
        z_q = z_q.transpose(1, 2)

        # Stack codes: [n_layers, batch, time]
        all_codes = torch.stack(all_codes, dim=0)

        # 模擬原始 quantizer 輸出格式
        # 原始格式: codes shape [1, batch, 1, time] (4D)
        # 我們需要合併多層 codes 為單一表示
        # 這裡簡單使用第一層的 codes 作為主要表示
        codes_output = all_codes[0:1].unsqueeze(2)  # [1, batch, 1, time]

        return {
            'quantized': z_q,
            'codes': codes_output,
            'all_layer_codes': all_codes,  # [n_layers, batch, time]
            'loss_commit': loss_commit,
            'loss_codebook': loss_codebook,
            'commitment_loss': (loss_commit + loss_codebook),
            'bandwidth': torch.tensor([bandwidth], device=z.device),
        }

    def get_codebook_usage(self, all_codes: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        分析每層 codebook 使用情況

        Args:
            all_codes: [n_layers, batch, time]

        Returns:
            Dict mapping layer_idx → usage_count [codebook_size]
        """
        usage = {}
        for layer_idx in range(self.n_layers):
            codes = all_codes[layer_idx].flatten()
            usage_count = torch.bincount(
                codes,
                minlength=self.codebook_size
            )
            usage[layer_idx] = usage_count
        return usage


class TeacherStudentRVQ(TeacherStudentIntermediate):
    """
    使用 RVQ 的 Teacher-Student 模型

    替換原有的單層 quantizer 為多層 RVQ
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        intermediate_indices: List[int] = [3, 6],
        device: str = 'cuda',
        n_rvq_layers: int = 4,
        rvq_codebook_size: int = 1024,
        rvq_update: str = "grad",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        ema_dead_code_threshold: int = 0,
        ema_usage_penalty: float = 0.0,
    ):
        """
        Args:
            ... (與 TeacherStudentIntermediate 相同)
            n_rvq_layers: RVQ 層數
            rvq_codebook_size: 每層 codebook 大小
        """
        # 先初始化父類
        super().__init__(
            wavtok_config=wavtok_config,
            wavtok_ckpt=wavtok_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            intermediate_indices=intermediate_indices,
            device=device
        )

        # 替換 student 的 quantizer 為 RVQ
        # 獲取原始 quantizer 的參數
        original_quantizer = self.student.feature_extractor.encodec.quantizer

        # 從原始 quantizer 獲取 dimension
        # EnCodec quantizer 的 dimension 通常在 vq.layers[0]._codebook.embed
        original_codebook = original_quantizer.vq.layers[0].codebook
        quantizer_dim = original_codebook.shape[1]

        # 創建 RVQ
        self.rvq = ResidualVectorQuantizer(
            n_layers=n_rvq_layers,
            codebook_size=rvq_codebook_size,
            dim=quantizer_dim,
            update_mode=rvq_update,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
            ema_dead_code_threshold=ema_dead_code_threshold,
            ema_usage_penalty=ema_usage_penalty,
        ).to(device)

        print(f"\n{'='*60}")
        print(f"RVQ Configuration:")
        print(f"  Layers: {n_rvq_layers}")
        print(f"  Codebook size per layer: {rvq_codebook_size}")
        print(f"  Dimension: {quantizer_dim}")
        print(f"  Total expressiveness: {rvq_codebook_size}^{n_rvq_layers}")
        print(f"  Update mode: {rvq_update}")
        if rvq_update == "ema":
            print(f"  EMA decay: {ema_decay}")
            print(f"  EMA eps: {ema_eps}")
            print(f"  Dead-code threshold: {ema_dead_code_threshold}")
            print(f"  Usage penalty (log cluster_size): {ema_usage_penalty}")
        print(f"{'='*60}\n")

        # 保存配置
        self.n_rvq_layers = n_rvq_layers
        self.rvq_codebook_size = rvq_codebook_size

    def forward(self, clean_audio: torch.Tensor, noisy_audio: torch.Tensor) -> dict:
        """
        Forward pass with RVQ

        覆寫父類的 forward，使用 RVQ 替代原始 quantizer
        """
        # Teacher forward (使用原始 quantizer)
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            teacher_encoder_out, teacher_intermediates = self.teacher_extractor(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward (使用 RVQ)
        student_encoder_out, student_intermediates = self.student_extractor(noisy_audio)

        # 使用 RVQ 而非原始 quantizer
        student_vq = self.rvq(student_encoder_out, frame_rate=75, bandwidth=0.075)
        student_codes = student_vq['codes']

        return {
            'teacher_codes': teacher_codes,
            'student_codes': student_codes,
            'teacher_intermediates': teacher_intermediates,
            'student_intermediates': student_intermediates,
            'teacher_encoder_out': teacher_encoder_out,
            'student_encoder_out': student_encoder_out,
            'student_quantized': student_vq['quantized'],
            'rvq_loss_commit': student_vq['loss_commit'],
            'rvq_loss_codebook': student_vq['loss_codebook'],
            'rvq_commitment_loss': student_vq['commitment_loss'],  # backward-compatible
            'all_layer_codes': student_vq['all_layer_codes'],  # [n_layers, batch, time]
        }

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        使用 teacher decoder 重建音頻

        Args:
            quantized: [batch, dim, time] - RVQ 量化後的向量

        Returns:
            audio: [batch, 1, audio_length] - 重建的音頻
        """
        with torch.no_grad():
            # 使用 teacher 的 top-level decode 方法
            # 這個方法接受 features [batch, dim, time] 並輸出 audio
            # bandwidth_id=0 對應標準帶寬設定
            bandwidth_id = torch.tensor([0], device=quantized.device)
            audio = self.teacher.decode(quantized, bandwidth_id=bandwidth_id)

            # 確保輸出形狀為 [batch, 1, time]
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # [batch, time] -> [batch, 1, time]

        return audio

    def get_rvq_usage(self, all_layer_codes: torch.Tensor) -> Dict[int, Dict]:
        """
        分析 RVQ 每層的使用情況

        Args:
            all_layer_codes: [n_layers, batch, time]

        Returns:
            Dict mapping layer_idx → {
                'usage_count': [codebook_size],
                'n_used': int,
                'entropy': float
            }
        """
        usage_stats = {}

        usage = self.rvq.get_codebook_usage(all_layer_codes)

        for layer_idx, counts in usage.items():
            n_used = (counts > 0).sum().item()

            # 計算 entropy
            probs = counts.float() / counts.sum()
            probs = probs[probs > 0]  # 只計算非零概率
            entropy = -(probs * probs.log()).sum().item()

            usage_stats[layer_idx] = {
                'usage_count': counts,
                'n_used': n_used,
                'entropy': entropy,
            }

        return usage_stats
