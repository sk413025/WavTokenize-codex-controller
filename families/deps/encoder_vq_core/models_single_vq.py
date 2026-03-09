"""
Minimal dependency surface for the single-VQ + EMA teacher-student model.

This extracts the live model path out of the historical
`families.compat_legacy.plan_ori_vq.plan_ori` shell while preserving the
existing runtime behavior.
"""

import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from families.deps.wavtokenizer_core.wavtok_lora_patch import apply_lora_patch
from families.compat_legacy.intermediate_stack.models import (
    TeacherStudentIntermediate,
    IntermediateExtractor,
    ALL_18_LAYERS,
    INTERMEDIATE_SUPERVISION_POINTS,
)

apply_lora_patch()


class SingleVQWithEMA(nn.Module):
    """Single-layer vector quantizer with EMA codebook updates."""

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
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.ema_dead_code_threshold = int(ema_dead_code_threshold)
        self.ema_usage_penalty = float(ema_usage_penalty)

        self.codebook = nn.Embedding(codebook_size, dim)

        if pretrained_codebook is not None:
            assert pretrained_codebook.shape == (codebook_size, dim), (
                f"expected shape ({codebook_size}, {dim}), got {pretrained_codebook.shape}"
            )
            self.codebook.weight.data = pretrained_codebook.clone()
        else:
            self.codebook.weight.data.uniform_(
                -1.0 / codebook_size, 1.0 / codebook_size
            )

        self.codebook.weight.requires_grad_(False)

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

        if pretrained_codebook is not None:
            self.ema_cluster_size.fill_(1.0)
            self.ema_embed_avg.copy_(pretrained_codebook.clone())

    @torch.no_grad()
    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        z_flat = z_flat.float()
        device = z_flat.device
        codebook_size = self.codebook_size

        counts = torch.bincount(indices, minlength=codebook_size).float()

        embed_sum = torch.zeros(codebook_size, self.dim, device=device)
        embed_sum.index_add_(0, indices, z_flat)

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            counts, alpha=(1.0 - self.ema_decay)
        )
        self.ema_embed_avg.mul_(self.ema_decay).add_(
            embed_sum, alpha=(1.0 - self.ema_decay)
        )

        total = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.ema_eps)
            / (total + codebook_size * self.ema_eps)
            * total
        )
        embed = self.ema_embed_avg / cluster_size.unsqueeze(1).clamp(min=1e-12)
        self.codebook.weight.data.copy_(embed)

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
        batch_size, dim, steps = z.shape

        z = z.transpose(1, 2)
        z_flat = z.reshape(-1, dim)

        x2 = (z_flat ** 2).sum(dim=1, keepdim=True)
        y2 = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)
        distances = x2 + y2 - 2.0 * (z_flat @ self.codebook.weight.t())

        if self.ema_usage_penalty > 0.0:
            usage_penalty = torch.log(self.ema_cluster_size.clamp(min=1.0))
            usage_penalty = usage_penalty * self.ema_usage_penalty
            distances = distances + usage_penalty.to(distances.dtype).unsqueeze(0)

        indices = torch.argmin(distances, dim=1)

        q = self.codebook(indices)
        q = q.reshape(batch_size, steps, dim)

        loss_commit = F.mse_loss(z, q.detach())

        if self.training:
            self._ema_update(z_flat.detach(), indices)

        q = z + (q - z).detach()

        z_q = q.transpose(1, 2)
        codes = indices.reshape(batch_size, steps).unsqueeze(0).unsqueeze(2)

        return {
            "quantized": z_q,
            "codes": codes,
            "loss_commit": loss_commit,
            "loss_codebook": torch.tensor(0.0, device=z.device),
            "commitment_loss": loss_commit,
            "bandwidth": torch.tensor([bandwidth], device=z.device),
        }

    def get_codebook_usage(self, codes: torch.Tensor) -> Dict:
        if codes.dim() == 4:
            codes = codes[0, :, 0, :]

        codes_flat = codes.flatten()
        usage_count = torch.bincount(codes_flat, minlength=self.codebook_size)

        n_used = (usage_count > 0).sum().item()
        probs = usage_count.float() / usage_count.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log2()).sum().item()

        return {
            "usage_count": usage_count,
            "n_used": n_used,
            "entropy": entropy,
        }


class TeacherStudentSingleVQ(TeacherStudentIntermediate):
    """Teacher-student model using the extracted single-VQ + EMA quantizer."""

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        intermediate_indices: List[int] = [3, 4, 6],
        device: str = "cuda",
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
        )

        original_quantizer = self.student.feature_extractor.encodec.quantizer
        pretrained_codebook = original_quantizer.vq.layers[0].codebook.detach().clone()
        codebook_size, quantizer_dim = pretrained_codebook.shape

        self.vq = SingleVQWithEMA(
            codebook_size=codebook_size,
            dim=quantizer_dim,
            pretrained_codebook=pretrained_codebook,
            ema_decay=vq_ema_decay,
            ema_eps=vq_ema_eps,
            ema_dead_code_threshold=vq_ema_threshold,
            ema_usage_penalty=vq_ema_usage_penalty,
        ).to(device)

        self.vq_codebook_size = codebook_size

        print(f"\n{'='*60}")
        print("SingleVQ + EMA Configuration:")
        print(f"  Codebook size: {codebook_size}")
        print(f"  Dimension: {quantizer_dim}")
        print("  Initialization: Pretrained (warm start)")
        print(f"  EMA decay: {vq_ema_decay}")
        print(f"  EMA eps: {vq_ema_eps}")
        print(f"  Dead-code threshold: {vq_ema_threshold}")
        print(f"  Usage penalty: {vq_ema_usage_penalty}")
        print(f"{'='*60}\n")

    def forward(self, clean_audio: torch.Tensor, noisy_audio: torch.Tensor) -> dict:
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            teacher_encoder_out, teacher_intermediates = self.teacher_extractor(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        student_encoder_out, student_intermediates = self.student_extractor(noisy_audio)
        student_vq = self.vq(student_encoder_out, frame_rate=75, bandwidth=0.075)
        student_codes = student_vq["codes"]

        return {
            "teacher_codes": teacher_codes,
            "student_codes": student_codes,
            "teacher_intermediates": teacher_intermediates,
            "student_intermediates": student_intermediates,
            "teacher_encoder_out": teacher_encoder_out,
            "student_encoder_out": student_encoder_out,
            "student_quantized": student_vq["quantized"],
            "vq_loss_commit": student_vq["loss_commit"],
            "vq_loss_codebook": student_vq["loss_codebook"],
            "vq_commitment_loss": student_vq["commitment_loss"],
        }

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bandwidth_id = torch.tensor([0], device=quantized.device)
            audio = self.teacher.decode(quantized, bandwidth_id=bandwidth_id)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
        return audio

    def get_vq_usage(self, codes: torch.Tensor) -> Dict:
        return self.vq.get_codebook_usage(codes)
