"""
exp_1226: Post-VQ Feature Loss

核心概念：
- 目前訓練優化的是 Pre-VQ encoder output (Cosine Sim = 0.495)
- 但解碼使用的是 Post-VQ quantized features (Cosine Sim = 0.9325)
- 直接優化 Post-VQ features 應該能更有效改善音質

新增 Loss:
1. PostVQFeatureLoss: MSE between student Post-VQ and teacher Post-VQ
2. PostVQCosineLoss: Cosine similarity between Post-VQ features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1219.losses import create_length_mask


class PostVQFeatureLoss(nn.Module):
    """
    Post-VQ Feature Loss - 直接優化 VQ 量化後的特徵

    Loss = MSE(VQ(student_encoder_out), VQ(teacher_encoder_out))

    關鍵：使用 straight-through estimator 讓梯度穿過 VQ
    """

    def __init__(self, encoder_stride=320):
        super().__init__()
        self.encoder_stride = encoder_stride

    def forward(self, student_encoder_out, teacher_encoder_out,
                student_quantized, teacher_quantized, lengths):
        """
        Args:
            student_encoder_out: (B, D, T) student encoder output (用於 straight-through)
            teacher_encoder_out: (B, D, T) teacher encoder output
            student_quantized: (B, D, T) student Post-VQ features
            teacher_quantized: (B, D, T) teacher Post-VQ features
            lengths: (B,) 有效 audio samples 數

        Returns:
            loss: scalar
        """
        B, D, T = student_quantized.shape
        max_audio_len = T * self.encoder_stride

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_quantized.device)

        # Straight-through estimator:
        # Forward: 使用 quantized features
        # Backward: 梯度直接傳到 encoder output
        student_st = student_encoder_out + (student_quantized - student_encoder_out).detach()

        # Target: teacher quantized (detached)
        target = teacher_quantized.detach()

        # MSE loss per frame
        diff_sq = (student_st - target) ** 2  # (B, D, T)
        diff_sq = diff_sq.mean(dim=1)  # (B, T)

        # Apply mask
        masked_diff = diff_sq * mask
        loss = masked_diff.sum() / (mask.sum() + 1e-8)

        return loss


class PostVQCosineLoss(nn.Module):
    """
    Post-VQ Cosine Loss - 最大化 Post-VQ features 的 cosine similarity

    Loss = 1 - cosine_similarity(VQ(student), VQ(teacher))
    """

    def __init__(self, encoder_stride=320):
        super().__init__()
        self.encoder_stride = encoder_stride

    def forward(self, student_encoder_out, teacher_encoder_out,
                student_quantized, teacher_quantized, lengths):
        """
        Args:
            Same as PostVQFeatureLoss

        Returns:
            loss: scalar
            stats: dict with cos_sim_mean, cos_sim_std
        """
        B, D, T = student_quantized.shape
        max_audio_len = T * self.encoder_stride

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_quantized.device)

        # Straight-through estimator
        student_st = student_encoder_out + (student_quantized - student_encoder_out).detach()
        target = teacher_quantized.detach()

        # Transpose for per-frame cosine similarity: (B, D, T) -> (B, T, D)
        student_t = student_st.permute(0, 2, 1)
        target_t = target.permute(0, 2, 1)

        # Cosine similarity per frame
        cos_sim = F.cosine_similarity(student_t, target_t, dim=2)  # (B, T)

        # Loss = 1 - cos_sim
        cos_loss = 1 - cos_sim  # (B, T)

        # Apply mask
        masked_loss = cos_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)

        # Stats
        masked_sim = cos_sim * mask
        mean_sim = masked_sim.sum() / (mask.sum() + 1e-8)

        return loss, {'cos_sim_mean': mean_sim.item()}


class MaskedCombinedLossV5(nn.Module):
    """
    組合版 Masked Loss V5 - 支援 Post-VQ Feature Loss

    新增:
    - Post-VQ Feature Loss: 直接優化 VQ 量化後的特徵
    - Post-VQ Cosine Loss: 最大化 Post-VQ cosine similarity
    """

    def __init__(self,
                 feature_weight=1.0,
                 cosine_weight=0.0,
                 triplet_weight=0.5,
                 triplet_margin=0.2,
                 ce_weight=0.0,
                 vq_commitment_weight=0.0,
                 vq_distortion_weight=0.0,
                 vq_temperature=1.0,
                 post_vq_feature_weight=0.0,   # 新增
                 post_vq_cosine_weight=0.0,    # 新增
                 encoder_stride=320):
        super().__init__()

        self.feature_weight = feature_weight
        self.cosine_weight = cosine_weight
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
        self.vq_commitment_weight = vq_commitment_weight
        self.vq_distortion_weight = vq_distortion_weight
        self.post_vq_feature_weight = post_vq_feature_weight
        self.post_vq_cosine_weight = post_vq_cosine_weight
        self.encoder_stride = encoder_stride

        # Import existing losses
        from exp_1219.losses import (
            MaskedFeatureLoss, MaskedCosineLoss, MaskedTripletLoss, MaskedCrossEntropyLoss
        )
        from exp_1226.losses import VQCommitmentLoss, VQDistortionLoss

        self.feature_loss = MaskedFeatureLoss(encoder_stride)
        self.cosine_loss = MaskedCosineLoss(encoder_stride)
        self.triplet_loss = MaskedTripletLoss(triplet_margin, encoder_stride)
        self.ce_loss = MaskedCrossEntropyLoss(encoder_stride)
        self.vq_commitment_loss = VQCommitmentLoss(encoder_stride)
        self.vq_distortion_loss = VQDistortionLoss(encoder_stride, vq_temperature)

        # New Post-VQ losses
        self.post_vq_feature_loss = PostVQFeatureLoss(encoder_stride)
        self.post_vq_cosine_loss = PostVQCosineLoss(encoder_stride)

    def forward(self, student_features, teacher_features, teacher_codes,
                codebook, lengths, logits=None,
                student_quantized=None, teacher_quantized=None):
        """
        Args:
            student_features: (B, D, T) student encoder output (Pre-VQ)
            teacher_features: (B, D, T) teacher encoder output (Pre-VQ)
            teacher_codes: (B, T) teacher VQ codes
            codebook: (num_codes, D) VQ codebook
            lengths: (B,) 有效 audio samples 數
            logits: (B, num_codes, T) optional, for CE loss
            student_quantized: (B, D, T) student Post-VQ features (新增)
            teacher_quantized: (B, D, T) teacher Post-VQ features (新增)

        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        loss_dict = {}
        total_loss = None

        teacher_features_detached = teacher_features.detach()

        # ==================== Pre-VQ Losses ====================

        # Feature Loss (MSE on Pre-VQ)
        if self.feature_weight > 0:
            feat_loss = self.feature_loss(student_features, teacher_features_detached, lengths)
            loss_dict['feature_loss'] = feat_loss.item()
            total_loss = self._add_loss(total_loss, self.feature_weight * feat_loss)

        # Cosine Loss (on Pre-VQ)
        if self.cosine_weight > 0:
            cos_loss = self.cosine_loss(student_features, teacher_features_detached, lengths)
            loss_dict['cosine_loss'] = cos_loss.item()
            cos_stats = self.cosine_loss.compute_stats(student_features, teacher_features_detached, lengths)
            loss_dict['pre_vq_cos_sim'] = cos_stats['cos_sim_mean']
            total_loss = self._add_loss(total_loss, self.cosine_weight * cos_loss)

        # Triplet Loss
        if self.triplet_weight > 0:
            trip_loss = self.triplet_loss(student_features, teacher_codes, codebook, lengths)
            loss_dict['triplet_loss'] = trip_loss.item()
            total_loss = self._add_loss(total_loss, self.triplet_weight * trip_loss)

        # CE Loss
        if self.ce_weight > 0 and logits is not None:
            t_codes = teacher_codes[0] if teacher_codes.dim() == 3 else teacher_codes
            ce_loss = self.ce_loss(logits, t_codes.long(), lengths)
            loss_dict['ce_loss'] = ce_loss.item()
            total_loss = self._add_loss(total_loss, self.ce_weight * ce_loss)

        # VQ Commitment Loss
        if self.vq_commitment_weight > 0:
            vq_commit = self.vq_commitment_loss(student_features, teacher_codes, codebook, lengths)
            loss_dict['vq_commitment_loss'] = vq_commit.item()
            total_loss = self._add_loss(total_loss, self.vq_commitment_weight * vq_commit)

        # VQ Distortion Loss
        if self.vq_distortion_weight > 0:
            vq_distort = self.vq_distortion_loss(student_features, teacher_codes, codebook, lengths)
            loss_dict['vq_distortion_loss'] = vq_distort.item()
            total_loss = self._add_loss(total_loss, self.vq_distortion_weight * vq_distort)

        # ==================== Post-VQ Losses (新增) ====================

        if student_quantized is not None and teacher_quantized is not None:
            # Post-VQ Feature Loss
            if self.post_vq_feature_weight > 0:
                post_vq_feat = self.post_vq_feature_loss(
                    student_features, teacher_features,
                    student_quantized, teacher_quantized, lengths
                )
                loss_dict['post_vq_feature_loss'] = post_vq_feat.item()
                total_loss = self._add_loss(total_loss, self.post_vq_feature_weight * post_vq_feat)

            # Post-VQ Cosine Loss
            if self.post_vq_cosine_weight > 0:
                post_vq_cos, cos_stats = self.post_vq_cosine_loss(
                    student_features, teacher_features,
                    student_quantized, teacher_quantized, lengths
                )
                loss_dict['post_vq_cosine_loss'] = post_vq_cos.item()
                loss_dict['post_vq_cos_sim'] = cos_stats['cos_sim_mean']
                total_loss = self._add_loss(total_loss, self.post_vq_cosine_weight * post_vq_cos)

        if total_loss is None:
            raise ValueError("No loss computed! At least one loss weight must be > 0")

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def _add_loss(self, total, new_loss):
        if total is None:
            return new_loss
        return total + new_loss


# ==================== 測試 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Post-VQ Feature Loss")
    print("=" * 60)

    B, D, T = 4, 512, 100
    num_codes = 4096
    encoder_stride = 320

    lengths = torch.tensor([32000, 24000, 16000, 8000])

    student_encoder_out = torch.randn(B, D, T, requires_grad=True)
    teacher_encoder_out = torch.randn(B, D, T)
    student_quantized = torch.randn(B, D, T)
    teacher_quantized = torch.randn(B, D, T)
    teacher_codes = torch.randint(0, num_codes, (B, T))
    codebook = torch.randn(num_codes, D)

    print(f"\nInput shapes:")
    print(f"  student_encoder_out: {student_encoder_out.shape}")
    print(f"  student_quantized: {student_quantized.shape}")

    # Test PostVQFeatureLoss
    print("\n--- Testing PostVQFeatureLoss ---")
    post_vq_feat_fn = PostVQFeatureLoss(encoder_stride)
    post_vq_feat = post_vq_feat_fn(
        student_encoder_out, teacher_encoder_out,
        student_quantized, teacher_quantized, lengths
    )
    print(f"Post-VQ Feature Loss: {post_vq_feat.item():.4f}")

    # Test gradient flow
    post_vq_feat.backward()
    print(f"Gradient flows to encoder: {student_encoder_out.grad is not None}")
    print(f"Gradient norm: {student_encoder_out.grad.norm().item():.4f}")

    # Reset grad
    student_encoder_out.grad = None

    # Test PostVQCosineLoss
    print("\n--- Testing PostVQCosineLoss ---")
    post_vq_cos_fn = PostVQCosineLoss(encoder_stride)
    post_vq_cos, stats = post_vq_cos_fn(
        student_encoder_out, teacher_encoder_out,
        student_quantized, teacher_quantized, lengths
    )
    print(f"Post-VQ Cosine Loss: {post_vq_cos.item():.4f}")
    print(f"Post-VQ Cos Sim Mean: {stats['cos_sim_mean']:.4f}")

    # Test Combined Loss V5
    print("\n--- Testing Combined Loss V5 ---")
    combined_loss_fn = MaskedCombinedLossV5(
        feature_weight=1.0,
        triplet_weight=1.0,
        post_vq_feature_weight=0.5,
        post_vq_cosine_weight=0.5,
        encoder_stride=encoder_stride
    )

    total_loss, loss_dict = combined_loss_fn(
        student_encoder_out, teacher_encoder_out, teacher_codes, codebook, lengths,
        student_quantized=student_quantized, teacher_quantized=teacher_quantized
    )

    print(f"Combined Loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
