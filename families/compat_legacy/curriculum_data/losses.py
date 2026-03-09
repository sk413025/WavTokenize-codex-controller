"""
exp_1226: 進階 Loss Functions

包含:
1. VQ-Aware Loss - 直接優化 VQ quantization compatibility
2. 繼承 exp_1219 的所有 loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 從 exp_1219 繼承基礎 loss functions
from exp_1219.losses import (
    create_length_mask,
    MaskedFeatureLoss,
    MaskedCosineLoss,
    MaskedTripletLoss,
    MaskedCrossEntropyLoss,
    compute_masked_accuracy,
)


class VQCommitmentLoss(nn.Module):
    """
    VQ Commitment Loss - 讓 student features 更接近 VQ centroids

    概念：
    - 原始 VQ-VAE 中，commitment loss = ||z - sg(e)||^2
    - 這裡我們讓 student features 接近 teacher 選擇的 codebook entry
    - 這樣可以減少 quantization error，改善 VQ decode 後的音質

    Loss = ||student_feature - codebook[teacher_code]||^2
    """

    def __init__(self, encoder_stride=320):
        super().__init__()
        self.encoder_stride = encoder_stride

    def forward(self, student_features, teacher_codes, codebook, lengths):
        """
        Args:
            student_features: (B, D, T) student encoder output
            teacher_codes: (B, T) or (1, B, T) teacher VQ codes
            codebook: (num_codes, D) VQ codebook
            lengths: (B,) 有效 audio samples 數

        Returns:
            loss: scalar, commitment loss
        """
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride

        # 處理 teacher_codes 維度
        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]  # (1, B, T) -> (B, T)

        # 創建 mask
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)

        # 獲取 teacher 選擇的 codebook entries
        # teacher_codes: (B, T) -> (B*T,)
        codes_flat = teacher_codes.reshape(-1).long()
        target_embeddings = codebook[codes_flat]  # (B*T, D)
        target_embeddings = target_embeddings.reshape(B, T, D)  # (B, T, D)

        # student_features: (B, D, T) -> (B, T, D)
        student_features_t = student_features.permute(0, 2, 1)

        # 計算 commitment loss: ||z - e||^2
        diff_sq = (student_features_t - target_embeddings.detach()) ** 2  # (B, T, D)
        diff_sq = diff_sq.mean(dim=2)  # (B, T) - 對 D 維度取平均

        # Apply mask
        masked_diff = diff_sq * mask
        loss = masked_diff.sum() / (mask.sum() + 1e-8)

        return loss


class VQDistortionLoss(nn.Module):
    """
    VQ Distortion Loss - 讓 student features quantize 後更接近 teacher features

    概念：
    - 直接最小化 VQ quantization 後的誤差
    - Loss = ||VQ(student_feature) - VQ(teacher_feature)||^2
    - 注意：這個 loss 需要 straight-through estimator 來傳遞梯度

    實作：
    - 使用 teacher codes 作為目標，避免重新 quantize
    - student 做 soft assignment 來保持可微分
    """

    def __init__(self, encoder_stride=320, temperature=1.0):
        super().__init__()
        self.encoder_stride = encoder_stride
        self.temperature = temperature

    def forward(self, student_features, teacher_codes, codebook, lengths):
        """
        計算 soft VQ distortion loss

        使用 soft assignment 而非 hard assignment 以保持可微分：
        soft_code = softmax(-dist(z, codebook) / temp) @ codebook
        """
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride

        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)

        # student_features: (B, D, T) -> (B*T, D)
        z = student_features.permute(0, 2, 1).reshape(-1, D)

        # 計算到所有 codebook entries 的距離
        # dists: (B*T, num_codes)
        dists = torch.cdist(z, codebook, p=2)

        # Soft assignment: (B*T, num_codes)
        soft_weights = F.softmax(-dists / self.temperature, dim=1)

        # Soft reconstructed features: (B*T, D)
        soft_recon = soft_weights @ codebook

        # Teacher target embeddings
        codes_flat = teacher_codes.reshape(-1).long()
        target_embeddings = codebook[codes_flat]  # (B*T, D)

        # Loss: ||soft_recon - target||^2
        diff_sq = (soft_recon - target_embeddings.detach()) ** 2
        diff_sq = diff_sq.mean(dim=1)  # (B*T,)

        # Reshape and apply mask
        diff_sq = diff_sq.reshape(B, T)
        masked_diff = diff_sq * mask
        loss = masked_diff.sum() / (mask.sum() + 1e-8)

        return loss


class MaskedCombinedLossV3(nn.Module):
    """
    組合版 Masked Loss V3 - 支援 VQ-Aware Losses

    新增:
    - VQ Commitment Loss: 讓 features 接近 codebook centroids
    - VQ Distortion Loss: 最小化 soft-VQ 後的誤差
    """

    def __init__(self,
                 feature_weight=1.0,
                 cosine_weight=0.0,
                 triplet_weight=0.5,
                 triplet_margin=0.2,
                 ce_weight=0.0,
                 vq_commitment_weight=0.0,   # 新增
                 vq_distortion_weight=0.0,   # 新增
                 vq_temperature=1.0,         # 新增
                 encoder_stride=320):
        super().__init__()

        self.feature_weight = feature_weight
        self.cosine_weight = cosine_weight
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
        self.vq_commitment_weight = vq_commitment_weight
        self.vq_distortion_weight = vq_distortion_weight
        self.encoder_stride = encoder_stride

        self.feature_loss = MaskedFeatureLoss(encoder_stride)
        self.cosine_loss = MaskedCosineLoss(encoder_stride)
        self.triplet_loss = MaskedTripletLoss(triplet_margin, encoder_stride)
        self.ce_loss = MaskedCrossEntropyLoss(encoder_stride)
        self.vq_commitment_loss = VQCommitmentLoss(encoder_stride)
        self.vq_distortion_loss = VQDistortionLoss(encoder_stride, vq_temperature)

    def forward(self, student_features, teacher_features, teacher_codes,
                codebook, lengths, logits=None):
        """
        Args:
            student_features: (B, D, T) student encoder output
            teacher_features: (B, D, T) teacher encoder output
            teacher_codes: (B, T) teacher VQ codes
            codebook: (num_codes, D) VQ codebook
            lengths: (B,) 有效 audio samples 數
            logits: (B, num_codes, T) optional, for CE loss

        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        loss_dict = {}
        total_loss = None

        teacher_features_detached = teacher_features.detach()

        # Feature Loss (MSE)
        if self.feature_weight > 0:
            feat_loss = self.feature_loss(student_features, teacher_features_detached, lengths)
            loss_dict['feature_loss'] = feat_loss.item()
            total_loss = self._add_loss(total_loss, self.feature_weight * feat_loss)

        # Cosine Loss
        if self.cosine_weight > 0:
            cos_loss = self.cosine_loss(student_features, teacher_features_detached, lengths)
            loss_dict['cosine_loss'] = cos_loss.item()
            cos_stats = self.cosine_loss.compute_stats(student_features, teacher_features_detached, lengths)
            loss_dict['cos_sim_mean'] = cos_stats['cos_sim_mean']
            loss_dict['cos_sim_std'] = cos_stats['cos_sim_std']
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

        # VQ Commitment Loss (新增)
        if self.vq_commitment_weight > 0:
            vq_commit = self.vq_commitment_loss(student_features, teacher_codes, codebook, lengths)
            loss_dict['vq_commitment_loss'] = vq_commit.item()
            total_loss = self._add_loss(total_loss, self.vq_commitment_weight * vq_commit)

        # VQ Distortion Loss (新增)
        if self.vq_distortion_weight > 0:
            vq_distort = self.vq_distortion_loss(student_features, teacher_codes, codebook, lengths)
            loss_dict['vq_distortion_loss'] = vq_distort.item()
            total_loss = self._add_loss(total_loss, self.vq_distortion_weight * vq_distort)

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
    print("Testing VQ-Aware Loss Functions")
    print("=" * 60)

    B, D, T = 4, 512, 100
    num_codes = 4096
    encoder_stride = 320

    lengths = torch.tensor([32000, 24000, 16000, 8000])

    student_features = torch.randn(B, D, T)
    teacher_features = torch.randn(B, D, T)
    teacher_codes = torch.randint(0, num_codes, (B, T))
    codebook = torch.randn(num_codes, D)

    print(f"\nInput shapes:")
    print(f"  student_features: {student_features.shape}")
    print(f"  teacher_codes: {teacher_codes.shape}")
    print(f"  codebook: {codebook.shape}")

    # 測試 VQ Commitment Loss
    print("\n--- Testing VQ Commitment Loss ---")
    vq_commit_fn = VQCommitmentLoss(encoder_stride)
    vq_commit = vq_commit_fn(student_features, teacher_codes, codebook, lengths)
    print(f"VQ Commitment Loss: {vq_commit.item():.4f}")

    # 測試 VQ Distortion Loss
    print("\n--- Testing VQ Distortion Loss ---")
    vq_distort_fn = VQDistortionLoss(encoder_stride, temperature=1.0)
    vq_distort = vq_distort_fn(student_features, teacher_codes, codebook, lengths)
    print(f"VQ Distortion Loss: {vq_distort.item():.4f}")

    # 測試 Combined Loss V3
    print("\n--- Testing Combined Loss V3 ---")
    combined_loss_fn = MaskedCombinedLossV3(
        feature_weight=1.0,
        triplet_weight=1.0,
        vq_commitment_weight=0.1,
        vq_distortion_weight=0.1,
        encoder_stride=encoder_stride
    )

    total_loss, loss_dict = combined_loss_fn(
        student_features, teacher_features, teacher_codes, codebook, lengths
    )

    print(f"Combined Loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
