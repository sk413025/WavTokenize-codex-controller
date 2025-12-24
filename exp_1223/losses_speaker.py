"""
exp_1223: Speaker-Weighted Loss Functions

在 Loss 層面加入 speaker information，而非改變 features。

核心概念:
- 計算當前樣本的 speaker embedding 與 training speakers 的相似度
- 相似度低 → 較難的樣本 → 給較低 weight（避免過度 penalize unseen speakers）
- 相似度高 → 較容易的樣本 → 正常 weight

這樣可以讓模型:
1. 對熟悉的 speakers 嚴格學習
2. 對陌生的 speakers 給予更多容忍度
3. 逐漸泛化到 unseen speakers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1219.losses import (
    MaskedFeatureLoss, MaskedCosineLoss, MaskedTripletLoss,
    MaskedCrossEntropyLoss, create_length_mask
)


class SpeakerWeightCalculator(nn.Module):
    """
    計算 speaker-based loss weight

    策略: 根據當前 speaker 與已知 speakers 的相似度計算 weight
    - 相似度高 → weight 接近 1.0
    - 相似度低 → weight 接近 min_weight (e.g., 0.5)
    """

    def __init__(self, min_weight: float = 0.5, temperature: float = 1.0):
        """
        Args:
            min_weight: 最低 weight (用於完全 unseen 的 speaker)
            temperature: 控制 weight 的敏感度
        """
        super().__init__()
        self.min_weight = min_weight
        self.temperature = temperature

        # 儲存 training speakers 的 centroid (running average)
        self.register_buffer('speaker_centroid', None)
        self.register_buffer('num_speakers_seen', torch.tensor(0))

    def update_centroid(self, speaker_embeddings: torch.Tensor):
        """
        更新 speaker centroid (用於 training)

        Args:
            speaker_embeddings: (B, D) batch of speaker embeddings
        """
        # L2 normalize
        speaker_embeddings = F.normalize(speaker_embeddings, p=2, dim=-1)
        batch_mean = speaker_embeddings.mean(dim=0)  # (D,)

        if self.speaker_centroid is None:
            self.speaker_centroid = batch_mean
            self.num_speakers_seen = torch.tensor(1, device=batch_mean.device)
        else:
            # Running average
            n = self.num_speakers_seen.float()
            self.speaker_centroid = (n * self.speaker_centroid + batch_mean) / (n + 1)
            self.num_speakers_seen = self.num_speakers_seen + 1

    def compute_weight(self, speaker_embeddings: torch.Tensor) -> torch.Tensor:
        """
        計算每個樣本的 loss weight

        Args:
            speaker_embeddings: (B, D) batch of speaker embeddings

        Returns:
            weights: (B,) weight for each sample, range [min_weight, 1.0]
        """
        if self.speaker_centroid is None:
            # 還沒有 centroid，返回全 1
            return torch.ones(speaker_embeddings.shape[0], device=speaker_embeddings.device)

        # L2 normalize
        speaker_embeddings = F.normalize(speaker_embeddings, p=2, dim=-1)
        centroid = F.normalize(self.speaker_centroid.unsqueeze(0), p=2, dim=-1)

        # Cosine similarity with centroid
        similarity = F.cosine_similarity(speaker_embeddings, centroid, dim=-1)  # (B,)

        # Map similarity [-1, 1] to weight [min_weight, 1.0]
        # similarity = 1 → weight = 1.0
        # similarity = -1 → weight = min_weight
        # similarity = 0 → weight = (1 + min_weight) / 2

        # Apply temperature: higher temp → more uniform weights
        similarity_scaled = similarity / self.temperature

        # Sigmoid-like mapping to [min_weight, 1.0]
        weight = self.min_weight + (1.0 - self.min_weight) * (similarity_scaled + 1) / 2

        # Clamp to valid range
        weight = torch.clamp(weight, self.min_weight, 1.0)

        return weight


class SpeakerWeightedCombinedLoss(nn.Module):
    """
    Speaker-Weighted Combined Loss

    基於 MaskedCombinedLossV2，加入 speaker-based weighting
    """

    def __init__(self,
                 feature_weight=1.0,
                 cosine_weight=0.0,
                 triplet_weight=1.0,
                 triplet_margin=0.2,
                 ce_weight=0.0,
                 encoder_stride=320,
                 # Speaker weighting params
                 speaker_min_weight=0.5,
                 speaker_temperature=1.0,
                 use_speaker_weighting=True):
        super().__init__()

        self.feature_weight = feature_weight
        self.cosine_weight = cosine_weight
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
        self.encoder_stride = encoder_stride
        self.use_speaker_weighting = use_speaker_weighting

        # Base losses
        self.feature_loss = MaskedFeatureLoss(encoder_stride)
        self.cosine_loss = MaskedCosineLoss(encoder_stride)
        self.triplet_loss = MaskedTripletLoss(triplet_margin, encoder_stride)
        self.ce_loss = MaskedCrossEntropyLoss(encoder_stride)

        # Speaker weight calculator
        if use_speaker_weighting:
            self.speaker_weight_calc = SpeakerWeightCalculator(
                min_weight=speaker_min_weight,
                temperature=speaker_temperature
            )
        else:
            self.speaker_weight_calc = None

    def forward(self, student_features, teacher_features, teacher_codes,
                codebook, lengths, logits=None, speaker_embeddings=None,
                update_speaker_centroid=True):
        """
        Args:
            student_features: (B, D, T)
            teacher_features: (B, D, T)
            teacher_codes: (B, T) or (1, B, T)
            codebook: (num_codes, D)
            lengths: (B,) valid audio samples
            logits: (B, num_codes, T) optional for CE loss
            speaker_embeddings: (B, speaker_dim) optional for speaker weighting
            update_speaker_centroid: whether to update centroid during training

        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses and weights
        """
        B = student_features.shape[0]
        device = student_features.device

        loss_dict = {}

        # Calculate speaker weights
        if self.use_speaker_weighting and speaker_embeddings is not None:
            if self.training and update_speaker_centroid:
                self.speaker_weight_calc.update_centroid(speaker_embeddings)

            speaker_weights = self.speaker_weight_calc.compute_weight(speaker_embeddings)  # (B,)
            loss_dict['speaker_weight_mean'] = speaker_weights.mean().item()
            loss_dict['speaker_weight_std'] = speaker_weights.std().item()
        else:
            speaker_weights = torch.ones(B, device=device)

        teacher_features_detached = teacher_features.detach()

        # Compute per-sample losses then apply speaker weights
        total_loss = torch.tensor(0.0, device=device)

        # Feature Loss (MSE) - per sample
        if self.feature_weight > 0:
            feat_loss_per_sample = self._compute_feature_loss_per_sample(
                student_features, teacher_features_detached, lengths
            )  # (B,)
            weighted_feat_loss = (feat_loss_per_sample * speaker_weights).mean()
            loss_dict['feature_loss'] = feat_loss_per_sample.mean().item()
            loss_dict['feature_loss_weighted'] = weighted_feat_loss.item()
            total_loss = total_loss + self.feature_weight * weighted_feat_loss

        # Cosine Loss - per sample
        if self.cosine_weight > 0:
            cos_loss_per_sample = self._compute_cosine_loss_per_sample(
                student_features, teacher_features_detached, lengths
            )  # (B,)
            weighted_cos_loss = (cos_loss_per_sample * speaker_weights).mean()
            loss_dict['cosine_loss'] = cos_loss_per_sample.mean().item()
            loss_dict['cosine_loss_weighted'] = weighted_cos_loss.item()
            total_loss = total_loss + self.cosine_weight * weighted_cos_loss

            # Compute cos_sim stats
            cos_stats = self.cosine_loss.compute_stats(
                student_features, teacher_features_detached, lengths
            )
            loss_dict['cos_sim_mean'] = cos_stats['cos_sim_mean']

        # Triplet Loss - per sample
        if self.triplet_weight > 0:
            trip_loss_per_sample = self._compute_triplet_loss_per_sample(
                student_features, teacher_codes, codebook, lengths
            )  # (B,)
            weighted_trip_loss = (trip_loss_per_sample * speaker_weights).mean()
            loss_dict['triplet_loss'] = trip_loss_per_sample.mean().item()
            loss_dict['triplet_loss_weighted'] = weighted_trip_loss.item()
            total_loss = total_loss + self.triplet_weight * weighted_trip_loss

        # CE Loss - per sample
        if self.ce_weight > 0 and logits is not None:
            t_codes = teacher_codes[0] if teacher_codes.dim() == 3 else teacher_codes
            ce_loss_per_sample = self._compute_ce_loss_per_sample(
                logits, t_codes.long(), lengths
            )  # (B,)
            weighted_ce_loss = (ce_loss_per_sample * speaker_weights).mean()
            loss_dict['ce_loss'] = ce_loss_per_sample.mean().item()
            loss_dict['ce_loss_weighted'] = weighted_ce_loss.item()
            total_loss = total_loss + self.ce_weight * weighted_ce_loss

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def _compute_feature_loss_per_sample(self, student_features, teacher_features, lengths):
        """Compute Feature Loss per sample (B,)"""
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                  device=student_features.device)  # (B, T)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, T)

        diff_sq = (student_features - teacher_features) ** 2  # (B, D, T)
        masked_diff = diff_sq * mask_expanded  # (B, D, T)

        # Sum over D and T, divide by valid count per sample
        loss_per_sample = masked_diff.sum(dim=(1, 2)) / (mask.sum(dim=1) * D + 1e-8)  # (B,)

        return loss_per_sample

    def _compute_cosine_loss_per_sample(self, student_features, teacher_features, lengths):
        """Compute Cosine Loss per sample (B,)"""
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                  device=student_features.device)  # (B, T)

        # Reshape: (B, D, T) -> (B, T, D)
        stu = student_features.permute(0, 2, 1)
        tea = teacher_features.permute(0, 2, 1)

        # Cosine similarity per frame: (B, T)
        cos_sim = F.cosine_similarity(stu, tea, dim=2)

        # Loss = 1 - cosine
        loss_per_frame = 1 - cos_sim  # (B, T)

        # Apply mask and compute per-sample mean
        masked_loss = loss_per_frame * mask
        loss_per_sample = masked_loss.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B,)

        return loss_per_sample

    def _compute_triplet_loss_per_sample(self, student_features, teacher_codes, codebook, lengths):
        """Compute Triplet Loss per sample (B,)"""
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                  device=student_features.device)  # (B, T)

        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]

        # Process each sample separately for clarity
        loss_per_sample = []
        for b in range(B):
            z = student_features[b].permute(1, 0)  # (T, D)
            codes = teacher_codes[b]  # (T,)
            sample_mask = mask[b]  # (T,)

            dists = torch.cdist(z, codebook, p=2)  # (T, num_codes)

            frame_indices = torch.arange(T, device=dists.device)
            pos_dist = dists[frame_indices, codes]  # (T,)

            dists_for_neg = dists.clone()
            dists_for_neg[frame_indices, codes] = float('inf')
            neg_dist = dists_for_neg.min(dim=1).values  # (T,)

            triplet = F.relu(pos_dist - neg_dist + self.triplet_loss.margin)  # (T,)

            masked_triplet = triplet * sample_mask
            sample_loss = masked_triplet.sum() / (sample_mask.sum() + 1e-8)
            loss_per_sample.append(sample_loss)

        return torch.stack(loss_per_sample)  # (B,)

    def _compute_ce_loss_per_sample(self, logits, targets, lengths):
        """Compute CE Loss per sample (B,)"""
        B, C, T = logits.shape
        max_audio_len = T * self.encoder_stride

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                  device=logits.device)  # (B, T)

        # (B, C, T) -> (B, T, C)
        logits_t = logits.permute(0, 2, 1)

        loss_per_sample = []
        for b in range(B):
            ce = F.cross_entropy(logits_t[b], targets[b], reduction='none')  # (T,)
            masked_ce = ce * mask[b]
            sample_loss = masked_ce.sum() / (mask[b].sum() + 1e-8)
            loss_per_sample.append(sample_loss)

        return torch.stack(loss_per_sample)  # (B,)


# ==================== 測試 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Speaker-Weighted Loss")
    print("=" * 60)

    B, D, T = 4, 512, 100
    num_codes = 1024
    speaker_dim = 256
    encoder_stride = 320

    lengths = torch.tensor([32000, 24000, 16000, 8000])

    student_features = torch.randn(B, D, T)
    teacher_features = torch.randn(B, D, T)
    teacher_codes = torch.randint(0, num_codes, (B, T))
    codebook = torch.randn(num_codes, D)
    speaker_embeddings = torch.randn(B, speaker_dim)

    print(f"\nInput shapes:")
    print(f"  student_features: {student_features.shape}")
    print(f"  speaker_embeddings: {speaker_embeddings.shape}")

    # Test Speaker Weight Calculator
    print("\n--- Testing Speaker Weight Calculator ---")
    weight_calc = SpeakerWeightCalculator(min_weight=0.5, temperature=1.0)

    # First batch (no centroid yet)
    weights1 = weight_calc.compute_weight(speaker_embeddings)
    print(f"Weights (no centroid): {weights1}")

    # Update centroid
    weight_calc.update_centroid(speaker_embeddings)

    # Second batch (with centroid)
    weights2 = weight_calc.compute_weight(speaker_embeddings)
    print(f"Weights (with centroid): {weights2}")

    # Different speakers
    different_speakers = torch.randn(B, speaker_dim) * 3  # More different
    weights3 = weight_calc.compute_weight(different_speakers)
    print(f"Weights (different speakers): {weights3}")

    # Test Combined Loss
    print("\n--- Testing Speaker-Weighted Combined Loss ---")
    loss_fn = SpeakerWeightedCombinedLoss(
        feature_weight=1.0,
        triplet_weight=1.0,
        speaker_min_weight=0.5,
        speaker_temperature=1.0,
        use_speaker_weighting=True
    )
    loss_fn.train()

    total_loss, loss_dict = loss_fn(
        student_features, teacher_features, teacher_codes, codebook, lengths,
        speaker_embeddings=speaker_embeddings
    )

    print(f"Total Loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
