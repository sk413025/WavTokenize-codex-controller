"""
exp_1226: Anti-Collapse / Diversity Loss

解決問題：Student encoder mode collapse
- Student 不論輸入什麼，都輸出相似的 features
- VQ 量化後集中在少數 codes (1760, 1834, 1623...)

解決方案：
1. Code Entropy Loss: 鼓勵 code distribution 更均勻
2. Feature Diversity Loss: 懲罰 batch 內 features 太相似
3. Contrastive Loss: 不同輸入應該產生不同輸出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1219.losses import (
    MaskedFeatureLoss,
    MaskedTripletLoss,
    MaskedCosineLoss,
    MaskedCrossEntropyLoss,
    create_length_mask,
    compute_masked_accuracy,
)
from families.compat_legacy.curriculum_data.losses_tolerant import FrameTolerantFeatureLoss, FrameTolerantTripletLoss


class CodeEntropyLoss(nn.Module):
    """
    Code Entropy Loss: 鼓勵 code distribution 更均勻

    原理：
    - 計算 batch 內 student codes 的 entropy
    - 高 entropy = 更多樣的 codes = 好
    - Loss = -entropy (最小化 = 最大化 entropy)

    目標：防止所有輸出都集中在少數 codes
    """

    def __init__(self, num_codes: int = 4096, temperature: float = 1.0):
        super().__init__()
        self.num_codes = num_codes
        self.temperature = temperature

    def forward(self, student_features: torch.Tensor,
                codebook: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                encoder_stride: int = 320) -> torch.Tensor:
        """
        Args:
            student_features: (B, D, T) encoder 輸出
            codebook: (num_codes, D)
            lengths: (B,) 有效長度
        """
        B, D, T = student_features.shape

        # 建立 mask
        if lengths is not None:
            max_audio_len = T * encoder_stride
            mask = create_length_mask(lengths, max_audio_len, encoder_stride,
                                       device=student_features.device)  # (B, T)
        else:
            mask = torch.ones(B, T, device=student_features.device)

        # 計算到 codebook 的距離 (soft assignment)
        z = student_features.permute(0, 2, 1)  # (B, T, D)
        z_flat = z.reshape(-1, D)  # (B*T, D)

        # 距離矩陣
        distances = torch.cdist(z_flat, codebook)  # (B*T, num_codes)

        # Soft assignment (使用 softmax)
        logits = -distances / self.temperature
        probs = F.softmax(logits, dim=-1)  # (B*T, num_codes)

        # 應用 mask
        mask_flat = mask.reshape(-1).unsqueeze(-1)  # (B*T, 1)
        probs_masked = probs * mask_flat

        # 計算 batch-level code distribution
        code_dist = probs_masked.sum(dim=0)  # (num_codes,)
        code_dist = code_dist / (code_dist.sum() + 1e-8)  # normalize

        # 計算 entropy
        entropy = -(code_dist * torch.log(code_dist + 1e-8)).sum()

        # 最大 entropy = log(num_codes)
        max_entropy = torch.log(torch.tensor(self.num_codes, dtype=torch.float32,
                                             device=student_features.device))

        # Normalized entropy loss (0 = max diversity, 1 = full collapse)
        entropy_loss = 1.0 - (entropy / max_entropy)

        return entropy_loss


class FeatureDiversityLoss(nn.Module):
    """
    Feature Diversity Loss: 懲罰 batch 內 features 太相似

    原理：
    - 計算 batch 內不同樣本的 feature cosine similarity
    - 相似度太高 = 不好 (collapse)
    - Loss = mean(similarity) for off-diagonal pairs
    """

    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: 允許的最大相似度，超過則懲罰
        """
        super().__init__()
        self.margin = margin

    def forward(self, student_features: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                encoder_stride: int = 320) -> torch.Tensor:
        """
        Args:
            student_features: (B, D, T) encoder 輸出
        """
        B, D, T = student_features.shape

        if B < 2:
            return torch.tensor(0.0, device=student_features.device)

        # 建立 mask
        if lengths is not None:
            max_audio_len = T * encoder_stride
            mask = create_length_mask(lengths, max_audio_len, encoder_stride,
                                       device=student_features.device)  # (B, T)
        else:
            mask = torch.ones(B, T, device=student_features.device)

        # 計算每個樣本的平均 feature
        z = student_features.permute(0, 2, 1)  # (B, T, D)

        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        z_masked = z * mask_expanded
        z_sum = z_masked.sum(dim=1)  # (B, D)
        z_count = mask.sum(dim=1, keepdim=True)  # (B, 1)
        z_mean = z_sum / (z_count + 1e-8)  # (B, D)

        # Normalize
        z_norm = F.normalize(z_mean, dim=-1)  # (B, D)

        # 計算 pairwise cosine similarity
        sim_matrix = torch.mm(z_norm, z_norm.t())  # (B, B)

        # 取 off-diagonal elements
        mask_diag = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
        off_diag_sim = sim_matrix[mask_diag]

        # Hinge loss: 懲罰超過 margin 的相似度
        loss = F.relu(off_diag_sim - self.margin).mean()

        return loss


class BatchContrastiveLoss(nn.Module):
    """
    Batch Contrastive Loss: 確保不同輸入產生不同輸出

    原理：
    - 對於每個樣本，其他樣本都是 negative
    - Student feature 應該與自己的 Teacher feature 相似
    - 與其他樣本的 Teacher feature 不相似
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_features: torch.Tensor,
                teacher_features: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                encoder_stride: int = 320) -> torch.Tensor:
        """
        Args:
            student_features: (B, D, T)
            teacher_features: (B, D, T)
        """
        B, D, T = student_features.shape

        if B < 2:
            return torch.tensor(0.0, device=student_features.device)

        # 建立 mask
        if lengths is not None:
            max_audio_len = T * encoder_stride
            mask = create_length_mask(lengths, max_audio_len, encoder_stride,
                                       device=student_features.device)
        else:
            mask = torch.ones(B, T, device=student_features.device)

        # Masked mean pooling
        def masked_mean(features, mask):
            z = features.permute(0, 2, 1)  # (B, T, D)
            mask_expanded = mask.unsqueeze(-1)
            z_masked = z * mask_expanded
            z_sum = z_masked.sum(dim=1)
            z_count = mask.sum(dim=1, keepdim=True)
            return z_sum / (z_count + 1e-8)

        s_mean = masked_mean(student_features, mask)  # (B, D)
        t_mean = masked_mean(teacher_features, mask)  # (B, D)

        # Normalize
        s_norm = F.normalize(s_mean, dim=-1)
        t_norm = F.normalize(t_mean, dim=-1)

        # Similarity matrix
        sim = torch.mm(s_norm, t_norm.t()) / self.temperature  # (B, B)

        # InfoNCE loss: diagonal should be high
        labels = torch.arange(B, device=sim.device)
        loss = F.cross_entropy(sim, labels)

        return loss


class MaskedCombinedLossV4(nn.Module):
    """
    組合版 Loss V4: Anti-Collapse + Frame-Tolerant

    包含:
    - Feature Loss (MSE) - 可選 Frame-Tolerant
    - Triplet Loss - 可選 Frame-Tolerant
    - Cosine Loss
    - CE Loss
    - Code Entropy Loss (Anti-Collapse)
    - Feature Diversity Loss (Anti-Collapse)
    - Batch Contrastive Loss (Anti-Collapse)
    """

    def __init__(self,
                 feature_weight: float = 1.0,
                 triplet_weight: float = 0.0,
                 triplet_margin: float = 0.2,
                 cosine_weight: float = 0.0,
                 ce_weight: float = 0.0,
                 entropy_weight: float = 0.1,
                 diversity_weight: float = 0.1,
                 contrastive_weight: float = 0.1,
                 diversity_margin: float = 0.5,
                 contrastive_temperature: float = 0.1,
                 encoder_stride: int = 320,
                 label_smoothing: float = 0.0,
                 use_frame_tolerant: bool = True,
                 frame_tolerance: int = 1):
        super().__init__()

        self.feature_weight = feature_weight
        self.triplet_weight = triplet_weight
        self.cosine_weight = cosine_weight
        self.ce_weight = ce_weight
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.contrastive_weight = contrastive_weight
        self.encoder_stride = encoder_stride
        self.use_frame_tolerant = use_frame_tolerant

        # 基礎 losses - 可選 Frame-Tolerant 版本
        if use_frame_tolerant:
            self.feature_loss = FrameTolerantFeatureLoss(encoder_stride, tolerance=frame_tolerance)
            self.triplet_loss = FrameTolerantTripletLoss(margin=triplet_margin, encoder_stride=encoder_stride, tolerance=frame_tolerance)
        else:
            self.feature_loss = MaskedFeatureLoss(encoder_stride)
            self.triplet_loss = MaskedTripletLoss(margin=triplet_margin, encoder_stride=encoder_stride)

        self.cosine_loss = MaskedCosineLoss(encoder_stride)
        self.ce_loss = MaskedCrossEntropyLoss(encoder_stride, label_smoothing)

        # Anti-Collapse losses
        self.entropy_loss = CodeEntropyLoss(num_codes=4096)
        self.diversity_loss = FeatureDiversityLoss(margin=diversity_margin)
        self.contrastive_loss = BatchContrastiveLoss(temperature=contrastive_temperature)

    def _add_loss(self, total, component):
        if total is None:
            return component
        return total + component

    def forward(self,
                student_features: torch.Tensor,
                teacher_features: torch.Tensor,
                teacher_codes: torch.Tensor,
                codebook: torch.Tensor,
                lengths: torch.Tensor,
                logits: Optional[torch.Tensor] = None) -> tuple:
        """
        Returns:
            loss: total loss
            loss_dict: 各 loss component 的值
        """
        total_loss = None
        loss_dict = {}

        # ========== 基礎 Losses ==========

        # Feature Loss (可能是 Frame-Tolerant 版本)
        if self.feature_weight > 0:
            if self.use_frame_tolerant:
                feat_loss, feat_info = self.feature_loss(student_features, teacher_features, lengths)
                loss_dict['feature_loss'] = feat_loss.item()
                loss_dict['offset_zero_ratio'] = feat_info.get('offset_zero_ratio', 0)
            else:
                feat_loss = self.feature_loss(student_features, teacher_features, lengths)
                loss_dict['feature_loss'] = feat_loss.item()
            total_loss = self._add_loss(total_loss, self.feature_weight * feat_loss)

        # Cosine Loss
        if self.cosine_weight > 0:
            cos_loss, cos_sim_mean, cos_sim_std = self.cosine_loss(
                student_features, teacher_features, lengths, return_stats=True
            )
            loss_dict['cosine_loss'] = cos_loss.item()
            loss_dict['cos_sim_mean'] = cos_sim_mean
            loss_dict['cos_sim_std'] = cos_sim_std
            total_loss = self._add_loss(total_loss, self.cosine_weight * cos_loss)

        # Triplet Loss (可能是 Frame-Tolerant 版本)
        if self.triplet_weight > 0:
            if self.use_frame_tolerant:
                trip_loss, trip_info = self.triplet_loss(student_features, teacher_codes, codebook, lengths)
                loss_dict['triplet_loss'] = trip_loss.item()
            else:
                trip_loss = self.triplet_loss(student_features, teacher_codes, codebook, lengths)
                loss_dict['triplet_loss'] = trip_loss.item()
            total_loss = self._add_loss(total_loss, self.triplet_weight * trip_loss)

        # CE Loss
        if self.ce_weight > 0 and logits is not None:
            t_codes = teacher_codes[0] if teacher_codes.dim() == 3 else teacher_codes
            ce_loss = self.ce_loss(logits, t_codes.long(), lengths)
            loss_dict['ce_loss'] = ce_loss.item()
            total_loss = self._add_loss(total_loss, self.ce_weight * ce_loss)

        # ========== Anti-Collapse Losses ==========

        # Code Entropy Loss
        if self.entropy_weight > 0:
            ent_loss = self.entropy_loss(student_features, codebook, lengths, self.encoder_stride)
            loss_dict['entropy_loss'] = ent_loss.item()
            total_loss = self._add_loss(total_loss, self.entropy_weight * ent_loss)

        # Feature Diversity Loss
        if self.diversity_weight > 0:
            div_loss = self.diversity_loss(student_features, lengths, self.encoder_stride)
            loss_dict['diversity_loss'] = div_loss.item()
            total_loss = self._add_loss(total_loss, self.diversity_weight * div_loss)

        # Batch Contrastive Loss
        if self.contrastive_weight > 0:
            contrast_loss = self.contrastive_loss(
                student_features, teacher_features, lengths, self.encoder_stride
            )
            loss_dict['contrastive_loss'] = contrast_loss.item()
            total_loss = self._add_loss(total_loss, self.contrastive_weight * contrast_loss)

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


# ==================== 測試 ====================
if __name__ == '__main__':
    print("Testing Anti-Collapse Losses...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, D, T = 4, 512, 100
    num_codes = 4096

    student_features = torch.randn(B, D, T, device=device)
    teacher_features = torch.randn(B, D, T, device=device)
    codebook = torch.randn(num_codes, D, device=device)
    lengths = torch.tensor([24000, 20000, 22000, 18000], device=device)

    # Test CodeEntropyLoss
    print("\n--- CodeEntropyLoss ---")
    entropy_loss = CodeEntropyLoss()
    loss = entropy_loss(student_features, codebook, lengths)
    print(f"Entropy Loss: {loss.item():.4f}")

    # Test with collapsed features (all same)
    collapsed = student_features[:1].expand(B, -1, -1).clone()
    loss_collapsed = entropy_loss(collapsed, codebook, lengths)
    print(f"Entropy Loss (collapsed): {loss_collapsed.item():.4f}")
    print(f"  → Collapsed should be higher (less diverse)")

    # Test FeatureDiversityLoss
    print("\n--- FeatureDiversityLoss ---")
    diversity_loss = FeatureDiversityLoss(margin=0.5)
    loss = diversity_loss(student_features, lengths)
    print(f"Diversity Loss: {loss.item():.4f}")

    loss_collapsed = diversity_loss(collapsed, lengths)
    print(f"Diversity Loss (collapsed): {loss_collapsed.item():.4f}")
    print(f"  → Collapsed should be higher (too similar)")

    # Test BatchContrastiveLoss
    print("\n--- BatchContrastiveLoss ---")
    contrastive_loss = BatchContrastiveLoss(temperature=0.1)
    loss = contrastive_loss(student_features, teacher_features, lengths)
    print(f"Contrastive Loss: {loss.item():.4f}")

    # Test MaskedCombinedLossV4
    print("\n--- MaskedCombinedLossV4 ---")
    teacher_codes = torch.randint(0, num_codes, (1, B, T), device=device)

    combined_loss = MaskedCombinedLossV4(
        feature_weight=1.0,
        triplet_weight=1.0,
        triplet_margin=0.2,
        entropy_weight=0.1,
        diversity_weight=0.1,
        contrastive_weight=0.1,
    )

    loss, loss_dict = combined_loss(
        student_features, teacher_features, teacher_codes, codebook, lengths
    )
    print(f"Total Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ All tests passed!")
