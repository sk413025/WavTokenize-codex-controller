"""
exp_1219: Masked Loss Functions with Cosine Similarity

基於 exp_1212/losses_masked.py，新增 MaskedCosineLoss
解決 Exp48 發現的特徵方向不對齊問題 (cos_sim = 0.21)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_length_mask(lengths, max_len, encoder_stride=320, device=None):
    """
    創建 frame-level mask

    Args:
        lengths: (B,) 每個樣本的有效 audio samples 數
        max_len: batch 內最長 audio samples 數
        encoder_stride: encoder 的 stride (WavTokenizer 約 320)
        device: 設備

    Returns:
        mask: (B, T) where T = max_len // encoder_stride
    """
    if device is None:
        device = lengths.device

    frame_lengths = lengths // encoder_stride
    max_frames = max_len // encoder_stride

    frame_indices = torch.arange(max_frames, device=device).unsqueeze(0)
    mask = frame_indices < frame_lengths.unsqueeze(1)

    return mask.float()


class MaskedFeatureLoss(nn.Module):
    """Masked Feature Loss (MSE)"""

    def __init__(self, encoder_stride=320):
        super().__init__()
        self.encoder_stride = encoder_stride

    def forward(self, student_features, teacher_features, lengths):
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)
        mask_expanded = mask.unsqueeze(1)

        diff_sq = (student_features - teacher_features) ** 2
        masked_diff = diff_sq * mask_expanded
        loss = masked_diff.sum() / (mask_expanded.sum() * D + 1e-8)

        return loss


class MaskedCosineLoss(nn.Module):
    """
    Masked Cosine Similarity Loss

    專門優化特徵方向對齊，解決 MSE 無法有效優化方向的問題。

    Cosine Similarity:
    - 1.0 = 完全對齊
    - 0.0 = 正交
    - -1.0 = 完全相反

    Loss = 1 - cos_sim (越接近 0 越好)
    """

    def __init__(self, encoder_stride=320):
        super().__init__()
        self.encoder_stride = encoder_stride

    def forward(self, student_features, teacher_features, lengths):
        """
        Args:
            student_features: (B, D, T) student encoder output
            teacher_features: (B, D, T) teacher encoder output (should be detached)
            lengths: (B,) 有效 audio samples 數

        Returns:
            loss: scalar, masked cosine loss
        """
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride

        # 創建 mask (B, T)
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)

        # Reshape: (B, D, T) -> (B, T, D) -> (B*T, D)
        stu = student_features.permute(0, 2, 1).reshape(-1, D)
        tea = teacher_features.permute(0, 2, 1).reshape(-1, D)

        # Cosine similarity per frame: (B*T,)
        cos_sim = F.cosine_similarity(stu, tea, dim=1)

        # Reshape mask: (B, T) -> (B*T,)
        mask_flat = mask.reshape(-1)

        # Loss = 1 - cosine (希望 cos_sim 接近 1)
        loss_per_frame = 1 - cos_sim

        # Apply mask
        masked_loss = loss_per_frame * mask_flat
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)

        return loss

    def compute_stats(self, student_features, teacher_features, lengths):
        """計算 cosine similarity 統計 (用於 logging)"""
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride

        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)

        stu = student_features.permute(0, 2, 1).reshape(-1, D)
        tea = teacher_features.permute(0, 2, 1).reshape(-1, D)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(stu, tea, dim=1)
            mask_flat = mask.reshape(-1)

            valid_cos_sim = cos_sim[mask_flat > 0]
            mean_cos = valid_cos_sim.mean().item()
            std_cos = valid_cos_sim.std().item()

        return {'cos_sim_mean': mean_cos, 'cos_sim_std': std_cos}


class MaskedTripletLoss(nn.Module):
    """Masked Triplet Loss for VQ Codebook"""

    def __init__(self, margin=0.2, encoder_stride=320):
        super().__init__()
        self.margin = margin
        self.encoder_stride = encoder_stride

    def forward(self, student_features, teacher_codes, codebook, lengths):
        B, D, T = student_features.shape
        max_audio_len = T * self.encoder_stride
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)

        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]

        z = student_features.permute(0, 2, 1).reshape(-1, D)
        dists = torch.cdist(z, codebook, p=2)

        teacher_flat = teacher_codes.reshape(-1)
        batch_indices = torch.arange(len(teacher_flat), device=dists.device)
        pos_dist = dists[batch_indices, teacher_flat]

        dists_for_neg = dists.clone()
        dists_for_neg[batch_indices, teacher_flat] = float('inf')
        neg_dist = dists_for_neg.min(dim=1).values

        pos_dist = pos_dist.reshape(B, T)
        neg_dist = neg_dist.reshape(B, T)

        triplet = F.relu(pos_dist - neg_dist + self.margin)
        masked_triplet = triplet * mask
        loss = masked_triplet.sum() / (mask.sum() + 1e-8)

        return loss


class MaskedCrossEntropyLoss(nn.Module):
    """Masked Cross-Entropy Loss"""

    def __init__(self, encoder_stride=320, label_smoothing=0.0):
        super().__init__()
        self.encoder_stride = encoder_stride
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, lengths):
        B, C, T = logits.shape
        max_audio_len = T * self.encoder_stride
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=logits.device)

        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        ce_loss = F.cross_entropy(logits_flat, targets_flat,
                                   label_smoothing=self.label_smoothing,
                                   reduction='none')

        masked_loss = ce_loss * mask_flat
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)

        return loss


def compute_masked_accuracy(predictions, targets, lengths, encoder_stride=320):
    """計算 Masked Token Accuracy"""
    B, T = predictions.shape
    max_audio_len = T * encoder_stride
    mask = create_length_mask(lengths, max_audio_len, encoder_stride,
                               device=predictions.device)

    correct = (predictions == targets).float()
    masked_correct = correct * mask

    num_correct = masked_correct.sum().item()
    num_total = mask.sum().item()
    accuracy = num_correct / (num_total + 1e-8)

    return accuracy, int(num_correct), int(num_total)


class MaskedCombinedLossV2(nn.Module):
    """
    組合版 Masked Loss V2

    支援:
    - Masked Feature Loss (MSE on encoder features)
    - Masked Cosine Loss (NEW: 優化方向對齊)
    - Masked Triplet Loss
    - Masked CE Loss
    """

    def __init__(self,
                 feature_weight=1.0,
                 cosine_weight=0.0,      # 新增
                 triplet_weight=0.5,
                 triplet_margin=0.2,
                 ce_weight=0.0,
                 encoder_stride=320):
        super().__init__()

        self.feature_weight = feature_weight
        self.cosine_weight = cosine_weight
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
        self.encoder_stride = encoder_stride

        self.feature_loss = MaskedFeatureLoss(encoder_stride)
        self.cosine_loss = MaskedCosineLoss(encoder_stride)
        self.triplet_loss = MaskedTripletLoss(triplet_margin, encoder_stride)
        self.ce_loss = MaskedCrossEntropyLoss(encoder_stride)

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
            if total_loss is None:
                total_loss = self.feature_weight * feat_loss
            else:
                total_loss = total_loss + self.feature_weight * feat_loss

        # Cosine Loss (NEW)
        if self.cosine_weight > 0:
            cos_loss = self.cosine_loss(student_features, teacher_features_detached, lengths)
            loss_dict['cosine_loss'] = cos_loss.item()

            # 同時記錄 cos_sim 統計
            cos_stats = self.cosine_loss.compute_stats(student_features, teacher_features_detached, lengths)
            loss_dict['cos_sim_mean'] = cos_stats['cos_sim_mean']
            loss_dict['cos_sim_std'] = cos_stats['cos_sim_std']

            if total_loss is None:
                total_loss = self.cosine_weight * cos_loss
            else:
                total_loss = total_loss + self.cosine_weight * cos_loss

        # Triplet Loss
        if self.triplet_weight > 0:
            trip_loss = self.triplet_loss(student_features, teacher_codes, codebook, lengths)
            loss_dict['triplet_loss'] = trip_loss.item()
            if total_loss is None:
                total_loss = self.triplet_weight * trip_loss
            else:
                total_loss = total_loss + self.triplet_weight * trip_loss

        # CE Loss
        if self.ce_weight > 0 and logits is not None:
            t_codes = teacher_codes
            if t_codes.dim() == 3:
                t_codes = t_codes[0]
            ce_loss = self.ce_loss(logits, t_codes.long(), lengths)
            loss_dict['ce_loss'] = ce_loss.item()
            if total_loss is None:
                total_loss = self.ce_weight * ce_loss
            else:
                total_loss = total_loss + self.ce_weight * ce_loss

        if total_loss is None:
            raise ValueError("No loss computed! At least one loss weight must be > 0")

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


# ==================== 測試 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Masked Loss Functions V2 (with Cosine Loss)")
    print("=" * 60)

    B, D, T = 4, 512, 100
    num_codes = 1024
    encoder_stride = 320

    lengths = torch.tensor([32000, 24000, 16000, 8000])

    student_features = torch.randn(B, D, T)
    teacher_features = torch.randn(B, D, T)
    teacher_codes = torch.randint(0, num_codes, (B, T))
    codebook = torch.randn(num_codes, D)

    print(f"\nInput shapes:")
    print(f"  student_features: {student_features.shape}")
    print(f"  teacher_features: {teacher_features.shape}")

    # 測試 Cosine Loss
    print("\n--- Testing Cosine Loss ---")
    cos_loss_fn = MaskedCosineLoss(encoder_stride)
    cos_loss = cos_loss_fn(student_features, teacher_features, lengths)
    cos_stats = cos_loss_fn.compute_stats(student_features, teacher_features, lengths)
    print(f"Masked Cosine Loss: {cos_loss.item():.4f}")
    print(f"Cosine Sim Mean: {cos_stats['cos_sim_mean']:.4f} ± {cos_stats['cos_sim_std']:.4f}")

    # 測試 Combined Loss V2
    print("\n--- Testing Combined Loss V2 ---")
    combined_loss_fn = MaskedCombinedLossV2(
        feature_weight=1.0,
        cosine_weight=0.5,
        triplet_weight=1.0,
        triplet_margin=0.2,
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
