"""
exp_1212: Masked Loss Functions

解決 Cross-Sample Mismatch (來源2) 問題
- 只在有效 frames 上計算 loss
- 忽略 padding 部分，避免稀釋 loss 信號

基於 exp_1210/DATASET_ALIGNMENT_REPORT.md 的分析
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

    # 轉換成 frame 數
    frame_lengths = lengths // encoder_stride  # (B,)
    max_frames = max_len // encoder_stride

    # 創建 mask: (B, T)
    frame_indices = torch.arange(max_frames, device=device).unsqueeze(0)  # (1, T)
    mask = frame_indices < frame_lengths.unsqueeze(1)  # (B, T)

    return mask.float()


class MaskedFeatureLoss(nn.Module):
    """
    Masked Feature Loss

    只在有效 frames 上計算 MSE loss，忽略 padding
    """

    def __init__(self, encoder_stride=320):
        super().__init__()
        self.encoder_stride = encoder_stride

    def forward(self, student_features, teacher_features, lengths):
        """
        Args:
            student_features: (B, D, T) student encoder output
            teacher_features: (B, D, T) teacher encoder output (detached)
            lengths: (B,) 有效 audio samples 數

        Returns:
            loss: scalar, masked MSE loss
        """
        B, D, T = student_features.shape

        # 計算最大 audio samples (反推)
        max_audio_len = T * self.encoder_stride

        # 創建 mask (B, T)
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)

        # Expand mask to (B, D, T)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, T)

        # 計算 MSE
        diff_sq = (student_features - teacher_features) ** 2  # (B, D, T)

        # Masked mean
        masked_diff = diff_sq * mask_expanded
        loss = masked_diff.sum() / (mask_expanded.sum() * D + 1e-8)

        return loss


class MaskedTripletLoss(nn.Module):
    """
    Masked Triplet Loss for VQ Codebook (Memory-Efficient Version)

    使用 torch.cdist 高效計算距離，只在有效 frames 上計算 loss
    """

    def __init__(self, margin=0.2, encoder_stride=320):
        super().__init__()
        self.margin = margin
        self.encoder_stride = encoder_stride

    def forward(self, student_features, teacher_codes, codebook, lengths):
        """
        Args:
            student_features: (B, D, T) student encoder output
            teacher_codes: (B, T) teacher VQ codes
            codebook: (num_codes, D) VQ codebook
            lengths: (B,) 有效 audio samples 數

        Returns:
            loss: scalar, masked triplet loss
        """
        B, D, T = student_features.shape

        # 創建 mask (B, T)
        max_audio_len = T * self.encoder_stride
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)

        # 處理 teacher_codes 維度
        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]  # (B, T)

        # Student features: (B, D, T) -> (B*T, D)
        z = student_features.permute(0, 2, 1).reshape(-1, D)  # (B*T, D)

        # 使用 cdist 高效計算到所有 code 的距離
        # dists: (B*T, num_codes)
        dists = torch.cdist(z, codebook, p=2)

        # 獲取正樣本距離 (到正確 code 的距離)
        teacher_flat = teacher_codes.reshape(-1)  # (B*T,)
        batch_indices = torch.arange(len(teacher_flat), device=dists.device)
        pos_dist = dists[batch_indices, teacher_flat]  # (B*T,)

        # 複製一份來找負樣本，避免 in-place 操作影響梯度
        dists_for_neg = dists.clone()
        dists_for_neg[batch_indices, teacher_flat] = float('inf')
        neg_dist = dists_for_neg.min(dim=1).values  # (B*T,)

        # Reshape back to (B, T)
        pos_dist = pos_dist.reshape(B, T)
        neg_dist = neg_dist.reshape(B, T)

        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        triplet = F.relu(pos_dist - neg_dist + self.margin)  # (B, T)

        # Apply mask
        masked_triplet = triplet * mask
        loss = masked_triplet.sum() / (mask.sum() + 1e-8)

        return loss


class MaskedCrossEntropyLoss(nn.Module):
    """
    Masked Cross-Entropy Loss

    只在有效 frames 上計算 CE loss
    """

    def __init__(self, encoder_stride=320, label_smoothing=0.0):
        super().__init__()
        self.encoder_stride = encoder_stride
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, lengths):
        """
        Args:
            logits: (B, num_classes, T) 預測 logits
            targets: (B, T) 目標 codes
            lengths: (B,) 有效 audio samples 數

        Returns:
            loss: scalar, masked CE loss
        """
        B, C, T = logits.shape

        # 創建 mask (B, T)
        max_audio_len = T * self.encoder_stride
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=logits.device)

        # Reshape for CE: (B*T, C) and (B*T,)
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
        targets_flat = targets.reshape(-1)  # (B*T,)
        mask_flat = mask.reshape(-1)  # (B*T,)

        # 計算 per-element CE loss
        ce_loss = F.cross_entropy(logits_flat, targets_flat,
                                   label_smoothing=self.label_smoothing,
                                   reduction='none')  # (B*T,)

        # Apply mask
        masked_loss = ce_loss * mask_flat
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)

        return loss


def compute_masked_accuracy(predictions, targets, lengths, encoder_stride=320):
    """
    計算 Masked Token Accuracy

    Args:
        predictions: (B, T) 預測的 codes
        targets: (B, T) 目標 codes
        lengths: (B,) 有效 audio samples 數

    Returns:
        accuracy: float, masked accuracy
        num_correct: int
        num_total: int
    """
    B, T = predictions.shape

    # 創建 mask (B, T)
    max_audio_len = T * encoder_stride
    mask = create_length_mask(lengths, max_audio_len, encoder_stride,
                               device=predictions.device)

    # 計算正確數
    correct = (predictions == targets).float()  # (B, T)
    masked_correct = correct * mask

    num_correct = masked_correct.sum().item()
    num_total = mask.sum().item()

    accuracy = num_correct / (num_total + 1e-8)

    return accuracy, int(num_correct), int(num_total)


class MaskedCombinedLoss(nn.Module):
    """
    組合版 Masked Loss

    支援:
    - Masked Feature Loss (MSE on encoder features)
    - Masked Triplet Loss (optional)
    - Masked CE Loss (optional)
    """

    def __init__(self,
                 feature_weight=1.0,
                 triplet_weight=0.5,
                 triplet_margin=0.2,
                 ce_weight=0.0,
                 encoder_stride=320):
        super().__init__()

        self.feature_weight = feature_weight
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
        self.encoder_stride = encoder_stride

        self.feature_loss = MaskedFeatureLoss(encoder_stride)
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
        total_loss = None  # Will be tensor after first loss is added

        # Feature Loss
        if self.feature_weight > 0:
            feat_loss = self.feature_loss(student_features, teacher_features.detach(), lengths)
            loss_dict['feature_loss'] = feat_loss.item()
            if total_loss is None:
                total_loss = self.feature_weight * feat_loss
            else:
                total_loss = total_loss + self.feature_weight * feat_loss

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
            # 處理 teacher_codes 維度 (可能是 3D)
            t_codes = teacher_codes
            if t_codes.dim() == 3:
                t_codes = t_codes[0]  # (B, T)
            ce_loss = self.ce_loss(logits, t_codes.long(), lengths)
            loss_dict['ce_loss'] = ce_loss.item()
            if total_loss is None:
                total_loss = self.ce_weight * ce_loss
            else:
                total_loss = total_loss + self.ce_weight * ce_loss

        # Safety check: ensure total_loss is a tensor
        if total_loss is None:
            raise ValueError("No loss computed! At least one loss weight must be > 0")

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


# ==================== 測試 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Masked Loss Functions")
    print("=" * 60)

    # 模擬數據
    B, D, T = 4, 512, 100
    num_codes = 1024
    encoder_stride = 320

    # 模擬不同長度的樣本
    # 假設 max audio length = T * stride = 32000 samples
    lengths = torch.tensor([32000, 24000, 16000, 8000])  # 不同長度

    student_features = torch.randn(B, D, T)
    teacher_features = torch.randn(B, D, T)
    teacher_codes = torch.randint(0, num_codes, (B, T))
    codebook = torch.randn(num_codes, D)

    print(f"\nInput shapes:")
    print(f"  student_features: {student_features.shape}")
    print(f"  teacher_features: {teacher_features.shape}")
    print(f"  teacher_codes: {teacher_codes.shape}")
    print(f"  lengths: {lengths}")

    # 測試 mask 創建
    mask = create_length_mask(lengths, 32000, encoder_stride)
    print(f"\nMask shape: {mask.shape}")
    print(f"Valid frames per sample: {mask.sum(dim=1).tolist()}")
    print(f"Total valid frames: {mask.sum().item()} / {B * T}")

    # 測試各種 loss
    print("\n--- Testing Individual Losses ---")

    # Feature Loss
    feat_loss_fn = MaskedFeatureLoss(encoder_stride)
    feat_loss = feat_loss_fn(student_features, teacher_features, lengths)
    print(f"Masked Feature Loss: {feat_loss.item():.4f}")

    # Triplet Loss
    trip_loss_fn = MaskedTripletLoss(margin=0.2, encoder_stride=encoder_stride)
    trip_loss = trip_loss_fn(student_features, teacher_codes, codebook, lengths)
    print(f"Masked Triplet Loss: {trip_loss.item():.4f}")

    # CE Loss
    logits = torch.randn(B, num_codes, T)
    ce_loss_fn = MaskedCrossEntropyLoss(encoder_stride)
    ce_loss = ce_loss_fn(logits, teacher_codes, lengths)
    print(f"Masked CE Loss: {ce_loss.item():.4f}")

    # Accuracy
    predictions = torch.randint(0, num_codes, (B, T))
    acc, correct, total = compute_masked_accuracy(predictions, teacher_codes, lengths, encoder_stride)
    print(f"Masked Accuracy: {acc*100:.2f}% ({correct}/{total})")

    # 測試組合 loss
    print("\n--- Testing Combined Loss ---")
    combined_loss_fn = MaskedCombinedLoss(
        feature_weight=1.0,
        triplet_weight=0.5,
        ce_weight=0.1,
        encoder_stride=encoder_stride
    )
    total_loss, loss_dict = combined_loss_fn(
        student_features, teacher_features, teacher_codes, codebook, lengths, logits
    )
    print(f"Combined Loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
