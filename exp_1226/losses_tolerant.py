"""
exp_1226: Frame-Tolerant Loss

解決問題：Noisy 和 Clean 音檔有時間偏移
- 73.3% 樣本有超過半個 frame (6.7ms) 的偏移
- 導致 token-level 對齊不準確

解決方案：
1. 允許 ±1 frame 的容忍度
2. 選擇最佳對齊的 loss

實作方式：
- 對於每個 student frame，計算與 teacher 的 t-1, t, t+1 三個 frame 的 loss
- 選擇最小的 loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1219.losses import create_length_mask


class FrameTolerantFeatureLoss(nn.Module):
    """
    Frame-Tolerant Feature Loss

    對於每個 student frame，與 teacher 的 t-1, t, t+1 比較，取最小 loss
    """

    def __init__(self, encoder_stride: int = 320, tolerance: int = 1):
        """
        Args:
            encoder_stride: Encoder 步長 (samples)
            tolerance: 容忍的 frame 偏移量 (預設 ±1)
        """
        super().__init__()
        self.encoder_stride = encoder_stride
        self.tolerance = tolerance

    def forward(self, student_features: torch.Tensor,
                teacher_features: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            student_features: (B, D, T)
            teacher_features: (B, D, T)
            lengths: (B,) 有效長度 (samples)

        Returns:
            loss: scalar
            info: dict with debug info
        """
        B, D, T = student_features.shape

        # 建立 mask
        max_audio_len = T * self.encoder_stride
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)  # (B, T)

        # Transpose to (B, T, D) for easier indexing
        s = student_features.permute(0, 2, 1)  # (B, T, D)
        t = teacher_features.permute(0, 2, 1)  # (B, T, D)

        # 計算每個 frame 與前後的 MSE
        # 選擇最小的
        best_mse = torch.zeros(B, T, device=s.device)
        best_offset = torch.zeros(B, T, dtype=torch.long, device=s.device)

        for offset in range(-self.tolerance, self.tolerance + 1):
            # 計算 student[t] 與 teacher[t+offset] 的 MSE
            if offset < 0:
                # teacher 領先
                s_slice = s[:, -offset:, :]  # (B, T+offset, D)
                t_slice = t[:, :T+offset, :]
                mse = ((s_slice - t_slice) ** 2).mean(dim=-1)  # (B, T+offset)

                # Pad 回 T
                mse_padded = F.pad(mse, (0, -offset), value=float('inf'))
            elif offset > 0:
                # student 領先
                s_slice = s[:, :T-offset, :]
                t_slice = t[:, offset:, :]
                mse = ((s_slice - t_slice) ** 2).mean(dim=-1)

                # Pad 回 T
                mse_padded = F.pad(mse, (offset, 0), value=float('inf'))
            else:
                mse_padded = ((s - t) ** 2).mean(dim=-1)  # (B, T)

            # 更新最佳
            if offset == -self.tolerance:
                best_mse = mse_padded
                best_offset.fill_(offset)
            else:
                better = mse_padded < best_mse
                best_mse = torch.where(better, mse_padded, best_mse)
                best_offset = torch.where(better, torch.tensor(offset, device=s.device), best_offset)

        # 應用 mask
        masked_mse = best_mse * mask
        loss = masked_mse.sum() / (mask.sum() + 1e-8)

        # Debug info
        info = {
            'mean_offset': best_offset.float().mean().item(),
            'offset_minus1_ratio': (best_offset == -1).float().mean().item(),
            'offset_zero_ratio': (best_offset == 0).float().mean().item(),
            'offset_plus1_ratio': (best_offset == 1).float().mean().item(),
        }

        return loss, info


class FrameTolerantTripletLoss(nn.Module):
    """
    Frame-Tolerant Triplet Loss

    Anchor: Student feature
    Positive: Teacher feature at best aligned position (t-1, t, t+1)
    Negative: Other codebook entries
    """

    def __init__(self, margin: float = 0.2, encoder_stride: int = 320, tolerance: int = 1):
        super().__init__()
        self.margin = margin
        self.encoder_stride = encoder_stride
        self.tolerance = tolerance

    def forward(self, student_features: torch.Tensor,
                teacher_codes: torch.Tensor,
                codebook: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            student_features: (B, D, T)
            teacher_codes: (1, B, T) or (B, T)
            codebook: (num_codes, D)
        """
        B, D, T = student_features.shape

        # 處理 teacher_codes
        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]  # (B, T)

        # 建立 mask
        max_audio_len = T * self.encoder_stride
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_features.device)  # (B, T)

        s = student_features.permute(0, 2, 1)  # (B, T, D)

        # 找最佳對齊的 positive
        best_pos_dist = torch.full((B, T), float('inf'), device=s.device)
        best_codes = teacher_codes.clone()

        for offset in range(-self.tolerance, self.tolerance + 1):
            if offset < 0:
                codes_shifted = F.pad(teacher_codes[:, :T+offset], (0, -offset), value=0)
            elif offset > 0:
                codes_shifted = F.pad(teacher_codes[:, offset:], (offset, 0), value=0)
            else:
                codes_shifted = teacher_codes

            # 獲取 positive embedding
            pos_embed = codebook[codes_shifted.long()]  # (B, T, D)

            # 計算距離
            pos_dist = ((s - pos_embed) ** 2).sum(dim=-1)  # (B, T)

            # 更新最佳
            better = pos_dist < best_pos_dist
            best_pos_dist = torch.where(better, pos_dist, best_pos_dist)
            best_codes = torch.where(better, codes_shifted, best_codes)

        # 使用最佳 codes 計算 triplet loss
        pos_embed = codebook[best_codes.long()]  # (B, T, D)
        pos_dist = ((s - pos_embed) ** 2).sum(dim=-1).sqrt()  # (B, T)

        # 隨機 negative
        neg_indices = torch.randint(0, codebook.shape[0], (B, T), device=s.device)
        neg_embed = codebook[neg_indices]
        neg_dist = ((s - neg_embed) ** 2).sum(dim=-1).sqrt()

        # Triplet loss
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)

        # 應用 mask
        masked_loss = triplet_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)

        info = {}
        return loss, info


class FrameTolerantAccuracy(nn.Module):
    """
    Frame-Tolerant Token Accuracy

    允許 ±tolerance 的偏移，只要命中就算對
    """

    def __init__(self, encoder_stride: int = 320, tolerance: int = 1):
        super().__init__()
        self.encoder_stride = encoder_stride
        self.tolerance = tolerance

    @torch.no_grad()
    def forward(self, student_codes: torch.Tensor,
                teacher_codes: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[float, dict]:
        """
        Returns:
            accuracy: float
            info: dict with strict_acc, tolerant_acc
        """
        if student_codes.dim() == 3:
            student_codes = student_codes[0]
        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]

        B, T = student_codes.shape

        # 建立 mask
        max_audio_len = T * self.encoder_stride
        mask = create_length_mask(lengths, max_audio_len, self.encoder_stride,
                                   device=student_codes.device)

        # Strict accuracy
        strict_correct = (student_codes == teacher_codes).float()
        strict_acc = (strict_correct * mask).sum() / (mask.sum() + 1e-8)

        # Tolerant accuracy
        tolerant_correct = torch.zeros_like(strict_correct)

        for offset in range(-self.tolerance, self.tolerance + 1):
            if offset < 0:
                # 比較 student[t] 與 teacher[t+offset]
                s_slice = student_codes[:, -offset:]
                t_slice = teacher_codes[:, :T+offset]
                match = (s_slice == t_slice).float()
                match_padded = F.pad(match, (0, -offset), value=0)
            elif offset > 0:
                s_slice = student_codes[:, :T-offset]
                t_slice = teacher_codes[:, offset:]
                match = (s_slice == t_slice).float()
                match_padded = F.pad(match, (offset, 0), value=0)
            else:
                match_padded = strict_correct

            tolerant_correct = torch.maximum(tolerant_correct, match_padded)

        tolerant_acc = (tolerant_correct * mask).sum() / (mask.sum() + 1e-8)

        info = {
            'strict_acc': strict_acc.item(),
            'tolerant_acc': tolerant_acc.item(),
            'improvement': (tolerant_acc - strict_acc).item(),
        }

        return tolerant_acc.item(), info


# ==================== 測試 ====================
if __name__ == '__main__':
    print("Testing Frame-Tolerant Losses...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, D, T = 4, 512, 100
    num_codes = 4096

    # 模擬有偏移的數據
    teacher_features = torch.randn(B, D, T, device=device)
    # Student 與 teacher 相似但有 1 frame 偏移
    student_features = torch.roll(teacher_features, shifts=1, dims=2) + 0.1 * torch.randn(B, D, T, device=device)

    teacher_codes = torch.randint(0, num_codes, (B, T), device=device)
    student_codes = torch.roll(teacher_codes, shifts=1, dims=1)  # 偏移 1 frame

    codebook = torch.randn(num_codes, D, device=device)
    lengths = torch.tensor([24000, 20000, 22000, 18000], device=device)

    # Test FrameTolerantFeatureLoss
    print("\n--- FrameTolerantFeatureLoss ---")
    feat_loss = FrameTolerantFeatureLoss(tolerance=1)
    loss, info = feat_loss(student_features, teacher_features, lengths)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")

    # 比較 standard loss
    from exp_1219.losses import MaskedFeatureLoss
    std_loss = MaskedFeatureLoss(320)
    std_loss_val = std_loss(student_features, teacher_features, lengths)
    print(f"Standard Loss: {std_loss_val.item():.4f}")
    print(f"Tolerant should be lower if there's offset alignment")

    # Test FrameTolerantAccuracy
    print("\n--- FrameTolerantAccuracy ---")
    acc_fn = FrameTolerantAccuracy(tolerance=1)
    acc, info = acc_fn(student_codes, teacher_codes, lengths)
    print(f"Tolerant Accuracy: {acc*100:.2f}%")
    print(f"Info: {info}")

    # Strict accuracy (應該接近 0 因為有偏移)
    from exp_1219.losses import compute_masked_accuracy
    strict_acc, _, _ = compute_masked_accuracy(student_codes, teacher_codes, lengths, 320)
    print(f"Strict Accuracy: {strict_acc*100:.2f}%")
    print(f"Improvement: {(acc - strict_acc)*100:.2f}%")

    print("\n✓ All tests passed!")
