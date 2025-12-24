"""
exp_1222: Audio Domain Loss Functions

支援:
1. Multi-Resolution STFT Loss (時頻域)
2. Mel Spectrogram Loss (感知加權)
3. Feature-level Loss (原有)
4. 組合 Loss

訓練策略: 連續 features 直接 decode，bypass VQ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, Tuple, Optional


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss

    在多個時頻解析度上計算 spectral convergence + magnitude loss
    參考: HiFi-GAN, Parallel WaveGAN
    """

    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [50, 120, 240],
        win_sizes: list = [240, 600, 1200],
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

        # 註冊 windows
        for fft_size, win_size in zip(fft_sizes, win_sizes):
            self.register_buffer(
                f'window_{fft_size}',
                torch.hann_window(win_size)
            )

    def stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_size: int) -> torch.Tensor:
        """計算 STFT magnitude"""
        window = getattr(self, f'window_{fft_size}')

        # 確保 window 在正確的 device
        if window.device != x.device:
            window = window.to(x.device)

        # Padding window if needed
        if win_size < fft_size:
            pad_size = fft_size - win_size
            window = F.pad(window, (pad_size // 2, pad_size - pad_size // 2))

        x_stft = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window[:win_size],
            return_complex=True,
        )

        return torch.abs(x_stft)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: (B, T) 預測音頻
            target: (B, T) 目標音頻

        Returns:
            loss: scalar
            info: dict with component losses
        """
        # 確保長度一致
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # Flatten if needed
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        sc_loss = 0  # Spectral Convergence
        mag_loss = 0  # Magnitude Loss

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            pred_mag = self.stft(pred, fft_size, hop_size, win_size)
            target_mag = self.stft(target, fft_size, hop_size, win_size)

            # Spectral Convergence: ||S_target - S_pred|| / ||S_target||
            sc_loss += torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)

            # Log Magnitude Loss: ||log(S_target) - log(S_pred)||_1
            mag_loss += F.l1_loss(torch.log(target_mag + 1e-8), torch.log(pred_mag + 1e-8))

        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)

        total_loss = sc_loss + mag_loss

        return total_loss, {
            'stft_loss': total_loss.item(),
            'spectral_convergence': sc_loss.item(),
            'magnitude_loss': mag_loss.item(),
        }


class MelSpectrogramLoss(nn.Module):
    """
    Mel Spectrogram Loss

    在 Mel 頻譜上計算 L1 Loss，更接近人耳感知
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = None,
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: (B, T) 預測音頻
            target: (B, T) 目標音頻
        """
        # 確保長度一致
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # Flatten if needed
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        # Move transform to correct device
        self.mel_transform = self.mel_transform.to(pred.device)

        # 計算 Mel spectrogram
        pred_mel = self.mel_transform(pred)
        target_mel = self.mel_transform(target)

        # 數值穩定性：clamp 避免 log(0) 或 log(負數)
        pred_mel = torch.clamp(pred_mel, min=1e-5)
        target_mel = torch.clamp(target_mel, min=1e-5)

        # 檢查 NaN
        if torch.isnan(pred_mel).any() or torch.isnan(target_mel).any():
            # 如果有 NaN，返回 0 loss 避免污染訓練
            return torch.tensor(0.0, device=pred.device, requires_grad=True), {'mel_loss': 0.0}

        # Log Mel Loss
        pred_log_mel = torch.log(pred_mel)
        target_log_mel = torch.log(target_mel)

        mel_loss = F.l1_loss(pred_log_mel, target_log_mel)

        # 最後再檢查一次 NaN
        if torch.isnan(mel_loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=True), {'mel_loss': 0.0}

        return mel_loss, {
            'mel_loss': mel_loss.item(),
        }


class FeatureLoss(nn.Module):
    """Feature-level MSE Loss (原有)"""

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            student_features: (B, C, L)
            teacher_features: (B, C, L)
        """
        loss = F.mse_loss(student_features, teacher_features)
        return loss, {'feature_loss': loss.item()}


class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss for VQ alignment

    確保 features 接近正確的 codebook entry
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_codes: torch.Tensor,
        codebook: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            student_features: (B, C, L) student encoder output
            teacher_codes: (B, L) or (1, B, L) teacher token indices
            codebook: (V, C) codebook embeddings
        """
        B, C, L = student_features.shape

        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]  # (B, L)

        # (B, L, C)
        features = student_features.permute(0, 2, 1).reshape(-1, C)  # (B*L, C)
        codes_flat = teacher_codes.reshape(-1)  # (B*L,)

        # Positive: correct codebook entry
        positive = codebook[codes_flat]  # (B*L, C)

        # Negative: random wrong entry
        V = codebook.shape[0]
        random_indices = torch.randint(0, V, (features.shape[0],), device=features.device)
        # 確保 negative 不等於 positive
        mask = random_indices == codes_flat
        random_indices[mask] = (random_indices[mask] + 1) % V
        negative = codebook[random_indices]  # (B*L, C)

        # Triplet loss: max(0, d(anchor, positive) - d(anchor, negative) + margin)
        d_pos = F.pairwise_distance(features, positive)
        d_neg = F.pairwise_distance(features, negative)

        loss = F.relu(d_pos - d_neg + self.margin).mean()

        return loss, {
            'triplet_loss': loss.item(),
            'd_pos_mean': d_pos.mean().item(),
            'd_neg_mean': d_neg.mean().item(),
        }


class AudioDomainLoss(nn.Module):
    """
    Audio Domain Loss - 組合所有 Loss

    支援:
    1. Audio-level: STFT + Mel Loss
    2. Feature-level: MSE + Triplet Loss
    """

    def __init__(
        self,
        # Audio-level weights
        stft_weight: float = 1.0,
        mel_weight: float = 1.0,
        # Feature-level weights
        feature_weight: float = 0.1,
        triplet_weight: float = 0.1,
        triplet_margin: float = 0.2,
        # STFT config
        stft_fft_sizes: list = [512, 1024, 2048],
        stft_hop_sizes: list = [50, 120, 240],
        stft_win_sizes: list = [240, 600, 1200],
        # Mel config
        mel_n_fft: int = 1024,
        mel_hop_length: int = 256,
        mel_n_mels: int = 80,
        sample_rate: int = 24000,
    ):
        super().__init__()

        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.feature_weight = feature_weight
        self.triplet_weight = triplet_weight

        # Audio losses
        if stft_weight > 0:
            self.stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=stft_fft_sizes,
                hop_sizes=stft_hop_sizes,
                win_sizes=stft_win_sizes,
                sample_rate=sample_rate,
            )

        if mel_weight > 0:
            self.mel_loss = MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_fft=mel_n_fft,
                hop_length=mel_hop_length,
                n_mels=mel_n_mels,
            )

        # Feature losses
        if feature_weight > 0:
            self.feature_loss = FeatureLoss()

        if triplet_weight > 0:
            self.triplet_loss = TripletMarginLoss(margin=triplet_margin)

    def forward(
        self,
        # Audio
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        # Features (optional)
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        teacher_codes: Optional[torch.Tensor] = None,
        codebook: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred_audio: (B, T) 預測音頻 (from student encoder -> decode)
            target_audio: (B, T) 目標音頻 (clean audio)
            student_features: (B, C, L) optional
            teacher_features: (B, C, L) optional
            teacher_codes: (B, L) optional
            codebook: (V, C) optional
        """
        total_loss = 0
        info = {}

        # Audio-level losses
        if self.stft_weight > 0:
            stft_loss, stft_info = self.stft_loss(pred_audio, target_audio)
            total_loss += self.stft_weight * stft_loss
            info.update(stft_info)

        if self.mel_weight > 0:
            mel_loss, mel_info = self.mel_loss(pred_audio, target_audio)
            total_loss += self.mel_weight * mel_loss
            info.update(mel_info)

        # Feature-level losses (optional)
        if self.feature_weight > 0 and student_features is not None and teacher_features is not None:
            feat_loss, feat_info = self.feature_loss(student_features, teacher_features)
            total_loss += self.feature_weight * feat_loss
            info.update(feat_info)

        if self.triplet_weight > 0 and student_features is not None and teacher_codes is not None and codebook is not None:
            trip_loss, trip_info = self.triplet_loss(student_features, teacher_codes, codebook)
            total_loss += self.triplet_weight * trip_loss
            info.update(trip_info)

        info['total_loss'] = total_loss.item()

        return total_loss, info
