from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class STFTLoss(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
        )
        return torch.abs(spec)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag_hat = self._stft(y_hat)
        mag = self._stft(y)
        sc_loss = torch.norm(mag - mag_hat, p="fro") / (torch.norm(mag, p="fro") + 1e-7)
        log_mag_loss = F.l1_loss(
            torch.log(mag.clamp(min=1e-7)),
            torch.log(mag_hat.clamp(min=1e-7)),
        )
        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512],
        hop_sizes: List[int] = [512, 256, 128],
        win_sizes: List[int] = [2048, 1024, 512],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.stft_losses = nn.ModuleList(
            [STFTLoss(n_fft, hop, win) for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)]
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)

        sc_loss = 0.0
        mag_loss = 0.0
        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(y_hat, y)
            sc_loss += sc
            mag_loss += mag

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss


class MelReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        mel_hat = torch.log(self.mel_spec(y_hat).clamp(min=1e-7))
        mel = torch.log(self.mel_spec(y).clamp(min=1e-7))
        return F.l1_loss(mel_hat, mel)
