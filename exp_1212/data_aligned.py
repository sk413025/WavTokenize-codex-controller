"""
Local exp_1212 compatibility module.

Purpose:
- Provide `AlignedNoisyCleanPairDataset` and `aligned_collate_fn` expected by
  `exp_1226.data_curriculum`.
- Use robust audio loading fallback (`soundfile + scipy`) when torchaudio backend
  cannot load due torchcodec/ffmpeg runtime issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset


def _load_audio_fallback_sf(path: Path, target_sr: int = 24000) -> torch.Tensor:
    import soundfile as sf

    audio_np, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    audio_np = audio_np[:, 0]  # first channel
    if sr != target_sr:
        audio_np = resample_poly(audio_np, target_sr, sr).astype("float32")
    return torch.from_numpy(audio_np)


class AlignedNoisyCleanPairDataset(Dataset):
    def __init__(self, cache_path, max_samples: Optional[int] = None, filter_clean_to_clean: bool = True):
        self.cache_path = Path(cache_path)
        assert self.cache_path.exists(), f"Cache not found: {cache_path}"

        data = torch.load(self.cache_path, weights_only=False)
        samples = data if isinstance(data, list) else [data]

        if filter_clean_to_clean:
            original_count = len(samples)
            samples = [s for s in samples if not self._is_clean_to_clean(s)]
            filtered_count = original_count - len(samples)
            print(f"Filtered {filtered_count} clean→clean samples ({filtered_count/original_count*100:.1f}%)")

        self.samples = samples[:max_samples] if max_samples is not None else samples
        print(f"Loaded {len(self.samples)} samples from {cache_path}")

    def _is_clean_to_clean(self, sample) -> bool:
        noisy_path = str(sample.get("noisy_path", ""))
        clean_path = str(sample.get("clean_path", ""))
        return noisy_path == clean_path or "_clean_" in noisy_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        noisy_audio = sample.get("noisy_audio", sample.get("audio", None))
        clean_audio = sample.get("clean_audio", sample.get("audio", None))

        if noisy_audio is None and "noisy_path" in sample:
            noisy_audio = self._load_audio_from_path(sample["noisy_path"])
        if clean_audio is None and "clean_path" in sample:
            clean_audio = self._load_audio_from_path(sample["clean_path"])

        if noisy_audio is None or clean_audio is None:
            audio = sample.get("audio", sample.get("waveform", None))
            if audio is None:
                raise ValueError(f"Sample {idx} has no audio data or valid paths")
            noisy_audio = audio if noisy_audio is None else noisy_audio
            clean_audio = audio if clean_audio is None else clean_audio

        noisy_audio = noisy_audio.squeeze()
        clean_audio = clean_audio.squeeze()

        min_len = min(len(noisy_audio), len(clean_audio))
        noisy_audio = noisy_audio[:min_len]
        clean_audio = clean_audio[:min_len]

        return {"noisy_audio": noisy_audio, "clean_audio": clean_audio, "length": min_len}

    def _resolve_path(self, audio_path) -> Path:
        audio_path = Path(audio_path)
        if audio_path.is_absolute() and audio_path.exists():
            return audio_path

        filename = audio_path.name
        possible_base_dirs = [
            Path("/home/sbplab/ruizi/WavTokenize/data"),
            Path("/home/sbplab/ruizi/c_code/data"),
        ]

        for base_dir in possible_base_dirs:
            if "_clean_" in filename:
                candidates = [base_dir / "clean" / "box2" / filename, base_dir / "clean" / filename]
            elif "_box_" in filename:
                candidates = [base_dir / "raw" / "box" / filename]
            elif "_papercup_" in filename:
                candidates = [base_dir / "raw" / "papercup" / filename]
            elif "_plastic_" in filename:
                candidates = [base_dir / "raw" / "plastic" / filename]
            else:
                candidates = [base_dir / filename]

            for c in candidates:
                if c.exists():
                    return c

        fallback = self.cache_path.parent.parent / filename
        return fallback

    def _load_audio_from_path(self, audio_path):
        path = self._resolve_path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        target_sr = 24000
        try:
            waveform, sr = torchaudio.load(str(path))
            if sr != target_sr:
                waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
            if waveform.dim() > 1:
                waveform = waveform[0]
            return waveform
        except Exception:
            # Covers torchcodec/ffmpeg runtime issues in some environments.
            return _load_audio_fallback_sf(path, target_sr=target_sr)


def aligned_collate_fn(batch):
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = max(item["noisy_audio"].shape[0] for item in batch)

    noisy_audios = []
    clean_audios = []
    for item in batch:
        noisy = item["noisy_audio"]
        clean = item["clean_audio"]
        if noisy.shape[0] < max_len:
            noisy = F.pad(noisy, (0, max_len - noisy.shape[0]))
        if clean.shape[0] < max_len:
            clean = F.pad(clean, (0, max_len - clean.shape[0]))
        noisy_audios.append(noisy)
        clean_audios.append(clean)

    return {"noisy_audio": torch.stack(noisy_audios), "clean_audio": torch.stack(clean_audios), "lengths": lengths}


def create_aligned_dataloaders(config):
    from exp_1201.config import TRAIN_CACHE, VAL_CACHE

    max_samples = getattr(config, "num_samples", None)
    train_dataset = AlignedNoisyCleanPairDataset(TRAIN_CACHE, max_samples=max_samples)
    val_dataset = AlignedNoisyCleanPairDataset(VAL_CACHE, max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, "num_workers", 0),
        pin_memory=getattr(config, "pin_memory", False),
        collate_fn=aligned_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, "num_workers", 0),
        pin_memory=getattr(config, "pin_memory", False),
        collate_fn=aligned_collate_fn,
    )
    return train_loader, val_loader

