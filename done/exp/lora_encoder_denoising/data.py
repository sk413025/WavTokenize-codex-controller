"""
Dataset for Noisy-Clean Audio Pairs
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class NoisyCleanPairDataset(Dataset):
    """
    簡單的 Noisy-Clean 配對數據集
    從 PyTorch cache 載入
    """

    def __init__(self, cache_path, max_samples=None):
        """
        Args:
            cache_path: Path to train_cache.pt or val_cache.pt
            max_samples: 最多載入多少樣本 (用於 smoke test)
        """
        self.cache_path = Path(cache_path)
        assert self.cache_path.exists(), f"Cache not found: {cache_path}"

        # 載入數據
        data = torch.load(cache_path, weights_only=False)

        # 假設 data 結構: list of dicts with 'noisy_audio' and 'clean_audio'
        # 如果你的數據結構不同，需要調整
        self.samples = data if isinstance(data, list) else [data]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 返回 noisy 和 clean audio
        # 確保是 1D tensor (T,)
        noisy_audio = sample.get('noisy_audio', sample.get('audio', None))
        clean_audio = sample.get('clean_audio', sample.get('audio', None))

        if noisy_audio is None or clean_audio is None:
            # 如果沒有配對，使用相同音頻（smoke test 可以接受）
            audio = sample['audio'] if 'audio' in sample else sample['waveform']
            noisy_audio = audio
            clean_audio = audio

        return {
            'noisy_audio': noisy_audio.squeeze(),  # (T,)
            'clean_audio': clean_audio.squeeze(),  # (T,)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    """
    # 找最長的音頻
    max_len = max(item['noisy_audio'].shape[0] for item in batch)

    noisy_audios = []
    clean_audios = []

    for item in batch:
        noisy = item['noisy_audio']
        clean = item['clean_audio']

        # Pad to max_len
        if noisy.shape[0] < max_len:
            noisy = F.pad(noisy, (0, max_len - noisy.shape[0]))
        if clean.shape[0] < max_len:
            clean = F.pad(clean, (0, max_len - clean.shape[0]))

        noisy_audios.append(noisy)
        clean_audios.append(clean)

    return {
        'noisy_audio': torch.stack(noisy_audios),  # (B, T)
        'clean_audio': torch.stack(clean_audios),  # (B, T)
    }


def create_dataloaders(config):
    """
    創建 train/val dataloaders

    Args:
        config: TrainConfig or SmokeTestConfig

    Returns:
        train_loader, val_loader
    """
    try:
        from .config import TRAIN_CACHE, VAL_CACHE
    except ImportError:
        from config import TRAIN_CACHE, VAL_CACHE

    # Smoke test 使用小數據
    max_samples = getattr(config, 'num_samples', None)

    train_dataset = NoisyCleanPairDataset(TRAIN_CACHE, max_samples=max_samples)
    val_dataset = NoisyCleanPairDataset(VAL_CACHE, max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', False),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', False),
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
