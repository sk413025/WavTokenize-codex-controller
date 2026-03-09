"""
exp_0128: Noise-Balanced DataLoader Creation

基於 exp_1226/data_curriculum.py，但使用 NoiseBalancedSampler 取代 CurriculumSampler。

目的：
- 保持 exp_k v6 的基本配置不變
- 只修改 sampler：CurriculumSampler → NoiseBalancedSampler
- 強制 papercup/plastic/box 各 1/3
"""

import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum
from exp_0128.noise_balanced_sampling.sampler import NoiseBalancedSampler


def create_noise_balanced_dataloaders(
    train_cache_path,
    val_cache_path,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
):
    """
    創建 Noise-Balanced DataLoader

    與 create_curriculum_dataloaders 的差異：
    - 使用 NoiseBalancedSampler 取代 CurriculumSampler
    - 不需要 curriculum_mode, initial_phase, phase_increment
    - 不計算 SNR (compute_snr=False)

    Args:
        train_cache_path: Path to train_cache.pt
        val_cache_path: Path to val_cache.pt
        batch_size: Batch size
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster transfer

    Returns:
        train_loader: Training DataLoader with NoiseBalancedSampler
        val_loader: Validation DataLoader (unchanged)
        train_sampler: NoiseBalancedSampler instance
    """
    print("=" * 60)
    print("Creating Noise-Balanced DataLoaders")
    print("=" * 60)

    # Training dataset (same config as exp_k v6)
    train_dataset = CurriculumDataset(
        train_cache_path,
        filter_clean_to_clean=True,  # Filter clean-to-clean (exp_k v6 uses True)
        compute_snr=False,  # exp_k v6 uses False
    )

    # Validation dataset (same config as exp_k v6)
    val_dataset = CurriculumDataset(
        val_cache_path,
        filter_clean_to_clean=True,  # Filter clean-to-clean (exp_k v6 uses True)
        compute_snr=False,  # exp_k v6 uses False
    )

    # Noise-balanced sampler
    train_sampler = NoiseBalancedSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_curriculum,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_curriculum,
        pin_memory=pin_memory,
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print("=" * 60)

    return train_loader, val_loader, train_sampler


# ==================== 測試 ====================
if __name__ == '__main__':
    from exp_1201.config import TRAIN_CACHE, VAL_CACHE

    print("\nTesting create_noise_balanced_dataloaders...")

    train_loader, val_loader, train_sampler = create_noise_balanced_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=8,
        num_workers=2,
    )

    print("\n--- Testing first batch ---")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Noisy audio shape: {batch['noisy_audio'].shape}")
        print(f"Clean audio shape: {batch['clean_audio'].shape}")
        break

    print("\n✅ Test complete!")
