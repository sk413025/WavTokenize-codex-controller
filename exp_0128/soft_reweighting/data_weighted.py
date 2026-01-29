"""
exp_0128: TracIn-Weighted DataLoader Creation (實驗 1)

基於 exp_1226/data_curriculum.py，但使用 TracInWeightedSampler。
"""

import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum
from exp_0128.soft_reweighting.tracin_weighted_sampler import TracInWeightedSampler


def create_tracin_weighted_dataloaders(
    train_cache_path,
    val_cache_path,
    tracin_scores_csv: str,
    alpha: float = 0.5,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
):
    """
    創建 TracIn-Weighted DataLoader

    Args:
        train_cache_path: Path to train_cache.pt
        val_cache_path: Path to val_cache.pt
        tracin_scores_csv: Path to TracIn scores CSV
        alpha: Reweighting strength
        batch_size: Batch size
        num_workers: DataLoader workers
        pin_memory: Pin memory

    Returns:
        train_loader: Training DataLoader with TracInWeightedSampler
        val_loader: Validation DataLoader (unchanged)
        train_sampler: TracInWeightedSampler instance
    """
    print("=" * 60)
    print("Creating TracIn-Weighted DataLoaders")
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

    # TracIn-weighted sampler
    train_sampler = TracInWeightedSampler(
        dataset=train_dataset,
        tracin_scores_csv=tracin_scores_csv,
        alpha=alpha,
        batch_size=batch_size,
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

    print("\nTesting create_tracin_weighted_dataloaders...")

    tracin_csv = "exp_0125/tracin_token_collapse_589e6d/tracin_scores_5ckpt.csv"

    train_loader, val_loader, train_sampler = create_tracin_weighted_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        tracin_scores_csv=tracin_csv,
        alpha=0.5,
        batch_size=8,
        num_workers=2,
    )

    print("\n--- Testing first batch ---")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Noisy audio shape: {batch['noisy_audio'].shape}")
        break

    print("\n✅ Test complete!")
