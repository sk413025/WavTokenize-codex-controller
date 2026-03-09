"""
exp_0128: Noise-Balanced Sampling for Token Collapse Fix

目的：
- 解決 exp_k v6 的 token collapse 問題
- TracIn 診斷結果：papercup 材質佔 proponents 57% (vs 全體 33%)
- 方法：強制每個 batch 的 papercup/plastic/box 各 1/3

使用：
    from exp_0128.noise_balanced_sampling.sampler import NoiseBalancedSampler

    train_sampler = NoiseBalancedSampler(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        ...
    )
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator
from pathlib import Path


def extract_noise_type(noisy_path: str) -> str:
    """
    從檔名提取 noise type

    檔名格式: {speaker}_{noise_type}_LDV_{id}.wav
    例如: nor_boy10_box_LDV_132.wav -> box

    Args:
        noisy_path: Noisy audio file path

    Returns:
        noise_type: 'box', 'papercup', or 'plastic'
    """
    filename = Path(noisy_path).stem  # e.g., nor_boy10_box_LDV_132
    parts = filename.split('_')

    # 找到 noise type (在 speaker 和 LDV 之間)
    # Format: {speaker}_{noise_type}_LDV_{id}
    if len(parts) >= 3:
        noise_type = parts[-3]  # e.g., box
        if noise_type in ['box', 'papercup', 'plastic']:
            return noise_type

    # Fallback: 嘗試在整個檔名中搜尋
    filename_lower = filename.lower()
    if 'papercup' in filename_lower:
        return 'papercup'
    elif 'plastic' in filename_lower:
        return 'plastic'
    elif 'box' in filename_lower:
        return 'box'

    raise ValueError(f"Cannot extract noise type from: {noisy_path}")


class NoiseBalancedSampler(Sampler):
    """
    平衡噪音材質的 Sampler

    確保每個 batch 的 papercup/plastic/box 各佔 1/3。

    實作策略：
    1. 將 dataset 按 noise type 分組
    2. 每個 batch 從三組中各抽 batch_size // 3 個樣本
    3. 如果 batch_size 無法整除 3，剩餘位置隨機分配

    Args:
        dataset: CurriculumDataset or AlignedNoisyCleanPairDataset
        batch_size: Batch size
        shuffle: 是否在每個 epoch shuffle
        drop_last: 是否丟棄最後不完整的 batch
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 按 noise type 分組
        self.noise_groups = {
            'box': [],
            'papercup': [],
            'plastic': [],
        }

        print("Grouping samples by noise type...")
        for idx in range(len(dataset)):
            sample = dataset.samples[idx]
            noisy_path = sample['noisy_path']
            noise_type = extract_noise_type(noisy_path)
            self.noise_groups[noise_type].append(idx)

        # 移除空的 noise type groups
        self.noise_groups = {k: v for k, v in self.noise_groups.items() if len(v) > 0}
        self.active_noise_types = sorted(self.noise_groups.keys())
        self.num_noise_types = len(self.active_noise_types)

        # 統計
        total = sum(len(v) for v in self.noise_groups.values())
        print(f"Noise type distribution:")
        for noise_type, indices in self.noise_groups.items():
            count = len(indices)
            pct = count / total * 100
            print(f"  {noise_type:8s}: {count:5d} ({pct:5.1f}%)")

        # 每個 batch 各 noise type 的樣本數
        self.samples_per_type = batch_size // self.num_noise_types
        self.remainder = batch_size % self.num_noise_types

        print(f"\nBatch composition:")
        print(f"  Batch size: {batch_size}")
        print(f"  Active noise types: {self.num_noise_types} ({', '.join(self.active_noise_types)})")
        print(f"  Samples per type: {self.samples_per_type}")
        print(f"  Remainder: {self.remainder} (will be randomly distributed)")

        # 計算總 batch 數
        min_group_size = min(len(v) for v in self.noise_groups.values())
        self.num_batches = min_group_size // self.samples_per_type
        if self.drop_last and min_group_size % self.samples_per_type != 0:
            self.num_batches -= 1

        print(f"  Total batches per epoch: {self.num_batches}")

    def __iter__(self) -> Iterator[int]:
        # Shuffle each group
        if self.shuffle:
            shuffled_groups = {
                k: np.random.permutation(v).tolist()
                for k, v in self.noise_groups.items()
            }
        else:
            shuffled_groups = {k: v.copy() for k, v in self.noise_groups.items()}

        # Create balanced batches
        batch_list = []
        for batch_idx in range(self.num_batches):
            batch_indices = []

            # Sample from each active noise type
            for noise_type in self.active_noise_types:
                start = batch_idx * self.samples_per_type
                end = start + self.samples_per_type
                batch_indices.extend(shuffled_groups[noise_type][start:end])

            # Handle remainder
            if self.remainder > 0:
                # Randomly select noise types for remainder positions
                remainder_types = np.random.choice(self.active_noise_types,
                                                   size=self.remainder, replace=True)
                for noise_type in remainder_types:
                    # Use next available sample from this type
                    idx = batch_idx * self.samples_per_type + self.samples_per_type
                    if idx < len(shuffled_groups[noise_type]):
                        batch_indices.append(shuffled_groups[noise_type][idx])

            # Shuffle within batch (optional, for randomness)
            if self.shuffle:
                np.random.shuffle(batch_indices)

            batch_list.extend(batch_indices)

        return iter(batch_list)

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


# ==================== 測試 ====================
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from exp_1226.data_curriculum import CurriculumDataset
    from exp_1201.config import TRAIN_CACHE

    print("=" * 60)
    print("Testing NoiseBalancedSampler")
    print("=" * 60)

    # Load dataset
    dataset = CurriculumDataset(
        TRAIN_CACHE,
        max_samples=1000,
        filter_clean_to_clean=True,
        compute_snr=False,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Create sampler
    sampler = NoiseBalancedSampler(
        dataset=dataset,
        batch_size=12,
        shuffle=True,
    )

    print(f"\nSampler length: {len(sampler)}")

    # Test first batch
    print("\n--- Testing first batch ---")
    indices = list(sampler)[:12]
    print(f"First batch indices: {indices}")

    # Check noise type distribution
    noise_counts = {'box': 0, 'papercup': 0, 'plastic': 0}
    for idx in indices:
        sample = dataset.samples[idx]
        noise_type = extract_noise_type(sample['noisy_path'])
        noise_counts[noise_type] += 1

    print(f"\nFirst batch noise distribution:")
    for noise_type, count in noise_counts.items():
        print(f"  {noise_type}: {count}")

    print("\n✅ Test complete!")
