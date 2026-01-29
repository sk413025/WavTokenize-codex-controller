"""
exp_0128: TracIn-Guided Weighted Sampler (實驗 1)

目的：
- 使用 TracIn influence scores 對訓練樣本做 soft reweighting
- 對高 influence (proponents) 樣本降權，而非硬刪除
- weight = 1 / (1 + α × tracin_score)

使用：
    from exp_0128.soft_reweighting.tracin_weighted_sampler import TracInWeightedSampler

    train_sampler = TracInWeightedSampler(
        dataset=train_dataset,
        tracin_scores_csv='exp_0125/.../tracin_scores_5ckpt.csv',
        alpha=0.5,
        batch_size=8,
    )
"""

import torch
from torch.utils.data import Sampler, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator


class TracInWeightedSampler(Sampler):
    """
    基於 TracIn influence scores 的加權採樣器

    實作策略：
    1. 讀取 TracIn scores (aggregated across checkpoints)
    2. 計算樣本權重：weight = 1 / (1 + α × score)
       - 高 score (proponents) → 低 weight
       - 低 score (opponents) → 高 weight
    3. 使用 WeightedRandomSampler

    Args:
        dataset: CurriculumDataset or AlignedNoisyCleanPairDataset
        tracin_scores_csv: Path to TracIn scores CSV (e.g., tracin_scores_5ckpt.csv)
        alpha: Reweighting strength (higher α = stronger downweighting)
        batch_size: Batch size
        replacement: Whether to sample with replacement
    """

    def __init__(
        self,
        dataset,
        tracin_scores_csv: str,
        alpha: float = 0.5,
        batch_size: int = 8,
        replacement: bool = True,
    ):
        self.dataset = dataset
        self.alpha = alpha
        self.batch_size = batch_size
        self.replacement = replacement

        print(f"Loading TracIn scores from: {tracin_scores_csv}")
        self.tracin_df = pd.read_csv(tracin_scores_csv)

        # Aggregate scores across checkpoints (if multiple checkpoints)
        # Format: train_idx, checkpoint, score
        print(f"TracIn CSV shape: {self.tracin_df.shape}")
        print(f"Columns: {self.tracin_df.columns.tolist()}")

        # Group by train_index and aggregate scores
        train_idx_col = 'train_index' if 'train_index' in self.tracin_df.columns else 'train_idx'

        if 'checkpoint' in self.tracin_df.columns:
            # Multiple checkpoints: aggregate (mean)
            self.aggregated_scores = (
                self.tracin_df.groupby(train_idx_col)['score']
                .mean()
                .to_dict()
            )
            print(f"Aggregated scores across checkpoints for {len(self.aggregated_scores)} samples")
        else:
            # Single checkpoint
            self.aggregated_scores = dict(
                zip(self.tracin_df[train_idx_col], self.tracin_df['score'])
            )
            print(f"Loaded scores for {len(self.aggregated_scores)} samples")

        # Compute weights for all dataset samples
        self.weights = self._compute_weights()

        # Stats
        print(f"\nWeight statistics:")
        weights_array = np.array(list(self.weights.values()))
        print(f"  Min: {weights_array.min():.6f}")
        print(f"  Max: {weights_array.max():.6f}")
        print(f"  Mean: {weights_array.mean():.6f}")
        print(f"  Std: {weights_array.std():.6f}")

        # Create WeightedRandomSampler
        weight_list = [self.weights.get(i, 1.0) for i in range(len(dataset))]
        self.sampler = WeightedRandomSampler(
            weights=weight_list,
            num_samples=len(dataset),
            replacement=replacement,
        )

        print(f"\nTracInWeightedSampler initialized:")
        print(f"  Alpha: {alpha}")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Num samples per epoch: {len(self.sampler)}")

    def _compute_weights(self):
        """
        計算每個樣本的權重

        weight = 1 / (1 + α × tracin_score)

        - tracin_score > 0 (proponents) → weight < 1 (降權)
        - tracin_score < 0 (opponents) → weight > 1 (提權)
        - tracin_score = 0 → weight = 1 (不變)
        """
        weights = {}

        for idx in range(len(self.dataset)):
            if idx in self.aggregated_scores:
                score = self.aggregated_scores[idx]
                weight = 1.0 / (1.0 + self.alpha * score)
            else:
                # 沒有 TracIn score 的樣本，給予預設權重 1.0
                weight = 1.0

            weights[idx] = weight

        return weights

    def __iter__(self) -> Iterator[int]:
        return iter(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)


# ==================== 測試 ====================
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
    sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

    from exp_1226.data_curriculum import CurriculumDataset
    from exp_1201.config import TRAIN_CACHE

    print("=" * 60)
    print("Testing TracInWeightedSampler")
    print("=" * 60)

    # Load dataset
    dataset = CurriculumDataset(
        TRAIN_CACHE,
        max_samples=2000,  # Only use samples with TracIn scores
        filter_clean_to_clean=True,
        compute_snr=False,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Create sampler
    tracin_csv = "exp_0125/tracin_token_collapse_589e6d/tracin_scores_5ckpt.csv"

    sampler = TracInWeightedSampler(
        dataset=dataset,
        tracin_scores_csv=tracin_csv,
        alpha=0.5,
        batch_size=8,
    )

    print(f"\nSampler length: {len(sampler)}")

    # Test sampling
    print("\n--- Testing sampling ---")
    indices = list(sampler)[:20]
    print(f"First 20 sampled indices: {indices}")

    print("\n✅ Test complete!")
