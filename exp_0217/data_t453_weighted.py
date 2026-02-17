"""
exp_0217: T453 Token 加權 Curriculum Dataset

背景：
  Token 453 是 WavTokenizer codebook 中最常見的 token，佔訓練集 clean_tokens 約 13%。
  這類 token 對應較為靜態/無聲的語音段，若早期訓練過多看到這類樣本，
  模型可能過度偏向輸出 T453，降低 token diversity。

策略：
  1. 預先計算每個樣本的 T453 濃度（clean_tokens 中 token=453 的比例）
  2. 訓練初期對高 T453 樣本降低採樣權重
  3. 隨訓練進行，T453 權重漸增至 1.0（完全平等採樣）

加權公式：
  w(sample, epoch_progress) = 1.0 - (1.0 - min_weight) * t453_ratio * (1 - epoch_progress)

  epoch_progress = current_epoch / total_epochs (0.0 ~ 1.0)

  例：t453_ratio=0.4, min_weight=0.3, epoch_progress=0.0
      → w = 1.0 - 0.7 * 0.4 = 0.72
  例：t453_ratio=0.4, min_weight=0.3, epoch_progress=0.5
      → w = 1.0 - 0.7 * 0.4 * 0.5 = 0.86
  例：t453_ratio=0.4, epoch_progress=1.0
      → w = 1.0 (完全平等)

實作方式：
  使用 WeightedRandomSampler，每次更新 epoch 時重新計算 weights。
  與 AugmentedCurriculumDataset 組合使用。
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_0216.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum

TARGET_TOKEN = 453   # T453


class T453WeightedSampler:
    """動態調整 T453 高濃度樣本的採樣權重。

    Args:
        dataset: AugmentedCurriculumDataset instance
        total_epochs: 總訓練 epoch 數
        min_weight: 第 0 epoch 時高 T453 樣本的最低相對權重（0~1）
        ramp_epochs: 從 min_weight 線性升到 1.0 需要的 epoch 數
                     若為 None，則整個訓練過程中線性升到 total_epochs
        batch_size: batch 大小
        num_samples_per_epoch: 每 epoch 採樣數（None 表示等於 dataset 大小）
    """

    def __init__(
        self,
        dataset: AugmentedCurriculumDataset,
        total_epochs: int,
        min_weight: float = 0.2,
        ramp_epochs: Optional[int] = None,
        batch_size: int = 8,
        num_samples_per_epoch: Optional[int] = None,
    ):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.min_weight = min_weight
        self.ramp_epochs = ramp_epochs if ramp_epochs is not None else total_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples_per_epoch or len(dataset)
        self.current_epoch = 0

        # Pre-compute T453 ratios from dataset (using clean_tokens from cache)
        self.t453_ratios = self._compute_t453_ratios()

        print(f"\n[T453WeightedSampler] Initialized:")
        print(f"  Target token: T{TARGET_TOKEN}")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  T453 ratio stats: mean={self.t453_ratios.mean():.4f}, "
              f"max={self.t453_ratios.max():.4f}")
        print(f"  min_weight: {min_weight:.2f}")
        print(f"  ramp_epochs: {self.ramp_epochs}")
        print(f"  Samples per epoch: {self.num_samples}")

        # Statistics
        high_t453 = (self.t453_ratios > 0.3).sum()
        print(f"  Samples with T453>30%: {high_t453} ({100*high_t453/len(dataset):.1f}%)")

    def _compute_t453_ratios(self) -> np.ndarray:
        """從 dataset.samples 中讀取 clean_tokens，計算 T453 比例。

        直接從快取讀取 token，不需要載入音訊，速度快。
        """
        print(f"Computing T453 ratios for {len(self.dataset.samples)} samples...")
        ratios = []

        for i, sample in enumerate(self.dataset.samples):
            tokens = sample.get('clean_tokens', None)
            if tokens is not None:
                total = tokens.numel()
                count = (tokens == TARGET_TOKEN).sum().item()
                ratios.append(count / max(total, 1))
            else:
                ratios.append(0.0)

            if (i + 1) % 2000 == 0:
                print(f"  Processed {i+1}/{len(self.dataset.samples)}")

        return np.array(ratios, dtype=np.float32)

    def _compute_weights(self, epoch: int) -> np.ndarray:
        """計算當前 epoch 的採樣權重。

        Args:
            epoch: 當前 epoch（從 0 開始）

        Returns:
            weights: [N] 每個樣本的採樣權重
        """
        # epoch_progress: 0.0（剛開始）→ 1.0（ramp 結束）
        epoch_progress = min(1.0, epoch / max(self.ramp_epochs, 1))

        # w = 1.0 - (1 - min_w) * t453_ratio * (1 - progress)
        # 當 progress=0: 高T453樣本 weight 最低
        # 當 progress=1: 所有樣本 weight=1.0
        penalty_scale = (1.0 - self.min_weight) * (1.0 - epoch_progress)
        weights = 1.0 - penalty_scale * self.t453_ratios

        # 確保最小值 > 0（WeightedRandomSampler 要求 weight > 0）
        weights = np.clip(weights, 1e-4, 1.0)
        return weights

    def get_sampler(self, epoch: int) -> WeightedRandomSampler:
        """建立當前 epoch 的 WeightedRandomSampler。

        Args:
            epoch: 當前 epoch

        Returns:
            WeightedRandomSampler instance
        """
        self.current_epoch = epoch
        weights = self._compute_weights(epoch)

        epoch_progress = min(1.0, epoch / max(self.ramp_epochs, 1))
        avg_w_high = float(weights[self.t453_ratios > 0.3].mean()) if (self.t453_ratios > 0.3).any() else 1.0
        print(f"  [T453 Sampler] Epoch {epoch}: progress={epoch_progress:.2f}, "
              f"high-T453 avg weight={avg_w_high:.3f}")

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=self.num_samples,
            replacement=True,
        )

    def is_fully_ramped(self, epoch: int) -> bool:
        """是否已完成 ramp（所有樣本等權）"""
        return epoch >= self.ramp_epochs


def create_t453_weighted_dataloaders(
    train_cache_path,
    val_cache_path,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    # T453 weighting 參數
    total_epochs: int = 300,
    t453_min_weight: float = 0.2,
    t453_ramp_epochs: Optional[int] = None,
    # 增強參數 (與 exp_0216 相同)
    snr_remix_prob: float = 0.5,
    snr_remix_range: tuple = (-5.0, 25.0),
    random_gain_prob: float = 0.3,
    random_gain_db: float = 3.0,
    random_crop_prob: float = 0.3,
    random_crop_min_ratio: float = 0.7,
    time_stretch_prob: float = 0.2,
    time_stretch_range: tuple = (0.95, 1.05),
):
    """建立 T453-aware 加權採樣的 DataLoader。

    注意：返回 (train_dataset, val_loader, t453_sampler)
          train DataLoader 需在每個 epoch 前用 t453_sampler.get_sampler(epoch) 更新。

    Args:
        train_cache_path: 訓練資料快取路徑
        val_cache_path: 驗證資料快取路徑
        batch_size: 批次大小
        num_workers: DataLoader worker 數
        pin_memory: 是否 pin memory
        total_epochs: 總訓練 epoch（用於計算 ramp）
        t453_min_weight: 第 0 epoch 時高 T453 樣本的最小相對權重
        t453_ramp_epochs: T453 weight 升到 1.0 所需的 epoch 數（None=total_epochs）
        snr_remix_prob ~ time_stretch_range: 資料增強參數

    Returns:
        (train_dataset, val_loader, t453_sampler)
    """
    # Training dataset（帶資料增強，不用 SNR curriculum）
    train_dataset = AugmentedCurriculumDataset(
        train_cache_path,
        augment=True,
        snr_remix_prob=snr_remix_prob,
        snr_remix_range=snr_remix_range,
        random_gain_prob=random_gain_prob,
        random_gain_db=random_gain_db,
        random_crop_prob=random_crop_prob,
        random_crop_min_ratio=random_crop_min_ratio,
        time_stretch_prob=time_stretch_prob,
        time_stretch_range=time_stretch_range,
        filter_clean_to_clean=True,
        compute_snr=False,   # T453 weighting 取代 SNR curriculum
    )

    # Validation dataset（不增強）
    val_dataset = AugmentedCurriculumDataset(
        val_cache_path,
        augment=False,
        filter_clean_to_clean=True,
        compute_snr=False,
    )

    # T453 sampler（跨 epoch 持續使用同一個 sampler 物件）
    t453_sampler = T453WeightedSampler(
        dataset=train_dataset,
        total_epochs=total_epochs,
        min_weight=t453_min_weight,
        ramp_epochs=t453_ramp_epochs,
        batch_size=batch_size,
        num_samples_per_epoch=len(train_dataset),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_curriculum,
        pin_memory=pin_memory,
    )

    return train_dataset, val_loader, t453_sampler


def make_train_loader(
    train_dataset: AugmentedCurriculumDataset,
    t453_sampler: T453WeightedSampler,
    epoch: int,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """每個 epoch 開始時呼叫，產生當前 epoch 的 train DataLoader。

    用法：
        train_dataset, val_loader, t453_sampler = create_t453_weighted_dataloaders(...)

        for epoch in range(total_epochs):
            train_loader = make_train_loader(train_dataset, t453_sampler, epoch)
            for batch in train_loader:
                ...

    Args:
        train_dataset: 訓練資料集
        t453_sampler: T453WeightedSampler（計算權重）
        epoch: 當前 epoch 號
        batch_size: 批次大小
        num_workers: DataLoader worker 數
        pin_memory: 是否 pin memory

    Returns:
        DataLoader with WeightedRandomSampler for this epoch
    """
    weighted_sampler = t453_sampler.get_sampler(epoch)

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=weighted_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_curriculum,
        pin_memory=pin_memory,
    )


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-feature-analysis')
    from exp_1201.config import TRAIN_CACHE, VAL_CACHE

    print("=" * 60)
    print("Testing T453 Weighted Sampler")
    print("=" * 60)

    train_dataset, val_loader, t453_sampler = create_t453_weighted_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=4,
        num_workers=0,
        total_epochs=300,
        t453_min_weight=0.2,
        t453_ramp_epochs=150,
    )

    # Epoch 0: 高T453樣本降權
    print("\n--- Epoch 0 ---")
    loader_e0 = make_train_loader(train_dataset, t453_sampler, epoch=0,
                                  batch_size=4, num_workers=0)
    batch = next(iter(loader_e0))
    print(f"  batch shape: {batch['noisy_audio'].shape}")

    # Epoch 150: 完全平等
    print("\n--- Epoch 150 ---")
    loader_e150 = make_train_loader(train_dataset, t453_sampler, epoch=150,
                                    batch_size=4, num_workers=0)
    batch = next(iter(loader_e150))
    print(f"  batch shape: {batch['noisy_audio'].shape}")

    # 驗證 weight 計算
    print("\n--- Weight Verification ---")
    for epoch in [0, 50, 100, 150, 200, 300]:
        weights = t453_sampler._compute_weights(epoch)
        high_t453_mask = t453_sampler.t453_ratios > 0.3
        if high_t453_mask.any():
            avg_high = weights[high_t453_mask].mean()
            avg_low = weights[~high_t453_mask].mean()
            print(f"  Epoch {epoch:3d}: high-T453 avg_w={avg_high:.4f}, "
                  f"low-T453 avg_w={avg_low:.4f}, ratio={avg_high/avg_low:.3f}")

    print("\n✅ T453 Weighted Sampler test passed!")
