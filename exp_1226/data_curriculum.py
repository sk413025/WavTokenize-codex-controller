"""
exp_1226: Curriculum Learning Dataset

概念：
- 從簡單樣本開始訓練，逐步增加難度
- 難度定義：SNR (Signal-to-Noise Ratio)
  - 高 SNR = 低噪音 = 簡單
  - 低 SNR = 高噪音 = 困難

實作策略：
1. Curriculum (從易到難): 先訓練簡單樣本，逐步加入困難樣本
2. Anti-Curriculum (從難到易): 先訓練困難樣本，再加入簡單樣本

繼承 exp_1212/data_aligned.py 的資料載入邏輯
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
import numpy as np
from typing import Optional, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn


def estimate_snr(noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> float:
    """
    估計 SNR (dB)

    SNR = 10 * log10(P_signal / P_noise)
    where P_noise = P_noisy - P_clean (假設 noisy = clean + noise)
    """
    if noisy_audio.dim() > 1:
        noisy_audio = noisy_audio.squeeze()
    if clean_audio.dim() > 1:
        clean_audio = clean_audio.squeeze()

    # 確保長度一致
    min_len = min(len(noisy_audio), len(clean_audio))
    noisy_audio = noisy_audio[:min_len]
    clean_audio = clean_audio[:min_len]

    # 計算 power
    signal_power = (clean_audio ** 2).mean()
    noise = noisy_audio - clean_audio
    noise_power = (noise ** 2).mean()

    # 避免除以零
    if noise_power < 1e-10:
        return 100.0  # 幾乎無噪音

    snr = 10 * torch.log10(signal_power / noise_power + 1e-10)
    return snr.item()


class CurriculumDataset(AlignedNoisyCleanPairDataset):
    """
    支援 Curriculum Learning 的資料集

    繼承 AlignedNoisyCleanPairDataset，增加：
    1. 預先計算所有樣本的 SNR
    2. 根據 curriculum phase 過濾樣本
    """

    def __init__(self,
                 cache_path,
                 max_samples: Optional[int] = None,
                 filter_clean_to_clean: bool = True,
                 compute_snr: bool = True):
        """
        Args:
            cache_path: Path to train_cache.pt or val_cache.pt
            max_samples: 最多載入多少樣本
            filter_clean_to_clean: 是否過濾掉 clean→clean 樣本
            compute_snr: 是否計算 SNR (用於 curriculum)
        """
        super().__init__(cache_path, max_samples, filter_clean_to_clean)

        self.snr_values = None
        self.difficulty_order = None

        # 計算 SNR
        if compute_snr:
            self._compute_snr_values()

    def _compute_snr_values(self):
        """計算所有樣本的 SNR"""
        print("Computing SNR values for curriculum learning...")
        snr_values = []
        valid_indices = []

        for i in range(len(self.samples)):
            try:
                item = super().__getitem__(i)
                noisy = item['noisy_audio']
                clean = item['clean_audio']
                snr = estimate_snr(noisy, clean)
                snr_values.append(snr)
                valid_indices.append(i)
            except Exception as e:
                # 無法載入的樣本給予中等 SNR
                snr_values.append(10.0)
                valid_indices.append(i)

            if (i + 1) % 500 == 0:
                print(f"  Computed SNR for {i + 1}/{len(self.samples)} samples")

        self.snr_values = np.array(snr_values)

        # 計算難度排序 (SNR 高 = 簡單，排後面)
        # difficulty_order[0] = 最難 (最低 SNR)
        self.difficulty_order = np.argsort(self.snr_values)

        print(f"SNR stats: min={self.snr_values.min():.1f}, max={self.snr_values.max():.1f}, "
              f"mean={self.snr_values.mean():.1f} dB")

    def get_curriculum_indices(self, phase: float, mode: str = 'curriculum') -> List[int]:
        """
        根據 curriculum phase 獲取樣本索引

        Args:
            phase: 0.0 ~ 1.0，表示訓練進度
            mode: 'curriculum' (易到難) or 'anti_curriculum' (難到易)

        Returns:
            indices: 這個 phase 應該使用的樣本索引
        """
        if self.difficulty_order is None:
            return list(range(len(self.samples)))

        n_samples = len(self.samples)
        n_active = max(1, int(n_samples * phase))

        if mode == 'curriculum':
            # 從易到難：phase 越大，包含越多困難樣本
            # difficulty_order 是從低 SNR (難) 到高 SNR (易)
            # 所以反過來取 (從高 SNR 開始)
            active_indices = self.difficulty_order[::-1][:n_active]
        elif mode == 'anti_curriculum':
            # 從難到易：phase 越大，包含越多簡單樣本
            active_indices = self.difficulty_order[:n_active]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return list(active_indices)

    def __getitem__(self, idx):
        """重寫以加入 SNR 資訊"""
        item = super().__getitem__(idx)

        # 添加 SNR 資訊
        if self.snr_values is not None and idx < len(self.snr_values):
            item['snr'] = self.snr_values[idx]

        return item


class CurriculumSampler(Sampler):
    """
    Curriculum Learning Sampler

    根據訓練進度調整採樣策略
    """

    def __init__(self,
                 dataset: CurriculumDataset,
                 mode: str = 'curriculum',
                 initial_phase: float = 0.3,
                 phase_increment: float = 0.1,
                 shuffle: bool = True):
        """
        Args:
            dataset: CurriculumDataset instance
            mode: 'curriculum' or 'anti_curriculum'
            initial_phase: 初始使用多少比例的資料
            phase_increment: 每次 advance() 增加多少比例
            shuffle: 是否打亂順序
        """
        self.dataset = dataset
        self.mode = mode
        self.current_phase = initial_phase
        self.phase_increment = phase_increment
        self.shuffle = shuffle

        self._update_indices()

    def _update_indices(self):
        """更新當前 phase 的樣本索引"""
        self.indices = self.dataset.get_curriculum_indices(
            self.current_phase, self.mode
        )
        if self.shuffle:
            np.random.shuffle(self.indices)

    def advance_phase(self):
        """進入下一個 phase"""
        old_phase = self.current_phase
        self.current_phase = min(1.0, self.current_phase + self.phase_increment)
        self._update_indices()
        print(f"Curriculum phase: {old_phase:.1%} -> {self.current_phase:.1%} "
              f"({len(self.indices)} samples)")

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_fn_curriculum(batch):
    """
    Curriculum 專用 collate function

    基於 aligned_collate_fn，增加 SNR 資訊
    """
    # 使用基礎的 aligned_collate_fn
    result = aligned_collate_fn(batch)

    # 添加 SNR 資訊
    if 'snr' in batch[0]:
        result['snr'] = torch.tensor([item['snr'] for item in batch])

    return result


def create_curriculum_dataloaders(
    train_cache_path,
    val_cache_path,
    batch_size: int = 8,
    num_workers: int = 4,
    curriculum_mode: str = 'curriculum',
    initial_phase: float = 0.3,
    phase_increment: float = 0.1,
    compute_snr: bool = True,
    pin_memory: bool = True,
    **kwargs
):
    """
    創建 Curriculum Learning 的 DataLoader

    Returns:
        train_loader: Training DataLoader with CurriculumSampler
        val_loader: Validation DataLoader (使用全部資料)
        train_sampler: CurriculumSampler (用於 advance_phase)
    """
    # Training dataset
    train_dataset = CurriculumDataset(
        train_cache_path,
        filter_clean_to_clean=True,
        compute_snr=compute_snr,
    )

    # Validation dataset (不需要 curriculum)
    val_dataset = CurriculumDataset(
        val_cache_path,
        filter_clean_to_clean=True,
        compute_snr=False,  # val 不需要 SNR
    )

    # Curriculum sampler
    train_sampler = CurriculumSampler(
        train_dataset,
        mode=curriculum_mode,
        initial_phase=initial_phase,
        phase_increment=phase_increment,
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

    return train_loader, val_loader, train_sampler


# ==================== 測試 ====================
if __name__ == '__main__':
    from exp_1201.config import TRAIN_CACHE, VAL_CACHE

    print("=" * 60)
    print("Testing Curriculum Learning Dataset")
    print("=" * 60)

    # 測試 CurriculumDataset
    print("\n--- Testing CurriculumDataset ---")
    dataset = CurriculumDataset(
        TRAIN_CACHE,
        max_samples=100,
        compute_snr=True,
    )

    print(f"Dataset size: {len(dataset)}")
    if dataset.snr_values is not None:
        print(f"SNR range: {dataset.snr_values.min():.1f} ~ {dataset.snr_values.max():.1f} dB")

    # 測試不同 phase
    for phase in [0.3, 0.5, 0.7, 1.0]:
        indices = dataset.get_curriculum_indices(phase, mode='curriculum')
        print(f"Phase {phase:.0%}: {len(indices)} samples")

    # 測試 DataLoader
    print("\n--- Testing Curriculum DataLoader ---")
    train_loader, val_loader, sampler = create_curriculum_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=4,
        num_workers=0,
        initial_phase=0.3,
        compute_snr=True,
    )

    print(f"Initial phase: {len(sampler)} samples")

    batch = next(iter(train_loader))
    print(f"Batch noisy_audio shape: {batch['noisy_audio'].shape}")
    print(f"Batch lengths: {batch['lengths']}")
    if 'snr' in batch:
        print(f"Batch SNR: {batch['snr']}")

    # 測試 advance_phase
    sampler.advance_phase()
    print(f"After advance: {len(sampler)} samples")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
