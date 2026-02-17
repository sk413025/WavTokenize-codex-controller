"""
exp_0216: 資料增強 Dataset

核心增強策略（應用於 train，不影響 val）：
1. SNR Remix: 分離 noise 成分，以隨機 SNR 重新混合
2. Random Gain: 隨機整體增益 (±3 dB)
3. Random Crop: 從 noisy/clean pair 隨機截取子段
4. Time Stretch (輕度): 0.95x ~ 1.05x 同步應用到 noisy+clean

繼承 exp_1226/data_curriculum.py 的 CurriculumDataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import random
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1226.data_curriculum import (
    CurriculumDataset,
    CurriculumSampler,
    collate_fn_curriculum,
)


class AugmentedCurriculumDataset(CurriculumDataset):
    """帶有資料增強功能的 CurriculumDataset。

    在訓練時對每個 noisy/clean pair 應用隨機增強，
    有效增加資料多樣性以緩解 overfitting。

    Args:
        cache_path: 資料快取路徑
        augment: 是否啟用增強（train=True, val=False）
        snr_remix_prob: SNR Remix 的機率
        snr_remix_range: SNR Remix 的 SNR 範圍 (dB)
        random_gain_prob: Random Gain 的機率
        random_gain_db: Random Gain 的範圍 (dB)
        random_crop_prob: Random Crop 的機率
        random_crop_min_ratio: 隨機裁切最小比例
        time_stretch_prob: Time Stretch 的機率
        time_stretch_range: Time Stretch 範圍
        max_samples: 最大樣本數
        filter_clean_to_clean: 是否過濾 clean→clean
        compute_snr: 是否計算 SNR
    """

    def __init__(
        self,
        cache_path,
        augment: bool = True,
        snr_remix_prob: float = 0.5,
        snr_remix_range: tuple = (-5.0, 25.0),
        random_gain_prob: float = 0.3,
        random_gain_db: float = 3.0,
        random_crop_prob: float = 0.3,
        random_crop_min_ratio: float = 0.7,
        time_stretch_prob: float = 0.2,
        time_stretch_range: tuple = (0.95, 1.05),
        max_samples: Optional[int] = None,
        filter_clean_to_clean: bool = True,
        compute_snr: bool = True,
    ):
        """初始化增強資料集。

        Args:
            cache_path: 資料快取的路徑
            augment: 是否啟用資料增強
            snr_remix_prob: SNR Remix 增強的觸發機率
            snr_remix_range: 重新混合時的 SNR 範圍 (dB)，格式為 (min, max)
            random_gain_prob: Random Gain 增強的觸發機率
            random_gain_db: 增益的最大幅度 (dB)
            random_crop_prob: Random Crop 增強的觸發機率
            random_crop_min_ratio: 裁切後保留的最小音訊比例
            time_stretch_prob: Time Stretch 增強的觸發機率
            time_stretch_range: 時間拉伸的範圍，格式為 (min_rate, max_rate)
            max_samples: 載入的最大樣本數量
            filter_clean_to_clean: 是否過濾掉 clean→clean 樣本
            compute_snr: 是否計算 SNR 用於 curriculum learning
        """
        super().__init__(
            cache_path,
            max_samples=max_samples,
            filter_clean_to_clean=filter_clean_to_clean,
            compute_snr=compute_snr,
        )

        self.augment = augment
        self.snr_remix_prob = snr_remix_prob
        self.snr_remix_range = snr_remix_range
        self.random_gain_prob = random_gain_prob
        self.random_gain_db = random_gain_db
        self.random_crop_prob = random_crop_prob
        self.random_crop_min_ratio = random_crop_min_ratio
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range

        if augment:
            print(f"[AugmentedDataset] 增強已啟用:")
            print(f"  SNR Remix:     p={snr_remix_prob:.1f}, range={snr_remix_range} dB")
            print(f"  Random Gain:   p={random_gain_prob:.1f}, ±{random_gain_db:.1f} dB")
            print(f"  Random Crop:   p={random_crop_prob:.1f}, min_ratio={random_crop_min_ratio:.2f}")
            print(f"  Time Stretch:  p={time_stretch_prob:.1f}, range={time_stretch_range}")

    def __getitem__(self, idx):
        """取得並增強一筆資料。

        Args:
            idx: 樣本索引

        Returns:
            包含 'noisy_audio', 'clean_audio', 'length' 的字典，
            可能還有 'snr'
        """
        item = super().__getitem__(idx)

        if not self.augment:
            return item

        noisy = item['noisy_audio']  # (T,)
        clean = item['clean_audio']  # (T,)

        # 1. SNR Remix: 分離 noise 成分，以隨機 SNR 重新混合
        if random.random() < self.snr_remix_prob:
            noisy, clean = self._snr_remix(noisy, clean)

        # 2. Random Gain: 隨機整體增益
        if random.random() < self.random_gain_prob:
            noisy, clean = self._random_gain(noisy, clean)

        # 3. Random Crop: 隨機截取子段
        if random.random() < self.random_crop_prob:
            noisy, clean = self._random_crop(noisy, clean)

        # 4. Time Stretch (透過重新取樣模擬)
        if random.random() < self.time_stretch_prob:
            noisy, clean = self._time_stretch(noisy, clean)

        item['noisy_audio'] = noisy
        item['clean_audio'] = clean
        item['length'] = len(noisy)

        return item

    def _snr_remix(self, noisy: torch.Tensor, clean: torch.Tensor):
        """從 noisy/clean 對中分離 noise 並以隨機 SNR 重新混合。

        這是最有效的增強：因為 cache 已有 noisy/clean pair，
        可以分離出 noise = noisy - clean，再用不同 SNR 重新混合，
        等於從有限資料產生無限多 SNR 變體。

        Args:
            noisy: noisy 音訊波形 (T,)
            clean: clean 音訊波形 (T,)

        Returns:
            (new_noisy, clean): 新的 noisy 波形和原始 clean 波形
        """
        noise = noisy - clean

        # 計算目標 SNR
        target_snr_db = random.uniform(*self.snr_remix_range)

        # 計算所需的 noise 增益
        signal_power = (clean ** 2).mean().clamp(min=1e-10)
        noise_power = (noise ** 2).mean().clamp(min=1e-10)

        # target_snr = 10 * log10(signal_power / (noise_power * gain^2))
        # => gain = sqrt(signal_power / (noise_power * 10^(target_snr/10)))
        target_noise_power = signal_power / (10 ** (target_snr_db / 10) + 1e-10)
        gain = torch.sqrt(target_noise_power / noise_power).clamp(max=10.0)

        new_noisy = clean + noise * gain

        # 防止 clipping
        max_val = max(new_noisy.abs().max().item(), 1e-6)
        if max_val > 1.0:
            new_noisy = new_noisy / max_val

        return new_noisy, clean

    def _random_gain(self, noisy: torch.Tensor, clean: torch.Tensor):
        """對 noisy 和 clean 同時施加相同的隨機增益。

        讓模型學習不依賴特定音量。

        Args:
            noisy: noisy 音訊波形 (T,)
            clean: clean 音訊波形 (T,)

        Returns:
            (gained_noisy, gained_clean): 增益後的波形對
        """
        gain_db = random.uniform(-self.random_gain_db, self.random_gain_db)
        gain = 10 ** (gain_db / 20)

        noisy = noisy * gain
        clean = clean * gain

        # 防止 clipping
        max_val = max(noisy.abs().max().item(), clean.abs().max().item(), 1e-6)
        if max_val > 1.0:
            scale = 0.99 / max_val
            noisy = noisy * scale
            clean = clean * scale

        return noisy, clean

    def _random_crop(self, noisy: torch.Tensor, clean: torch.Tensor):
        """從 noisy/clean pair 隨機截取同步子段。

        增加位置多樣性，讓模型接觸同一音訊的不同片段。

        Args:
            noisy: noisy 音訊波形 (T,)
            clean: clean 音訊波形 (T,)

        Returns:
            (cropped_noisy, cropped_clean): 裁切後的波形對
        """
        T = len(noisy)
        min_len = int(T * self.random_crop_min_ratio)
        if min_len >= T:
            return noisy, clean

        crop_len = random.randint(min_len, T)
        start = random.randint(0, T - crop_len)

        return noisy[start:start + crop_len], clean[start:start + crop_len]

    def _time_stretch(self, noisy: torch.Tensor, clean: torch.Tensor):
        """透過線性插值模擬時間拉伸/壓縮。

        同步應用到 noisy 和 clean，保持對齊。

        Args:
            noisy: noisy 音訊波形 (T,)
            clean: clean 音訊波形 (T,)

        Returns:
            (stretched_noisy, stretched_clean): 拉伸後的波形對
        """
        rate = random.uniform(*self.time_stretch_range)
        T = len(noisy)
        new_T = int(T * rate)

        if new_T < 100 or new_T == T:
            return noisy, clean

        # 使用 F.interpolate 做線性插值
        noisy_stretched = F.interpolate(
            noisy.unsqueeze(0).unsqueeze(0),  # (1, 1, T)
            size=new_T,
            mode='linear',
            align_corners=False,
        ).squeeze()  # (new_T,)

        clean_stretched = F.interpolate(
            clean.unsqueeze(0).unsqueeze(0),
            size=new_T,
            mode='linear',
            align_corners=False,
        ).squeeze()  # (new_T,)

        return noisy_stretched, clean_stretched


def create_augmented_curriculum_dataloaders(
    train_cache_path,
    val_cache_path,
    batch_size: int = 8,
    num_workers: int = 2,
    curriculum_mode: str = 'curriculum',
    initial_phase: float = 0.3,
    phase_increment: float = 0.1,
    compute_snr: bool = True,
    pin_memory: bool = True,
    # 增強參數
    snr_remix_prob: float = 0.5,
    snr_remix_range: tuple = (-5.0, 25.0),
    random_gain_prob: float = 0.3,
    random_gain_db: float = 3.0,
    random_crop_prob: float = 0.3,
    random_crop_min_ratio: float = 0.7,
    time_stretch_prob: float = 0.2,
    time_stretch_range: tuple = (0.95, 1.05),
    **kwargs,
):
    """建立帶有資料增強的 Curriculum Learning DataLoader。

    Train set 使用增強，Val set 不使用增強。

    Args:
        train_cache_path: 訓練資料快取路徑
        val_cache_path: 驗證資料快取路徑
        batch_size: 批次大小
        num_workers: DataLoader 工作程序數
        curriculum_mode: curriculum 模式
        initial_phase: 初始 curriculum phase
        phase_increment: 每次 advance 增加的 phase
        compute_snr: 是否計算 SNR
        pin_memory: 是否使用 pin_memory
        snr_remix_prob: SNR Remix 機率
        snr_remix_range: SNR Remix 的 SNR 範圍
        random_gain_prob: Random Gain 機率
        random_gain_db: Random Gain 範圍
        random_crop_prob: Random Crop 機率
        random_crop_min_ratio: Random Crop 最小保留比例
        time_stretch_prob: Time Stretch 機率
        time_stretch_range: Time Stretch 速率範圍

    Returns:
        (train_loader, val_loader, train_sampler): DataLoader 和 Sampler 三元組
    """
    # Training dataset (帶增強)
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
        compute_snr=compute_snr,
    )

    # Validation dataset (不增強)
    val_dataset = AugmentedCurriculumDataset(
        val_cache_path,
        augment=False,
        filter_clean_to_clean=True,
        compute_snr=False,
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


if __name__ == '__main__':
    from exp_1201.config import TRAIN_CACHE, VAL_CACHE

    print("=" * 60)
    print("Testing Augmented Curriculum Dataset")
    print("=" * 60)

    dataset = AugmentedCurriculumDataset(
        TRAIN_CACHE,
        augment=True,
        max_samples=20,
        compute_snr=False,
    )

    print(f"\nDataset size: {len(dataset)}")

    # 取同一筆資料兩次，增強後應該不同
    item1 = dataset[0]
    item2 = dataset[0]
    diff = (item1['noisy_audio'] - item2['noisy_audio']).abs().mean()
    print(f"Same index, different augmentation? diff={diff:.6f} "
          f"({'Yes ✅' if diff > 0.001 else 'No ❌'})")

    # 測試 DataLoader
    print("\n--- Testing Augmented DataLoader ---")
    train_loader, val_loader, sampler = create_augmented_curriculum_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=4,
        num_workers=0,
        compute_snr=False,
    )

    batch = next(iter(train_loader))
    print(f"Batch noisy shape: {batch['noisy_audio'].shape}")
    print(f"Batch clean shape: {batch['clean_audio'].shape}")
    print(f"Batch lengths: {batch['lengths']}")

    print("\n✅ All augmentation tests passed!")
