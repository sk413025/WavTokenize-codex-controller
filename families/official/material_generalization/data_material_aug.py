"""
exp_0304: 材質泛化資料增強 Dataset

新增增強策略（目標：模擬未知材質的 LDV 頻率響應）：
5. Random Frequency Response: 隨機 parametric EQ 模擬不同材質共振
6. Spectral Normalization: 輸入頻譜正規化到 canonical LDV 分佈
7. Random Low-pass: 隨機低通截止頻率模擬不同材質的高頻衰減
8. Resonance Injection: 隨機注入共振峰模擬材質共振

繼承 families/deps/encoder_aug/data_augmented.py 的 AugmentedCurriculumDataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import random
from typing import Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from families.deps.encoder_aug.data_augmented import AugmentedCurriculumDataset
from families.compat_legacy.curriculum_data.data_curriculum import collate_fn_curriculum


class MaterialAugDataset(AugmentedCurriculumDataset):
    """帶有材質泛化增強的資料集。

    在 AugmentedCurriculumDataset 的 4 種增強之上，
    新增模擬未知材質 LDV 頻率響應的增強策略。

    新增增強：
        1. Random Frequency Response: 隨機 parametric EQ（2~5 bands）
        2. Spectral Envelope Normalization: 頻譜包絡正規化
        3. Random Low-pass Cutoff: 模擬不同材質高頻衰減
        4. Resonance Injection: 隨機共振峰注入

    Args:
        cache_path: 資料快取路徑。
        augment: 是否啟用增強。
        freq_response_prob: Random Frequency Response 觸發機率。
        freq_response_n_bands: EQ band 數量範圍 (min, max)。
        freq_response_gain_db: 每個 band 的最大增益 (dB)。
        spectral_norm_prob: Spectral Normalization 觸發機率。
        spectral_norm_n_fft: 頻譜正規化 FFT 大小。
        random_lowpass_prob: Random Low-pass 觸發機率。
        random_lowpass_range: 截止頻率範圍 (Hz)。
        resonance_prob: Resonance Injection 觸發機率。
        resonance_n_peaks: 共振峰數量範圍 (min, max)。
        resonance_q_range: 共振峰 Q factor 範圍。
        resonance_gain_range: 共振峰增益範圍 (dB)。
        sample_rate: 音訊採樣率。
        **kwargs: 傳遞給 AugmentedCurriculumDataset 的參數。
    """

    def __init__(
        self,
        cache_path,
        augment: bool = True,
        # 原有增強參數
        snr_remix_prob: float = 0.5,
        snr_remix_range: tuple = (-5.0, 25.0),
        random_gain_prob: float = 0.3,
        random_gain_db: float = 3.0,
        random_crop_prob: float = 0.3,
        random_crop_min_ratio: float = 0.7,
        time_stretch_prob: float = 0.2,
        time_stretch_range: tuple = (0.95, 1.05),
        # 新增材質增強參數
        freq_response_prob: float = 0.5,
        freq_response_n_bands: tuple = (2, 5),
        freq_response_gain_db: float = 10.0,
        spectral_norm_prob: float = 0.3,
        spectral_norm_n_fft: int = 2048,
        random_lowpass_prob: float = 0.3,
        random_lowpass_range: tuple = (2000.0, 6000.0),
        resonance_prob: float = 0.3,
        resonance_n_peaks: tuple = (1, 3),
        resonance_q_range: tuple = (5.0, 30.0),
        resonance_gain_range: tuple = (3.0, 12.0),
        sample_rate: int = 24000,
        # 其他
        max_samples: Optional[int] = None,
        filter_clean_to_clean: bool = True,
        compute_snr: bool = True,
    ):
        """初始化材質泛化增強資料集。

        Args:
            cache_path: 資料快取路徑。
            augment: 是否啟用增強。
            snr_remix_prob: SNR Remix 觸發機率。
            snr_remix_range: SNR Remix 範圍 (dB)。
            random_gain_prob: Random Gain 觸發機率。
            random_gain_db: Random Gain 範圍 (dB)。
            random_crop_prob: Random Crop 觸發機率。
            random_crop_min_ratio: Random Crop 最小保留比例。
            time_stretch_prob: Time Stretch 觸發機率。
            time_stretch_range: Time Stretch 範圍。
            freq_response_prob: Random Frequency Response 觸發機率。
            freq_response_n_bands: EQ band 數量範圍。
            freq_response_gain_db: 每個 band 的最大增益 (dB)。
            spectral_norm_prob: Spectral Normalization 觸發機率。
            spectral_norm_n_fft: 頻譜正規化 FFT 大小。
            random_lowpass_prob: Random Low-pass 觸發機率。
            random_lowpass_range: 截止頻率範圍 (Hz)。
            resonance_prob: Resonance Injection 觸發機率。
            resonance_n_peaks: 共振峰數量範圍。
            resonance_q_range: 共振峰 Q factor 範圍。
            resonance_gain_range: 共振峰增益範圍 (dB)。
            sample_rate: 音訊採樣率。
            max_samples: 最大樣本數。
            filter_clean_to_clean: 是否過濾 clean→clean。
            compute_snr: 是否計算 SNR。
        """
        super().__init__(
            cache_path,
            augment=augment,
            snr_remix_prob=snr_remix_prob,
            snr_remix_range=snr_remix_range,
            random_gain_prob=random_gain_prob,
            random_gain_db=random_gain_db,
            random_crop_prob=random_crop_prob,
            random_crop_min_ratio=random_crop_min_ratio,
            time_stretch_prob=time_stretch_prob,
            time_stretch_range=time_stretch_range,
            max_samples=max_samples,
            filter_clean_to_clean=filter_clean_to_clean,
            compute_snr=compute_snr,
        )

        # 材質增強參數
        self.freq_response_prob = freq_response_prob
        self.freq_response_n_bands = freq_response_n_bands
        self.freq_response_gain_db = freq_response_gain_db
        self.spectral_norm_prob = spectral_norm_prob
        self.spectral_norm_n_fft = spectral_norm_n_fft
        self.random_lowpass_prob = random_lowpass_prob
        self.random_lowpass_range = random_lowpass_range
        self.resonance_prob = resonance_prob
        self.resonance_n_peaks = resonance_n_peaks
        self.resonance_q_range = resonance_q_range
        self.resonance_gain_range = resonance_gain_range
        self.sample_rate = sample_rate

        if augment:
            print(f"[MaterialAugDataset] 材質泛化增強已啟用:")
            print(f"  Freq Response: p={freq_response_prob:.1f}, "
                  f"bands={freq_response_n_bands}, ±{freq_response_gain_db:.0f}dB")
            print(f"  Spectral Norm: p={spectral_norm_prob:.1f}, "
                  f"n_fft={spectral_norm_n_fft}")
            print(f"  Random LP:     p={random_lowpass_prob:.1f}, "
                  f"cutoff={random_lowpass_range} Hz")
            print(f"  Resonance:     p={resonance_prob:.1f}, "
                  f"peaks={resonance_n_peaks}, Q={resonance_q_range}, "
                  f"gain={resonance_gain_range}dB")

    def __getitem__(self, idx):
        """取得並增強一筆資料，包含材質泛化增強。

        先執行父類的 4 種增強（SNR remix, gain, crop, stretch），
        再對 noisy 施加材質模擬增強。
        注意：材質增強只作用在 noisy 上，clean 不變。

        Args:
            idx: 樣本索引。

        Returns:
            包含增強後 'noisy_audio', 'clean_audio', 'length' 的字典。
        """
        item = super().__getitem__(idx)

        if not self.augment:
            return item

        noisy = item['noisy_audio']  # (T,)

        # 5. Random Frequency Response（只對 noisy）
        if random.random() < self.freq_response_prob:
            noisy = self._random_freq_response(noisy)

        # 6. Spectral Envelope Normalization（只對 noisy）
        if random.random() < self.spectral_norm_prob:
            noisy = self._spectral_normalize(noisy)

        # 7. Random Low-pass（只對 noisy）
        if random.random() < self.random_lowpass_prob:
            noisy = self._random_lowpass(noisy)

        # 8. Resonance Injection（只對 noisy）
        if random.random() < self.resonance_prob:
            noisy = self._resonance_inject(noisy)

        item['noisy_audio'] = noisy
        item['length'] = len(noisy)
        return item

    def _random_freq_response(self, noisy: torch.Tensor) -> torch.Tensor:
        """施加隨機頻率響應（parametric EQ）模擬不同材質共振。

        在頻率域對 noisy 施加隨機的頻段增益/衰減，
        模擬不同材質的 h(f) 頻率響應差異。
        使用 2~5 個隨機頻段，每個頻段獨立的 center freq, bandwidth, gain。

        Args:
            noisy: 輸入 noisy 波形 (T,)。

        Returns:
            經過隨機頻率響應變換的波形 (T,)。
        """
        T = len(noisy)
        n_fft = min(2048, T)
        hop = n_fft // 4

        # STFT
        noisy_3d = noisy.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        # 使用 torch.stft
        window = torch.hann_window(n_fft, device=noisy.device)
        spec = torch.stft(noisy, n_fft, hop_length=hop, win_length=n_fft,
                          window=window, return_complex=True)  # (F, Frames)

        F_bins = spec.shape[0]
        freqs = torch.linspace(0, self.sample_rate / 2, F_bins, device=noisy.device)

        # 建立頻率增益曲線
        gain_curve = torch.ones(F_bins, device=noisy.device)
        n_bands = random.randint(*self.freq_response_n_bands)

        for _ in range(n_bands):
            center_hz = random.uniform(100, self.sample_rate / 2 - 100)
            bandwidth_hz = random.uniform(200, 2000)
            gain_db = random.uniform(-self.freq_response_gain_db,
                                     self.freq_response_gain_db)
            gain_linear = 10 ** (gain_db / 20)

            # Gaussian-shaped band
            sigma = bandwidth_hz / (2 * 2.355)  # FWHM → sigma
            band_gain = torch.exp(-0.5 * ((freqs - center_hz) / max(sigma, 1.0)) ** 2)
            gain_curve = gain_curve * (1 + band_gain * (gain_linear - 1))

        # 應用增益到頻譜
        spec = spec * gain_curve.unsqueeze(-1)

        # iSTFT
        result = torch.istft(spec, n_fft, hop_length=hop, win_length=n_fft,
                             window=window, length=T)

        # 防止 clipping
        max_val = result.abs().max().item()
        if max_val > 1.0:
            result = result * (0.99 / max_val)

        return result

    def _spectral_normalize(self, noisy: torch.Tensor) -> torch.Tensor:
        """頻譜包絡正規化，將輸入映射到平均 LDV 頻譜分佈。

        計算輸入的長時頻譜包絡，除以自身包絡再乘以
        一個合成的「canonical LDV」包絡（低頻強、高頻弱）。
        這讓不同材質的 LDV 信號趨向統一的頻譜 shape。

        Args:
            noisy: 輸入 noisy 波形 (T,)。

        Returns:
            頻譜正規化後的波形 (T,)。
        """
        T = len(noisy)
        n_fft = min(self.spectral_norm_n_fft, T)
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=noisy.device)

        spec = torch.stft(noisy, n_fft, hop_length=hop, win_length=n_fft,
                          window=window, return_complex=True)
        mag = spec.abs()
        phase = spec.angle()

        # 計算長時頻譜包絡（時間平均）
        envelope = mag.mean(dim=-1, keepdim=True).clamp(min=1e-7)  # (F, 1)

        # 合成 canonical LDV 頻譜包絡：
        # 低頻(< 1kHz)能量最強，中頻(1-4kHz)衰減，高頻(>4kHz)幾乎消失
        F_bins = spec.shape[0]
        freqs = torch.linspace(0, self.sample_rate / 2, F_bins, device=noisy.device)

        # 使用指數衰減模擬 LDV 特徵 + 隨機化
        decay_rate = random.uniform(0.0005, 0.002)  # 隨機衰減率
        canonical = torch.exp(-decay_rate * freqs)
        # 加一些隨機抖動
        jitter = 1.0 + 0.1 * torch.randn(F_bins, device=noisy.device)
        canonical = (canonical * jitter).clamp(min=1e-4).unsqueeze(-1)  # (F, 1)

        # 正規化：除以自身包絡，乘以 canonical
        normalized_mag = (mag / envelope) * canonical
        normalized_spec = normalized_mag * torch.exp(1j * phase)

        result = torch.istft(normalized_spec, n_fft, hop_length=hop,
                             win_length=n_fft, window=window, length=T)

        max_val = result.abs().max().item()
        if max_val > 1.0:
            result = result * (0.99 / max_val)
        if max_val < 1e-6:
            return noisy  # 避免靜音

        return result

    def _random_lowpass(self, noisy: torch.Tensor) -> torch.Tensor:
        """隨機低通濾波，模擬不同材質的高頻衰減程度。

        不同材質的反射率和共振特性導致高頻衰減程度不同。
        隨機选取一個截止頻率，施加 smooth low-pass 濾波。

        Args:
            noisy: 輸入 noisy 波形 (T,)。

        Returns:
            低通濾波後的波形 (T,)。
        """
        T = len(noisy)
        n_fft = min(2048, T)
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=noisy.device)

        spec = torch.stft(noisy, n_fft, hop_length=hop, win_length=n_fft,
                          window=window, return_complex=True)

        F_bins = spec.shape[0]
        freqs = torch.linspace(0, self.sample_rate / 2, F_bins, device=noisy.device)

        cutoff = random.uniform(*self.random_lowpass_range)
        # Smooth sigmoid roll-off (不是硬截斷)
        rolloff_width = random.uniform(500, 1500)  # Hz
        lp_filter = torch.sigmoid(-(freqs - cutoff) / rolloff_width)
        lp_filter = lp_filter.unsqueeze(-1)

        spec = spec * lp_filter

        result = torch.istft(spec, n_fft, hop_length=hop, win_length=n_fft,
                             window=window, length=T)

        max_val = result.abs().max().item()
        if max_val > 1.0:
            result = result * (0.99 / max_val)
        if max_val < 1e-6:
            return noisy

        return result

    def _resonance_inject(self, noisy: torch.Tensor) -> torch.Tensor:
        """注入隨機共振峰，模擬不同材質的機械共振。

        不同材質有不同的機械共振頻率（例如木板 ~200-800Hz，
        玻璃 ~1-3kHz，金屬 ~500Hz-5kHz）。
        隨機在若干頻率位置注入窄帶共振（高 Q factor），
        模擬未知材質的共振特性。

        Args:
            noisy: 輸入 noisy 波形 (T,)。

        Returns:
            注入共振後的波形 (T,)。
        """
        T = len(noisy)
        n_fft = min(2048, T)
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=noisy.device)

        spec = torch.stft(noisy, n_fft, hop_length=hop, win_length=n_fft,
                          window=window, return_complex=True)

        F_bins = spec.shape[0]
        freqs = torch.linspace(0, self.sample_rate / 2, F_bins, device=noisy.device)

        n_peaks = random.randint(*self.resonance_n_peaks)
        resonance_curve = torch.ones(F_bins, device=noisy.device)

        for _ in range(n_peaks):
            center_hz = random.uniform(150, 5000)
            q = random.uniform(*self.resonance_q_range)
            gain_db = random.uniform(*self.resonance_gain_range)
            gain_linear = 10 ** (gain_db / 20)

            bandwidth = center_hz / q
            sigma = bandwidth / (2 * 2.355)
            peak = torch.exp(-0.5 * ((freqs - center_hz) / max(sigma, 1.0)) ** 2)
            resonance_curve = resonance_curve + peak * (gain_linear - 1)

        spec = spec * resonance_curve.clamp(min=0.01).unsqueeze(-1)

        result = torch.istft(spec, n_fft, hop_length=hop, win_length=n_fft,
                             window=window, length=T)

        max_val = result.abs().max().item()
        if max_val > 1.0:
            result = result * (0.99 / max_val)
        if max_val < 1e-6:
            return noisy

        return result


def create_material_aug_dataloaders(
    train_cache_path,
    val_cache_path,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    sample_rate: int = 24000,
    # 原有增強
    snr_remix_prob: float = 0.5,
    snr_remix_range: tuple = (-5.0, 25.0),
    random_gain_prob: float = 0.3,
    random_gain_db: float = 3.0,
    random_crop_prob: float = 0.3,
    random_crop_min_ratio: float = 0.7,
    time_stretch_prob: float = 0.2,
    time_stretch_range: tuple = (0.95, 1.05),
    # 材質增強
    freq_response_prob: float = 0.5,
    freq_response_n_bands: tuple = (2, 5),
    freq_response_gain_db: float = 10.0,
    spectral_norm_prob: float = 0.3,
    spectral_norm_n_fft: int = 2048,
    random_lowpass_prob: float = 0.3,
    random_lowpass_range: tuple = (2000.0, 6000.0),
    resonance_prob: float = 0.3,
    resonance_n_peaks: tuple = (1, 3),
    resonance_q_range: tuple = (5.0, 30.0),
    resonance_gain_range: tuple = (3.0, 12.0),
):
    """建立帶有材質泛化增強的 DataLoader。

    Args:
        train_cache_path: 訓練資料快取路徑。
        val_cache_path: 驗證資料快取路徑。
        batch_size: 批次大小。
        num_workers: DataLoader 工作程序數。
        pin_memory: 是否使用 pin_memory。
        sample_rate: 音訊採樣率。
        snr_remix_prob: SNR Remix 機率。
        snr_remix_range: SNR Remix 的 SNR 範圍。
        random_gain_prob: Random Gain 機率。
        random_gain_db: Random Gain 範圍。
        random_crop_prob: Random Crop 機率。
        random_crop_min_ratio: Random Crop 最小保留比例。
        time_stretch_prob: Time Stretch 機率。
        time_stretch_range: Time Stretch 速率範圍。
        freq_response_prob: Random Frequency Response 機率。
        freq_response_n_bands: EQ band 數量範圍。
        freq_response_gain_db: 每個 band 最大增益 (dB)。
        spectral_norm_prob: Spectral Normalization 機率。
        spectral_norm_n_fft: 頻譜正規化 FFT 大小。
        random_lowpass_prob: Random Low-pass 機率。
        random_lowpass_range: 截止頻率範圍 (Hz)。
        resonance_prob: Resonance Injection 機率。
        resonance_n_peaks: 共振峰數量範圍。
        resonance_q_range: 共振峰 Q factor 範圍。
        resonance_gain_range: 共振峰增益範圍 (dB)。

    Returns:
        (train_loader, val_loader) 二元組。
    """
    train_ds = MaterialAugDataset(
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
        freq_response_prob=freq_response_prob,
        freq_response_n_bands=freq_response_n_bands,
        freq_response_gain_db=freq_response_gain_db,
        spectral_norm_prob=spectral_norm_prob,
        spectral_norm_n_fft=spectral_norm_n_fft,
        random_lowpass_prob=random_lowpass_prob,
        random_lowpass_range=random_lowpass_range,
        resonance_prob=resonance_prob,
        resonance_n_peaks=resonance_n_peaks,
        resonance_q_range=resonance_q_range,
        resonance_gain_range=resonance_gain_range,
        sample_rate=sample_rate,
        filter_clean_to_clean=True,
        compute_snr=False,
    )

    val_ds = MaterialAugDataset(
        val_cache_path,
        augment=False,
        filter_clean_to_clean=True,
        compute_snr=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_curriculum,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_curriculum,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


if __name__ == '__main__':
    from families.deps.wavtokenizer_core.config import TRAIN_CACHE, VAL_CACHE

    print("=" * 60)
    print("Testing MaterialAugDataset")
    print("=" * 60)

    ds = MaterialAugDataset(
        TRAIN_CACHE,
        augment=True,
        max_samples=20,
        compute_snr=False,
    )

    print(f"\nDataset size: {len(ds)}")

    # 測試增強是否正常
    item1 = ds[0]
    item2 = ds[0]
    diff = (item1['noisy_audio'] - item2['noisy_audio']).abs().mean()
    print(f"Same index, different augmentation? diff={diff:.6f} "
          f"({'Yes ✅' if diff > 0.001 else 'No ❌'})")

    # 測試 DataLoader
    train_loader, val_loader = create_material_aug_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=4, num_workers=0,
    )
    batch = next(iter(train_loader))
    print(f"Batch noisy: {batch['noisy_audio'].shape}")
    print(f"Batch clean: {batch['clean_audio'].shape}")
    print("✅ MaterialAugDataset test passed!")
