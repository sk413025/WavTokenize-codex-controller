"""
exp_1212: 修復對齊問題的 Dataset

修復內容:
1. 來源1 修復: Per-pair min-length 截斷
2. 來源2 修復: 返回 lengths，支援 masked loss

基於 exp_1210/DATASET_ALIGNMENT_REPORT.md 的分析
"""

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class AlignedNoisyCleanPairDataset(Dataset):
    """
    對齊版 Noisy-Clean 配對數據集

    修復:
    1. 每對 noisy/clean 截到相同長度 (min_len)
    2. 返回 length 資訊供 masked loss 使用
    3. 過濾 Clean→Clean 樣本 (Student 應只訓練 denoising)
    """

    def __init__(self, cache_path, max_samples=None, filter_clean_to_clean=True):
        """
        Args:
            cache_path: Path to train_cache.pt or val_cache.pt
            max_samples: 最多載入多少樣本 (用於 smoke test)
            filter_clean_to_clean: 是否過濾掉 clean→clean 樣本 (預設 True)
        """
        self.cache_path = Path(cache_path)
        assert self.cache_path.exists(), f"Cache not found: {cache_path}"

        # 載入數據
        data = torch.load(cache_path, weights_only=False)

        # 假設 data 結構: list of dicts with 'noisy_audio' and 'clean_audio'
        samples = data if isinstance(data, list) else [data]

        # 過濾 Clean→Clean 樣本
        if filter_clean_to_clean:
            original_count = len(samples)
            samples = [
                s for s in samples
                if not self._is_clean_to_clean(s)
            ]
            filtered_count = original_count - len(samples)
            print(f"Filtered {filtered_count} clean→clean samples ({filtered_count/original_count*100:.1f}%)")

        self.samples = samples

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {cache_path}")

    def _is_clean_to_clean(self, sample):
        """判斷是否為 clean→clean 樣本"""
        noisy_path = sample.get('noisy_path', '')
        clean_path = sample.get('clean_path', '')

        # 方法1: 路徑相同
        if noisy_path == clean_path:
            return True

        # 方法2: noisy_path 包含 '_clean_' (表示是 clean 音訊)
        if '_clean_' in noisy_path:
            return True

        return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 策略 1: 如果 cache 中已有 audio waveforms，直接使用
        noisy_audio = sample.get('noisy_audio', sample.get('audio', None))
        clean_audio = sample.get('clean_audio', sample.get('audio', None))

        # 策略 2: 如果沒有 audio，嘗試從 file paths 載入
        if noisy_audio is None and 'noisy_path' in sample:
            noisy_audio = self._load_audio_from_path(sample['noisy_path'])

        if clean_audio is None and 'clean_path' in sample:
            clean_audio = self._load_audio_from_path(sample['clean_path'])

        # 策略 3: Fallback - 使用相同音頻（smoke test）
        if noisy_audio is None or clean_audio is None:
            audio = sample.get('audio', sample.get('waveform', None))
            if audio is not None:
                noisy_audio = audio if noisy_audio is None else noisy_audio
                clean_audio = audio if clean_audio is None else clean_audio
            else:
                raise ValueError(f"Sample {idx} has no audio data or valid paths")

        noisy_audio = noisy_audio.squeeze()  # (T,)
        clean_audio = clean_audio.squeeze()  # (T,)

        # ========== 修復: Per-Pair Min-Length 截斷 ==========
        min_len = min(len(noisy_audio), len(clean_audio))
        noisy_audio = noisy_audio[:min_len]
        clean_audio = clean_audio[:min_len]
        # ===================================================

        return {
            'noisy_audio': noisy_audio,  # (T,)
            'clean_audio': clean_audio,  # (T,)
            'length': min_len,  # 新增: 有效長度
        }

    def _load_audio_from_path(self, audio_path):
        """
        從檔案路徑載入音訊

        Args:
            audio_path: str or Path, 可能是相對路徑或只是檔名

        Returns:
            audio: torch.Tensor (T,) @ 24kHz
        """
        audio_path = Path(audio_path)

        # 如果是絕對路徑且存在，直接使用
        if audio_path.is_absolute() and audio_path.exists():
            pass
        else:
            # 從檔名推斷目錄
            filename = audio_path.name
            base_dir = Path("/home/sbplab/ruizi/c_code/data")

            # 根據檔名特徵判斷目錄
            if "_clean_" in filename:
                # clean 音頻在 clean/box2/
                audio_path = base_dir / "clean" / "box2" / filename
            elif "_box_" in filename:
                audio_path = base_dir / "raw" / "box" / filename
            elif "_papercup_" in filename:
                audio_path = base_dir / "raw" / "papercup" / filename
            elif "_plastic_" in filename:
                audio_path = base_dir / "raw" / "plastic" / filename
            else:
                # Fallback: 嘗試從 cache 目錄解析
                audio_path = self.cache_path.parent.parent / filename

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 載入音訊
        waveform, sr = torchaudio.load(str(audio_path))

        # Resample to 24kHz if needed
        target_sr = 24000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Squeeze to 1D (取第一個 channel)
        if waveform.dim() > 1:
            waveform = waveform[0]

        return waveform


def aligned_collate_fn(batch):
    """
    修復版 Collate function

    新增: 返回 lengths 供 masked loss 使用
    """
    # 記錄每個樣本的有效長度
    lengths = torch.tensor([item['length'] for item in batch])

    # 找最長的音頻
    max_len = max(item['noisy_audio'].shape[0] for item in batch)

    noisy_audios = []
    clean_audios = []

    for item in batch:
        noisy = item['noisy_audio']
        clean = item['clean_audio']

        # Pad if shorter (不需要 truncate，因為已經對齊)
        if noisy.shape[0] < max_len:
            noisy = F.pad(noisy, (0, max_len - noisy.shape[0]))
        if clean.shape[0] < max_len:
            clean = F.pad(clean, (0, max_len - clean.shape[0]))

        noisy_audios.append(noisy)
        clean_audios.append(clean)

    return {
        'noisy_audio': torch.stack(noisy_audios),  # (B, T)
        'clean_audio': torch.stack(clean_audios),  # (B, T)
        'lengths': lengths,  # (B,) 新增: 每個樣本的有效長度
    }


def create_aligned_dataloaders(config):
    """
    創建修復版 train/val dataloaders

    Args:
        config: TrainConfig or SmokeTestConfig

    Returns:
        train_loader, val_loader
    """
    try:
        from exp_1201.config import TRAIN_CACHE, VAL_CACHE
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from exp_1201.config import TRAIN_CACHE, VAL_CACHE

    # Smoke test 使用小數據
    max_samples = getattr(config, 'num_samples', None)

    train_dataset = AlignedNoisyCleanPairDataset(TRAIN_CACHE, max_samples=max_samples)
    val_dataset = AlignedNoisyCleanPairDataset(VAL_CACHE, max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', False),
        collate_fn=aligned_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', False),
        collate_fn=aligned_collate_fn,
    )

    return train_loader, val_loader
