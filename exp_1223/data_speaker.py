"""
exp_1223: Speaker-Aware Dataset

擴展 exp_1212/data_aligned.py，增加 speaker embedding 支援
用於 Exp60 Speaker 適應實驗
"""

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SpeakerAwareDataset(Dataset):
    """
    支援 Speaker Embedding 的 Noisy-Clean 配對數據集

    新增功能:
    1. 返回 speaker_embedding (256-dim, from ECAPA-TDNN)
    2. 返回 speaker_id (用於分析)
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

        # 統計 speaker 分佈
        speaker_counts = {}
        for s in self.samples:
            spk = s.get('speaker_id', 'unknown')
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        print(f"  Speakers: {len(speaker_counts)}")

    def _is_clean_to_clean(self, sample):
        """判斷是否為 clean→clean 樣本"""
        noisy_path = sample.get('noisy_path', '')
        clean_path = sample.get('clean_path', '')

        if noisy_path == clean_path:
            return True
        if '_clean_' in noisy_path:
            return True
        return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 獲取 audio
        noisy_audio = sample.get('noisy_audio', sample.get('audio', None))
        clean_audio = sample.get('clean_audio', sample.get('audio', None))

        if noisy_audio is None and 'noisy_path' in sample:
            noisy_audio = self._load_audio_from_path(sample['noisy_path'])

        if clean_audio is None and 'clean_path' in sample:
            clean_audio = self._load_audio_from_path(sample['clean_path'])

        if noisy_audio is None or clean_audio is None:
            audio = sample.get('audio', sample.get('waveform', None))
            if audio is not None:
                noisy_audio = audio if noisy_audio is None else noisy_audio
                clean_audio = audio if clean_audio is None else clean_audio
            else:
                raise ValueError(f"Sample {idx} has no audio data or valid paths")

        # 確保張量格式正確
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.squeeze(0)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.squeeze(0)

        # 對齊長度
        min_len = min(len(noisy_audio), len(clean_audio))
        noisy_audio = noisy_audio[:min_len]
        clean_audio = clean_audio[:min_len]

        # 獲取 speaker embedding
        speaker_embedding = sample.get('speaker_embedding', None)
        if speaker_embedding is None:
            # Fallback: 使用零向量
            speaker_embedding = torch.zeros(256)

        # 獲取 speaker id
        speaker_id = sample.get('speaker_id', 'unknown')

        return {
            'noisy_audio': noisy_audio,
            'clean_audio': clean_audio,
            'lengths': min_len,
            'speaker_embedding': speaker_embedding,
            'speaker_id': speaker_id,
        }

    def _load_audio_from_path(self, audio_path, target_sr=24000):
        """
        從檔案路徑載入音訊
        (與 exp_1212/data_aligned.py 相同的邏輯)
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
                audio_path = base_dir / "clean" / "box2" / filename
            elif "_box_" in filename:
                audio_path = base_dir / "raw" / "box" / filename
            elif "_papercup_" in filename:
                audio_path = base_dir / "raw" / "papercup" / filename
            elif "_plastic_" in filename:
                audio_path = base_dir / "raw" / "plastic" / filename
            else:
                audio_path = self.cache_path.parent.parent / filename

        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            return None

        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            if waveform.dim() > 1:
                waveform = waveform[0]
            return waveform
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None


def collate_fn_speaker(batch):
    """自定義 collate function，支援變長音頻和 speaker embedding"""
    # 找出最長的音頻長度
    max_len = max(item['lengths'] for item in batch)

    batch_size = len(batch)

    # 初始化 padded tensors
    noisy_padded = torch.zeros(batch_size, max_len)
    clean_padded = torch.zeros(batch_size, max_len)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    speaker_embeddings = torch.zeros(batch_size, 256)
    speaker_ids = []

    for i, item in enumerate(batch):
        length = item['lengths']
        noisy_padded[i, :length] = item['noisy_audio']
        clean_padded[i, :length] = item['clean_audio']
        lengths[i] = length
        speaker_embeddings[i] = item['speaker_embedding']
        speaker_ids.append(item['speaker_id'])

    return {
        'noisy_audio': noisy_padded,
        'clean_audio': clean_padded,
        'lengths': lengths,
        'speaker_embedding': speaker_embeddings,
        'speaker_ids': speaker_ids,
    }


def create_speaker_aware_dataloaders(config):
    """創建支援 speaker embedding 的 dataloaders"""
    from exp_1201.config import TRAIN_CACHE, VAL_CACHE

    train_dataset = SpeakerAwareDataset(
        TRAIN_CACHE,
        filter_clean_to_clean=True
    )

    val_dataset = SpeakerAwareDataset(
        VAL_CACHE,
        filter_clean_to_clean=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn_speaker,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn_speaker,
        drop_last=False,
    )

    return train_loader, val_loader
