"""
Zero-Shot Dataset: 返回 audio waveform + tokens

與 Baseline Dataset 的差異:
1. __getitem__ 返回 (noisy_audio, clean_audio, noisy_tokens, clean_tokens)
2. 保留 audio waveform 用於 speaker encoder
3. 其餘邏輯完全相同
"""

import os
import sys
from pathlib import Path

# 添加父目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import AudioDataset as BaseAudioDataset
import torch
import torchaudio


class ZeroShotAudioDataset(BaseAudioDataset):
    """
    Zero-Shot Audio Dataset

    繼承 BaseAudioDataset，只修改 __getitem__ 返回格式

    Returns:
        noisy_audio: (audio_len,) waveform tensor
        clean_audio: (audio_len,) waveform tensor
        content_id: str
    """

    def __getitem__(self, idx):
        """
        Args:
            idx: index

        Returns:
            noisy_audio: (T,) waveform @ 24kHz
            clean_audio: (T,) waveform @ 24kHz
            content_id: str (句子 ID)
        """
        # 獲取文件路徑
        pair = self.paired_files[idx]
        noisy_path = pair['input']
        clean_path = pair['target']

        # 載入音頻
        noisy_audio, sr_noisy = torchaudio.load(noisy_path)
        clean_audio, sr_clean = torchaudio.load(clean_path)

        # Resample if needed
        if sr_noisy != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_noisy, self.sample_rate)
            noisy_audio = resampler(noisy_audio)

        if sr_clean != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_clean, self.sample_rate)
            clean_audio = resampler(clean_audio)

        # Squeeze to 1D (if stereo, take first channel)
        if noisy_audio.dim() > 1:
            noisy_audio = noisy_audio[0]
        if clean_audio.dim() > 1:
            clean_audio = clean_audio[0]

        # 提取 content_id (句子 ID)
        filename = os.path.basename(noisy_path)
        parts = filename.split('_')
        if len(parts) >= 5:
            content_id = parts[4].replace('.wav', '')  # LDV_001 -> 001
        elif len(parts) >= 4:
            content_id = parts[3].replace('.wav', '')  # 001
        else:
            content_id = 'unknown'

        return noisy_audio, clean_audio, content_id


# ============================================================================
#                            Collate Function
# ============================================================================

def zeroshot_collate_fn(batch, wavtokenizer, device):
    """
    Collate function for ZeroShotAudioDataset

    將 batch 中的 audio 編碼為 tokens，同時保留 waveform

    Args:
        batch: list of (noisy_audio, clean_audio, content_id)
        wavtokenizer: WavTokenizer model
        device: torch.device

    Returns:
        noisy_audio_batch: (B, max_audio_len) padded waveforms
        clean_audio_batch: (B, max_audio_len) padded waveforms
        noisy_tokens_batch: (B, max_token_len) padded tokens
        clean_tokens_batch: (B, max_token_len) padded tokens
        content_ids_batch: (B,) tensor of content IDs
    """
    noisy_audio_list = []
    clean_audio_list = []
    noisy_tokens_list = []
    clean_tokens_list = []
    content_ids_list = []

    # Process each sample
    for noisy_audio, clean_audio, content_id in batch:
        # Move audio to device
        noisy_audio = noisy_audio.to(device).unsqueeze(0)  # (1, T)
        clean_audio = clean_audio.to(device).unsqueeze(0)  # (1, T)

        # Encode to tokens
        with torch.no_grad():
            _, noisy_tokens = wavtokenizer.encode_infer(
                noisy_audio,
                bandwidth_id=torch.tensor([0], device=device)
            )
            _, clean_tokens = wavtokenizer.encode_infer(
                clean_audio,
                bandwidth_id=torch.tensor([0], device=device)
            )

        # Store
        noisy_audio_list.append(noisy_audio.squeeze(0))  # (T,)
        clean_audio_list.append(clean_audio.squeeze(0))  # (T,)
        noisy_tokens_list.append(noisy_tokens[0])  # (1, seq_len)
        clean_tokens_list.append(clean_tokens[0])  # (1, seq_len)
        content_ids_list.append(content_id)

    # Pad audio waveforms to max length in batch
    max_audio_len = max(audio.shape[0] for audio in noisy_audio_list)
    padded_noisy_audio = []
    padded_clean_audio = []

    for noisy_audio, clean_audio in zip(noisy_audio_list, clean_audio_list):
        if noisy_audio.shape[0] < max_audio_len:
            pad_size = max_audio_len - noisy_audio.shape[0]
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_size), value=0)
        if clean_audio.shape[0] < max_audio_len:
            pad_size = max_audio_len - clean_audio.shape[0]
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_size), value=0)

        padded_noisy_audio.append(noisy_audio)
        padded_clean_audio.append(clean_audio)

    noisy_audio_batch = torch.stack(padded_noisy_audio, dim=0)  # (B, max_audio_len)
    clean_audio_batch = torch.stack(padded_clean_audio, dim=0)  # (B, max_audio_len)

    # Pad tokens to max length in batch
    max_token_len = max(
        max(t.shape[1] for t in noisy_tokens_list),
        max(t.shape[1] for t in clean_tokens_list)
    )

    padded_noisy_tokens = []
    padded_clean_tokens = []

    for noisy_t, clean_t in zip(noisy_tokens_list, clean_tokens_list):
        curr_noisy = noisy_t.squeeze(0)  # (seq_len,)
        curr_clean = clean_t.squeeze(0)  # (seq_len,)

        if curr_noisy.shape[0] < max_token_len:
            pad_size = max_token_len - curr_noisy.shape[0]
            curr_noisy = torch.nn.functional.pad(curr_noisy, (0, pad_size), value=0)

        if curr_clean.shape[0] < max_token_len:
            pad_size = max_token_len - curr_clean.shape[0]
            curr_clean = torch.nn.functional.pad(curr_clean, (0, pad_size), value=0)

        padded_noisy_tokens.append(curr_noisy)
        padded_clean_tokens.append(curr_clean)

    noisy_tokens_batch = torch.stack(padded_noisy_tokens, dim=0)  # (B, max_token_len)
    clean_tokens_batch = torch.stack(padded_clean_tokens, dim=0)  # (B, max_token_len)

    # Content IDs (convert to numeric if needed)
    numeric_ids = []
    for cid in content_ids_list:
        if isinstance(cid, str):
            digits = ''.join(c for c in cid if c.isdigit())
            numeric_ids.append(int(digits) if digits else hash(cid) % 1000)
        else:
            numeric_ids.append(int(cid))

    content_ids_batch = torch.tensor(numeric_ids, dtype=torch.long)

    return (
        noisy_audio_batch,
        clean_audio_batch,
        noisy_tokens_batch,
        clean_tokens_batch,
        content_ids_batch
    )


# ============================================================================
#                            測試代碼
# ============================================================================

if __name__ == '__main__':
    print("測試 ZeroShotAudioDataset...")

    # 創建 dataset
    dataset = ZeroShotAudioDataset(
        input_dirs=['../../data/raw/box'],
        target_dir='../../data/clean/box2',
        max_sentences_per_speaker=10
    )

    print(f"Dataset size: {len(dataset)}")

    # 測試單個樣本
    if len(dataset) > 0:
        noisy_audio, clean_audio, content_id = dataset[0]
        print(f"\nSample 0:")
        print(f"  - Noisy audio shape: {noisy_audio.shape}")
        print(f"  - Clean audio shape: {clean_audio.shape}")
        print(f"  - Content ID: {content_id}")

    print("\n✅ Dataset test passed!")
