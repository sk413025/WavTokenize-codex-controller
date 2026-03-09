"""
Zero-Shot Audio Dataset (HDF5 版本 V2 - 單一文件包含 train/val)

特點:
- Memory-mapped 訪問（不需要載入到 RAM）
- 支持變長序列（使用 seq_lengths）
- 支持多進程 DataLoader
- 包含 VQ distances
- 單一 HDF5 文件包含 train/val 兩個 split

使用:
    from data_zeroshot_hdf5_v2 import HDF5ZeroShotDataset
    
    train_dataset = HDF5ZeroShotDataset(
        h5_path='./data_with_distances/cache_with_distances.h5',
        split='train'
    )
    
    val_dataset = HDF5ZeroShotDataset(
        h5_path='./data_with_distances/cache_with_distances.h5',
        split='val'
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # 支持多進程
        collate_fn=cached_collate_fn_with_distances
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAD_TOKEN


class HDF5ZeroShotDataset(Dataset):
    """
    HDF5-based Zero-Shot Audio Dataset (V2)
    
    優勢:
    - Memory-mapped 訪問，RAM 使用量 <500MB
    - 快速隨機訪問
    - 支持多進程 DataLoader
    - 單一文件包含 train/val 兩個 split
    
    Args:
        h5_path: HDF5 文件路徑
        split: 'train' or 'val'
    """
    
    def __init__(self, h5_path, split='train'):
        self.h5_path = Path(h5_path)
        self.split = split
        self.h5_file = None
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 文件不存在: {h5_path}")
        
        # 打開文件讀取 metadata（初始化時）
        with h5py.File(self.h5_path, 'r') as f:
            if split not in f:
                raise ValueError(f"Split '{split}' not found in {h5_path}. Available: {list(f.keys())}")
            
            self.num_samples = f[split].attrs['num_samples']
            self.input_dirs = f.attrs.get('input_dirs', 'N/A')
            self.target_dir = f.attrs.get('target_dir', 'N/A')
        
        file_size_gb = self.h5_path.stat().st_size / (1024**3)
        
        print(f"載入 HDF5 數據集: {self.h5_path}")
        print(f"  文件大小: {file_size_gb:.2f} GB")
        print(f"  Split: {split}")
        print(f"  樣本數: {self.num_samples}")
        print(f"  模式: Memory-Mapped (RAM 使用量 ≈ 0)")
    
    def _ensure_file_open(self):
        """確保 HDF5 文件已打開（多進程安全）"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.group = self.h5_file[self.split]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        返回一個樣本
        
        Returns:
            dict: {
                'noisy_tokens': (T,) long
                'clean_tokens': (T,) long
                'noisy_distances': (T, 4096) float
                'clean_distances': (T, 4096) float
                'speaker_emb': (D,) float
                'metadata': dict
            }
        """
        self._ensure_file_open()
        
        # 讀取序列長度
        seq_len = int(self.group['seq_lengths'][idx])
        
        # 讀取數據（只取有效部分，去除 padding）
        noisy_tokens = torch.from_numpy(
            self.group['noisy_tokens'][idx, :seq_len].astype(np.int64)
        )
        clean_tokens = torch.from_numpy(
            self.group['clean_tokens'][idx, :seq_len].astype(np.int64)
        )
        noisy_distances = torch.from_numpy(
            self.group['noisy_distances'][idx, :seq_len].astype(np.float32)
        )
        clean_distances = torch.from_numpy(
            self.group['clean_distances'][idx, :seq_len].astype(np.float32)
        )
        speaker_emb = torch.from_numpy(
            self.group['speaker_emb'][idx].astype(np.float32)
        )
        
        # 讀取 metadata
        metadata = {
            'content_id': self.group['content_id'][idx],
            'speaker_id': self.group['speaker_id'][idx],
            'material': self.group['material'][idx],
            'sentence_id': self.group['sentence_id'][idx],
            'filename': self.group['filename'][idx]
        }
        
        return {
            'noisy_tokens': noisy_tokens,
            'clean_tokens': clean_tokens,
            'noisy_distances': noisy_distances,
            'clean_distances': clean_distances,
            'speaker_emb': speaker_emb,
            'metadata': metadata
        }


def cached_collate_fn_with_distances(batch):
    """
    Collate function for batching with distances
    
    處理變長序列的 padding
    
    Args:
        batch: list of dict from __getitem__
    
    Returns:
        dict: batched tensors
    """
    # 找到最大長度
    max_len = max(item['noisy_tokens'].shape[0] for item in batch)
    
    # Padding
    noisy_tokens_padded = []
    clean_tokens_padded = []
    noisy_distances_padded = []
    clean_distances_padded = []
    speaker_embs = []
    metadatas = []
    
    for item in batch:
        seq_len = item['noisy_tokens'].shape[0]
        pad_len = max_len - seq_len
        
        # Pad tokens
        noisy_tok = torch.cat([
            item['noisy_tokens'],
            torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)
        ])
        clean_tok = torch.cat([
            item['clean_tokens'],
            torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)
        ])
        
        # Pad distances
        noisy_dist = torch.cat([
            item['noisy_distances'],
            torch.zeros(pad_len, item['noisy_distances'].shape[1])
        ])
        clean_dist = torch.cat([
            item['clean_distances'],
            torch.zeros(pad_len, item['clean_distances'].shape[1])
        ])
        
        noisy_tokens_padded.append(noisy_tok)
        clean_tokens_padded.append(clean_tok)
        noisy_distances_padded.append(noisy_dist)
        clean_distances_padded.append(clean_dist)
        speaker_embs.append(item['speaker_emb'])
        metadatas.append(item['metadata'])
    
    return {
        'noisy_tokens': torch.stack(noisy_tokens_padded),
        'clean_tokens': torch.stack(clean_tokens_padded),
        'noisy_distances': torch.stack(noisy_distances_padded),
        'clean_distances': torch.stack(clean_distances_padded),
        'speaker_emb': torch.stack(speaker_embs),
        'metadata': metadatas
    }


if __name__ == '__main__':
    """測試 HDF5 dataset"""
    
    print("="*80)
    print("測試 HDF5ZeroShotDataset V2")
    print("="*80)
    
    # 假設已經轉換了數據
    h5_path = './data_with_distances/cache_with_distances.h5'
    
    if not Path(h5_path).exists():
        print(f"❌ 文件不存在: {h5_path}")
        print("\n請先運行:")
        print("  python preprocess_zeroshot_cache_with_distances_hdf5.py \\")
        print("    --input_dirs ../../data/raw/box ../../data/raw/papercup \\")
        print("    --target_dir ../../data/clean/box2 \\")
        print("    --output_dir ./data_with_distances \\")
        print("    --batch_size 16")
        exit(1)
    
    # 測試訓練集
    print("\n[測試訓練集]")
    train_dataset = HDF5ZeroShotDataset(h5_path, split='train')
    
    print(f"\n✓ Dataset 創建成功")
    print(f"  樣本數: {len(train_dataset)}")
    
    # 測試單個樣本訪問
    print("\n測試單個樣本訪問...")
    sample = train_dataset[0]
    
    print(f"  noisy_tokens: {sample['noisy_tokens'].shape}, dtype={sample['noisy_tokens'].dtype}")
    print(f"  clean_tokens: {sample['clean_tokens'].shape}, dtype={sample['clean_tokens'].dtype}")
    print(f"  noisy_distances: {sample['noisy_distances'].shape}, dtype={sample['noisy_distances'].dtype}")
    print(f"  clean_distances: {sample['clean_distances'].shape}, dtype={sample['clean_distances'].dtype}")
    print(f"  speaker_emb: {sample['speaker_emb'].shape}, dtype={sample['speaker_emb'].dtype}")
    print(f"  metadata: {sample['metadata']}")
    
    # 驗證 distance 與 token 的一致性
    print("\n驗證 distance 與 token 的一致性...")
    noisy_tok = sample['noisy_tokens'][0].item()
    noisy_dist_argmax = sample['noisy_distances'][0].argmax().item()
    
    print(f"  第一個 token: {noisy_tok}")
    print(f"  Distance argmax: {noisy_dist_argmax}")
    
    if noisy_tok == noisy_dist_argmax:
        print("  ✅ 一致！")
    else:
        print(f"  ❌ 不一致！Token={noisy_tok}, Argmax={noisy_dist_argmax}")
    
    # 測試隨機訪問速度
    print("\n測試隨機訪問速度...")
    import time
    
    indices = np.random.choice(len(train_dataset), size=100, replace=False)
    start = time.time()
    for idx in indices:
        _ = train_dataset[idx]
    elapsed = time.time() - start
    
    print(f"  訪問 100 個樣本耗時: {elapsed:.3f} 秒")
    print(f"  平均每樣本: {elapsed/100*1000:.1f} ms")
    
    # 測試 DataLoader（多進程）
    print("\n測試 DataLoader（多進程）...")
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=cached_collate_fn_with_distances
    )
    
    batch = next(iter(dataloader))
    
    print(f"  Batch 大小: {batch['noisy_tokens'].shape[0]}")
    print(f"  noisy_tokens: {batch['noisy_tokens'].shape}")
    print(f"  noisy_distances: {batch['noisy_distances'].shape}")
    print(f"  speaker_emb: {batch['speaker_emb'].shape}")
    
    # 測試驗證集
    print("\n[測試驗證集]")
    val_dataset = HDF5ZeroShotDataset(h5_path, split='val')
    print(f"  樣本數: {len(val_dataset)}")
    
    val_sample = val_dataset[0]
    print(f"  ✓ 可以正常訪問驗證集樣本")
    
    print("\n" + "="*80)
    print("✅ 測試通過！")
    print("="*80)
    print("\n下一步:")
    print("  修改 train_with_distances.py 使用 HDF5ZeroShotDataset")
