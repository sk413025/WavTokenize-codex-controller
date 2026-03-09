"""
Zero-Shot Speaker Denoising 預處理腳本（直接寫入 HDF5）

改進:
  - 串流式寫入 HDF5，不累積在記憶體
  - 自動 gzip 壓縮，節省 20-30% 磁碟空間
  - 支持變長序列的高效存儲
  - 避免 63GB 一次性 torch.save() 的問題

使用:
  python preprocess_zeroshot_cache_with_distances_hdf5.py \
    --input_dirs ../../data/raw/box ../../data/raw/papercup \
    --target_dir ../../data/clean/box2 \
    --output_dir ./data_with_distances \
    --batch_size 16
"""

import torch
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import h5py

# 添加必要的路徑
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from decoder.pretrained import WavTokenizer
from speaker_encoder import create_speaker_encoder
from data_zeroshot import ZeroShotAudioDataset


class DistanceCapture:
    """
    捕獲 VQ-VAE quantize 過程中的 distance 矩陣
    
    使用 PyTorch Hook 攔截 EuclideanCodebook.quantize() 的 distance 計算
    """
    
    # 類級別的 hook 註冊表（避免重複註冊）
    _hooks_registry = {}
    
    def __init__(self, hook_id: str):
        """
        Args:
            hook_id: 唯一標識符 (e.g., 'noisy', 'clean')
        """
        self.hook_id = hook_id
        self.distances = None
        self.is_active = False
        
    @classmethod
    def install_hook(cls, codebook, hook_id: str):
        """
        安裝 hook 到 codebook.quantize 方法（類方法，全局只安裝一次）
        
        Args:
            codebook: EuclideanCodebook 實例
            hook_id: 此 hook 的唯一標識
        """
        if hook_id in cls._hooks_registry:
            return cls._hooks_registry[hook_id]
        
        # 保存原始方法
        original_quantize = codebook.quantize
        
        def hooked_quantize(features):
            """包裝 quantize 方法，捕獲 distance"""
            # 調用原始 quantize
            result = original_quantize(features)
            
            # 檢查所有已註冊的 capture 實例
            for capture_id, capture_instance in cls._hooks_registry.items():
                if capture_instance.is_active:
                    # 重新計算 distance（因為原始 quantize 已經返回）
                    # features: (B, T, D) or (B*T, D)
                    # codebook.embeddings: (num_codes, D)
                    if features.dim() == 3:
                        B, T, D = features.shape
                        features_flat = features.reshape(-1, D)
                    else:
                        features_flat = features
                    
                    # 計算 distance: -||features - codebook||²
                    # dist[i, j] = -||features[i] - codebook[j]||²
                    embed = codebook.embed.t()  # (K, D)
                    dist = -(
                        features_flat.pow(2).sum(1, keepdim=True)
                        - 2 * features_flat @ embed
                        + embed.pow(2).sum(0, keepdim=True)
                    )  # (N, K)
                    
                    # 保存到對應的 capture 實例
                    capture_instance.distances = dist.detach().cpu()
            
            return result
        
        # 替換方法
        codebook.quantize = hooked_quantize
        
        return hooked_quantize
    
    @classmethod
    def register(cls, hook_id: str):
        """
        註冊一個 DistanceCapture 實例
        
        Args:
            hook_id: 唯一標識符
        
        Returns:
            DistanceCapture 實例
        """
        if hook_id in cls._hooks_registry:
            raise ValueError(f"Hook ID '{hook_id}' 已經被註冊！")
        
        instance = cls(hook_id)
        cls._hooks_registry[hook_id] = instance
        return instance
    
    def activate(self):
        """激活此 capture（開始記錄 distances）"""
        self.is_active = True
    
    def deactivate(self):
        """停用此 capture（停止記錄）"""
        self.is_active = False
    
    def clear(self):
        """清空已捕獲的 distances"""
        self.distances = None
    
    def get_distances(self):
        """
        獲取捕獲的 distances
        
        Returns:
            torch.Tensor: (N, num_codes) distance 矩陣，或 None（如果未捕獲）
        """
        return self.distances


def preprocess_batch_with_distances(
    batch_data, batch_indices, full_dataset,
    wavtokenizer, speaker_encoder, device,
    distance_capture_noisy, distance_capture_clean
):
    """
    預處理一個 batch 的數據（含 distances）
    
    Returns:
        list of dict: 每個樣本包含:
            - noisy_tokens: (T,) token 序列
            - clean_tokens: (T,) token 序列
            - noisy_distances: (T, 4096) distance 矩陣
            - clean_distances: (T, 4096) distance 矩陣
            - speaker_emb: (D,) speaker embedding
            - metadata: dict
    """
    noisy_audios = []
    clean_audios = []
    metadatas = []
    content_ids = []

    # 收集 batch 數據
    for idx_in_batch, (noisy_audio, clean_audio, content_id_str) in enumerate(batch_data):
        # 獲取實際的 data_idx
        data_idx = batch_indices[idx_in_batch]
        
        # 獲取 metadata
        pair = full_dataset.paired_files[data_idx]
        noisy_path = os.path.join(pair['input_dir'], pair['input'])
        clean_path = os.path.join(full_dataset.target_dir, pair['target'])
        
        filename = os.path.basename(noisy_path)
        parts = filename.split('_')
        
        if len(parts) >= 5:
            material = parts[0]
            speaker_id = parts[1]
            sentence_id = parts[4].split('.')[0]
            content_id = f"{material}_{speaker_id}_{sentence_id}"
        else:
            material = "unknown"
            speaker_id = "unknown"
            sentence_id = "unknown"
            content_id = filename
        
        metadata = {
            'content_id': content_id,
            'speaker_id': speaker_id,
            'material': material,
            'sentence_id': sentence_id,
            'noisy_path': noisy_path,
            'clean_path': clean_path,
            'filename': filename
        }
        
        metadatas.append(metadata)
        content_ids.append(content_id)
        
        noisy_audios.append(noisy_audio)
        clean_audios.append(clean_audio)

    # Padding
    max_len_noisy = max(a.shape[0] for a in noisy_audios)
    max_len_clean = max(a.shape[0] for a in clean_audios)
    max_len = max(max_len_noisy, max_len_clean)

    padded_noisy = []
    padded_clean = []

    for noisy, clean in zip(noisy_audios, clean_audios):
        if noisy.shape[0] < max_len:
            noisy = torch.nn.functional.pad(noisy, (0, max_len - noisy.shape[0]), value=0)
        if clean.shape[0] < max_len:
            clean = torch.nn.functional.pad(clean, (0, max_len - clean.shape[0]), value=0)

        padded_noisy.append(noisy)
        padded_clean.append(clean)

    noisy_batch = torch.stack(padded_noisy).to(device)
    clean_batch = torch.stack(padded_clean).to(device)

    # 批量編碼（並捕獲 distances）
    with torch.no_grad():
        batch_size = len(batch_data)
        bandwidth_id = torch.tensor([0] * batch_size, device=device)

        # ⭐ Noisy audio: 激活 capture，清空，編碼
        distance_capture_noisy.clear()
        distance_capture_noisy.activate()
        _, noisy_tokens = wavtokenizer.encode_infer(noisy_batch, bandwidth_id=bandwidth_id)
        noisy_distances = distance_capture_noisy.get_distances()  # (B×T, 4096)
        distance_capture_noisy.deactivate()
        
        # ⭐ Clean audio: 激活 capture，清空，編碼
        distance_capture_clean.clear()
        distance_capture_clean.activate()
        _, clean_tokens = wavtokenizer.encode_infer(clean_batch, bandwidth_id=bandwidth_id)
        clean_distances = distance_capture_clean.get_distances()  # (B×T, 4096)
        distance_capture_clean.deactivate()

        # ⭐ 錯誤檢查
        if noisy_distances is None or clean_distances is None:
            raise RuntimeError("❌ Distances 未被捕獲！")

        # 處理 tokens 形狀
        if noisy_tokens.dim() == 3:
            if noisy_tokens.shape[0] == 1:
                noisy_tokens = noisy_tokens.squeeze(0)
            elif noisy_tokens.shape[1] == 1:
                noisy_tokens = noisy_tokens.squeeze(1)

        if clean_tokens.dim() == 3:
            if clean_tokens.shape[0] == 1:
                clean_tokens = clean_tokens.squeeze(0)
            elif clean_tokens.shape[1] == 1:
                clean_tokens = clean_tokens.squeeze(1)

        noisy_tokens = noisy_tokens.cpu()
        clean_tokens = clean_tokens.cpu()

        # ⭐ Reshape distances: (B×T, 4096) → list of (T, 4096)
        noisy_distances_list = []
        clean_distances_list = []
        
        for i in range(batch_size):
            T_noisy = noisy_tokens.shape[1] if noisy_tokens.dim() == 2 else noisy_tokens[i].shape[0]
            T_clean = clean_tokens.shape[1] if clean_tokens.dim() == 2 else clean_tokens[i].shape[0]
            
            # 從 (B×T, 4096) 中提取對應的 (T, 4096)
            start_idx_noisy = i * T_noisy
            end_idx_noisy = start_idx_noisy + T_noisy
            noisy_dist_i = noisy_distances[start_idx_noisy:end_idx_noisy]  # (T, 4096)
            
            start_idx_clean = i * T_clean
            end_idx_clean = start_idx_clean + T_clean
            clean_dist_i = clean_distances[start_idx_clean:end_idx_clean]  # (T, 4096)
            
            noisy_distances_list.append(noisy_dist_i)
            clean_distances_list.append(clean_dist_i)

        # 提取 speaker embedding
        speaker_embs = speaker_encoder(clean_batch)

    # 組裝樣本
    samples = []
    for i in range(batch_size):
        if noisy_tokens.dim() == 2:
            noisy_tok = noisy_tokens[i]
        else:
            noisy_tok = noisy_tokens

        if clean_tokens.dim() == 2:
            clean_tok = clean_tokens[i]
        else:
            clean_tok = clean_tokens

        sample = {
            'noisy_tokens': noisy_tok,
            'clean_tokens': clean_tok,
            'noisy_distances': noisy_distances_list[i],
            'clean_distances': clean_distances_list[i],
            'speaker_emb': speaker_embs[i],
            'metadata': metadatas[i]
        }
        samples.append(sample)

    return samples


def create_hdf5_dataset(h5_file, split_name, estimated_samples, max_seq_len=512, num_codes=4096):
    """
    創建 HDF5 dataset（支持動態增長）
    
    Args:
        h5_file: h5py.File 對象
        split_name: 'train' or 'val'
        estimated_samples: 預估的樣本數（用於初始化）
        max_seq_len: 最大序列長度
        num_codes: codebook 大小
    
    Returns:
        dict: 包含所有 dataset 的字典
    """
    group = h5_file.create_group(split_name)
    group.attrs['num_samples'] = 0  # 將動態更新
    
    # 創建可變長 datasets (使用 maxshape 支持動態增長)
    datasets = {
        'noisy_tokens': group.create_dataset(
            'noisy_tokens',
            shape=(0, max_seq_len),
            maxshape=(None, max_seq_len),
            dtype='i4',
            chunks=(1, max_seq_len),
            compression='gzip',
            compression_opts=4
        ),
        'clean_tokens': group.create_dataset(
            'clean_tokens',
            shape=(0, max_seq_len),
            maxshape=(None, max_seq_len),
            dtype='i4',
            chunks=(1, max_seq_len),
            compression='gzip',
            compression_opts=4
        ),
        'noisy_distances': group.create_dataset(
            'noisy_distances',
            shape=(0, max_seq_len, num_codes),
            maxshape=(None, max_seq_len, num_codes),
            dtype='f4',
            chunks=(1, max_seq_len, num_codes),
            compression='gzip',
            compression_opts=4
        ),
        'clean_distances': group.create_dataset(
            'clean_distances',
            shape=(0, max_seq_len, num_codes),
            maxshape=(None, max_seq_len, num_codes),
            dtype='f4',
            chunks=(1, max_seq_len, num_codes),
            compression='gzip',
            compression_opts=4
        ),
        'speaker_emb': group.create_dataset(
            'speaker_emb',
            shape=(0, 192),  # ECAPA-TDNN 輸出 192 維
            maxshape=(None, 192),
            dtype='f4',
            chunks=(1, 192),
            compression='gzip',
            compression_opts=4
        ),
        'seq_lengths': group.create_dataset(
            'seq_lengths',
            shape=(0,),
            maxshape=(None,),
            dtype='i4',
            chunks=(100,)
        )
    }
    
    # Metadata (variable-length strings)
    dt = h5py.special_dtype(vlen=str)
    datasets['content_id'] = group.create_dataset('content_id', shape=(0,), maxshape=(None,), dtype=dt, chunks=(100,))
    datasets['speaker_id'] = group.create_dataset('speaker_id', shape=(0,), maxshape=(None,), dtype=dt, chunks=(100,))
    datasets['material'] = group.create_dataset('material', shape=(0,), maxshape=(None,), dtype=dt, chunks=(100,))
    datasets['sentence_id'] = group.create_dataset('sentence_id', shape=(0,), maxshape=(None,), dtype=dt, chunks=(100,))
    datasets['filename'] = group.create_dataset('filename', shape=(0,), maxshape=(None,), dtype=dt, chunks=(100,))
    
    return group, datasets


def append_to_hdf5(datasets, samples):
    """
    將樣本批次追加到 HDF5 datasets
    
    Args:
        datasets: dict of h5py.Dataset
        samples: list of dict (來自 preprocess_batch_with_distances)
    """
    if not samples:
        return
    
    current_size = datasets['noisy_tokens'].shape[0]
    new_size = current_size + len(samples)
    
    # Resize all datasets
    for key in datasets:
        datasets[key].resize(new_size, axis=0)
    
    # 寫入數據
    for i, sample in enumerate(samples):
        idx = current_size + i
        
        # 獲取實際序列長度
        seq_len = sample['noisy_tokens'].shape[0]
        max_len = datasets['noisy_tokens'].shape[1]
        
        # Pad to max_len
        noisy_tok_padded = np.zeros(max_len, dtype=np.int32)
        clean_tok_padded = np.zeros(max_len, dtype=np.int32)
        noisy_dist_padded = np.zeros((max_len, datasets['noisy_distances'].shape[2]), dtype=np.float32)
        clean_dist_padded = np.zeros((max_len, datasets['clean_distances'].shape[2]), dtype=np.float32)
        
        noisy_tok_padded[:seq_len] = sample['noisy_tokens'].numpy()
        clean_tok_padded[:seq_len] = sample['clean_tokens'].numpy()
        noisy_dist_padded[:seq_len] = sample['noisy_distances'].numpy()
        clean_dist_padded[:seq_len] = sample['clean_distances'].numpy()
        
        # 寫入
        datasets['noisy_tokens'][idx] = noisy_tok_padded
        datasets['clean_tokens'][idx] = clean_tok_padded
        datasets['noisy_distances'][idx] = noisy_dist_padded
        datasets['clean_distances'][idx] = clean_dist_padded
        datasets['speaker_emb'][idx] = sample['speaker_emb'].cpu().numpy()
        datasets['seq_lengths'][idx] = seq_len
        
        # Metadata
        meta = sample['metadata']
        datasets['content_id'][idx] = meta['content_id']
        datasets['speaker_id'][idx] = meta['speaker_id']
        datasets['material'][idx] = meta['material']
        datasets['sentence_id'][idx] = meta['sentence_id']
        datasets['filename'][idx] = meta['filename']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', required=True)
    parser.add_argument('--target_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--wavtokenizer_config', default='../../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml')
    parser.add_argument('--wavtokenizer_checkpoint', default='../../models/wavtokenizer_large_speech_320_24k.ckpt')
    parser.add_argument('--speaker_encoder_path', default='pretrained_models/spkrec-ecapa-voxceleb')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("預處理 Zero-Shot 數據集（直接寫入 HDF5）")
    print("="*80)
    print(f"輸出目錄: {output_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入 WavTokenizer
    print("\n加載 WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(args.wavtokenizer_config, args.wavtokenizer_checkpoint)
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()
    print("✓ WavTokenizer 載入完成")
    
    # ⭐ 安裝 Distance Capture Hook
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0]._codebook
    DistanceCapture.install_hook(codebook, hook_id='noisy')
    print("✓ Distance capture hook 機制已安裝到 codebook.quantize")
    
    distance_capture_noisy = DistanceCapture.register('noisy')
    distance_capture_clean = DistanceCapture.register('clean')
    print(f"✓ Distance capture 實例已註冊 (ID: noisy)")
    print(f"✓ Distance capture 實例已註冊 (ID: clean)")
    
    distance_capture_noisy.activate()
    distance_capture_clean.activate()
    print("⭐ Distance capture 機制已啟用")
    
    # 載入 Speaker Encoder
    print("\n加載 Speaker Encoder...")
    speaker_encoder = create_speaker_encoder(
        model_type='ecapa',
        freeze=True,
        output_dim=192
    )
    # 不需要 .to(device)，ECAPA 模型會在 CPU 上運行
    speaker_encoder.eval()
    print("✓ Speaker Encoder (ecapa) 載入完成")
    
    # 創建數據集
    print("\n創建數據集...")
    full_dataset = ZeroShotAudioDataset(
        input_dirs=args.input_dirs,
        target_dir=args.target_dir,
        max_sentences_per_speaker=None
    )
    print(f"✓ 數據集大小: {len(full_dataset)} 個音頻對")
    
    # 分割訓練集和驗證集
    print("\n分割數據集...")
    val_speakers = ['girl9', 'boy7', 'boy8']
    excluded_speakers = ['girl6']
    
    train_indices = []
    val_indices = []
    excluded_count = 0
    
    for idx in range(len(full_dataset)):
        pair = full_dataset.paired_files[idx]
        noisy_filename = pair['input']
        parts = os.path.basename(noisy_filename).split('_')
        
        if len(parts) >= 2:
            speaker = parts[1]
        else:
            continue
        
        if speaker in excluded_speakers:
            excluded_count += 1
            continue
        
        if speaker in val_speakers:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    
    print(f"✓ 訓練集: {len(train_indices)} 樣本")
    print(f"✓ 驗證集: {len(val_indices)} 樣本")
    print(f"✓ 排除: {excluded_count} 樣本 (speakers: {excluded_speakers})")
    
    # ⭐ 創建 HDF5 文件
    h5_path = output_dir / 'cache_with_distances.h5'
    print(f"\n創建 HDF5 文件: {h5_path}")
    
    with h5py.File(h5_path, 'w') as h5f:
        # 文件級別 metadata
        h5f.attrs['input_dirs'] = str(args.input_dirs)
        h5f.attrs['target_dir'] = args.target_dir
        h5f.attrs['total_samples'] = len(train_indices) + len(val_indices)
        
        # 創建訓練集 datasets
        print("\n創建訓練集 HDF5 datasets...")
        train_group, train_datasets = create_hdf5_dataset(h5f, 'train', len(train_indices))
        
        # 創建驗證集 datasets
        print("創建驗證集 HDF5 datasets...")
        val_group, val_datasets = create_hdf5_dataset(h5f, 'val', len(val_indices))
        
        # ⭐ 預處理訓練集（串流式寫入）
        print("\n" + "=" * 80)
        print("預處理訓練集（串流式寫入 HDF5）...")
        print("=" * 80)
        
        for i in tqdm(range(0, len(train_indices), args.batch_size), desc="訓練集"):
            batch_indices = train_indices[i:i + args.batch_size]
            batch_data = [full_dataset[idx] for idx in batch_indices]
            
            processed = preprocess_batch_with_distances(
                batch_data, batch_indices, full_dataset,
                wavtokenizer, speaker_encoder, device,
                distance_capture_noisy, distance_capture_clean
            )
            
            # ⭐ 立即寫入 HDF5（不累積在記憶體）
            append_to_hdf5(train_datasets, processed)
            
            if (i // args.batch_size) % 50 == 0:
                torch.cuda.empty_cache()
                h5f.flush()  # 定期同步到磁碟
        
        # 更新訓練集樣本數
        train_group.attrs['num_samples'] = train_datasets['noisy_tokens'].shape[0]
        print(f"✓ 訓練集寫入完成: {train_group.attrs['num_samples']} 樣本")
        
        # ⭐ 預處理驗證集（串流式寫入）
        print("\n" + "=" * 80)
        print("預處理驗證集（串流式寫入 HDF5）...")
        print("=" * 80)
        
        for i in tqdm(range(0, len(val_indices), args.batch_size), desc="驗證集"):
            batch_indices = val_indices[i:i + args.batch_size]
            batch_data = [full_dataset[idx] for idx in batch_indices]
            
            processed = preprocess_batch_with_distances(
                batch_data, batch_indices, full_dataset,
                wavtokenizer, speaker_encoder, device,
                distance_capture_noisy, distance_capture_clean
            )
            
            # ⭐ 立即寫入 HDF5（不累積在記憶體）
            append_to_hdf5(val_datasets, processed)
            
            if (i // args.batch_size) % 50 == 0:
                torch.cuda.empty_cache()
                h5f.flush()
        
        # 更新驗證集樣本數
        val_group.attrs['num_samples'] = val_datasets['noisy_tokens'].shape[0]
        print(f"✓ 驗證集寫入完成: {val_group.attrs['num_samples']} 樣本")
    
    print("\n" + "=" * 80)
    print("✅ 預處理完成！")
    print("=" * 80)
    print(f"HDF5 文件: {h5_path}")
    
    # 顯示文件大小
    file_size_gb = h5_path.stat().st_size / (1024**3)
    print(f"文件大小: {file_size_gb:.2f} GB (含 gzip 壓縮)")
    
    print("\n使用方式:")
    print("  from data_zeroshot_hdf5 import HDF5ZeroShotDataset")
    print(f"  train_dataset = HDF5ZeroShotDataset('{h5_path}', split='train')")
    print(f"  val_dataset = HDF5ZeroShotDataset('{h5_path}', split='val')")


if __name__ == '__main__':
    main()
