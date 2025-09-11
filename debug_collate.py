#!/usr/bin/env python3
"""
調試collate_fn函數以查看確切的數據結構
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from ttdata import AudioDataset
from ttt2 import collate_fn

def debug_collate_fn(batch, trim_to_shortest=True):
    """Debug版本的collate_fn，詳細打印數據結構"""
    print(f"\n=== DEBUG collate_fn ===")
    print(f"批次大小: {len(batch)}")
    print(f"批次類型: {type(batch)}")
    
    for i, item in enumerate(batch):
        print(f"  項目 {i}:")
        print(f"    類型: {type(item)}")
        print(f"    長度: {len(item) if hasattr(item, '__len__') else '無長度'}")
        
        if isinstance(item, (tuple, list)):
            for j, elem in enumerate(item):
                print(f"      索引 {j}: 類型={type(elem)}, 形狀={getattr(elem, 'shape', '無形狀')}")
        else:
            print(f"      內容: {item}")
    
    # 嘗試原來的處理方式
    try:
        input_wavs = [item[0] for item in batch]
        print(f"✓ 成功提取 input_wavs，長度: {len(input_wavs)}")
        return collate_fn(batch, trim_to_shortest)
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        print(f"錯誤類型: {type(e)}")
        raise e

def main():
    # 創建數據集（與主程序相同）
    input_dirs = [os.path.join(os.getcwd(), 'data', 'raw', 'box')]
    target_dir = os.path.join(os.getcwd(), 'data', 'clean', 'box2')
    
    dataset = AudioDataset(input_dirs, target_dir, max_sentences_per_speaker=100)
    print(f'數據集大小: {len(dataset)}')
    
    # 創建speaker-based split（與主程序相同）
    val_speakers = ['girl9', 'boy7']
    train_indices = []
    val_indices = []
    
    for idx, (input_wav, target_wav, content_id) in enumerate(dataset):
        input_path = dataset.paired_files[idx]['input']
        filename = os.path.basename(input_path)
        parts = filename.split('_')
        if len(parts) >= 2:
            speaker = parts[1]
            if speaker in val_speakers:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
    
    print(f'訓練集索引數量: {len(train_indices)}')
    print(f'驗證集索引數量: {len(val_indices)}')
    
    # 創建train子集（只取前幾個進行調試）
    train_dataset = Subset(dataset, train_indices[:4])
    
    # 創建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,  # 設置為0以便調試
        collate_fn=debug_collate_fn
    )
    
    print('\n準備測試DataLoader...')
    
    # 測試第一個批次
    try:
        batch = next(iter(train_loader))
        print(f'\n成功！批次類型: {type(batch)}')
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            print(f'input shape: {batch[0].shape}')
            print(f'target shape: {batch[1].shape}')
    except Exception as e:
        print(f'\n錯誤: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
