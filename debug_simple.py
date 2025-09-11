import torch
import torch.nn.functional as F
from ttdata import AudioDataset  
from torch.utils.data import DataLoader, Subset
import os


def debug_collate_fn_simple(batch):
    """簡化的調試函數，檢查batch格式"""
    print(f"DEBUG: batch type: {type(batch)}")
    print(f"DEBUG: batch length: {len(batch)}")
    
    for i, item in enumerate(batch):
        print(f"  Item {i}: type={type(item)}, content={item}")
        if hasattr(item, '__len__'):
            print(f"    Length: {len(item)}")
        
    # 停止執行避免進一步錯誤
    raise RuntimeError("Debug completed - stopping execution")


def main():
    print("創建數據集...")
    input_dirs = [os.path.join(os.getcwd(), "data", "raw", "box")]
    target_dir = os.path.join(os.getcwd(), "data", "clean", "box2")
    
    dataset = AudioDataset(input_dirs, target_dir)

    print(f"數據集大小: {len(dataset)}")
    
    # 驗證集語者
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
    
    print(f"訓練集索引數量: {len(train_indices)}")
    print(f"驗證集索引數量: {len(val_indices)}")

    # 創建一個小子集來調試
    train_dataset = Subset(dataset, train_indices[:4])
    
    # 測試數據項直接獲取
    print("\n直接測試數據項:")
    item = train_dataset[0]
    print(f"Dataset[0]: type={type(item)}, length={len(item)}")
    
    # 創建DataLoader使用簡化調試函數  
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=debug_collate_fn_simple
    )
    
    print("\n準備測試DataLoader...")
    
    try:
        for batch in train_loader:
            print("成功獲得批次")
            break
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
