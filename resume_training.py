import torch
from torch.utils.data import DataLoader
from tsne import *  # 導入 tsne.py 的所有內容
from try3 import AudioDataset, collate_fn  # 從 try3.py 導入
from tsne import EnhancedWavTokenizer  # 從 tsne.py 導入

def resume_training():
    # 1. 配置設定
    config = {
        'config_path': "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        'model_path': "./wavtokenizer_large_speech_320_24k.ckpt",        
        'save_dir': './tout2',
        'epochs': 2000,
        'batch_size': 4,
        'learning_rate': 2e-4,
        'weight_decay': 0.001,
        'feature_scale': 1.5
    }

    # 2. 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 3. 初始化模型
    model = EnhancedWavTokenizer(config['config_path'], config['model_path']).to(device)

    # 4. 初始化優化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 5. 載入檢查點
    checkpoint_path = "./tout2/checkpoint_epoch_600.pt"
    print(f"正在載入檢查點: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"成功載入檢查點，從第 {start_epoch} 個 epoch 繼續訓練")
    except Exception as e:
        print(f"載入檢查點時發生錯誤: {str(e)}")
        return

    # 6. 初始化數據集和加載器
    dataset = AudioDataset(
        input_dirs=["./box", "./plastic", "./papercup"],
        target_dir="./box2"
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 7. 繼續訓練
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        save_dir=config['save_dir'],
        num_epochs=config['epochs']
    )

if __name__ == "__main__":
    resume_training()