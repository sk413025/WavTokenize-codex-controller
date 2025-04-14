import torch
import os
from tsne import EnhancedWavTokenizer, train_model
from try3 import AudioDataset, collate_fn
from torch.utils.data import DataLoader

def resume_training():
    TOTAL_EPOCHS = 2000  # 設定總共要訓練的輪數
    
    config = {
        'config_path': "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        'model_path': "./wavtokenizer_large_speech_320_24k.ckpt",        
        'save_dir': './tout2',
        # 移除手動設定的 epochs
        'batch_size': 4,
        'learning_rate': 2e-4,
        'weight_decay': 0.001,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2,
    }
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 在載入前顯示檢查點路徑並等待確認
    best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
    print(f"\nPreparing to load checkpoint from: {best_model_path}")
    input("Press Enter to continue...")  # 等待使用者按下 Enter

    
    # 初始化模型
    model = EnhancedWavTokenizer(
        config['config_path'], 
        config['model_path']
    ).to(device)
    
    # 載入 checkpoint
    best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device)

    # 顯示檢查點內容
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"- model_state_dict: [Parameters dictionary]")
        elif key == 'optimizer_state_dict':
            print(f"- optimizer_state_dict: [Optimizer dictionary]")
        else:
            print(f"- {key}: {checkpoint[key]}")
    
    input("\nPress Enter to continue with training...")  # 再次等待確認
 

    # 計算剩餘要訓練的輪數
    completed_epochs = checkpoint['epoch']
    remaining_epochs = TOTAL_EPOCHS - completed_epochs
    
    print(f"\nTraining Progress:")
    print(f"Total planned epochs: {TOTAL_EPOCHS}")
    print(f"Completed epochs: {completed_epochs}")
    print(f"Remaining epochs: {remaining_epochs}")
    
    
    # 載入模型狀態
    model.load_state_dict(checkpoint['model_state_dict'])
    best_loss = checkpoint['loss']  # 獲取之前的最佳 loss
    start_epoch = checkpoint['epoch']

    print(f"Loaded checkpoint from epoch {start_epoch}")
    print(f"Previous best loss: {best_loss:.6f}")
    
    # 初始化優化器
    optimizer = torch.optim.AdamW(
        model.feature_extractor.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 載入優化器狀態
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 載入資料集
    dataset = AudioDataset(
        input_dirs=["./box", "./plastic", "./papercup"],
        target_dir="./box2"
    )
    
    # 初始化 DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True
    )
    
    # 修正這行，移除對 config['epochs'] 的引用
    print(f"Starting training from epoch {checkpoint['epoch']} for {remaining_epochs} more epochs")
    # 修正這行，直接使用 checkpoint['loss']
    print(f"Previous best loss: {checkpoint['loss']:.4f}")
    
    # 繼續訓練
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        save_dir=config['save_dir'],
        num_epochs=remaining_epochs,  # 使用計算出的剩餘輪數
    )

if __name__ == "__main__":
    resume_training()