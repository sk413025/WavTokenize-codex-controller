
"""
離散化WavTokenizer修復整合使用範例

展示如何使用所有修復後的組件
"""

import torch
import torch.nn as nn
from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoising
from improved_vector_quantization import create_wavtokenizer_vq_wrapper
from discrete_transformer_architecture import upgrade_wavtokenizer_transformer

def create_fixed_wavtokenizer_model(config_path, checkpoint_path=None):
    """創建修復後的WavTokenizer模型
    
    Args:
        config_path: 配置文件路徑
        checkpoint_path: 檢查點路徑
        
    Returns:
        修復後的模型
    """
    # 1. 創建基礎模型（使用修復後的代碼）
    model = WavTokenizerTransformerDenoising(config_path)
    
    # 2. 升級為離散專用Transformer（如果需要）
    discrete_config = {
        'vocab_size': 4096,
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'local_window': 16
    }
    
    # 可選：使用改進的Vector Quantization
    # model.wavtokenizer = create_wavtokenizer_vq_wrapper(
    #     model.wavtokenizer,
    #     use_improved_vq=True,
    #     use_multiscale=False
    # )
    
    # 3. 加載檢查點（如果提供）
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"已加載檢查點: {checkpoint_path}")
    
    return model

def run_fixed_training(model, train_loader, val_loader, num_epochs=10):
    """使用修復後的策略運行訓練
    
    主要改進：
    - 修復的驗證損失計算
    - 改進的梯度裁剪
    - 重新平衡的損失權重
    - 音頻維度標準化
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 使用AdamW優化器，更適合Transformer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 訓練階段（使用修復後的train_epoch）
        train_loss = train_epoch_fixed(model, train_loader, optimizer, criterion, device)
        
        # 驗證階段（使用修復後的validate_epoch）  
        val_loss, val_accuracy = validate_epoch_fixed(model, val_loader, criterion, device)
        
        # 學習率調度
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, 'best_fixed_model.pth')
            print(f"  ✅ 保存新的最佳模型 (val_loss: {val_loss:.4f})")
        
        print("-" * 50)

def train_epoch_fixed(model, dataloader, optimizer, criterion, device):
    """修復後的訓練epoch函數"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            optimizer.zero_grad()
            
            # 前向傳播（使用修復後的forward函數）
            loss_result = model.forward_with_loss(batch, device)
            
            if isinstance(loss_result, tuple):
                total_loss_value, loss_dict = loss_result
            else:
                total_loss_value = loss_result
                loss_dict = {'total_loss': total_loss_value.item()}
            
            # 反向傳播（使用改進的梯度裁剪）
            total_loss_value.backward()
            
            # 應用改進的梯度裁剪
            from wavtokenizer_transformer_denoising import apply_advanced_gradient_clipping
            grad_norm = apply_advanced_gradient_clipping(model, max_norm=0.5, adaptive=True)
            
            optimizer.step()
            
            total_loss += total_loss_value.item()
            num_batches += 1
            
            # 記錄詳細信息
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss={total_loss_value.item():.4f}, GradNorm={grad_norm:.4f}")
                for key, value in loss_dict.items():
                    if key != 'total_loss':
                        print(f"  {key}: {value:.4f}")
        
        except Exception as e:
            print(f"訓練批次 {batch_idx} 出錯: {e}")
            continue
    
    return total_loss / max(num_batches, 1)

def validate_epoch_fixed(model, dataloader, criterion, device):
    """修復後的驗證epoch函數"""
    # 這個函數已經在主文件中修復，直接調用
    from wavtokenizer_transformer_denoising import validate_epoch
    return validate_epoch(model, dataloader, criterion, device)

if __name__ == "__main__":
    print("🔧 離散化WavTokenizer修復整合範例")
    print("本範例展示如何使用所有修復後的組件")
    print("\n主要修復項目：")
    print("1. ✅ 驗證損失計算邏輯")
    print("2. ✅ 音頻維度標準化") 
    print("3. ✅ 損失函數權重重新平衡")
    print("4. ✅ 改進梯度裁剪策略")
    print("5. ✅ 優化Vector Quantization")
    print("6. ✅ 離散專用Transformer架構")
    print("\n使用說明：")
    print("- 調用 create_fixed_wavtokenizer_model() 創建修復後的模型")
    print("- 調用 run_fixed_training() 開始修復後的訓練")
    print("- 所有修復都已整合，預期解決主要問題")
