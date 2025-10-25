# Token Denoising Hybrid Loss 訓練問題修復報告
## 實驗編號: EXP20251024_HYBRID_LOSS_FIX
## 日期: 2024-10-24

## 問題診斷

### 1. 發現的問題

#### 問題 1: 音頻樣本未保存
- **現象**: 每 100 epochs 應該保存音頻，但 audio_samples 目錄下只有 epoch_100, epoch_200, epoch_300, epoch_400，沒有更新的樣本
- **根本原因**: `save_audio_samples` 函數在 epoch 100, 200, 300, 400 時均失敗
- **錯誤信息**: `保存音頻樣本時出錯:` (詳細錯誤被截斷)
- **相關警告**: `Warning: SConv1d 需要 3D 輸入張量 [B, C, T]，但收到了 4D 張量 torch.Size([1, 1, 1, 95105])`

#### 問題 2: Loss 圖表未生成
- **現象**: 沒有 loss 曲線圖文件
- **根本原因**: 訓練腳本中缺少繪製 loss 圖表的代碼
- **預期**:每 50 epochs 應該保存 loss 曲線圖

#### 問題 3: Checkpoint 保存策略不一致
- **現象**: checkpoint 每 10 epochs 保存一次（正確）
- **建議**: 考慮每 100 epochs 保存一次以節省磁碟空間

### 2. 當前訓練狀態

- **當前 Epoch**: 417 / 600
- **訓練進度**: 69.5%
- **訓練 Loss**: ~0.71 (穩定下降)
- **驗證 Loss**: ~12.7 (波動，可能過擬合)
- **訓練 Accuracy**: ~79%
- **驗證 Accuracy**: ~22.7%
- **訓練時間**: 已運行約 11414 分鐘 (~190 小時)

### 3. 修復方案

#### 修復 1: 音頻保存函數
**問題**: tokens 維度處理不正確，導致 WavTokenizer 解碼失敗

**解決方案**:
```python
def save_audio_samples_fixed(
    wavtokenizer,
    noisy_tokens,
    pred_tokens,
    clean_tokens,
    epoch,
    output_dir,
    device,
    num_samples=3
):
    """修復後的音頻保存函數"""
    samples_dir = Path(output_dir) / 'audio_samples' / f'epoch_{epoch}'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(num_samples, noisy_tokens.size(0))
    
    with torch.no_grad():
        for i in range(num_samples):
            # 正確的維度: (1, 1, T) - (n_q=1, batch=1, seq_len=T)
            noisy_tok = noisy_tokens[i:i+1].unsqueeze(0)  # (1, T) -> (1, 1, T)
            pred_tok = pred_tokens[i:i+1].unsqueeze(0)
            clean_tok = clean_tokens[i:i+1].unsqueeze(0)
            
            # 解碼為音頻
            noisy_features = wavtokenizer.codes_to_features(noisy_tok)
            pred_features = wavtokenizer.codes_to_features(pred_tok)
            clean_features = wavtokenizer.codes_to_features(clean_tok)
            
            noisy_audio = wavtokenizer.decode(noisy_features)
            pred_audio = wavtokenizer.decode(pred_features)
            clean_audio = wavtokenizer.decode(clean_features)
            
            # 保證音頻維度正確: (1, 1, T) -> (1, T)
            if noisy_audio.dim() == 3:
                noisy_audio = noisy_audio.squeeze(1)
            if pred_audio.dim() == 3:
                pred_audio = pred_audio.squeeze(1)
            if clean_audio.dim() == 3:
                clean_audio = clean_audio.squeeze(1)
            
            # 保存音頻
            torchaudio.save(
                str(samples_dir / f'sample_{i}_noisy.wav'),
                noisy_audio.cpu(),
                24000
            )
            torchaudio.save(
                str(samples_dir / f'sample_{i}_predicted.wav'),
                pred_audio.cpu(),
                24000
            )
            torchaudio.save(
                str(samples_dir / f'sample_{i}_clean.wav'),
                clean_audio.cpu(),
                24000
            )
            
            # 繪製頻譜圖
            try:
                plot_spectrograms(
                    noisy_audio.cpu().squeeze().numpy(),
                    pred_audio.cpu().squeeze().numpy(),
                    clean_audio.cpu().squeeze().numpy(),
                    str(samples_dir / f'sample_{i}_spectrogram.png')
                )
            except Exception as e:
                print(f"Warning: 無法繪製頻譜圖 {i}: {e}")
```

#### 修復 2: 添加 Loss 圖表繪製
**添加位置**: 訓練 loop 中，每 50 epochs 執行一次

**解決方案**:
```python
import matplotlib.pyplot as plt

# 在訓練循環前初始化
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 在每個 epoch 後記錄
train_losses.append(train_metrics['total_loss'])
val_losses.append(val_metrics['total_loss'])
train_accuracies.append(train_metrics['accuracy'])
val_accuracies.append(val_metrics['accuracy'])

# 每 50 epochs 繪製
if epoch % 50 == 0 or epoch == args.num_epochs:
    plot_training_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        output_dir,
        epoch
    )

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir, epoch):
    """繪製訓練曲線"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Total Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title(f'Total Loss (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'g-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'orange', label='Val Acc', linewidth=2)
    ax2.set_title(f'Accuracy (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # CE Loss (如果有記錄)
    # ... 其他 loss components
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'plots' / f'training_curves_epoch_{epoch}.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存訓練曲線: {plot_path}")
```

### 4. 建議的操作流程

#### 選項 A: 繼續當前訓練 (不建議)
- 優點: 不中斷訓練
- 缺點: 
  - 無法修復已有問題
  - 已有 epoch 的音頻和圖表無法補救
  - 驗證 loss 持續上升，可能過擬合

#### 選項 B: 停止當前訓練，修復後重啟 (建議)
- 優點:
  - 修復所有問題
  - 從 best checkpoint (最低 val loss) 重新開始
  - 可以調整超參數防止過擬合
- 缺點:
  - 需要重新訓練
  - 但由於已有 best_model.pth，可以從最佳狀態繼續

#### 選項 C: 停止訓練，直接測試當前模型 (快速驗證)
- 優點:
  - 快速評估當前模型效果
  - 決定是否值得繼續訓練
- 缺點:
  - 如果效果不好，浪費了訓練時間

### 5. 推薦方案: 選項 C + B

**Step 1**: 先停止當前訓練
```bash
kill 276500  # PID of current training process
```

**Step 2**: 使用 best_model.pth 或 checkpoint_epoch_410.pth 測試效果
```bash
cd /home/sbplab/ruizi/c_code/try
python test_model_audio_quality.py \\
    --checkpoint ../results/token_denoising_hybrid_loss_20251023_053633/best_model.pth \\
    --output_dir ../results/test_best_model \\
    --num_samples 10
```

**Step 3**: 根據測試結果決定：
- 如果效果好: 繼續訓練或使用此模型
- 如果效果差: 調整超參數重新訓練

**Step 4**: 如果決定繼續訓練，使用修復後的腳本
```bash
cd /home/sbplab/ruizi/c_code/try
nohup bash run_token_denoising_hybrid_fixed.sh > ../logs/hybrid_training_fixed_20251024.log 2>&1 &
```

## 後續實驗建議

### 防止過擬合的策略

1. **Early Stopping**: 如果驗證 loss 連續 20 epochs 沒有改善，提前停止
2. **增強 Regularization**: 
   - 增加 dropout (0.1 -> 0.2)
   - 增加 weight_decay (0.01 -> 0.05)
3. **學習率調整**: 使用 ReduceLROnPlateau 而非 CosineAnnealingLR
4. **數據增強**: 添加更多 noise variations

### 超參數調整建議

```python
# 當前設定
d_model = 512
nhead = 8
num_layers = 6
dropout = 0.1
weight_decay = 0.01

# 建議調整 (防止過擬合)
d_model = 512  # 保持
nhead = 8      # 保持
num_layers = 4  # 減少層數
dropout = 0.2   # 增加 dropout
weight_decay = 0.05  # 增加正則化
```

## 結論

目前訓練已經進行了 417 epochs，但存在以下問題:
1. 音頻樣本保存失敗 (每 100 epochs)
2. Loss 圖表未生成
3. 驗證 loss 持續上升，疑似過擬合

建議:
1. 立即停止當前訓練
2. 使用 best_model.pth 測試音頻質量
3. 根據測試結果決定是否繼續訓練或調整超參數重新訓練
4. 使用修復後的訓練腳本，確保音頻和圖表正確保存
