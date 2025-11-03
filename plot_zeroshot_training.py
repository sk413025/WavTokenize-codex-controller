"""
繪製 Zero-Shot 訓練曲線

從訓練日誌中提取 loss 和 accuracy，繪製訓練曲線
"""

import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_training_log(log_path):
    """
    解析訓練日誌
    
    Returns:
        dict: {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    """
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    current_epoch = None
    
    with open(log_path, 'r') as f:
        for line in f:
            # 匹配 Epoch 行: "Epoch 1/100"
            epoch_match = re.search(r'Epoch (\d+)/\d+', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            
            # 匹配 Train 行: "Train - Loss: 5.6763, Acc: 23.86%"
            train_match = re.search(r'Train - Loss: ([\d.]+), Acc: ([\d.]+)%', line)
            if train_match and current_epoch is not None:
                train_losses.append(float(train_match.group(1)))
                train_accs.append(float(train_match.group(2)))
                continue
            
            # 匹配 Val 行: "Val   - Loss: 5.6381, Acc: 17.30%"
            val_match = re.search(r'Val\s+- Loss: ([\d.]+), Acc: ([\d.]+)%', line)
            if val_match and current_epoch is not None:
                val_losses.append(float(val_match.group(1)))
                val_accs.append(float(val_match.group(2)))
                epochs.append(current_epoch)
                continue
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }


def plot_training_curves(data, output_path):
    """
    繪製訓練曲線
    
    Args:
        data: 訓練數據字典
        output_path: 輸出圖片路徑
    """
    epochs = data['epochs']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Loss 曲線
    axes[0].plot(epochs, data['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, data['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('CrossEntropy Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 標註最佳點
    min_val_loss_idx = data['val_loss'].index(min(data['val_loss']))
    min_val_loss_epoch = epochs[min_val_loss_idx]
    min_val_loss = data['val_loss'][min_val_loss_idx]
    axes[0].plot(min_val_loss_epoch, min_val_loss, 'r*', markersize=15, 
                label=f'Best Val Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch})')
    axes[0].legend(fontsize=10)
    
    # 2. Accuracy 曲線
    axes[1].plot(epochs, data['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, data['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Token Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # 標註最佳點
    max_val_acc_idx = data['val_acc'].index(max(data['val_acc']))
    max_val_acc_epoch = epochs[max_val_acc_idx]
    max_val_acc = data['val_acc'][max_val_acc_idx]
    axes[1].plot(max_val_acc_epoch, max_val_acc, 'r*', markersize=15,
                label=f'Best Val Acc: {max_val_acc:.2f}% (Epoch {max_val_acc_epoch})')
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 訓練曲線已保存至: {output_path}")


def print_training_summary(data):
    """打印訓練摘要"""
    epochs = data['epochs']
    
    print("=" * 80)
    print("Zero-Shot 訓練結果摘要")
    print("=" * 80)
    print(f"總訓練 Epochs: {len(epochs)}")
    print()
    
    # 最佳 Val Loss
    min_val_loss_idx = data['val_loss'].index(min(data['val_loss']))
    print(f"最佳驗證損失:")
    print(f"  - Epoch: {epochs[min_val_loss_idx]}")
    print(f"  - Val Loss: {data['val_loss'][min_val_loss_idx]:.4f}")
    print(f"  - Val Acc: {data['val_acc'][min_val_loss_idx]:.2f}%")
    print()
    
    # 最佳 Val Acc
    max_val_acc_idx = data['val_acc'].index(max(data['val_acc']))
    print(f"最佳驗證準確率:")
    print(f"  - Epoch: {epochs[max_val_acc_idx]}")
    print(f"  - Val Loss: {data['val_loss'][max_val_acc_idx]:.4f}")
    print(f"  - Val Acc: {data['val_acc'][max_val_acc_idx]:.2f}%")
    print()
    
    # 最終 Epoch 結果
    print(f"最終 Epoch ({epochs[-1]}) 結果:")
    print(f"  - Train Loss: {data['train_loss'][-1]:.4f}")
    print(f"  - Train Acc: {data['train_acc'][-1]:.2f}%")
    print(f"  - Val Loss: {data['val_loss'][-1]:.4f}")
    print(f"  - Val Acc: {data['val_acc'][-1]:.2f}%")
    print()
    
    # 訓練改善幅度
    initial_train_loss = data['train_loss'][0]
    final_train_loss = data['train_loss'][-1]
    initial_val_acc = data['val_acc'][0]
    final_val_acc = data['val_acc'][-1]
    
    print(f"訓練改善:")
    print(f"  - Train Loss: {initial_train_loss:.4f} → {final_train_loss:.4f} "
          f"({(initial_train_loss - final_train_loss) / initial_train_loss * 100:.1f}% 改善)")
    print(f"  - Val Acc: {initial_val_acc:.2f}% → {final_val_acc:.2f}% "
          f"(+{final_val_acc - initial_val_acc:.2f}%)")
    print("=" * 80)


def main():
    # 設定路徑
    log_path = Path('/home/sbplab/ruizi/c_code/done/exp/results/zeroshot_full_20251101_083849/training.log')
    output_path = Path('/home/sbplab/ruizi/c_code/done/exp/results/zeroshot_full_20251101_083849/training_curves.png')
    
    print(f"解析訓練日誌: {log_path}")
    
    # 解析日誌
    data = parse_training_log(log_path)
    
    if not data['epochs']:
        print("❌ 錯誤: 未能從日誌中解析出訓練數據")
        return
    
    print(f"✓ 成功解析 {len(data['epochs'])} 個 epochs 的數據")
    print()
    
    # 打印摘要
    print_training_summary(data)
    print()
    
    # 繪製曲線
    print(f"繪製訓練曲線...")
    plot_training_curves(data, output_path)
    
    print()
    print(f"完成！結果保存在:")
    print(f"  - 訓練曲線圖: {output_path}")


if __name__ == '__main__':
    main()
