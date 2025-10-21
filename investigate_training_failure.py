#!/usr/bin/env python3
"""
檢查為什麼模型沒有學會預測 tokens

實驗編號: EXP20251021_01
生成函式: investigate_training_failure
"""

import torch

print("="*80)
print("調查訓練失敗原因 - EXP20251021_01")
print("="*80)

# 檢查不同 epochs 的 checkpoints
epochs_to_check = [100, 200, 300, 400]

base_path = "results/transformer_large_tokenloss_large_tokenloss_202510200359"

print("\n檢查各 epoch 的 checkpoint:")
print("-" * 80)

for epoch in epochs_to_check:
    ckpt_path = f"{base_path}/checkpoint_epoch_{epoch}.pth"
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        print(f"\nEpoch {epoch}:")
        print(f"  Loss (儲存值): {checkpoint.get('loss', 'N/A')}")
        print(f"  Config keys:   {list(checkpoint.get('config', {}).keys())}")
        
        # 檢查 optimizer 狀態
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            if 'param_groups' in opt_state:
                lr = opt_state['param_groups'][0].get('lr', 'N/A')
                print(f"  Learning Rate: {lr}")
        
    except FileNotFoundError:
        print(f"\nEpoch {epoch}: ❌ Checkpoint 不存在")
    except Exception as e:
        print(f"\nEpoch {epoch}: ❌ 載入失敗 - {e}")

# 檢查配置
print("\n" + "="*80)
print("檢查訓練腳本配置")
print("="*80)

import subprocess

result = subprocess.run(
    ['grep', '-A', '5', '--ce_weight', 'run_transformer_large_tokenloss.sh'],
    capture_output=True,
    text=True
)

print(result.stdout)

# 檢查模型初始化時的參數
print("\n" + "="*80)
print("檢查模型訓練參數")
print("="*80)

result2 = subprocess.run(
    ['ps', 'aux'],
    capture_output=True,
    text=True
)

for line in result2.stdout.split('\n'):
    if 'wavtokenizer_transformer_denoising.py' in line and 'grep' not in line:
        print("當前訓練程序:")
        # 解析參數
        if '--ce_weight' in line:
            ce_idx = line.find('--ce_weight')
            ce_part = line[ce_idx:ce_idx+30]
            print(f"  {ce_part}")
        
        if '--learning_rate' in line or '--lr' in line:
            lr_idx = line.find('--learning_rate') if '--learning_rate' in line else line.find('--lr')
            lr_part = line[lr_idx:lr_idx+30] if lr_idx > 0 else "未找到"
            print(f"  {lr_part}")
        
        break

print("\n" + "="*80)
print("🔍 可能的問題")
print("="*80)

print("\n1. Learning Rate 問題：")
print("   如果 LR 過小，模型學習太慢")
print("   如果 LR 過大，可能不收斂")
print("   建議檢查: 1e-4 到 1e-3 之間較合適")

print("\n2. Checkpoint Loss = 1000000.0：")
print("   這是一個異常值！")
print("   可能原因：")
print("   - Validation loss 計算錯誤")
print("   - 儲存時使用了錯誤的 loss 值")
print("   - 實際訓練 loss 可能正常")

print("\n3. CE Loss = 8.59 (接近隨機)：")
print("   log(4096) = 8.32")
print("   當前 CE = 8.59 只比隨機好一點點")
print("   表示模型幾乎沒學到任何東西")

print("\n4. 可能的根本原因：")
print("   ❌ Wavtokenizer embeddings 被凍結")
print("   ❌ Transformer 參數沒有被訓練")
print("   ❌ Gradient 沒有正確傳遞")
print("   ❌ Learning rate 過小")

print("\n" + "="*80)
print("💡 建議診斷步驟")
print("="*80)

print("\n1. 檢查模型參數是否真的有更新：")
print("   比較 Epoch 100 和 Epoch 400 的參數值")

print("\n2. 檢查梯度：")
print("   在訓練過程中打印各層的梯度範數")

print("\n3. 檢查 learning rate schedule：")
print("   確認 LR 沒有衰減到太小")

print("\n4. 檢查損失計算：")
print("   確認 CE Loss 真的被用於反向傳播")

print("\n" + "="*80)
