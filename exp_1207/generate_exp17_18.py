#!/usr/bin/env python3
"""
自動生成 exp17 (Margin Loss) 和 exp18 (Curriculum) 的訓練程式

這個腳本會讀取 train_with_ce.py，進行必要的修改，然後生成兩個新的訓練程式
"""

import re
from pathlib import Path

def create_exp17_margin_loss():
    """創建 exp17: Margin Loss 訓練程式"""
    
    # 讀取原始檔案
    base_file = Path(__file__).parent / 'train_with_ce.py'
    with open(base_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 修改文檔字串
    content = content.replace(
        '"""exp16: Feature Loss + Cross-Entropy Loss 訓練',
        '"""exp17: Margin-based Contrastive Loss 訓練'
    )
    
    # 2. 在 import 後添加 margin loss 函式
    import_end = content.find('from exp_1201.wavtok_lora_patch import apply_lora_patch')
    insert_pos = content.find('\n', import_end) + 1
    
    margin_loss_code = '''
# ==================== Margin Loss 函式 (exp17) ====================

def compute_margin_loss(student_encoder_out, teacher_codes, codebook, margin=0.5):
    """
    計算 Margin-based Contrastive Loss
    
    確保 student embedding 在正確的 Voronoi cell 內部：
    loss = max(0, d_correct - d_nearest_wrong + margin)
    """
    B, C, T = student_encoder_out.shape
    
    # student_encoder_out: (B, C, T) -> (B, T, C)
    z = student_encoder_out.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
    
    # 計算到所有 codebook 的距離
    distances = torch.cdist(z, codebook, p=2)  # (B*T, num_codes)
    
    # 獲取 teacher codes
    if teacher_codes.dim() == 3:
        t_codes = teacher_codes[0]
    else:
        t_codes = teacher_codes.squeeze(1) if teacher_codes.dim() > 2 else teacher_codes
    
    correct_codes = t_codes.reshape(-1).long()  # (B*T,)
    
    # 到正確 token 的距離
    correct_dist = distances.gather(1, correct_codes.unsqueeze(1)).squeeze(1)
    
    # 到最近錯誤 token 的距離
    mask = torch.ones_like(distances, dtype=torch.bool)
    mask.scatter_(1, correct_codes.unsqueeze(1), False)
    masked_distances = distances.clone()
    masked_distances[~mask] = float('inf')
    nearest_wrong_dist = masked_distances.min(dim=1)[0]
    
    # Margin Loss
    margin_loss = torch.clamp(correct_dist - nearest_wrong_dist + margin, min=0).mean()
    
    return margin_loss, correct_dist.mean().item(), nearest_wrong_dist.mean().item()

'''
    content = content[:insert_pos] + margin_loss_code + content[insert_pos:]
    
    # 3. 修改 compute_losses 函式
    # 找到函式定義
    pattern = r'def compute_losses\(model, output, distance_matrix, feature_weight, ce_weight, ce_temperature=0\.1\):'
    replacement = 'def compute_losses(model, output, distance_matrix, margin, ce_weight, ce_temperature=0.1):'
    content = re.sub(pattern, replacement, content)
    
    # 修改函式內部 - 替換 feature_loss 為 margin_loss
    pattern = r'# 1\. Feature Loss.*?feature_loss = F\.mse_loss\(student_encoder_out, teacher_encoder_out\)'
    replacement = '''# 1. Margin Loss（替換 MSE Loss）
    margin_loss_val, mean_correct_dist, mean_nearest_wrong_dist = compute_margin_loss(
        student_encoder_out, teacher_codes, model.codebook, margin=margin
    )'''
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 修改 total_loss 計算
    pattern = r'total_loss = feature_weight \* feature_loss \+ ce_weight \* ce_loss'
    replacement = 'total_loss = margin_loss_val + ce_weight * ce_loss'
    content = re.sub(pattern, replacement, content)
    
    # 修改返回值
    pattern = r"'total_loss': total_loss,\s+'feature_loss': feature_loss,"
    replacement = "'total_loss': total_loss,\\n        'margin_loss': margin_loss_val,\\n        'mean_correct_dist': mean_correct_dist,\\n        'mean_nearest_wrong_dist': mean_nearest_wrong_dist,"
    content = re.sub(pattern, replacement, content)
    
    # 4. 修改 train_epoch 簽名
    pattern = r'def train_epoch\(model, dataloader, optimizer, scheduler, device, epoch,\s+distance_matrix, feature_weight, ce_weight,'
    replacement = 'def train_epoch(model, dataloader, optimizer, scheduler, device, epoch,\\n                distance_matrix, margin, ce_weight,'
    content = re.sub(pattern, replacement, content)
    
    # 修改 train_epoch 內的呼叫
    pattern = r'losses = compute_losses\(model, output, distance_matrix, feature_weight, ce_weight, ce_temperature\)'
    replacement = 'losses = compute_losses(model, output, distance_matrix, margin, ce_weight, ce_temperature)'
    content = re.sub(pattern, replacement, content)
    
    # 修改進度條顯示
    pattern = r"'feat': f\"{losses\['feature_loss'\]\.item\(\):.4f}\","
    replacement = "'margin': f\"{losses['margin_loss'].item():.4f}\","
    content = re.sub(pattern, replacement, content)
    
    # 5. 修改 validate 簽名和呼叫
    pattern = r'def validate\(model, dataloader, device, distance_matrix, feature_weight, ce_weight,'
    replacement = 'def validate(model, dataloader, device, distance_matrix, margin, ce_weight,'
    content = re.sub(pattern, replacement, content)
    
    pattern = r'losses = compute_losses\(model, output, distance_matrix, feature_weight, ce_weight, ce_temperature\)'
    replacement = 'losses = compute_losses(model, output, distance_matrix, margin, ce_weight, ce_temperature)'
    content = re.sub(pattern, replacement, content)
    
    # 6. 修改 argparse
    pattern = r"parser\.add_argument\('--feature_weight'.*?\)"
    replacement = "parser.add_argument('--margin', type=float, default=0.5,\\n                       help='Margin for contrastive loss')"
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 7. 修改 history 記錄
    content = content.replace("'train_feature_loss'", "'train_margin_loss'")
    content = content.replace("'val_feature_loss'", "'val_margin_loss'")
    content = content.replace("history['train_feature_loss']", "history['train_margin_loss']")
    content = content.replace("history['val_feature_loss']", "history['val_margin_loss']")
    
    # 8. 修改 main() 中的呼叫
    pattern = r'args\.feature_weight, args\.ce_weight'
    replacement = 'args.margin, args.ce_weight'
    content = re.sub(pattern, replacement, content)
    
    # 9. 修改實驗名稱預設值
    pattern = r"default='exp16_feature_ce'"
    replacement = "default='exp17_margin_loss'"
    content = re.sub(pattern, replacement, content)
    
    # 寫入新檔案
    output_file = Path(__file__).parent / 'train_margin_loss.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已生成 {output_file}")
    return output_file


def create_exp18_curriculum():
    """創建 exp18: Curriculum Learning 訓練程式"""
    
    # 讀取原始檔案
    base_file = Path(__file__).parent / 'train_with_ce.py'
    with open(base_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 修改文檔字串
    content = content.replace(
        '"""exp16: Feature Loss + Cross-Entropy Loss 訓練',
        '"""exp18: Curriculum Learning (Reverse) 訓練'
    )
    
    # 2. 在 import 後添加 curriculum 函式
    import_end = content.find('from exp_1201.wavtok_lora_patch import apply_lora_patch')
    insert_pos = content.find('\n', import_end) + 1
    
    curriculum_code = '''
# ==================== Curriculum Learning 函式 (exp18) ====================

def get_curriculum_weights(epoch, args):
    """
    根據 epoch 返回當前階段的 loss 權重
    
    Stage 1: CE only - 快速定位正確 Voronoi cell
    Stage 2: CE + MSE - 在 cell 內部精細調整
    Stage 3: MSE dominant - 穩定 embedding
    """
    if epoch <= args.stage1_epochs:
        return args.stage1_ce_weight, args.stage1_mse_weight, "Stage1_CE_only"
    elif epoch <= args.stage1_epochs + args.stage2_epochs:
        return args.stage2_ce_weight, args.stage2_mse_weight, "Stage2_CE+MSE"
    else:
        return args.stage3_ce_weight, args.stage3_mse_weight, "Stage3_MSE_dominant"

'''
    content = content[:insert_pos] + curriculum_code + content[insert_pos:]
    
    # 3. 修改 argparse - 添加 curriculum 參數
    # 找到 parser 定義的位置
    parser_pos = content.find("def main():")
    parser_pos = content.find("parser = argparse.ArgumentParser", parser_pos)
    
    # 在 LoRA 參數之前插入 curriculum 參數
    lora_pos = content.find("# LoRA", parser_pos)
    
    curriculum_args = """
    # Curriculum Learning Stages
    parser.add_argument('--stage1_epochs', type=int, default=10,
                       help='Stage 1: CE only')
    parser.add_argument('--stage1_ce_weight', type=float, default=1.0)
    parser.add_argument('--stage1_mse_weight', type=float, default=0.0)
    
    parser.add_argument('--stage2_epochs', type=int, default=20,
                       help='Stage 2: CE + MSE')
    parser.add_argument('--stage2_ce_weight', type=float, default=0.5)
    parser.add_argument('--stage2_mse_weight', type=float, default=1.0)
    
    parser.add_argument('--stage3_epochs', type=int, default=20,
                       help='Stage 3: MSE dominant')
    parser.add_argument('--stage3_ce_weight', type=float, default=0.1)
    parser.add_argument('--stage3_mse_weight', type=float, default=1.0)

    """
    content = content[:lora_pos] + curriculum_args + "    " + content[lora_pos:]
    
    # 4. 修改 main() 中的訓練迴圈 - 動態獲取權重
    # 找到 for epoch in range 的位置
    pattern = r'(for epoch in range\(1, args\.num_epochs \+ 1\):)\s+(# Train)'
    replacement = r'''\1
        # 動態獲取當前階段的權重
        ce_weight, mse_weight, stage_name = get_curriculum_weights(epoch, args)
        
        print(f"\\nEpoch {epoch}/{args.num_epochs} - {stage_name}")
        print(f"  Loss weights: MSE={mse_weight:.1f}, CE={ce_weight:.1f}")
        
        \2'''
    content = re.sub(pattern, replacement, content)
    
    # 修改 train_epoch 和 validate 的呼叫，使用動態權重
    pattern = r'train_epoch\(\s+model, train_loader, optimizer, scheduler, device, epoch, distance_matrix,\s+args\.feature_weight, args\.ce_weight,'
    replacement = 'train_epoch(\\n            model, train_loader, optimizer, scheduler, device, epoch, distance_matrix,\\n            mse_weight, ce_weight,'
    content = re.sub(pattern, replacement, content)
    
    pattern = r'validate\(model, val_loader, device, distance_matrix,\s+args\.feature_weight, args\.ce_weight,'
    replacement = 'validate(model, val_loader, device, distance_matrix,\\n                              mse_weight, ce_weight,'
    content = re.sub(pattern, replacement, content)
    
    # 5. 修改實驗名稱預設值
    pattern = r"default='exp16_feature_ce'"
    replacement = "default='exp18_reverse_curriculum'"
    content = re.sub(pattern, replacement, content)
    
    # 6. 修改總 epochs 預設值為 50
    pattern = r"parser\.add_argument\('--num_epochs', type=int, default=30\)"
    replacement = "parser.add_argument('--num_epochs', type=int, default=50,\\n                       help='Total epochs (should match stage1 + stage2 + stage3)')"
    content = re.sub(pattern, replacement, content)
    
    # 寫入新檔案
    output_file = Path(__file__).parent / 'train_curriculum.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已生成 {output_file}")
    return output_file


if __name__ == '__main__':
    print("="*60)
    print("生成 exp17 和 exp18 訓練程式")
    print("="*60)
    print()
    
    # 生成 exp17
    print("生成 exp17 (Margin Loss)...")
    exp17_file = create_exp17_margin_loss()
    
    print()
    
    # 生成 exp18
    print("生成 exp18 (Curriculum Learning)...")
    exp18_file = create_exp18_curriculum()
    
    print()
    print("="*60)
    print("完成！")
    print("="*60)
    print()
    print("下一步：")
    print("  1. 執行 exp17: ./run_exp17_margin.sh")
    print("  2. 執行 exp18: ./run_exp18_curriculum.sh")
    print()
