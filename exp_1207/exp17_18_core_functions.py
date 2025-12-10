"""
exp17 & exp18 核心函式

這個文件包含兩個實驗的核心修改函式，可以直接複製到訓練程式中
"""

import torch
import torch.nn.functional as F


# ==================== exp17: Margin Loss ====================

def compute_margin_loss(student_encoder_out, teacher_codes, codebook, margin=0.5):
    """
    計算 Margin-based Contrastive Loss
    
    確保 student embedding 在正確的 Voronoi cell 內部：
    loss = max(0, d_correct - d_nearest_wrong + margin)
    
    Args:
        student_encoder_out: (B, C, T) Student embedding (VQ 前)
        teacher_codes: (B, T) 或 (B, 1, T) Teacher token indices
        codebook: (num_codes, C) Codebook vectors
        margin: 安全邊距（預設 0.5）
        
    Returns:
        margin_loss: Scalar loss value
        mean_correct_dist: 到正確 token 的平均距離（監控）
        mean_nearest_wrong_dist: 到最近錯誤 token 的平均距離（監控）
    """
    B, C, T = student_encoder_out.shape
    num_codes = codebook.shape[0]
    
    # student_encoder_out: (B, C, T) -> (B, T, C)
    z = student_encoder_out.permute(0, 2, 1)  # (B, T, C)
    z_flat = z.reshape(-1, C)  # (B*T, C)
    
    # 計算到所有 codebook 的距離
    # z_flat: (B*T, C), codebook: (num_codes, C)
    # distances: (B*T, num_codes)
    distances = torch.cdist(z_flat, codebook, p=2)  # L2 distance
    
    # 獲取 teacher codes
    if teacher_codes.dim() == 3:
        t_codes = teacher_codes[0]  # (B, T)
    else:
        t_codes = teacher_codes.squeeze(1) if teacher_codes.dim() > 2 else teacher_codes
    
    correct_codes = t_codes.reshape(-1).long()  # (B*T,)
    
    # 到正確 token 的距離
    correct_dist = distances.gather(1, correct_codes.unsqueeze(1)).squeeze(1)  # (B*T,)
    
    # 到最近錯誤 token 的距離
    # Mask out correct token
    mask = torch.ones_like(distances, dtype=torch.bool)  # (B*T, num_codes)
    mask.scatter_(1, correct_codes.unsqueeze(1), False)  # Set correct token to False
    
    # 只考慮錯誤的 tokens
    masked_distances = distances.clone()
    masked_distances[~mask] = float('inf')  # 將正確 token 設為無窮大
    nearest_wrong_dist = masked_distances.min(dim=1)[0]  # (B*T,)
    
    # Margin Loss: 希望 correct_dist < nearest_wrong_dist - margin
    # loss = max(0, correct_dist - nearest_wrong_dist + margin)
    margin_loss = torch.clamp(correct_dist - nearest_wrong_dist + margin, min=0).mean()
    
    # 監控指標
    mean_correct_dist = correct_dist.mean().item()
    mean_nearest_wrong_dist = nearest_wrong_dist.mean().item()
    
    return margin_loss, mean_correct_dist, mean_nearest_wrong_dist


# ==================== exp18: Curriculum Learning ====================

def get_curriculum_weights(epoch, args):
    """
    根據 epoch 返回當前階段的 loss 權重
    
    Stage 1 (epoch 1-stage1_epochs): CE only
    Stage 2 (epoch stage1+1 to stage1+stage2): CE + MSE
    Stage 3 (epoch stage1+stage2+1 to end): MSE dominant
    
    Args:
        epoch: 當前 epoch (1-based)
        args: 包含 stage1_epochs, stage2_epochs, 
              stage1_ce_weight, stage1_mse_weight,
              stage2_ce_weight, stage2_mse_weight,
              stage3_ce_weight, stage3_mse_weight
              
    Returns:
        (ce_weight, mse_weight, stage_name)
    """
    if epoch <= args.stage1_epochs:
        # Stage 1: CE only
        return args.stage1_ce_weight, args.stage1_mse_weight, "Stage1_CE_only"
    elif epoch <= args.stage1_epochs + args.stage2_epochs:
        # Stage 2: CE + MSE
        return args.stage2_ce_weight, args.stage2_mse_weight, "Stage2_CE+MSE"
    else:
        # Stage 3: MSE dominant
        return args.stage3_ce_weight, args.stage3_mse_weight, "Stage3_MSE_dominant"


# ==================== 修改版 compute_losses (exp17) ====================

def compute_losses_margin(model, output, distance_matrix, margin, ce_weight, ce_temperature=0.1):
    """
    計算所有 losses (exp17: Margin Loss)
    
    取代原本的 compute_losses 函式
    
    Returns:
        dict with:
        - total_loss: margin_loss + ce_weight * ce_loss
        - margin_loss: Margin contrastive loss
        - ce_loss: CrossEntropy loss
        - mean_correct_dist: 監控指標
        - mean_nearest_wrong_dist: 監控指標
        - distance_loss: 監控指標
        - vq_loss: 監控指標
        - token_acc: 監控指標
    """
    student_encoder_out = output['student_encoder_out']
    teacher_encoder_out = output['teacher_encoder_out']
    student_codes = output['student_codes']
    teacher_codes = output['teacher_codes']
    vq_loss = output['vq_loss']

    # 1. Margin Loss（替換 MSE Loss）
    margin_loss_val, mean_correct_dist, mean_nearest_wrong_dist = compute_margin_loss(
        student_encoder_out, 
        teacher_codes, 
        model.codebook, 
        margin=margin
    )

    # 2. Cross-Entropy Loss
    logits = model.compute_ce_logits(student_encoder_out)
    
    if teacher_codes.dim() == 3:
        t_codes = teacher_codes[0]
    else:
        t_codes = teacher_codes.squeeze(1)

    B, T, num_codes = logits.shape
    logits_scaled = logits / ce_temperature
    logits_flat = logits_scaled.reshape(B * T, num_codes)
    targets_flat = t_codes.reshape(B * T).long()
    ce_loss = F.cross_entropy(logits_flat, targets_flat)

    # 3. Total Loss = Margin + CE（不使用 MSE）
    total_loss = margin_loss_val + ce_weight * ce_loss

    # 4. 監控指標
    with torch.no_grad():
        if student_codes.dim() == 3:
            s_codes = student_codes[0]
        else:
            s_codes = student_codes.squeeze(1)

        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        distances = distance_matrix[s_flat, t_flat]
        distance_loss = distances.mean().item()
        token_acc = (s_codes == t_codes).float().mean().item()

    vq_loss_val = vq_loss.item() if torch.is_tensor(vq_loss) else vq_loss

    return {
        'total_loss': total_loss,
        'margin_loss': margin_loss_val,
        'ce_loss': ce_loss,
        'mean_correct_dist': mean_correct_dist,
        'mean_nearest_wrong_dist': mean_nearest_wrong_dist,
        'distance_loss': distance_loss,
        'vq_loss': vq_loss_val,
        'token_acc': token_acc,
    }


# ==================== 修改版 compute_losses (exp18) ====================

def compute_losses_curriculum(model, output, distance_matrix, mse_weight, ce_weight, ce_temperature=0.1):
    """
    計算所有 losses (exp18: Curriculum Learning)
    
    與原本的 compute_losses 相同，但 weights 由外部動態傳入
    
    Returns:
        dict with:
        - total_loss: mse_weight * mse_loss + ce_weight * ce_loss
        - mse_loss: Feature MSE loss
        - ce_loss: CrossEntropy loss
        - distance_loss: 監控指標
        - vq_loss: 監控指標
        - token_acc: 監控指標
    """
    student_encoder_out = output['student_encoder_out']
    teacher_encoder_out = output['teacher_encoder_out']
    student_codes = output['student_codes']
    teacher_codes = output['teacher_codes']
    vq_loss = output['vq_loss']

    # 1. MSE Loss
    mse_loss = F.mse_loss(student_encoder_out, teacher_encoder_out)

    # 2. Cross-Entropy Loss
    logits = model.compute_ce_logits(student_encoder_out)
    
    if teacher_codes.dim() == 3:
        t_codes = teacher_codes[0]
    else:
        t_codes = teacher_codes.squeeze(1)

    B, T, num_codes = logits.shape
    logits_scaled = logits / ce_temperature
    logits_flat = logits_scaled.reshape(B * T, num_codes)
    targets_flat = t_codes.reshape(B * T).long()
    ce_loss = F.cross_entropy(logits_flat, targets_flat)

    # 3. Total Loss（動態權重）
    total_loss = mse_weight * mse_loss + ce_weight * ce_loss

    # 4. 監控指標
    with torch.no_grad():
        if student_codes.dim() == 3:
            s_codes = student_codes[0]
        else:
            s_codes = student_codes.squeeze(1)

        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        distances = distance_matrix[s_flat, t_flat]
        distance_loss = distances.mean().item()
        token_acc = (s_codes == t_codes).float().mean().item()

    vq_loss_val = vq_loss.item() if torch.is_tensor(vq_loss) else vq_loss

    return {
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'ce_loss': ce_loss,
        'distance_loss': distance_loss,
        'vq_loss': vq_loss_val,
        'token_acc': token_acc,
    }


# ==================== 使用說明 ====================
"""
exp17 (train_margin_loss.py) 使用方式：
1. 複製 compute_margin_loss 和 compute_losses_margin 函式
2. 將原本的 compute_losses 替換為 compute_losses_margin
3. 修改 train_epoch 和 validate 的呼叫：
   - 將 feature_weight 參數改為 margin
   - 更新 losses dict 的 key ('feature_loss' -> 'margin_loss')
4. 修改 argparse：
   - 移除 --feature_weight
   - 新增 --margin (default=0.5)

exp18 (train_curriculum.py) 使用方式：
1. 複製 get_curriculum_weights 和 compute_losses_curriculum 函式
2. 將原本的 compute_losses 替換為 compute_losses_curriculum
3. 在訓練迴圈中，每個 epoch 開始時呼叫：
   ce_weight, mse_weight, stage_name = get_curriculum_weights(epoch, args)
4. 將 weights 傳入 compute_losses_curriculum
5. 修改 argparse：
   - 新增 --stage1_epochs, --stage2_epochs, --stage3_epochs
   - 新增 --stage1_ce_weight, --stage1_mse_weight (for each stage)
"""
