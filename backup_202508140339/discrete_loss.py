#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
離散編碼損失函數模組

為WavTokenizer的離散編碼實現L2和內容一致性損失函數，
用於增強離散編碼的泛化能力和語義一致性。

實驗編號: EXP06
日期: 2025-08-08
作者: 實驗腳本生成器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_discrete_l2_loss(enhanced_discrete, target_discrete, device):
    """
    計算離散編碼的L2損失

    將離散編碼轉換為嵌入表示，然後計算L2距離。
    這種方法可以在離散空間中創建有意義的距離度量。

    Args:
        enhanced_discrete (torch.Tensor): 增強模型產生的離散編碼 [batch_size, n_q, seq_len]
        target_discrete (torch.Tensor): 目標離散編碼 [batch_size, n_q, seq_len]
        device (torch.device): 計算設備

    Returns:
        torch.Tensor: L2損失值
    """
    # 確保輸入維度正確
    if enhanced_discrete.dim() == 2:
        enhanced_discrete = enhanced_discrete.unsqueeze(0)
    if target_discrete.dim() == 2:
        target_discrete = target_discrete.unsqueeze(0)

    # 處理可變長度序列
    # 找出最短序列長度
    min_length = min(enhanced_discrete.size(-1), target_discrete.size(-1))
    enhanced_discrete = enhanced_discrete[..., :min_length]
    target_discrete = target_discrete[..., :min_length]

    # 統計特徵轉換 - 計算每個離散編碼的統計特徵
    def compute_statistical_features(discrete_codes):
        # 將編碼轉為浮點型以便計算
        discrete_float = discrete_codes.float()
        
        # 批次大小
        batch_size = discrete_float.size(0)
        
        # 對每個批次樣本，計算統計特徵
        features = []
        for b in range(batch_size):
            sample_code = discrete_float[b].flatten()
            
            # 計算統計特徵
            mean_val = torch.mean(sample_code)
            std_val = torch.std(sample_code)
            median_val = torch.median(sample_code)
            min_val = torch.min(sample_code)
            max_val = torch.max(sample_code)
            
            # 組合特徵
            sample_features = torch.stack([mean_val, std_val, median_val, min_val, max_val])
            features.append(sample_features)
        
        return torch.stack(features)
    
    # 計算統計特徵
    enhanced_features = compute_statistical_features(enhanced_discrete)
    target_features = compute_statistical_features(target_discrete)
    
    # 計算L2距離
    l2_dist = torch.norm(enhanced_features - target_features, dim=1)
    distance_loss = l2_dist.mean()
    
    return distance_loss


def compute_discrete_content_consistency_loss(discrete_codes, content_ids, device):
    """
    計算離散編碼的內容一致性損失

    對相同內容ID的離散編碼計算統計特徵，然後用餘弦相似度衡量它們的一致性。
    目標是使相同語義內容的編碼在統計層面上更相似。

    Args:
        discrete_codes (torch.Tensor): 離散編碼張量 [batch_size, n_q, seq_len]
        content_ids (list or torch.Tensor): 批次中每個樣本的內容ID
        device (torch.device): 計算設備

    Returns:
        torch.Tensor: 內容一致性損失值
    """
    # 確保content_ids是在正確的設備上的張量，並且是數值類型
    if content_ids is None:
        # 如果沒有content_ids，創建虛擬的不同ID
        content_ids = torch.arange(discrete_codes.size(0), device=device)
    elif not isinstance(content_ids, torch.Tensor):
        # 將字符串ID轉換為數字ID
        try:
            # 嘗試直接將字符串轉換為數字
            numeric_ids = []
            for cid in content_ids:
                # 從字符串中提取數字部分
                if isinstance(cid, str):
                    # 提取所有數字
                    digits = ''.join(c for c in cid if c.isdigit())
                    if digits:
                        numeric_ids.append(int(digits))
                    else:
                        # 如果沒有數字，使用哈希值的一部分
                        numeric_ids.append(hash(cid) % 10000)
                else:
                    # 已經是數字或其他類型，嘗試直接轉換
                    numeric_ids.append(int(cid) if cid is not None else 0)
            content_ids = torch.tensor(numeric_ids, device=device)
        except Exception as e:
            print(f"無法將content_ids轉換為張量: {e}")
            # 出錯時創建虛擬的不同ID
            content_ids = torch.arange(discrete_codes.size(0), device=device)
    
    # 將離散編碼轉換為統計特徵向量，以便計算相似度
    def compute_embedding(codes):
        # 將編碼轉為浮點型以便計算
        codes_float = codes.float()
        
        # 平展以進行統計特徵計算
        if codes_float.dim() >= 3:
            flat_codes = codes_float.reshape(codes_float.size(0), -1)
        else:
            flat_codes = codes_float
        
        # 計算簡單的統計特徵作為嵌入
        means = torch.mean(flat_codes, dim=1, keepdim=True)
        stds = torch.std(flat_codes, dim=1, keepdim=True)
        medians = torch.median(flat_codes, dim=1).values.unsqueeze(1)
        maxes = torch.max(flat_codes, dim=1).values.unsqueeze(1)
        mines = torch.min(flat_codes, dim=1).values.unsqueeze(1)
        
        # 組合所有特徵作為嵌入向量
        embeddings = torch.cat([means, stds, medians, maxes, mines], dim=1)
        return embeddings
    
    # 獲取特徵嵌入
    embeddings = compute_embedding(discrete_codes)
    
    # 初始化損失為0
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 計算不同批次內相同content_id的樣本數
    unique_content_ids = torch.unique(content_ids)
    valid_groups = 0
    
    # 對於每個唯一的內容ID，找到具有該ID的所有樣本
    for content_id in unique_content_ids:
        # 找到對應此content_id的所有樣本索引
        indices = (content_ids == content_id).nonzero(as_tuple=True)[0]
        
        # 如果此內容ID只出現一次，則跳過（至少需要2個樣本才能計算一致性）
        if len(indices) < 2:
            continue
            
        # 提取這些樣本的嵌入
        group_embeddings = embeddings[indices]
        
        # 計算這組嵌入的平均值
        mean_embedding = torch.mean(group_embeddings, dim=0, keepdim=True)
        
        # 正規化嵌入向量，準備計算餘弦相似度
        norm_group_embeddings = F.normalize(group_embeddings, p=2, dim=1)
        norm_mean_embedding = F.normalize(mean_embedding, p=2, dim=1)
        
        # 計算餘弦相似度 (1 - similarity 轉為距離)
        # cos相似度範圍為[-1, 1]，1表示完全相同，-1表示完全相反，0表示正交
        # 將其轉換為[0, 2]的距離，0表示完全相似
        cos_sim = F.cosine_similarity(norm_group_embeddings, norm_mean_embedding, dim=1)
        distances = 1.0 - cos_sim  # 轉換為距離，範圍[0, 2]，0表示完全相似（我們想要最小化這個距離）
        
        # 累加這組樣本的平均距離到損失值
        group_loss = torch.mean(distances)
        loss = loss + group_loss
        
        # 計數有效的組
        valid_groups += 1
    
    # 如果有有效的組，則取平均值；否則損失為0
    if valid_groups > 0:
        loss = loss / valid_groups
        
    return loss


def compute_hybrid_discrete_loss(enhanced_discrete, target_discrete, content_ids, device, alpha=0.1, beta=0.9):
    """
    計算混合離散編碼損失，結合L2損失和內容一致性損失

    Args:
        enhanced_discrete (torch.Tensor): 增強模型產生的離散編碼
        target_discrete (torch.Tensor): 目標離散編碼
        content_ids (list or torch.Tensor): 批次中每個樣本的內容ID
        device (torch.device): 計算設備
        alpha (float): 內容一致性損失權重
        beta (float): L2損失權重

    Returns:
        tuple: (total_loss, loss_details)，其中loss_details是包含各損失組件的字典
    """
    # 計算L2損失
    l2_loss = compute_discrete_l2_loss(enhanced_discrete, target_discrete, device)
    
    # 計算內容一致性損失
    content_consistency_loss = compute_discrete_content_consistency_loss(enhanced_discrete, content_ids, device)
    
    # 混合損失
    total_loss = alpha * content_consistency_loss + beta * l2_loss
    
    # 返回總損失和詳細信息
    return total_loss, {
        'discrete_l2_loss': l2_loss.item(),
        'discrete_content_consistency_loss': content_consistency_loss.item()
    }
