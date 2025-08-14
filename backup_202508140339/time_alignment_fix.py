"""
基於commit 38f072d1c9756b8a2c5701f3912c0bdf809d23f0的時間對齊解決方案實現

解決語者語速不同時的內容一致性對齊問題的實現
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

def compute_time_invariant_content_consistency_loss(features, content_ids, device, pool_method='mean'):
    """
    基於時間池化的內容一致性損失，解決語速對齊問題
    
    核心思想：通過時間池化消除時間對齊假設，使用對比式學習
    
    Args:
        features: 中間層特徵 [B, C, T]
        content_ids: 內容ID列表
        device: 計算設備
        pool_method: 池化方式 ('mean', 'max', 'attention')
    
    Returns:
        torch.Tensor: 時間不變的內容一致性損失
    """
    if content_ids is None or len(content_ids) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 第一步：時間池化，消除時間對齊假設
    if pool_method == 'mean':
        # 全序列平均池化: [B, C, T] -> [B, C]
        pooled_features = torch.mean(features, dim=2)
    elif pool_method == 'max':
        # 最大池化: [B, C, T] -> [B, C]
        pooled_features, _ = torch.max(features, dim=2)
    elif pool_method == 'attention':
        # 簡化的注意力池化
        attention_weights = torch.softmax(torch.mean(features, dim=1, keepdim=True), dim=2)  # [B, 1, T]
        pooled_features = torch.sum(features * attention_weights, dim=2)  # [B, C]
    else:
        pooled_features = torch.mean(features, dim=2)
    
    # L2歸一化，使特徵在單位球面上
    pooled_features = F.normalize(pooled_features, p=2, dim=1)
    
    # 第二步：基於content_id分組進行對比式學習
    content_groups = defaultdict(list)
    for i, cid in enumerate(content_ids):
        content_groups[cid].append(i)
    
    # SupCon風格的對比損失
    total_loss = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    for cid, indices in content_groups.items():
        if len(indices) < 2:  # 需要至少2個樣本才能計算對比損失
            continue
            
        # 同一內容的所有樣本
        group_features = pooled_features[indices]  # [K, C]
        
        # 計算組內平均作為錨點
        anchor = torch.mean(group_features, dim=0, keepdim=True)  # [1, C]
        anchor = F.normalize(anchor, p=2, dim=1)
        
        # 計算每個樣本與錨點的相似度
        similarities = torch.matmul(group_features, anchor.T).squeeze()  # [K]
        
        # 使用負log似然促進相似度
        loss_per_group = -torch.log(torch.sigmoid(similarities * 10.0)).mean()  # 溫度參數=0.1
        
        total_loss += loss_per_group
        valid_samples += 1
    
    if valid_samples > 0:
        return total_loss / valid_samples
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)

def compute_supervised_contrastive_loss(features, content_ids, device, temperature=0.1):
    """
    實現Supervised Contrastive Learning，解決時間對齊問題
    
    基於論文: "Supervised Contrastive Learning" (Khosla et al., 2020)
    """
    if content_ids is None or len(content_ids) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 時間池化 [B, C, T] -> [B, C]
    pooled_features = torch.mean(features, dim=2)
    # L2歸一化
    normalized_features = F.normalize(pooled_features, p=2, dim=1)
    
    batch_size = normalized_features.shape[0]
    
    # 計算所有對的相似度矩陣
    similarity_matrix = torch.matmul(normalized_features, normalized_features.T) / temperature
    
    # 創建標籤矩陣
    labels = torch.tensor(content_ids, device=device)
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
    
    # 移除對角線（自己與自己的相似度）
    identity_mask = torch.eye(batch_size, device=device).bool()
    label_mask = label_mask & ~identity_mask
    
    # 計算SupCon損失
    total_loss = torch.tensor(0.0, device=device)
    num_positives = 0
    
    for i in range(batch_size):
        positive_mask = label_mask[i]  # 與i有相同標籤的樣本
        if not positive_mask.any():
            continue
            
        # 所有樣本（除了自己）
        negative_mask = ~identity_mask[i]
        
        # 正樣本的logits
        positive_logits = similarity_matrix[i][positive_mask]
        # 所有樣本的logits（用於分母）
        all_logits = similarity_matrix[i][negative_mask]
        
        # 對每個正樣本計算損失
        for pos_logit in positive_logits:
            loss_i = -pos_logit + torch.logsumexp(all_logits, dim=0)
            total_loss += loss_i
            num_positives += 1
    
    if num_positives > 0:
        return total_loss / num_positives
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)

def compute_dtw_aligned_loss(features1, features2, content_ids, device):
    """
    使用動態時間規整(DTW)對齊後計算損失
    
    注意：這是簡化版本，實際DTW需要專門的庫如fastdtw
    """
    # 這裡給出概念性實現，實際需要安裝dtw庫
    # 由於DTW計算複雜度高，建議只在特殊情況下使用
    
    # 簡化版：使用滑動窗口找最佳對齊
    def find_best_alignment(seq1, seq2, window_size=5):
        """找到兩個序列的最佳時間對齊"""
        seq1 = F.normalize(seq1, p=2, dim=0)  # [C, T1]
        seq2 = F.normalize(seq2, p=2, dim=0)  # [C, T2]
        
        best_similarity = -float('inf')
        best_offset = 0
        
        max_offset = min(window_size, abs(seq1.size(1) - seq2.size(1)))
        
        for offset in range(-max_offset, max_offset + 1):
            if offset >= 0:
                s1 = seq1[:, offset:]
                s2 = seq2[:, :seq1.size(1) - offset]
            else:
                s1 = seq1[:, :seq1.size(1) + offset]
                s2 = seq2[:, -offset:]
            
            if s1.size(1) > 0 and s2.size(1) > 0:
                min_len = min(s1.size(1), s2.size(1))
                s1 = s1[:, :min_len]
                s2 = s2[:, :min_len]
                
                # 計算餘弦相似度
                similarity = F.cosine_similarity(s1.flatten(), s2.flatten(), dim=0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_offset = offset
        
        return best_offset, best_similarity
    
    # 分組處理相同content_id的樣本
    content_groups = defaultdict(list)
    for i, cid in enumerate(content_ids):
        content_groups[cid].append(i)
    
    total_loss = torch.tensor(0.0, device=device)
    valid_pairs = 0
    
    for cid, indices in content_groups.items():
        if len(indices) < 2:
            continue
            
        # 對同一內容的所有配對進行DTW對齊
        group_features = features[indices]  # [K, C, T]
        
        for i in range(len(group_features)):
            for j in range(i + 1, len(group_features)):
                offset, similarity = find_best_alignment(
                    group_features[i], group_features[j]
                )
                
                # 使用1-similarity作為損失
                loss = 1.0 - similarity
                total_loss += loss
                valid_pairs += 1
    
    if valid_pairs > 0:
        return total_loss / valid_pairs
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)

# 中文註釋的實現指南
"""
實現要點總結：

1. 時間池化方法：
   - 全序列平均：適合大多數情況，計算簡單
   - 最大池化：突出顯著特徵，適合短語識別
   - 注意力池化：學習重要時間段，更智能但計算複雜

2. 對比式學習：
   - SupCon：適合有明確標籤的情況
   - InfoNCE：適合自監督場景
   - 溫度參數：控制分離強度，通常0.1效果較好

3. DTW對齊：
   - 精確但計算昂貴
   - 適合高品質要求的場景
   - 可考慮使用GPU加速版本

4. 選擇建議：
   - 訓練階段：使用時間池化+對比學習（快速）
   - 關鍵評估：使用DTW對齊（精確）
   - 實時應用：使用簡化的滑動窗口對齊
"""
