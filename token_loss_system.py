#!/usr/bin/env python3
"""
Token 空間的 Loss 系統
將 ttt2.py 的 loss 運算邏輯應用到離散 token 序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_combined_token_loss(predicted_logits, target_tokens, 
                              input_tokens, embedding_layer,
                              weights={'ce': 1.0, 'l2_embed': 0.5, 'coherence': 0.2, 'manifold': 0.1}):
    """
    重構後的組合 Token 損失函數
    
    設計理念：
    1. **CE Loss (1.0)**: 主要監督信號，確保正確預測 Token ID
    2. **L2 Embed (0.5)**: 輔助損失，即使預測錯誤也要錯得「聲學相似」
    3. **Coherence (0.2)**: 時間連貫性，讓相鄰 Token 平滑過渡
    4. **Manifold (0.1)**: 正則化，防止模型偏離輸入太遠
    
    優勢：
    - 權重更合理：CE 為主 (1.0)，其他為輔
    - 邏輯更清晰：每個損失職責明確
    - 代碼更簡潔：去除冗餘計算
    - PAD 處理更優雅：使用 ignore_index

    Args:
        predicted_logits: [B, L, VocabSize] 模型的原始輸出
        target_tokens: [B, L] 正確答案 Token ID
        input_tokens: [B, L] 輸入的帶噪聲 Token ID
        embedding_layer: 用於將 Token ID 轉為向量的 Embedding 層
        weights: 各損失的權重字典
        
    Returns:
        tuple: (total_loss, loss_dict)
    """
    import logging
    losses = {}
    device = predicted_logits.device
    vocab_size = predicted_logits.size(-1)
    
    # ==================== 1. 主要監督信號：交叉熵損失 (CE Loss) ====================
    # 職責：確保模型能準確預測出正確的 Token ID。這是最重要的損失。
    if weights.get('ce', 0) > 0:
        logits_flat = predicted_logits.reshape(-1, vocab_size)
        target_flat = target_tokens.reshape(-1)
        
        # 忽略 PAD token (假設 PAD_TOKEN_ID = 4096)
        losses['ce_loss'] = F.cross_entropy(logits_flat, target_flat, ignore_index=4096)

    # 從 logits 中獲取預測的 token ID，用於後續計算
    predicted_tokens = torch.argmax(predicted_logits, dim=-1)
    
    # 詳細調試日誌
    logging.debug(f"[Token Loss] predicted_logits shape: {predicted_logits.shape}")
    logging.debug(f"[Token Loss] predicted_tokens shape: {predicted_tokens.shape}")
    logging.debug(f"[Token Loss] target_tokens shape: {target_tokens.shape}")
    logging.debug(f"[Token Loss] input_tokens shape: {input_tokens.shape}")

    # 一次性計算所有需要的 embeddings，避免重複計算和形狀不匹配
    predicted_embed = None
    target_embed = None
    input_embed = None
    
    if embedding_layer is not None and (weights.get('l2_embed', 0) > 0 or 
                                        weights.get('coherence', 0) > 0 or 
                                        weights.get('manifold', 0) > 0):
        # 計算預測的 embedding
        predicted_embed = embedding_layer(predicted_tokens)
        logging.debug(f"[Token Loss] predicted_embed shape: {predicted_embed.shape}")
        
        # 計算目標和輸入的 embedding（不需要梯度）
        with torch.no_grad():
            target_embed = embedding_layer(target_tokens)
            input_embed = embedding_layer(input_tokens)
            logging.debug(f"[Token Loss] target_embed shape: {target_embed.shape}")
            logging.debug(f"[Token Loss] input_embed shape: {input_embed.shape}")

    # ==================== 2. 聲學相似性損失：Embedding L2 Loss ====================
    # 職責：讓預測錯誤時，也盡量錯得「比較像」。鼓勵聲學上的相似性。
    if weights.get('l2_embed', 0) > 0 and predicted_embed is not None:
        losses['l2_embed_loss'] = F.mse_loss(predicted_embed, target_embed)

    # ==================== 3. 時間連貫性損失：Coherence Loss ====================
    # 職責：讓相鄰 Token 之間的過渡更平滑，解決頻譜破碎問題。
    if weights.get('coherence', 0) > 0 and predicted_embed is not None:
        try:
            logging.debug(f"[Token Loss] Computing coherence loss, predicted_embed shape: {predicted_embed.shape}")
            # 計算相鄰 embedding 的差異的 L2 範數
            slice1 = predicted_embed[:, 1:, :]
            slice2 = predicted_embed[:, :-1, :]
            logging.debug(f"[Token Loss] slice1 shape: {slice1.shape}, slice2 shape: {slice2.shape}")
            pred_diff = slice1 - slice2
            logging.debug(f"[Token Loss] pred_diff shape: {pred_diff.shape}")
            # 我們的目標是最小化這個差異，讓序列更平滑
            losses['coherence_loss'] = pred_diff.pow(2).mean()
            logging.debug(f"[Token Loss] coherence_loss computed: {losses['coherence_loss'].item():.4f}")
        except Exception as e:
            # 如果 coherence loss 計算失敗，記錄錯誤但不中斷訓練
            logging.error(f"❌ Coherence loss 計算失敗（predicted_embed shape: {predicted_embed.shape}）: {e}")
            import traceback
            traceback.print_exc()
            # 設置為 0 以避免影響訓練
            losses['coherence_loss'] = torch.tensor(0.0, device=device)

    # ==================== 4. 正則化項：Manifold Loss ====================
    # 職責：防止模型偏離輸入太遠，只做「降噪」而非「創造」。
    if weights.get('manifold', 0) > 0 and predicted_embed is not None:
        # 簡單版本：直接懲罰與輸入 embedding 的距離
        losses['manifold_loss'] = F.mse_loss(predicted_embed, input_embed)
    
    # ==================== 計算總損失 ====================
    total_loss = torch.tensor(0.0, device=device)
    if 'ce_loss' in losses:
        total_loss += weights.get('ce', 0) * losses['ce_loss']
    if 'l2_embed_loss' in losses:
        total_loss += weights.get('l2_embed', 0) * losses['l2_embed_loss']
    if 'coherence_loss' in losses:
        total_loss += weights.get('coherence', 0) * losses['coherence_loss']
    if 'manifold_loss' in losses:
        total_loss += weights.get('manifold', 0) * losses['manifold_loss']
    
    losses['total_loss'] = total_loss
    
    # 轉換為數值用於記錄
    loss_dict = {name: loss.item() for name, loss in losses.items()}
    
    return total_loss, loss_dict

if __name__ == "__main__":
    print("Token 空間 Loss 系統設計完成！")
    print("\n核心概念：")
    print("1. L2 距離：將 token 嵌入到連續空間計算歐式距離")
    print("2. 內容一致性：確保 token 預測的準確性和分佈合理性")  
    print("3. Manifold 正則化：防止預測偏離輸入 manifold 太遠")
    print("4. 正則化：控制 logits 的大小和分佈")
    print("5. 連貫性：保證序列的語義連續性")
    print("\n完全按照 ttt2.py 的運算邏輯，但應用在 token 空間！")
