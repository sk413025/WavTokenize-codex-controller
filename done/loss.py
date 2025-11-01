import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleCELoss(nn.Module):
    """
    簡化的 CrossEntropy 損失函數 (僅 CE Loss)

    參考 debug_single_sample.py 的簡潔做法：
    只使用 Token CrossEntropy Loss，不包含其他複雜損失
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # CrossEntropy Loss
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred_logits, target_tokens):
        """
        計算 CrossEntropy Loss (簡化版)

        Args:
            pred_logits: (B, T, 4096) 模型預測的 logits
            target_tokens: (B, T) Ground truth clean tokens

        Returns:
            loss: scalar tensor - CrossEntropy Loss
        """
        B, T, vocab_size = pred_logits.shape

        # Reshape: (B, T, 4096) -> (B*T, 4096)
        # 參考 debug_single_sample.py line 264
        logits_flat = pred_logits.view(-1, vocab_size)
        target_flat = target_tokens.view(-1).long()

        # CrossEntropy Loss
        loss = self.ce_loss(logits_flat, target_flat)

        return loss