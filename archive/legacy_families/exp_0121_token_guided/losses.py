"""
Token-Guided Loss Functions

基於 token 分析結果設計的 loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class TokenWeightedMSELoss(nn.Module):
    """
    Token-Weighted MSE Loss

    對「容易產生錯誤 token」的 activation 給更高的 loss 權重

    原理:
    1. 從 token_error_rates.pt 載入每個 token 的 error rate
    2. 將 error rate 映射到 activation 空間
    3. 對容易錯的區域給更高的 loss weight
    """

    def __init__(self, error_rates_path: str = None, base_weight: float = 1.0,
                 high_error_multiplier: float = 2.0, high_error_threshold: float = 0.7):
        """
        Args:
            error_rates_path: token_error_rates.pt 的路徑
            base_weight: 基礎權重
            high_error_multiplier: 高錯誤率 token 的權重倍數
            high_error_threshold: 高錯誤率的閾值
        """
        super().__init__()
        self.base_weight = base_weight
        self.high_error_multiplier = high_error_multiplier
        self.high_error_threshold = high_error_threshold

        # 載入 error rates
        if error_rates_path and Path(error_rates_path).exists():
            self.register_buffer('error_rates', torch.load(error_rates_path))
            print(f"Loaded token error rates from {error_rates_path}")
        else:
            # 預設: 所有 token 權重相同
            self.register_buffer('error_rates', torch.full((4096,), 0.5))
            print("Using default token error rates (0.5 for all)")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                target_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: 預測的 activation [B, C, T] 或 [B, T, C]
            target: 目標 activation [B, C, T] 或 [B, T, C]
            target_tokens: 目標 token ids [B, T] (用於計算權重)

        Returns:
            weighted MSE loss
        """
        # 基本 MSE
        mse = F.mse_loss(pred, target, reduction='none')

        if target_tokens is not None:
            # 根據 target token 的 error rate 計算權重
            # target_tokens: [B, T]
            error_rates = self.error_rates[target_tokens.long()]  # [B, T]

            # 映射到權重
            weights = torch.where(
                error_rates > self.high_error_threshold,
                torch.full_like(error_rates, self.high_error_multiplier),
                torch.full_like(error_rates, self.base_weight)
            )  # [B, T]

            # 擴展權重以匹配 activation shape
            if pred.dim() == 3:
                if pred.shape[1] > pred.shape[2]:  # [B, C, T]
                    weights = weights.unsqueeze(1)  # [B, 1, T]
                else:  # [B, T, C]
                    weights = weights.unsqueeze(2)  # [B, T, 1]

            return (mse * weights).mean()
        else:
            return mse.mean()


class NoiseTypeAwareLoss(nn.Module):
    """
    Noise-Type Aware Loss

    對不同噪音類型給不同的 loss 權重
    plastic 最難 → 給最高權重
    """

    def __init__(self, noise_weights: dict = None):
        """
        Args:
            noise_weights: 各噪音類型的權重，如 {'plastic': 2.0, 'box': 1.5, 'papercup': 1.0}
        """
        super().__init__()
        self.noise_weights = noise_weights or {
            'plastic': 2.0,   # 最難
            'box': 1.5,       # 中等
            'papercup': 1.2,  # 較易
            'clean': 0.0,     # clean 不計入 loss
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                noise_types: list) -> torch.Tensor:
        """
        Args:
            pred: [B, ...] 預測
            target: [B, ...] 目標
            noise_types: list of str, 長度為 B

        Returns:
            weighted loss
        """
        batch_size = pred.shape[0]
        weights = torch.tensor(
            [self.noise_weights.get(nt, 1.0) for nt in noise_types],
            device=pred.device
        )

        # 計算每個樣本的 loss
        mse_per_sample = F.mse_loss(pred, target, reduction='none')
        # 展平後取 mean per sample
        mse_per_sample = mse_per_sample.view(batch_size, -1).mean(dim=1)

        # 加權平均
        weighted_loss = (mse_per_sample * weights).sum() / weights.sum()
        return weighted_loss


class FocalTokenLoss(nn.Module):
    """
    Focal Loss for Token Classification

    對難以正確分類的 token 給更高的權重
    類似 Focal Loss 的設計，但應用在 token matching 上
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        """
        Args:
            gamma: focusing parameter (越大越關注難樣本)
            alpha: 每個 token 的權重 [4096]
        """
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, pred_logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, T, 4096] 預測的 logits
            target_tokens: [B, T] 目標 token ids

        Returns:
            focal loss
        """
        B, T, C = pred_logits.shape

        # Softmax
        probs = F.softmax(pred_logits, dim=-1)  # [B, T, 4096]

        # 取出正確類別的機率
        target_flat = target_tokens.view(-1)  # [B*T]
        probs_flat = probs.view(-1, C)  # [B*T, 4096]
        p_t = probs_flat[torch.arange(B*T, device=pred_logits.device), target_flat]  # [B*T]

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Cross entropy
        ce_loss = F.cross_entropy(pred_logits.view(-1, C), target_flat, reduction='none')

        # Alpha weighting (if provided)
        if self.alpha is not None:
            alpha_t = self.alpha[target_flat]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class CombinedTokenGuidedLoss(nn.Module):
    """
    組合 Loss: MSE + Token-Weighted + Noise-Aware

    Total Loss = λ1 * MSE_loss + λ2 * Token_weighted_loss + λ3 * Noise_aware_loss
    """

    def __init__(self, error_rates_path: str = None,
                 lambda_mse: float = 1.0,
                 lambda_token: float = 0.5,
                 lambda_noise: float = 0.3):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_token = lambda_token
        self.lambda_noise = lambda_noise

        self.mse_loss = nn.MSELoss()
        self.token_weighted_loss = TokenWeightedMSELoss(error_rates_path)
        self.noise_aware_loss = NoiseTypeAwareLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                target_tokens: torch.Tensor = None,
                noise_types: list = None) -> dict:
        """
        Returns:
            dict with 'total', 'mse', 'token_weighted', 'noise_aware'
        """
        losses = {}

        # 基本 MSE
        losses['mse'] = self.mse_loss(pred, target)

        # Token-weighted (如果有 token 資訊)
        if target_tokens is not None:
            losses['token_weighted'] = self.token_weighted_loss(pred, target, target_tokens)
        else:
            losses['token_weighted'] = losses['mse']

        # Noise-aware (如果有噪音類型資訊)
        if noise_types is not None:
            losses['noise_aware'] = self.noise_aware_loss(pred, target, noise_types)
        else:
            losses['noise_aware'] = losses['mse']

        # 總 loss
        losses['total'] = (
            self.lambda_mse * losses['mse'] +
            self.lambda_token * losses['token_weighted'] +
            self.lambda_noise * losses['noise_aware']
        )

        return losses
