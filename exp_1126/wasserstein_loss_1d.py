"""
1D Wasserstein Distance Loss for Token Prediction (Memory-Efficient)

使用 1D Wasserstein Distance (Optimal Transport on 1D)
假設 tokens 有自然順序，距離 = |i - j|

優勢:
- 內存需求: O(n) vs 2D 的 O(n²)
- 計算速度: 快速 closed-form solution
- GPU 友好: 可用更大 batch size

缺點:
- 假設 token 順序有意義（對 WavTokenizer 可能不完全適用）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Wasserstein1DLoss(nn.Module):
    """
    1D Wasserstein Distance Loss (Earth Mover's Distance on 1D)

    使用快速的 1D optimal transport 算法
    適合 token index 有序的情況
    """

    def __init__(self, num_classes=4096, reduction='mean', scale_factor=1.0):
        """
        Args:
            num_classes: 類別數量 (tokens)
            reduction: 'mean' or 'sum'
            scale_factor: 縮放因子，用於匹配 CE loss 的 magnitude
        """
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.scale_factor = scale_factor

    def wasserstein_1d(self, pred_dist, target_dist):
        """
        計算 1D Wasserstein Distance (closed-form solution)

        W1(P, Q) = ∫|F_P(x) - F_Q(x)| dx
        其中 F_P, F_Q 是累積分佈函數

        Args:
            pred_dist: (batch_size, num_classes) - 預測分佈
            target_dist: (batch_size, num_classes) - 目標分佈

        Returns:
            (batch_size,) - 每個樣本的 Wasserstein distance
        """
        # 計算累積分佈函數 (CDF)
        pred_cdf = torch.cumsum(pred_dist, dim=1)
        target_cdf = torch.cumsum(target_dist, dim=1)

        # W1 distance = L1 距離 between CDFs
        # 乘以 (1 / num_classes) 作為正規化
        w1_dist = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1) / self.num_classes

        return w1_dist

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,) - class indices
        """
        # 過濾掉 PAD_TOKEN (4096)
        valid_mask = targets < self.num_classes

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        # 將 logits 轉為概率分佈
        pred_dist = F.softmax(valid_logits, dim=-1)

        # 將 targets 轉為 one-hot distribution
        target_dist = F.one_hot(valid_targets, num_classes=self.num_classes).float()

        # 計算 1D Wasserstein distance
        loss = self.wasserstein_1d(pred_dist, target_dist)

        # 應用縮放因子
        loss = loss * self.scale_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HybridWasserstein1DCELoss(nn.Module):
    """
    Hybrid loss: 1D Wasserstein + CrossEntropy

    結合兩者優勢:
    - 1D Wasserstein: 考慮 token 距離，內存友好
    - CrossEntropy: 快速收斂，類別區分
    """

    def __init__(self, num_classes=4096, alpha=0.5, scale_factor=1.0):
        """
        Args:
            num_classes: 類別數量
            alpha: Wasserstein weight (1-alpha for CE)
            scale_factor: Wasserstein 縮放因子，用於匹配 CE loss magnitude
        """
        super().__init__()
        self.alpha = alpha
        self.wasserstein_loss = Wasserstein1DLoss(num_classes, scale_factor=scale_factor)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,) - class indices
        """
        # 過濾 PAD tokens for CE loss
        valid_mask = targets < 4096
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # CE loss (only on valid tokens)
        ce_loss = self.ce_loss(logits[valid_mask], targets[valid_mask])

        # 1D Wasserstein loss
        w_loss = self.wasserstein_loss(logits, targets)

        return self.alpha * w_loss + (1 - self.alpha) * ce_loss


# 測試
if __name__ == '__main__':
    # 測試 1D Wasserstein Loss
    num_classes = 4096
    batch_size = 32  # 1D 版本可以用更大的 batch size!

    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Pure 1D Wasserstein
    print("=" * 60)
    print("1D Wasserstein Loss 測試")
    print("=" * 60)

    w1d_loss = Wasserstein1DLoss(num_classes=num_classes)
    loss_w1d = w1d_loss(logits, targets)
    print(f"1D Wasserstein Loss: {loss_w1d.item():.6f}")

    # Hybrid
    hybrid_loss = HybridWasserstein1DCELoss(num_classes=num_classes, alpha=0.5)
    loss_hybrid = hybrid_loss(logits, targets)
    print(f"Hybrid Loss (α=0.5): {loss_hybrid.item():.6f}")

    # Compare with CE
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(logits, targets)
    print(f"CrossEntropy Loss: {loss_ce.item():.6f}")

    print("\n" + "=" * 60)
    print("內存優勢:")
    print("=" * 60)
    print(f"✓ 1D Wasserstein: O(n) = {num_classes} elements")
    print(f"✗ 2D Wasserstein: O(n²) = {num_classes**2:,} elements")
    print(f"✓ 內存節省: {num_classes**2 / num_classes:.0f}x")
    print(f"✓ 可用 batch size: 32 (vs 2D 的 8)")

    print("\n" + "=" * 60)
    print("速度測試:")
    print("=" * 60)

    import time

    # 預熱
    for _ in range(10):
        _ = w1d_loss(logits, targets)

    # 測速
    start = time.time()
    for _ in range(100):
        loss = w1d_loss(logits, targets)
        loss.backward()
    end = time.time()

    print(f"✓ 100 次前向+反向傳播: {end - start:.3f}s")
    print(f"✓ 平均每次: {(end - start) / 100 * 1000:.2f}ms")
