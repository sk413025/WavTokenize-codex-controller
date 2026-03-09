"""
使用 VQ Distances 的訓練範例

展示三種使用 VQ distances 的方法:
1. Soft Target Training (Knowledge Distillation)
2. Distance-based Wasserstein Loss
3. Hybrid: Soft Target + Hard Target

相比原始的 hard token target，soft target 包含了更豐富的資訊
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetLoss(nn.Module):
    """
    Soft Target Loss (Knowledge Distillation from VQ distances)

    使用 VQ-VAE 的 distance 分佈作為 soft target
    相比 hard token，soft target 包含了"接近但未選中"的 tokens 資訊

    優勢:
      - 保留了 VQ 量化過程中的相似度信息
      - 比 one-hot target 更豐富
      - 類似 Knowledge Distillation，但 teacher 是 VQ-VAE

    新增 (v2):
      - Class weights: 降低 majority class (token 453) 的權重，防止 collapse
      - Entropy regularization: 鼓勵多樣化預測，防止過度集中

    使用:
      loss_fn = SoftTargetLoss(temperature=2.0, alpha=0.5,
                               class_weights=weights, entropy_weight=0.01)
      loss = loss_fn(pred_logits, clean_distances, clean_tokens)
    """

    def __init__(self, temperature=2.0, alpha=0.5, ignore_index=4096,
                 class_weights=None, entropy_weight=0.0):
        """
        Args:
            temperature: Softmax 溫度（越高越平滑）
            alpha: soft target 的權重 (1-alpha 給 hard target)
            ignore_index: 忽略的 token index (用於 padding)
            class_weights: (C,) tensor，類別權重（降低 majority class 權重）
            entropy_weight: 熵正則化權重（>0 鼓勵多樣化預測）
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.entropy_weight = entropy_weight

        # Cross Entropy with class weights
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=class_weights
        )

    def forward(self, pred_logits, target_distances, target_tokens):
        """
        Args:
            pred_logits: (B, T, 4096) - 模型預測的 logits
            target_distances: (B, T, 4096) - VQ-VAE 的 distances
            target_tokens: (B, T) - hard tokens (用於混合 loss)

        Returns:
            loss: scalar
        """
        B, T, C = pred_logits.shape

        # Soft target from distances
        # distances 越大 = 越接近，所以直接 softmax
        soft_targets = F.softmax(target_distances / self.temperature, dim=-1)  # (B, T, 4096)

        # Soft loss (KL divergence)
        pred_log_probs = F.log_softmax(pred_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(
            pred_log_probs.reshape(-1, C),
            soft_targets.reshape(-1, C),
            reduction='batchmean'
        ) * (self.temperature ** 2)  # 溫度補償

        # Hard loss (Cross Entropy with class weights)
        hard_loss = self.ce_loss(pred_logits.reshape(-1, C), target_tokens.reshape(-1))

        # Entropy regularization (鼓勵多樣化預測)
        entropy_reg = 0.0
        if self.entropy_weight > 0:
            # 計算預測分佈的熵（熵越高 = 越多樣化）
            pred_probs = F.softmax(pred_logits, dim=-1)  # (B, T, C)
            entropy = -(pred_probs * torch.log(pred_probs + 1e-10)).sum(dim=-1).mean()
            # 負熵作為懲罰（最大化熵 = 最小化負熵）
            entropy_reg = -self.entropy_weight * entropy

        # 混合
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss + entropy_reg

        return total_loss


class DistanceBasedWassersteinLoss(nn.Module):
    """
    基於 VQ Distances 的 Wasserstein Loss
    
    與原始 Wasserstein Loss 的差異:
      原始: Cost Matrix 基於 token 索引距離 |i - j|  (錯誤!)
      改進: Cost Matrix 基於 VQ embedding 空間的真實距離
    
    優勢:
      - 使用 VQ-VAE 學到的語義距離
      - 比索引距離更有意義
      - 反映真實的聲學相似度
    
    使用:
      loss_fn = DistanceBasedWassersteinLoss(num_classes=4096)
      loss = loss_fn(pred_logits, clean_distances, clean_tokens)
    """
    
    def __init__(self, num_classes=4096, reg=0.1, max_iter=50):
        """
        Args:
            num_classes: Token 數量
            reg: Sinkhorn regularization
            max_iter: Sinkhorn iterations
        """
        super().__init__()
        self.num_classes = num_classes
        self.reg = reg
        self.max_iter = max_iter
    
    def compute_cost_matrix_from_distances(self, distances):
        """
        從 distances 計算 Cost Matrix
        
        Args:
            distances: (B, T, 4096) - VQ distances for all frames
        
        Returns:
            cost_matrix: (4096, 4096) - pairwise token distances
        
        原理:
          如果 token i 和 token j 經常出現相似的 distance pattern，
          則它們在語義上接近
        """
        # 這裡使用簡化方法：
        # 假設我們已經預先計算好了 codebook embedding 的距離矩陣
        # 實際使用時應該從 wavtokenizer.get_codebook() 計算
        
        # Placeholder: 使用單位矩陣（需要替換為真實的 codebook distances）
        cost_matrix = torch.eye(self.num_classes, device=distances.device)
        
        return cost_matrix
    
    def forward(self, pred_logits, target_distances, target_tokens):
        """
        Args:
            pred_logits: (B, T, 4096)
            target_distances: (B, T, 4096) - 這裡可用於 soft target
            target_tokens: (B, T) - hard tokens
        
        Returns:
            loss: scalar
        """
        # 簡化實現：使用 soft target 作為 Wasserstein 的目標分佈
        B, T, C = pred_logits.shape
        
        pred_probs = F.softmax(pred_logits, dim=-1)  # (B, T, 4096)
        target_probs = F.softmax(target_distances, dim=-1)  # (B, T, 4096)
        
        # 使用 1D Wasserstein (簡化版本)
        # 完整版本需要 Sinkhorn 算法 + codebook-based cost matrix
        loss = F.mse_loss(pred_probs, target_probs)  # 簡化為 MSE
        
        return loss


class HybridDistanceLoss(nn.Module):
    """
    混合 Loss：Soft Target + Hard Target + Wasserstein

    結合三種 loss 的優勢:
      1. Soft Target: 豐富的相似度信息
      2. Hard Target: 確保準確預測
      3. Wasserstein: 考慮 token 間的語義距離

    新增 (v2):
      - Class weights: 降低 majority class (token 453) 的權重，防止 collapse
      - Entropy regularization: 鼓勵多樣化預測，防止過度集中

    使用:
      loss_fn = HybridDistanceLoss(alpha=0.3, beta=0.3, gamma=0.4,
                                   class_weights=weights, entropy_weight=0.01)
      loss = loss_fn(pred_logits, clean_distances, clean_tokens)
    """

    def __init__(self, alpha=0.3, beta=0.3, gamma=0.4, temperature=2.0,
                 class_weights=None, entropy_weight=0.0):
        """
        Args:
            alpha: Soft target 權重
            beta: Hard target 權重
            gamma: Wasserstein 權重
            temperature: Soft target 溫度
            class_weights: (C,) tensor，類別權重（降低 majority class 權重）
            entropy_weight: 熵正則化權重（>0 鼓勵多樣化預測）
        """
        super().__init__()
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "權重總和必須為 1"

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.entropy_weight = entropy_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, pred_logits, target_distances, target_tokens):
        """
        Args:
            pred_logits: (B, T, 4096)
            target_distances: (B, T, 4096)
            target_tokens: (B, T)
        
        Returns:
            loss_dict: Dict of individual losses and total loss
        """
        B, T, C = pred_logits.shape
        
        # 1. Soft Target Loss (KL divergence)
        soft_targets = F.softmax(target_distances / self.temperature, dim=-1)
        pred_log_probs = F.log_softmax(pred_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(
            pred_log_probs.reshape(-1, C),
            soft_targets.reshape(-1, C),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 2. Hard Target Loss (Cross Entropy)
        hard_loss = self.ce_loss(pred_logits.reshape(-1, C), target_tokens.reshape(-1))
        
        # 3. Wasserstein-like Loss (簡化為 distribution matching)
        pred_probs = F.softmax(pred_logits, dim=-1)
        target_probs = F.softmax(target_distances, dim=-1)
        wasserstein_loss = F.mse_loss(pred_probs, target_probs)

        # 4. Entropy regularization (鼓勵多樣化預測)
        entropy_reg = 0.0
        if self.entropy_weight > 0:
            entropy = -(pred_probs * torch.log(pred_probs + 1e-10)).sum(dim=-1).mean()
            entropy_reg = -self.entropy_weight * entropy

        # 總 Loss
        total_loss = (
            self.alpha * soft_loss +
            self.beta * hard_loss +
            self.gamma * wasserstein_loss +
            entropy_reg
        )

        return {
            'total_loss': total_loss,
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item(),
            'wasserstein_loss': wasserstein_loss.item(),
            'entropy_reg': entropy_reg if isinstance(entropy_reg, float) else entropy_reg.item()
        }


def example_training_step_with_distances(model, batch, loss_fn, device):
    """
    訓練範例：使用帶 distances 的數據
    
    Args:
        model: Denoising Transformer
        batch: 從 DataLoader 獲取的 batch (包含 distances)
        loss_fn: 上述任一 loss function
        device: torch.device
    
    Returns:
        loss: scalar tensor
    """
    # 解包 batch
    noisy_tokens = batch['noisy_tokens'].to(device)           # (B, T)
    clean_tokens = batch['clean_tokens'].to(device)           # (B, T)
    clean_distances = batch['clean_distances'].to(device)     # (B, T, 4096)
    speaker_embeddings = batch['speaker_embeddings'].to(device)  # (B, 256)
    
    # Forward
    pred_logits = model(noisy_tokens, speaker_embeddings, return_logits=True)  # (B, T, 4096)
    
    # 計算 loss（使用 distances）
    if isinstance(loss_fn, HybridDistanceLoss):
        loss_dict = loss_fn(pred_logits, clean_distances, clean_tokens)
        loss = loss_dict['total_loss']
        
        # 可以記錄各個 loss 分量
        print(f"  Soft Loss: {loss_dict['soft_loss']:.4f}")
        print(f"  Hard Loss: {loss_dict['hard_loss']:.4f}")
        print(f"  Wasserstein Loss: {loss_dict['wasserstein_loss']:.4f}")
    else:
        loss = loss_fn(pred_logits, clean_distances, clean_tokens)
    
    return loss


if __name__ == '__main__':
    """測試各種 loss functions"""
    
    # 模擬數據
    B, T, C = 4, 75, 4096
    
    pred_logits = torch.randn(B, T, C)
    target_distances = torch.randn(B, T, C)  # 模擬 VQ distances
    target_tokens = torch.randint(0, C, (B, T))
    
    print("=" * 80)
    print("測試 Soft Target Loss")
    print("=" * 80)
    
    loss_fn = SoftTargetLoss(temperature=2.0, alpha=0.5)
    loss = loss_fn(pred_logits, target_distances, target_tokens)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 80)
    print("測試 Hybrid Distance Loss")
    print("=" * 80)
    
    loss_fn = HybridDistanceLoss(alpha=0.3, beta=0.3, gamma=0.4, temperature=2.0)
    loss_dict = loss_fn(pred_logits, target_distances, target_tokens)
    print(f"Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  - Soft Loss: {loss_dict['soft_loss']:.4f}")
    print(f"  - Hard Loss: {loss_dict['hard_loss']:.4f}")
    print(f"  - Wasserstein Loss: {loss_dict['wasserstein_loss']:.4f}")
    
    print("\n✅ 測試通過！")
    
    print("\n" + "=" * 80)
    print("使用建議")
    print("=" * 80)
    print("""
    1. Soft Target Loss (推薦起手):
       - 簡單有效
       - temperature=2.0, alpha=0.5 是好的起點
       - 適合快速驗證 distances 的價值
    
    2. Hybrid Loss (推薦進階):
       - 結合多種信號
       - 需要調整權重 (alpha, beta, gamma)
       - 可能達到最好效果
    
    3. Distance-based Wasserstein (實驗性):
       - 需要完整實現 Sinkhorn 算法
       - 需要預計算 codebook cost matrix
       - 計算成本較高
    
    實驗建議:
      A. Baseline: 只用 hard target (CE loss)
      B. Exp1: Soft target (alpha=0.5)
      C. Exp2: Soft target (alpha=0.7)
      D. Exp3: Hybrid loss (0.3/0.3/0.4)
      
      比較 Val Acc 和音頻品質
    """)
