"""
exp_1209: Loss 函數定義

包含:
1. TripletLoss: 三元組損失，使用 hard negative mining
2. ContrastiveLoss: 對比損失
3. DirectionAwareLoss: 方向感知損失
4. CombinedLoss: 組合多個損失函數

Triplet Loss vs Contrastive Loss:
---------------------------------
| 特性 | Triplet Loss | Contrastive Loss |
|------|--------------|------------------|
| 組成 | (anchor, positive, negative) | (sample1, sample2, label) |
| 目標 | d(a,p) + margin < d(a,n) | 相似拉近，不相似推遠 |
| 優點 | 精確控制 margin | 可以用多個負樣本 |
| 難度 | 需要選擇好的負樣本 | 相對簡單 |

本實驗選擇 Triplet Loss + Hard Negative Mining，因為:
1. 可以精確控制「正確 code」和「錯誤 code」的距離關係
2. Hard negative 確保模型學習精細區分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TripletLoss(nn.Module):
    """
    三元組損失函數，使用 Hard Negative Mining

    目標:
        使 student 特徵靠近 teacher 特徵（positive），
        遠離錯誤的 code（negative）

    公式:
        L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    其中:
        - anchor: teacher encoder output（正確的目標）
        - positive: student adapted output（應該靠近 anchor）
        - negative: 最近的錯誤 code embedding（應該遠離）

    Args:
        margin: triplet margin，預設 0.5
        distance_type: 距離度量，'l2' 或 'cosine'
        reduction: 'mean', 'sum', 或 'none'
    """

    def __init__(
        self,
        margin: float = 0.5,
        distance_type: str = 'l2',
        reduction: str = 'mean',
    ):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
        self.reduction = reduction

    def _compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        計算兩個張量之間的距離

        Args:
            x: shape (..., dim)
            y: shape (..., dim)

        Returns:
            距離, shape (...)
        """
        if self.distance_type == 'l2':
            return torch.norm(x - y, p=2, dim=-1)
        elif self.distance_type == 'cosine':
            # cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(x, y, dim=-1)
            return 1 - cos_sim
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def get_hard_negatives(
        self,
        student_out: torch.Tensor,
        codebook: torch.Tensor,
        teacher_codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用 Hard Negative Mining 選擇最難區分的負樣本

        策略: 選擇距離 student output 最近的「錯誤」code
        這是最難區分的負樣本，強迫模型學習精細邊界

        Args:
            student_out: student encoder/adapter 輸出, shape (B, C, T)
            codebook: VQ codebook, shape (num_codes, C)
            teacher_codes: 正確的 token, shape (1, B, T) 或 (B, T)

        Returns:
            hard_negatives: 最難區分的錯誤 code embedding, shape (B, T, C)
        """
        B, C, T = student_out.shape

        # 處理 teacher_codes 的維度
        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]  # (B, T)

        # (B, C, T) -> (B, T, C) -> (B*T, C)
        z = student_out.permute(0, 2, 1).reshape(-1, C)

        # 計算到所有 code 的 L2 距離
        # dists: (B*T, num_codes)
        dists = torch.cdist(z, codebook, p=2)

        # 把正確 code 的距離設為無窮大（排除正確答案）
        teacher_flat = teacher_codes.reshape(-1)  # (B*T,)
        batch_indices = torch.arange(len(teacher_flat), device=dists.device)
        dists[batch_indices, teacher_flat] = float('inf')

        # 選擇最近的錯誤 code（最難區分的負樣本）
        hard_neg_idx = dists.argmin(dim=1)  # (B*T,)
        hard_negatives = codebook[hard_neg_idx]  # (B*T, C)

        return hard_negatives.reshape(B, T, C)

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        codebook: torch.Tensor,
        teacher_codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        計算 Triplet Loss

        Args:
            student_out: student encoder/adapter 輸出, shape (B, C, T)
            teacher_out: teacher encoder 輸出, shape (B, C, T)
            codebook: VQ codebook, shape (num_codes, C)
            teacher_codes: 正確的 token

        Returns:
            loss: triplet loss
            info: 包含額外資訊的 dict
        """
        B, C, T = student_out.shape

        # 獲取 hard negatives
        hard_negatives = self.get_hard_negatives(student_out, codebook, teacher_codes)
        # hard_negatives: (B, T, C)

        # 轉換維度以計算距離
        # anchor: teacher_out (B, C, T) -> (B, T, C)
        # positive: student_out (B, C, T) -> (B, T, C)
        anchor = teacher_out.permute(0, 2, 1)      # (B, T, C)
        positive = student_out.permute(0, 2, 1)    # (B, T, C)
        negative = hard_negatives                   # (B, T, C)

        # 計算距離
        d_pos = self._compute_distance(anchor, positive)  # (B, T)
        d_neg = self._compute_distance(anchor, negative)  # (B, T)

        # Triplet loss: max(0, d_pos - d_neg + margin)
        losses = F.relu(d_pos - d_neg + self.margin)

        # Reduction
        if self.reduction == 'mean':
            loss = losses.mean()
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses

        # 額外資訊
        with torch.no_grad():
            info = {
                'd_positive': d_pos.mean().item(),
                'd_negative': d_neg.mean().item(),
                'margin_satisfied': (d_neg > d_pos + self.margin).float().mean().item(),
            }

        return loss, info


class ContrastiveLoss(nn.Module):
    """
    對比損失函數

    使用 InfoNCE 風格的對比學習:
    - 正樣本: teacher encoder output（對應同一時間步）
    - 負樣本: batch 內其他樣本的 teacher output + 錯誤的 codes

    公式:
        L = -log(exp(sim(z, z+)/τ) / Σexp(sim(z, z-)/τ))

    Args:
        temperature: softmax 溫度參數
        use_codebook_negatives: 是否使用 codebook 中的錯誤 codes 作為負樣本
    """

    def __init__(
        self,
        temperature: float = 0.1,
        use_codebook_negatives: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_codebook_negatives = use_codebook_negatives

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        codebook: Optional[torch.Tensor] = None,
        teacher_codes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        計算 Contrastive Loss

        Args:
            student_out: (B, C, T)
            teacher_out: (B, C, T)
            codebook: (num_codes, C), 可選
            teacher_codes: 正確的 token, 可選

        Returns:
            loss, info
        """
        B, C, T = student_out.shape

        # 展平
        # student: (B, C, T) -> (B*T, C)
        # teacher: (B, C, T) -> (B*T, C)
        student_flat = student_out.permute(0, 2, 1).reshape(-1, C)
        teacher_flat = teacher_out.permute(0, 2, 1).reshape(-1, C)

        # 正規化
        student_norm = F.normalize(student_flat, dim=-1)
        teacher_norm = F.normalize(teacher_flat, dim=-1)

        # 正樣本相似度: (B*T,)
        pos_sim = (student_norm * teacher_norm).sum(dim=-1) / self.temperature

        # 負樣本
        if self.use_codebook_negatives and codebook is not None:
            # 使用 codebook 中的所有 codes 作為負樣本
            codebook_norm = F.normalize(codebook, dim=-1)  # (num_codes, C)
            neg_sim = torch.matmul(student_norm, codebook_norm.t()) / self.temperature
            # neg_sim: (B*T, num_codes)

            # 移除正確 code 的相似度
            if teacher_codes is not None:
                if teacher_codes.dim() == 3:
                    teacher_codes = teacher_codes[0]
                teacher_flat_codes = teacher_codes.reshape(-1)
                batch_idx = torch.arange(len(teacher_flat_codes), device=neg_sim.device)
                neg_sim[batch_idx, teacher_flat_codes] = float('-inf')
        else:
            # 使用 batch 內其他樣本作為負樣本
            neg_sim = torch.matmul(student_norm, teacher_norm.t()) / self.temperature
            # 移除對角線（自己和自己）
            mask = torch.eye(len(student_norm), device=neg_sim.device).bool()
            neg_sim = neg_sim.masked_fill(mask, float('-inf'))

        # InfoNCE loss
        # logits = [pos_sim, neg_sim]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B*T, 1+num_neg)
        labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # 額外資訊
        with torch.no_grad():
            info = {
                'pos_sim': pos_sim.mean().item(),
                'neg_sim_max': neg_sim.max(dim=1)[0].mean().item(),
            }

        return loss, info


class DirectionAwareLoss(nn.Module):
    """
    方向感知損失

    不只優化距離，還要確保移動方向正確

    公式:
        L = MSE + λ * (1 - cosine(movement, correct_direction))

    其中:
        - movement: student_out - original_encoder_out（實際移動方向）
        - correct_direction: teacher_out - original_encoder_out（正確方向）

    Args:
        direction_weight: 方向損失的權重
    """

    def __init__(self, direction_weight: float = 1.0):
        super().__init__()
        self.direction_weight = direction_weight

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        original_encoder_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        計算方向感知損失

        Args:
            student_out: adapter/LoRA 輸出, (B, C, T)
            teacher_out: teacher encoder 輸出, (B, C, T)
            original_encoder_out: 原始 encoder 輸出（無 adapter/LoRA）, (B, C, T)

        Returns:
            loss, info
        """
        # MSE loss
        mse_loss = F.mse_loss(student_out, teacher_out)

        # 方向損失
        # movement: student 的實際移動
        movement = student_out - original_encoder_out
        # correct_direction: 理想的移動方向
        correct_direction = teacher_out - original_encoder_out

        # 展平計算 cosine similarity
        movement_flat = movement.permute(0, 2, 1).reshape(-1, student_out.shape[1])
        direction_flat = correct_direction.permute(0, 2, 1).reshape(-1, student_out.shape[1])

        # 只在有明顯移動的地方計算
        movement_norm = movement_flat.norm(dim=-1, keepdim=True)
        direction_norm = direction_flat.norm(dim=-1, keepdim=True)

        # 避免除以零
        valid_mask = (movement_norm.squeeze() > 1e-6) & (direction_norm.squeeze() > 1e-6)

        if valid_mask.sum() > 0:
            movement_normalized = movement_flat[valid_mask] / (movement_norm[valid_mask] + 1e-8)
            direction_normalized = direction_flat[valid_mask] / (direction_norm[valid_mask] + 1e-8)
            cosine_sim = (movement_normalized * direction_normalized).sum(dim=-1)
            direction_loss = (1 - cosine_sim).mean()
        else:
            direction_loss = torch.tensor(0.0, device=student_out.device)

        # 總損失
        loss = mse_loss + self.direction_weight * direction_loss

        # 額外資訊
        with torch.no_grad():
            info = {
                'mse_loss': mse_loss.item(),
                'direction_loss': direction_loss.item() if torch.is_tensor(direction_loss) else direction_loss,
                'mean_cosine': cosine_sim.mean().item() if valid_mask.sum() > 0 else 0.0,
            }

        return loss, info


class CombinedLoss(nn.Module):
    """
    組合多個損失函數

    Total Loss = λ₁ * Feature_Loss + λ₂ * Triplet_Loss + λ₃ * CE_Loss

    Args:
        feature_weight: Feature MSE Loss 權重
        triplet_weight: Triplet Loss 權重
        triplet_margin: Triplet margin
        ce_weight: Cross-Entropy Loss 權重
        ce_temperature: CE temperature
        direction_weight: Direction Loss 權重（可選）
    """

    def __init__(
        self,
        feature_weight: float = 1.0,
        triplet_weight: float = 1.0,
        triplet_margin: float = 0.5,
        ce_weight: float = 0.0,
        ce_temperature: float = 0.1,
        direction_weight: float = 0.0,
    ):
        super().__init__()
        self.feature_weight = feature_weight
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
        self.ce_temperature = ce_temperature
        self.direction_weight = direction_weight

        # 創建各個 loss
        self.triplet_loss = TripletLoss(margin=triplet_margin)

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        codebook: torch.Tensor,
        teacher_codes: torch.Tensor,
        original_encoder_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        計算組合損失

        Args:
            student_out: student encoder/adapter 輸出, (B, C, T)
            teacher_out: teacher encoder 輸出, (B, C, T)
            codebook: VQ codebook, (num_codes, C)
            teacher_codes: 正確的 token
            original_encoder_out: 原始 encoder 輸出（用於 direction loss）

        Returns:
            total_loss, info
        """
        losses = {}
        info = {}

        # 1. Feature MSE Loss
        if self.feature_weight > 0:
            feature_loss = F.mse_loss(student_out, teacher_out)
            losses['feature'] = self.feature_weight * feature_loss
            info['feature_loss'] = feature_loss.item()

        # 2. Triplet Loss
        if self.triplet_weight > 0:
            triplet_loss, triplet_info = self.triplet_loss(
                student_out, teacher_out, codebook, teacher_codes
            )
            losses['triplet'] = self.triplet_weight * triplet_loss
            info['triplet_loss'] = triplet_loss.item()
            info.update({f'triplet_{k}': v for k, v in triplet_info.items()})

        # 3. Cross-Entropy Loss
        if self.ce_weight > 0:
            B, C, T = student_out.shape

            # 計算 logits: -||z - c||^2
            z = student_out.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
            # 使用簡化的 logits 計算
            logits = 2 * torch.matmul(z, codebook.t())  # (B*T, num_codes)
            c_sq = (codebook ** 2).sum(dim=1)  # (num_codes,)
            logits = logits - c_sq.unsqueeze(0)
            logits = logits / self.ce_temperature

            # 處理 teacher_codes
            if teacher_codes.dim() == 3:
                teacher_codes = teacher_codes[0]
            targets = teacher_codes.reshape(-1).long()

            ce_loss = F.cross_entropy(logits, targets)
            losses['ce'] = self.ce_weight * ce_loss
            info['ce_loss'] = ce_loss.item()

        # 4. Direction Loss（可選）
        if self.direction_weight > 0 and original_encoder_out is not None:
            direction_loss_fn = DirectionAwareLoss(direction_weight=1.0)
            dir_loss, dir_info = direction_loss_fn(student_out, teacher_out, original_encoder_out)
            losses['direction'] = self.direction_weight * dir_loss
            info.update({f'direction_{k}': v for k, v in dir_info.items()})

        # 總損失
        total_loss = sum(losses.values())
        info['total_loss'] = total_loss.item()

        return total_loss, info


def compute_token_accuracy(student_codes: torch.Tensor, teacher_codes: torch.Tensor) -> float:
    """
    計算 token 準確率

    Args:
        student_codes: student tokens, (1, B, T) 或 (B, T)
        teacher_codes: teacher tokens, (1, B, T) 或 (B, T)

    Returns:
        準確率 (0-1)
    """
    if student_codes.dim() == 3:
        student_codes = student_codes[0]
    if teacher_codes.dim() == 3:
        teacher_codes = teacher_codes[0]

    return (student_codes == teacher_codes).float().mean().item()
