"""
exp_1231: 優化方向測試 - Loss Functions

包含：
- SoftTokenLoss: KL Divergence 監督 logits 分布
- ContrastiveTokenLoss: 對比學習 loss
- MultiScaleFeatureLoss: 多尺度特徵監督
- ProgressiveLossScheduler: 漸進式 loss 權重調度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class SoftTokenLoss(nn.Module):
    """
    Exp71: Soft Token Loss using KL Divergence

    不只監督 argmax token，監督整個 logits 分布。
    讓 student 的 feature 分布接近 teacher 的分布。
    """

    def __init__(self, temperature: float = 1.0, reduction: str = 'batchmean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        codebook: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        encoder_stride: int = 320,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            student_features: (B, D, T) or (B, T, D) - Student encoder output
            teacher_features: (B, D, T) or (B, T, D) - Teacher encoder output
            codebook: (K, D) - VQ codebook, K=4096
            lengths: (B,) - Original audio lengths
            encoder_stride: Encoder stride for calculating valid frames

        Returns:
            loss: KL Divergence loss
            info: Dict with metrics
        """
        # 處理維度：encoder output 可能是 (B, D, T) 或 (B, T, D)
        # codebook 的 D=512，所以我們用它來判斷
        D_codebook = codebook.shape[1]  # 512

        # 如果 student_features 的最後一維不是 D，需要 transpose
        if student_features.shape[-1] != D_codebook:
            student_features = student_features.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        if teacher_features.shape[-1] != D_codebook:
            teacher_features = teacher_features.transpose(1, 2)

        B, T, D = student_features.shape
        K = codebook.shape[0]

        # 計算 features 到 codebook 的距離
        # Reshape for cdist: (B*T, D) vs (K, D)
        student_flat = student_features.reshape(-1, D)  # (B*T, D)
        teacher_flat = teacher_features.reshape(-1, D)  # (B*T, D)

        # 計算歐氏距離的平方
        # student_dist: (B*T, K)
        student_dist = torch.cdist(student_flat, codebook, p=2).pow(2)
        teacher_dist = torch.cdist(teacher_flat, codebook, p=2).pow(2)

        # 轉換為 logits (負距離 / temperature)
        student_logits = -student_dist / self.temperature
        teacher_logits = -teacher_dist / self.temperature

        # Softmax / Log-softmax
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()  # Teacher 不需要梯度

        # 建立 mask 用於有效區域
        if lengths is not None:
            mask = torch.zeros(B, T, device=student_features.device)
            for i in range(B):
                valid_frames = lengths[i].item() // encoder_stride
                valid_frames = min(valid_frames, T)
                mask[i, :valid_frames] = 1.0
            mask = mask.reshape(-1)  # (B*T,)
        else:
            mask = torch.ones(B * T, device=student_features.device)

        # 計算 KL Divergence (只計算有效區域)
        # KL(P || Q) = sum(P * log(P/Q)) = sum(P * log(P) - P * log(Q))
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)

        # 加權平均
        valid_tokens = mask.sum()
        if valid_tokens > 0:
            loss = (kl_per_token * mask).sum() / valid_tokens
        else:
            loss = kl_per_token.mean()

        # 計算額外指標
        student_codes = student_logits.argmax(dim=-1)
        teacher_codes = teacher_logits.argmax(dim=-1)
        accuracy = ((student_codes == teacher_codes).float() * mask).sum() / valid_tokens if valid_tokens > 0 else 0

        info = {
            'soft_token_loss': loss.item(),
            'soft_token_accuracy': accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
            'temperature': self.temperature,
        }

        return loss, info


class ContrastiveTokenLoss(nn.Module):
    """
    Exp72: Contrastive Token Loss (InfoNCE)

    讓 student feature 靠近正確的 codebook entry，遠離錯誤的。
    使用 Hard Negative Mining 選擇最有挑戰性的負樣本。
    """

    def __init__(
        self,
        temperature: float = 0.1,
        num_negatives: int = 16,
        hard_negative_mining: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.hard_negative_mining = hard_negative_mining

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_codes: torch.Tensor,
        codebook: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        encoder_stride: int = 320,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            student_features: (B, D, T) or (B, T, D)
            teacher_codes: (B, T) - Teacher 選擇的 VQ codes
            codebook: (K, D)
            lengths: (B,)

        Returns:
            loss: InfoNCE loss
            info: Dict with metrics
        """
        # 處理維度
        D_codebook = codebook.shape[1]
        if student_features.shape[-1] != D_codebook:
            student_features = student_features.transpose(1, 2)

        B, T, D = student_features.shape
        K = codebook.shape[0]
        device = student_features.device

        # 正樣本：Teacher 選擇的 codes
        # positive: (B, T, D)
        positive = codebook[teacher_codes]

        # 計算 student 到所有 codes 的距離 (用於 hard negative mining)
        student_flat = student_features.reshape(-1, D)  # (B*T, D)
        distances = torch.cdist(student_flat, codebook, p=2)  # (B*T, K)

        # 選擇負樣本
        if self.hard_negative_mining:
            # Hard Negative: 排除正確答案後，選擇最近的 k 個
            teacher_codes_flat = teacher_codes.reshape(-1)  # (B*T,)

            # 將正確答案的距離設為無限大
            mask = torch.zeros_like(distances)
            mask.scatter_(1, teacher_codes_flat.unsqueeze(1), float('inf'))
            masked_distances = distances + mask

            # 選擇最近的 k 個
            _, neg_indices = masked_distances.topk(self.num_negatives, dim=-1, largest=False)
        else:
            # Random Negative: 隨機選擇
            neg_indices = torch.randint(0, K, (B * T, self.num_negatives), device=device)

        # 取得負樣本 embeddings
        # negatives: (B*T, num_negatives, D)
        negatives = codebook[neg_indices]

        # 正規化 features (for cosine similarity)
        student_flat_norm = F.normalize(student_flat, dim=-1)
        positive_flat = positive.reshape(-1, D)
        positive_norm = F.normalize(positive_flat, dim=-1)
        negatives_norm = F.normalize(negatives, dim=-1)

        # 計算相似度
        # pos_sim: (B*T,)
        pos_sim = (student_flat_norm * positive_norm).sum(dim=-1) / self.temperature

        # neg_sim: (B*T, num_negatives)
        neg_sim = torch.bmm(
            student_flat_norm.unsqueeze(1),  # (B*T, 1, D)
            negatives_norm.transpose(1, 2)   # (B*T, D, num_negatives)
        ).squeeze(1) / self.temperature

        # InfoNCE Loss
        # log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B*T, 1+num_negatives)
        labels = torch.zeros(B * T, dtype=torch.long, device=device)  # 正樣本在第 0 位

        # 建立 mask
        if lengths is not None:
            mask = torch.zeros(B, T, device=device)
            for i in range(B):
                valid_frames = lengths[i].item() // encoder_stride
                valid_frames = min(valid_frames, T)
                mask[i, :valid_frames] = 1.0
            mask = mask.reshape(-1)
        else:
            mask = torch.ones(B * T, device=device)

        # Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        valid_tokens = mask.sum()
        loss = (ce_loss * mask).sum() / valid_tokens if valid_tokens > 0 else ce_loss.mean()

        # 計算準確率 (正樣本相似度是否最高)
        predictions = logits.argmax(dim=-1)
        accuracy = ((predictions == 0).float() * mask).sum() / valid_tokens if valid_tokens > 0 else 0

        info = {
            'contrastive_loss': loss.item(),
            'contrastive_accuracy': accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
            'avg_pos_sim': (pos_sim * mask).sum().item() / valid_tokens.item() if valid_tokens > 0 else 0,
            'avg_neg_sim': (neg_sim.mean(dim=-1) * mask).sum().item() / valid_tokens.item() if valid_tokens > 0 else 0,
        }

        return loss, info


class MultiScaleFeatureLoss(nn.Module):
    """
    Exp74: Multi-Scale Feature Loss

    監督多個 encoder 中間層的特徵，不只是最後一層。
    """

    def __init__(
        self,
        layer_weights: List[float] = [0.1, 0.3, 0.6],
        normalize: bool = True,
    ):
        super().__init__()
        self.layer_weights = layer_weights
        self.normalize = normalize

    def forward(
        self,
        student_features_list: List[torch.Tensor],
        teacher_features_list: List[torch.Tensor],
        lengths: Optional[torch.Tensor] = None,
        encoder_stride: int = 320,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            student_features_list: List of (B, T, D) tensors from different layers
            teacher_features_list: List of (B, T, D) tensors from different layers
            lengths: (B,)

        Returns:
            loss: Multi-scale feature loss
            info: Dict with per-layer losses
        """
        assert len(student_features_list) == len(teacher_features_list)
        assert len(student_features_list) == len(self.layer_weights)

        device = student_features_list[0].device
        B, T, D = student_features_list[0].shape

        # 建立 mask
        if lengths is not None:
            mask = torch.zeros(B, T, device=device)
            for i in range(B):
                valid_frames = lengths[i].item() // encoder_stride
                valid_frames = min(valid_frames, T)
                mask[i, :valid_frames] = 1.0
        else:
            mask = torch.ones(B, T, device=device)

        total_loss = 0
        layer_losses = {}

        for i, (s_feat, t_feat, weight) in enumerate(zip(
            student_features_list, teacher_features_list, self.layer_weights
        )):
            if self.normalize:
                s_feat = F.normalize(s_feat, dim=-1)
                t_feat = F.normalize(t_feat, dim=-1)

            # MSE Loss
            diff = (s_feat - t_feat).pow(2).mean(dim=-1)  # (B, T)
            layer_loss = (diff * mask).sum() / mask.sum()

            total_loss += weight * layer_loss
            layer_losses[f'layer_{i}_loss'] = layer_loss.item()

        info = {
            'multi_scale_loss': total_loss.item(),
            **layer_losses,
        }

        return total_loss, info


class ProgressiveLossScheduler:
    """
    Exp75: Progressive Loss Schedule

    根據訓練進度動態調整各個 loss 的權重。
    """

    def __init__(
        self,
        total_epochs: int = 300,
        phase1_ratio: float = 0.33,
        phase2_ratio: float = 0.33,
        # Phase 3 = 1 - phase1 - phase2
    ):
        self.total_epochs = total_epochs
        self.phase1_end = int(total_epochs * phase1_ratio)
        self.phase2_end = int(total_epochs * (phase1_ratio + phase2_ratio))

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        返回當前 epoch 的 loss 權重

        Phase 1: 只訓練 Feature Loss
        Phase 2: 加入 Soft Token Loss
        Phase 3: Soft Token 為主
        """
        if epoch <= self.phase1_end:
            # Phase 1: 純 feature learning
            return {
                'feature_weight': 1.0,
                'triplet_weight': 1.0,
                'soft_token_weight': 0.0,
                'phase': 1,
            }
        elif epoch <= self.phase2_end:
            # Phase 2: 漸進加入 soft token
            progress = (epoch - self.phase1_end) / (self.phase2_end - self.phase1_end)
            return {
                'feature_weight': 1.0,
                'triplet_weight': 1.0,
                'soft_token_weight': progress,  # 0 → 1
                'phase': 2,
            }
        else:
            # Phase 3: soft token 為主
            progress = (epoch - self.phase2_end) / (self.total_epochs - self.phase2_end)
            return {
                'feature_weight': max(0.5, 1.0 - progress * 0.5),  # 1 → 0.5
                'triplet_weight': max(0.5, 1.0 - progress * 0.5),
                'soft_token_weight': 1.0,
                'phase': 3,
            }


class CombinedLossExp71(nn.Module):
    """
    Exp71 專用：Feature Loss + Triplet Loss + Soft Token Loss
    """

    def __init__(
        self,
        feature_weight: float = 1.0,
        triplet_weight: float = 1.0,
        soft_token_weight: float = 1.0,
        triplet_margin: float = 0.2,
        soft_token_temperature: float = 1.0,
        vq_commitment_weight: float = 0.1,
        vq_distortion_weight: float = 0.1,
    ):
        super().__init__()
        self.feature_weight = feature_weight
        self.triplet_weight = triplet_weight
        self.soft_token_weight = soft_token_weight
        self.triplet_margin = triplet_margin
        self.vq_commitment_weight = vq_commitment_weight
        self.vq_distortion_weight = vq_distortion_weight

        self.soft_token_loss = SoftTokenLoss(temperature=soft_token_temperature)

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_codes: torch.Tensor,
        codebook: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        encoder_stride: int = 320,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        計算組合 loss
        """
        # 處理維度：encoder output 可能是 (B, D, T) 或 (B, T, D)
        D_codebook = codebook.shape[1]
        if student_features.shape[-1] != D_codebook:
            student_features = student_features.transpose(1, 2)
        if teacher_features.shape[-1] != D_codebook:
            teacher_features = teacher_features.transpose(1, 2)

        B, T, D = student_features.shape
        device = student_features.device

        # 建立 mask
        if lengths is not None:
            mask = torch.zeros(B, T, device=device)
            for i in range(B):
                valid_frames = lengths[i].item() // encoder_stride
                valid_frames = min(valid_frames, T)
                mask[i, :valid_frames] = 1.0
        else:
            mask = torch.ones(B, T, device=device)

        total_loss = 0
        info = {}

        # 1. Feature Loss (MSE)
        feature_diff = (student_features - teacher_features).pow(2).mean(dim=-1)
        feature_loss = (feature_diff * mask).sum() / mask.sum()
        total_loss += self.feature_weight * feature_loss
        info['feature_loss'] = feature_loss.item()

        # 2. Triplet Loss
        if self.triplet_weight > 0:
            # 正樣本距離
            pos_dist = (student_features - teacher_features).pow(2).sum(dim=-1).sqrt()

            # 負樣本：shift 後的 teacher features
            neg_teacher = torch.roll(teacher_features, shifts=1, dims=0)
            neg_dist = (student_features - neg_teacher).pow(2).sum(dim=-1).sqrt()

            triplet = F.relu(pos_dist - neg_dist + self.triplet_margin)
            triplet_loss = (triplet * mask).sum() / mask.sum()
            total_loss += self.triplet_weight * triplet_loss
            info['triplet_loss'] = triplet_loss.item()

        # 3. Soft Token Loss
        if self.soft_token_weight > 0:
            soft_loss, soft_info = self.soft_token_loss(
                student_features, teacher_features, codebook, lengths, encoder_stride
            )
            total_loss += self.soft_token_weight * soft_loss
            info.update(soft_info)

        # 4. VQ Commitment Loss
        if self.vq_commitment_weight > 0:
            student_codes = self._get_codes(student_features, codebook)
            quantized = codebook[student_codes]
            commitment_loss = (student_features - quantized.detach()).pow(2).mean()
            total_loss += self.vq_commitment_weight * commitment_loss
            info['vq_commitment_loss'] = commitment_loss.item()

        # 5. VQ Distortion Loss
        if self.vq_distortion_weight > 0:
            student_codes = self._get_codes(student_features, codebook)
            quantized = codebook[student_codes]
            distortion_loss = (student_features.detach() - quantized).pow(2).mean()
            total_loss += self.vq_distortion_weight * distortion_loss
            info['vq_distortion_loss'] = distortion_loss.item()

        info['total_loss'] = total_loss.item()
        return total_loss, info

    def _get_codes(self, features: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """計算最近的 codebook indices"""
        B, T, D = features.shape
        flat = features.reshape(-1, D)
        distances = torch.cdist(flat, codebook, p=2)
        codes = distances.argmin(dim=-1)
        return codes.reshape(B, T)


class CombinedLossExp72(nn.Module):
    """
    Exp72 專用：Feature Loss + Triplet Loss + Contrastive Token Loss
    """

    def __init__(
        self,
        feature_weight: float = 1.0,
        triplet_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        triplet_margin: float = 0.2,
        contrastive_temperature: float = 0.1,
        num_negatives: int = 16,
        hard_negative_mining: bool = True,
    ):
        super().__init__()
        self.feature_weight = feature_weight
        self.triplet_weight = triplet_weight
        self.contrastive_weight = contrastive_weight
        self.triplet_margin = triplet_margin

        self.contrastive_loss = ContrastiveTokenLoss(
            temperature=contrastive_temperature,
            num_negatives=num_negatives,
            hard_negative_mining=hard_negative_mining,
        )

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_codes: torch.Tensor,
        codebook: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        encoder_stride: int = 320,
    ) -> Tuple[torch.Tensor, Dict]:
        # 處理維度：encoder output 可能是 (B, D, T) 或 (B, T, D)
        D_codebook = codebook.shape[1]
        if student_features.shape[-1] != D_codebook:
            student_features = student_features.transpose(1, 2)
        if teacher_features.shape[-1] != D_codebook:
            teacher_features = teacher_features.transpose(1, 2)

        B, T, D = student_features.shape
        device = student_features.device

        # 建立 mask
        if lengths is not None:
            mask = torch.zeros(B, T, device=device)
            for i in range(B):
                valid_frames = lengths[i].item() // encoder_stride
                valid_frames = min(valid_frames, T)
                mask[i, :valid_frames] = 1.0
        else:
            mask = torch.ones(B, T, device=device)

        total_loss = 0
        info = {}

        # 1. Feature Loss
        feature_diff = (student_features - teacher_features).pow(2).mean(dim=-1)
        feature_loss = (feature_diff * mask).sum() / mask.sum()
        total_loss += self.feature_weight * feature_loss
        info['feature_loss'] = feature_loss.item()

        # 2. Triplet Loss
        if self.triplet_weight > 0:
            pos_dist = (student_features - teacher_features).pow(2).sum(dim=-1).sqrt()
            neg_teacher = torch.roll(teacher_features, shifts=1, dims=0)
            neg_dist = (student_features - neg_teacher).pow(2).sum(dim=-1).sqrt()
            triplet = F.relu(pos_dist - neg_dist + self.triplet_margin)
            triplet_loss = (triplet * mask).sum() / mask.sum()
            total_loss += self.triplet_weight * triplet_loss
            info['triplet_loss'] = triplet_loss.item()

        # 3. Contrastive Token Loss
        if self.contrastive_weight > 0:
            contrastive_loss, contrastive_info = self.contrastive_loss(
                student_features, teacher_codes, codebook, lengths, encoder_stride
            )
            total_loss += self.contrastive_weight * contrastive_loss
            info.update(contrastive_info)

        info['total_loss'] = total_loss.item()
        return total_loss, info
