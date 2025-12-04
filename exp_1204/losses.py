"""
exp_1204: Curriculum Learning + Mixed Loss (MSE + CE)

核心改進：
1. 漸進式學習 (Curriculum Learning):
   - 階段 1: 高溫度 MSE，讓 emb 大致靠近
   - 階段 2: 中溫度 MSE + CE 混合
   - 階段 3: 低溫度 CE 為主，精確對齊

2. 混合 Loss:
   - MSE Loss: 穩定訓練，提供平滑梯度
   - CE Loss: 直接監督 token 選擇，強烈梯度

3. Temperature Annealing:
   - 訓練初期用高溫度（軟監督）
   - 逐漸降低到低溫度（硬監督）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CurriculumEmbDistillationLoss(nn.Module):
    """
    漸進式 Embedding Distillation Loss

    結合：
    1. MSE Loss: student_emb → codebook[teacher_codes]
    2. CE Loss: 直接監督 token 選擇
    3. Temperature Annealing: 隨訓練進度調整溫度

    漸進策略：
    - 階段 1 (warm-up): 高溫度 MSE 為主，CE 權重小
    - 階段 2 (transition): 降低溫度，增加 CE 權重
    - 階段 3 (refinement): 低溫度，CE 為主
    """

    def __init__(
        self,
        # Loss 權重
        mse_weight=1.0,           # MSE Loss 初始權重
        ce_weight=0.0,            # CE Loss 初始權重
        feature_loss_weight=0.0,  # 量化後特徵 MSE

        # Temperature 設定
        initial_temperature=2.0,  # 初始溫度（軟監督）
        final_temperature=0.1,    # 最終溫度（硬監督）

        # Curriculum 設定
        curriculum_mode='linear', # 'linear', 'cosine', 'step'
        warmup_epochs=5,          # MSE 為主的 warm-up 階段
        transition_epochs=20,     # 漸進轉換階段

        # 其他
        label_smoothing=0.0,
    ):
        """
        Args:
            mse_weight: MSE Loss 權重
            ce_weight: CE Loss 最終權重（會從 0 漸進到這個值）
            feature_loss_weight: 量化後特徵 MSE 權重
            initial_temperature: 初始溫度
            final_temperature: 最終溫度
            curriculum_mode: 漸進模式 ('linear', 'cosine', 'step')
            warmup_epochs: warm-up 階段（只用 MSE）
            transition_epochs: 轉換階段
            label_smoothing: CE Loss label smoothing
        """
        super().__init__()

        # Loss 權重
        self.mse_weight = mse_weight
        self.ce_weight_final = ce_weight
        self.feature_loss_weight = feature_loss_weight

        # Temperature
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature

        # Curriculum
        self.curriculum_mode = curriculum_mode
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs

        # Other
        self.label_smoothing = label_smoothing

        # 當前狀態（由 trainer 更新）
        self.current_epoch = 0
        self.total_epochs = 50

    def set_epoch(self, epoch, total_epochs=None):
        """更新當前 epoch（由 trainer 調用）"""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs

    def get_current_temperature(self):
        """根據當前 epoch 計算溫度"""
        if self.current_epoch < self.warmup_epochs:
            # Warm-up: 使用初始溫度
            return self.initial_temperature

        # 計算 transition 進度 (0 → 1)
        progress = min(1.0, (self.current_epoch - self.warmup_epochs) /
                       max(1, self.transition_epochs))

        if self.curriculum_mode == 'linear':
            temp = self.initial_temperature + progress * (
                self.final_temperature - self.initial_temperature)
        elif self.curriculum_mode == 'cosine':
            temp = self.final_temperature + 0.5 * (
                self.initial_temperature - self.final_temperature) * (
                1 + math.cos(math.pi * progress))
        elif self.curriculum_mode == 'step':
            # 分三階段
            if progress < 0.33:
                temp = self.initial_temperature
            elif progress < 0.66:
                temp = (self.initial_temperature + self.final_temperature) / 2
            else:
                temp = self.final_temperature
        else:
            temp = self.initial_temperature

        return max(temp, self.final_temperature)

    def get_current_ce_weight(self):
        """根據當前 epoch 計算 CE Loss 權重"""
        if self.current_epoch < self.warmup_epochs:
            # Warm-up: CE 權重為 0
            return 0.0

        # 計算 transition 進度
        progress = min(1.0, (self.current_epoch - self.warmup_epochs) /
                       max(1, self.transition_epochs))

        if self.curriculum_mode == 'linear':
            return progress * self.ce_weight_final
        elif self.curriculum_mode == 'cosine':
            return self.ce_weight_final * (1 - math.cos(math.pi * progress)) / 2
        elif self.curriculum_mode == 'step':
            if progress < 0.33:
                return 0.0
            elif progress < 0.66:
                return self.ce_weight_final * 0.5
            else:
                return self.ce_weight_final
        else:
            return self.ce_weight_final * progress

    def forward(self, model_output, distance_matrix, codebook):
        """
        Args:
            model_output: dict with student_emb, teacher_codes, etc.
            distance_matrix: (4096, 4096) codebook distances
            codebook: (4096, 512) VQ codebook

        Returns:
            total_loss: scalar
            metrics: dict
        """
        # 獲取當前 curriculum 參數
        current_temp = self.get_current_temperature()
        current_ce_weight = self.get_current_ce_weight()

        # 必要的輸出
        student_emb = model_output['student_emb']      # (B, C, T)
        teacher_codes = model_output['teacher_codes']  # (n_q, B, T)

        B, C, T_emb = student_emb.shape

        # 處理 teacher_codes 格式
        if teacher_codes.dim() == 3:
            teacher_codes_2d = teacher_codes[0]  # (B, T)
        else:
            teacher_codes_2d = teacher_codes.squeeze(1)

        T_code = teacher_codes_2d.shape[1]
        T = min(T_emb, T_code)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. MSE Loss: student_emb → codebook[teacher_codes]
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        teacher_flat = teacher_codes_2d[:, :T].reshape(-1).long()  # (B*T,)
        target_embeddings = codebook[teacher_flat]  # (B*T, C)

        emb_truncated = student_emb[:, :, :T]
        emb_flat = emb_truncated.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)

        mse_loss = F.mse_loss(emb_flat, target_embeddings)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. CE Loss: 直接監督 token 選擇
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ce_loss = torch.tensor(0.0, device=student_emb.device)

        if current_ce_weight > 0:
            # 計算距離
            distances = torch.cdist(
                emb_flat.unsqueeze(0),
                codebook.unsqueeze(0)
            ).squeeze(0)  # (B*T, 4096)

            # 轉換為 logits（使用當前溫度）
            logits = -distances / current_temp

            # CE Loss
            ce_loss = F.cross_entropy(
                logits,
                teacher_flat,
                label_smoothing=self.label_smoothing
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. Feature Loss (VQ 後特徵監控)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        feature_loss = torch.tensor(0.0, device=student_emb.device)
        vq_feature_loss_monitor = torch.tensor(0.0, device=student_emb.device)

        if 'student_features' in model_output and 'teacher_features' in model_output:
            student_features = model_output['student_features']
            teacher_features = model_output['teacher_features']

            T_sf = student_features.shape[-1]
            T_tf = teacher_features.shape[-1]
            T_feat = min(T_sf, T_tf)

            # 監控用
            with torch.no_grad():
                vq_feature_loss_monitor = F.mse_loss(
                    student_features[:, :, :T_feat],
                    teacher_features[:, :, :T_feat]
                )

            if self.feature_loss_weight > 0:
                feature_loss = F.mse_loss(
                    student_features[:, :, :T_feat],
                    teacher_features[:, :, :T_feat]
                )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Total Loss
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        total_loss = (
            self.mse_weight * mse_loss +
            current_ce_weight * ce_loss +
            self.feature_loss_weight * feature_loss
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Metrics
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with torch.no_grad():
            # Token Accuracy
            distances = torch.cdist(
                emb_flat.unsqueeze(0),
                codebook.unsqueeze(0)
            ).squeeze(0)
            predictions = distances.argmin(dim=-1)
            token_accuracy = (predictions == teacher_flat).float().mean()

            # Emb 到 target 的距離
            emb_to_target_dist = torch.sqrt(
                ((emb_flat - target_embeddings) ** 2).sum(dim=-1)
            ).mean()

            # Hard distance loss (如果有 student_codes)
            hard_distance_loss = torch.tensor(0.0)
            if 'student_codes' in model_output:
                student_codes = model_output['student_codes']
                if student_codes.dim() == 3:
                    student_codes_2d = student_codes[0]
                else:
                    student_codes_2d = student_codes.squeeze(1)
                student_flat = student_codes_2d[:, :T].reshape(-1).long()
                hard_distances = distance_matrix[student_flat, teacher_flat]
                hard_distance_loss = hard_distances.mean()

        metrics = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'ce_loss': ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            'feature_loss': feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss,
            'vq_feature_loss_monitor': vq_feature_loss_monitor.item(),
            'hard_distance_loss': hard_distance_loss.item(),
            'code_match_rate': token_accuracy.item(),
            'emb_to_target_dist': emb_to_target_dist.item(),
            # Curriculum 狀態
            'current_temperature': current_temp,
            'current_ce_weight': current_ce_weight,
            'curriculum_epoch': self.current_epoch,
        }

        return total_loss, metrics


# 為了向後相容，保留原有的 EmbDistillationLoss
class EmbDistillationLoss(CurriculumEmbDistillationLoss):
    """
    向後相容的 wrapper，使用 CurriculumEmbDistillationLoss
    但預設不啟用 curriculum（保持原行為）
    """
    def __init__(
        self,
        emb_to_codebook_weight=1.0,
        ce_token_weight=0.0,
        feature_loss_weight=0.0,
        vq_loss_weight=0.0,  # 向後相容，但不使用
        temperature=1.0,
        label_smoothing=0.0,
    ):
        super().__init__(
            mse_weight=emb_to_codebook_weight,
            ce_weight=ce_token_weight,
            feature_loss_weight=feature_loss_weight,
            initial_temperature=temperature,
            final_temperature=temperature,  # 不變
            warmup_epochs=0,  # 不使用 curriculum
            transition_epochs=1,
            label_smoothing=label_smoothing,
        )
