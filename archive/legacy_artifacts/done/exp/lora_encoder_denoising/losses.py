"""
Loss Functions for LoRA Encoder Denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDistillationLoss(nn.Module):
    """
    Feature-level + Code-level Distillation Loss
    """

    def __init__(
        self,
        feature_loss_weight=1.0,
        distance_loss_weight=0.1,
        vq_loss_weight=0.01,
    ):
        super().__init__()
        self.feature_loss_weight = feature_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.vq_loss_weight = vq_loss_weight

    def forward(self, model_output, distance_matrix):
        """
        Args:
            model_output: dict from TeacherStudentModel.forward()
            distance_matrix: (4096, 4096) codebook distances

        Returns:
            total_loss, metrics_dict
        """
        student_features = model_output['student_features']  # (B, 512, T)
        teacher_features = model_output['teacher_features']  # (B, 512, T)
        student_codes = model_output['student_codes']        # (B, 1, T)
        teacher_codes = model_output['teacher_codes']        # (B, 1, T)
        vq_loss = model_output['vq_loss']

        # Feature-level MSE
        feature_loss = F.mse_loss(student_features, teacher_features)

        # Distance-based code loss
        B, _, T = student_codes.shape
        student_flat = student_codes[:, 0, :].reshape(-1).long()
        teacher_flat = teacher_codes[:, 0, :].reshape(-1).long()

        distances = distance_matrix[student_flat, teacher_flat]
        distance_loss = distances.mean()

        # Code match rate (for monitoring)
        code_match_rate = (student_flat == teacher_flat).float().mean()

        # Total loss
        total_loss = (
            self.feature_loss_weight * feature_loss +
            self.distance_loss_weight * distance_loss +
            self.vq_loss_weight * vq_loss
        )

        metrics = {
            'total_loss': total_loss.item(),
            'feature_loss': feature_loss.item(),
            'distance_loss': distance_loss.item(),
            'vq_loss': vq_loss.item(),
            'code_match_rate': code_match_rate.item(),
        }

        return total_loss, metrics
