"""
Loss Functions for exp_1201: Soft Distance Loss

改進 exp_1128 的不可微 distance loss 問題
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDistanceLoss(nn.Module):
    """
    可微的 Soft Distance Loss

    原理：
    1. 計算 student features 到所有 codes 的距離
    2. 用 softmax 將距離轉換為機率分布 (soft codes)
    3. 計算 soft codes 與 teacher codes 之間的期望距離

    相比 exp_1128 的 hard distance loss:
    - 舊方法: distance_matrix[argmax(student), teacher] → 不可微
    - 新方法: softmax(-dist/τ) @ distance_matrix[teacher] → 可微
    """

    def __init__(self, temperature=1.0):
        """
        Args:
            temperature: Softmax temperature (τ)
                - τ → 0: 接近 hard (one-hot)
                - τ → ∞: 接近 uniform
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, student_features, teacher_codes, codebook, distance_matrix):
        """
        Args:
            student_features: (B, 512, T) - Student encoder 輸出
            teacher_codes: (B, 1, T) - Teacher 的離散 codes
            codebook: (4096, 512) - VQ codebook
            distance_matrix: (4096, 4096) - 預計算的 code 間距離

        Returns:
            soft_distance_loss: scalar
            metrics: dict with debugging info
        """
        B, C, T = student_features.shape
        device = student_features.device

        # 1. Reshape features: (B, 512, T) -> (B*T, 512)
        features_flat = student_features.permute(0, 2, 1).reshape(-1, C)

        # 2. 計算到所有 codes 的距離: (B*T, 4096)
        # 使用 cdist 計算 L2 距離
        distances = torch.cdist(
            features_flat.unsqueeze(0),
            codebook.unsqueeze(0)
        ).squeeze(0)  # (B*T, 4096)

        # 3. Softmax over distances (負距離，距離越小機率越大)
        logits = -distances / self.temperature
        soft_codes = F.softmax(logits, dim=-1)  # (B*T, 4096)

        # 4. 獲取 teacher codes: (B*T,)
        teacher_flat = teacher_codes[:, 0, :].reshape(-1).long()

        # 5. 獲取 teacher code 到所有 codes 的距離: (B*T, 4096)
        teacher_distances = distance_matrix[teacher_flat]

        # 6. 計算期望距離 (weighted sum)
        expected_distance = (soft_codes * teacher_distances).sum(dim=-1)  # (B*T,)

        # 7. 平均
        soft_distance_loss = expected_distance.mean()

        # Metrics for monitoring
        # Hard codes (for comparison)
        hard_codes = distances.argmin(dim=-1)
        code_match_rate = (hard_codes == teacher_flat).float().mean()

        # Entropy of soft codes (高熵 = 更不確定)
        entropy = -(soft_codes * torch.log(soft_codes + 1e-10)).sum(dim=-1).mean()

        metrics = {
            'soft_distance_loss': soft_distance_loss.item(),
            'code_match_rate': code_match_rate.item(),
            'soft_code_entropy': entropy.item(),
            'expected_distance_mean': expected_distance.mean().item(),
            'expected_distance_std': expected_distance.std().item(),
        }

        return soft_distance_loss, metrics


class EncoderDistillationLoss(nn.Module):
    """
    Complete Loss Function for LoRA Encoder Denoising

    組成:
    - Feature MSE Loss: ||student_features - teacher_features||²
    - Soft Distance Loss: 可微的 code alignment loss
    - VQ Loss: commitment loss (通常設為 0，凍結 codebook)
    """

    def __init__(
        self,
        feature_loss_weight=1.0,
        soft_dist_loss_weight=0.1,
        vq_loss_weight=0.0,
        temperature=1.0,
    ):
        super().__init__()
        self.feature_loss_weight = feature_loss_weight
        self.soft_dist_loss_weight = soft_dist_loss_weight
        self.vq_loss_weight = vq_loss_weight

        self.soft_distance_loss_fn = SoftDistanceLoss(temperature=temperature)

    def forward(self, model_output, distance_matrix, codebook):
        """
        Args:
            model_output: dict from TeacherStudentModel.forward()
            distance_matrix: (4096, 4096) codebook distances
            codebook: (4096, 512) VQ codebook

        Returns:
            total_loss, metrics_dict
        """
        student_features = model_output['student_features']  # (B, 512, T)
        teacher_features = model_output['teacher_features']  # (B, 512, T)
        teacher_codes = model_output['teacher_codes']        # (B, 1, T)
        vq_loss = model_output['vq_loss']

        # 1. Feature-level MSE Loss
        feature_loss = F.mse_loss(student_features, teacher_features)

        # 2. Soft Distance Loss (可微!)
        soft_dist_loss, soft_dist_metrics = self.soft_distance_loss_fn(
            student_features, teacher_codes, codebook, distance_matrix
        )

        # 3. Hard Distance Loss (for monitoring only, 不參與梯度)
        with torch.no_grad():
            student_codes = model_output['student_codes']
            B, _, T = student_codes.shape
            student_flat = student_codes[:, 0, :].reshape(-1).long()
            teacher_flat = teacher_codes[:, 0, :].reshape(-1).long()
            hard_distances = distance_matrix[student_flat, teacher_flat]
            hard_distance_loss = hard_distances.mean()

        # Total loss
        total_loss = (
            self.feature_loss_weight * feature_loss +
            self.soft_dist_loss_weight * soft_dist_loss +
            self.vq_loss_weight * vq_loss
        )

        metrics = {
            'total_loss': total_loss.item(),
            'feature_loss': feature_loss.item(),
            'soft_distance_loss': soft_dist_loss.item(),
            'hard_distance_loss': hard_distance_loss.item(),  # monitoring
            'vq_loss': vq_loss.item(),
            'code_match_rate': soft_dist_metrics['code_match_rate'],
            'soft_code_entropy': soft_dist_metrics['soft_code_entropy'],
        }

        return total_loss, metrics


def test_gradient_flow():
    """測試 Soft Distance Loss 的梯度流"""
    print("=" * 60)
    print("Testing Soft Distance Loss Gradient Flow")
    print("=" * 60)

    B, T, C = 2, 10, 512
    codebook_size = 4096

    # 創建測試數據
    student_features = torch.randn(B, C, T, requires_grad=True)
    teacher_codes = torch.randint(0, codebook_size, (B, 1, T))
    codebook = torch.randn(codebook_size, C)
    distance_matrix = torch.cdist(codebook, codebook, p=2)

    # 創建 loss function
    loss_fn = SoftDistanceLoss(temperature=1.0)

    # Forward
    loss, metrics = loss_fn(student_features, teacher_codes, codebook, distance_matrix)

    print(f"\nSoft Distance Loss: {loss.item():.4f}")
    print(f"  requires_grad: {loss.requires_grad}")
    print(f"  grad_fn: {loss.grad_fn}")

    # Backward
    print("\nAttempting backward...")
    try:
        loss.backward()
        if student_features.grad is not None:
            print(f"  ✅ SUCCESS! grad norm: {student_features.grad.norm().item():.6f}")
        else:
            print(f"  ❌ FAILED: grad is None")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_gradient_flow()
