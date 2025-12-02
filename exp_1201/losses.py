"""
Loss Functions for exp_1201: Soft Distance Loss

改進 exp_1128 的不可微 distance loss 問題

方法比較:
1. Soft Distance Loss (softmax): 用 softmax 將距離轉換為機率分布
2. Gumbel-Softmax: 直接修改 VQ，forward=hard, backward=soft
3. STE (Straight-Through): 已有，但不傳遞 code 選擇資訊

本實驗同時提供兩種方法供比較。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax_sample(logits, temperature=1.0, hard=True):
    """
    Gumbel-Softmax: 可微的離散採樣

    Args:
        logits: (B*T, codebook_size) - 距離的負值 (或任何 logits)
        temperature: τ，越小越接近 one-hot
        hard: 如果 True，forward 用 hard (argmax)，backward 用 soft

    Returns:
        soft_codes: (B*T, codebook_size) - soft or hard one-hot
    """
    # Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature

    # Softmax
    y_soft = F.softmax(gumbels, dim=-1)

    if hard:
        # Straight-through: forward 用 hard，backward 用 soft
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


class TokenClassificationLoss(nn.Module):
    """
    Token Classification Loss - 直接監督 token 選擇
    
    原理：
    將 token 選擇視為 4096 類分類問題，用 Cross-Entropy 直接監督。
    
    這是最直接的方式：
    - 距離越小 → logits 越大 → 被選中的機率越高
    - Cross-Entropy 會懲罰任何選錯的情況
    
    優點：
    - 直接優化 Token Accuracy（本質問題）
    - 梯度清晰：選錯 token 就會被懲罰
    
    缺點：
    - 不考慮 token 間的語義相似性
    - 可能過於「硬」
    """
    
    def __init__(self, temperature=1.0, label_smoothing=0.0):
        """
        Args:
            temperature: 距離到 logits 的縮放因子
            label_smoothing: Label smoothing 係數 (0.0 = 無)
        """
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(self, student_features, teacher_codes, codebook, distance_matrix=None):
        """
        Args:
            student_features: (B, 512, T) - Student encoder 輸出
            teacher_codes: (n_q, B, T) - Teacher 的離散 codes
            codebook: (4096, 512) - VQ codebook
            distance_matrix: 不使用，保持介面一致
            
        Returns:
            ce_loss: scalar - Cross-Entropy Loss
            metrics: dict - 監控指標
        """
        B, C, T_feat = student_features.shape
        
        # 處理 teacher_codes 格式 (n_q, B, T) → (B, T)
        if teacher_codes.dim() == 3:
            teacher_codes_2d = teacher_codes[0]
        else:
            teacher_codes_2d = teacher_codes.squeeze(1)
        
        T_code = teacher_codes_2d.shape[1]
        T = min(T_feat, T_code)
        
        # 1. Reshape features: (B, 512, T) -> (B*T, 512)
        features_truncated = student_features[:, :, :T]
        features_flat = features_truncated.permute(0, 2, 1).reshape(-1, C)
        
        # 2. 計算到所有 codes 的距離: (B*T, 4096)
        distances = torch.cdist(
            features_flat.unsqueeze(0),
            codebook.unsqueeze(0)
        ).squeeze(0)
        
        # 3. 轉換為 logits (負距離 / temperature)
        logits = -distances / self.temperature  # (B*T, 4096)
        
        # 4. 獲取 teacher codes: (B*T,)
        teacher_truncated = teacher_codes_2d[:, :T]
        teacher_flat = teacher_truncated.reshape(-1).long()
        
        # 5. Cross-Entropy Loss
        ce_loss = F.cross_entropy(
            logits, 
            teacher_flat, 
            label_smoothing=self.label_smoothing
        )
        
        # Metrics
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == teacher_flat).float().mean()
            
            # Top-5 Accuracy
            top5_preds = logits.topk(5, dim=-1).indices
            top5_correct = (top5_preds == teacher_flat.unsqueeze(-1)).any(dim=-1)
            top5_accuracy = top5_correct.float().mean()
            
            # Confidence (softmax probability of correct class)
            probs = F.softmax(logits, dim=-1)
            confidence = probs[torch.arange(len(teacher_flat)), teacher_flat].mean()
        
        metrics = {
            'ce_loss': ce_loss.item(),
            'code_match_rate': accuracy.item(),
            'top5_accuracy': top5_accuracy.item(),
            'confidence': confidence.item(),
        }
        
        return ce_loss, metrics


class MarginLoss(nn.Module):
    """
    Margin Loss - 確保正確 token 與錯誤 token 有足夠距離
    
    原理：
    不只要接近正確 token，還要遠離最近的錯誤 token
    loss = max(0, d(x, positive) - d(x, negative) + margin)
    
    優點：
    - 明確的決策邊界優化
    - 對 hard negatives 敏感
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: 正確和錯誤 token 的最小距離差
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, student_features, teacher_codes, codebook, distance_matrix=None):
        """
        Args:
            student_features: (B, 512, T) - Student encoder 輸出
            teacher_codes: (n_q, B, T) - Teacher 的離散 codes
            codebook: (4096, 512) - VQ codebook
            distance_matrix: 不使用
            
        Returns:
            margin_loss: scalar
            metrics: dict
        """
        B, C, T_feat = student_features.shape
        
        # 處理 teacher_codes 格式
        if teacher_codes.dim() == 3:
            teacher_codes_2d = teacher_codes[0]
        else:
            teacher_codes_2d = teacher_codes.squeeze(1)
        
        T_code = teacher_codes_2d.shape[1]
        T = min(T_feat, T_code)
        
        # Reshape features
        features_truncated = student_features[:, :, :T]
        features_flat = features_truncated.permute(0, 2, 1).reshape(-1, C)
        
        # 計算到所有 codes 的距離
        distances = torch.cdist(
            features_flat.unsqueeze(0),
            codebook.unsqueeze(0)
        ).squeeze(0)  # (B*T, 4096)
        
        # 獲取 teacher codes
        teacher_truncated = teacher_codes_2d[:, :T]
        teacher_flat = teacher_truncated.reshape(-1).long()
        
        batch_size = features_flat.shape[0]
        
        # 正確 token 的距離
        correct_dist = distances[torch.arange(batch_size), teacher_flat]  # (B*T,)
        
        # 將正確 token 的距離設為無窮大，找最近的錯誤 token
        distances_masked = distances.clone()
        distances_masked[torch.arange(batch_size), teacher_flat] = float('inf')
        wrong_dist = distances_masked.min(dim=-1)[0]  # (B*T,)
        
        # Margin loss: correct_dist + margin < wrong_dist
        margin_loss = F.relu(correct_dist - wrong_dist + self.margin).mean()
        
        # Metrics
        with torch.no_grad():
            predictions = distances.argmin(dim=-1)
            accuracy = (predictions == teacher_flat).float().mean()
            margin_satisfied = (wrong_dist - correct_dist > self.margin).float().mean()
        
        metrics = {
            'margin_loss': margin_loss.item(),
            'code_match_rate': accuracy.item(),
            'margin_satisfied_rate': margin_satisfied.item(),
            'avg_correct_dist': correct_dist.mean().item(),
            'avg_wrong_dist': wrong_dist.mean().item(),
        }
        
        return margin_loss, metrics


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
            teacher_codes: (n_q, B, T) - Teacher 的離散 codes (WavTokenizer 格式)
            codebook: (4096, 512) - VQ codebook
            distance_matrix: (4096, 4096) - 預計算的 code 間距離

        Returns:
            soft_distance_loss: scalar
            metrics: dict with debugging info
        """
        B, C, T_feat = student_features.shape
        device = student_features.device

        # teacher_codes 格式是 (n_q, B, T)，取第一個 quantizer
        # 轉換為 (B, T)
        if teacher_codes.dim() == 3:
            teacher_codes_2d = teacher_codes[0]  # (B, T)
        else:
            teacher_codes_2d = teacher_codes.squeeze(1)  # (B, T)

        T_code = teacher_codes_2d.shape[1]

        # 確保時間維度對齊 (features 和 codes 可能有不同的時間長度)
        T = min(T_feat, T_code)

        # 1. Reshape features: (B, 512, T) -> (B*T, 512)
        features_truncated = student_features[:, :, :T]
        features_flat = features_truncated.permute(0, 2, 1).reshape(-1, C)

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
        teacher_truncated = teacher_codes_2d[:, :T]  # (B, T)
        teacher_flat = teacher_truncated.reshape(-1).long()  # (B*T,)

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


class STEDistanceLoss(nn.Module):
    """
    使用 Straight-Through Estimator (STE) 的 Distance Loss

    原理：
    1. 計算 student features 到所有 codes 的距離
    2. 用 argmax 選擇最近的 code (forward)
    3. 梯度直接傳回 soft distances (backward)

    STE 公式：
    hard_one_hot = softmax(-dist/τ).argmax().one_hot()
    soft_probs = softmax(-dist/τ)
    codes = hard_one_hot - soft_probs.detach() + soft_probs

    這樣 forward 是 hard one-hot，backward 是 soft gradient
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_features, teacher_codes, codebook, distance_matrix):
        """
        Args:
            student_features: (B, 512, T) - Student encoder 輸出
            teacher_codes: (n_q, B, T) - Teacher 的離散 codes
            codebook: (4096, 512) - VQ codebook
            distance_matrix: (4096, 4096) - 預計算的 code 間距離

        Returns:
            ste_distance_loss: scalar
            metrics: dict with debugging info
        """
        B, C, T_feat = student_features.shape

        # 處理 teacher_codes 格式 (n_q, B, T) → (B, T)
        if teacher_codes.dim() == 3:
            teacher_codes_2d = teacher_codes[0]
        else:
            teacher_codes_2d = teacher_codes.squeeze(1)

        T_code = teacher_codes_2d.shape[1]
        T = min(T_feat, T_code)

        # 1. Reshape features: (B, 512, T) -> (B*T, 512)
        features_truncated = student_features[:, :, :T]
        features_flat = features_truncated.permute(0, 2, 1).reshape(-1, C)

        # 2. 計算到所有 codes 的距離: (B*T, 4096)
        distances = torch.cdist(
            features_flat.unsqueeze(0),
            codebook.unsqueeze(0)
        ).squeeze(0)

        # 3. Softmax 機率 (用於梯度)
        soft_probs = F.softmax(-distances / self.temperature, dim=-1)  # (B*T, 4096)

        # 4. Hard one-hot (用於 forward)
        hard_indices = distances.argmin(dim=-1)  # (B*T,)
        hard_one_hot = torch.zeros_like(soft_probs)
        hard_one_hot.scatter_(-1, hard_indices.unsqueeze(-1), 1.0)

        # 5. STE: forward=hard, backward=soft
        ste_codes = hard_one_hot - soft_probs.detach() + soft_probs  # (B*T, 4096)

        # 6. 獲取 teacher codes: (B*T,)
        teacher_truncated = teacher_codes_2d[:, :T]
        teacher_flat = teacher_truncated.reshape(-1).long()

        # 7. 計算距離
        teacher_distances = distance_matrix[teacher_flat]  # (B*T, 4096)
        ste_distance = (ste_codes * teacher_distances).sum(dim=-1)  # (B*T,)
        ste_distance_loss = ste_distance.mean()

        # Metrics
        code_match_rate = (hard_indices == teacher_flat).float().mean()
        entropy = -(soft_probs * torch.log(soft_probs + 1e-10)).sum(dim=-1).mean()

        metrics = {
            'ste_distance_loss': ste_distance_loss.item(),
            'code_match_rate': code_match_rate.item(),
            'ste_code_entropy': entropy.item(),
            'ste_distance_mean': ste_distance.mean().item(),
            'ste_distance_std': ste_distance.std().item(),
        }

        return ste_distance_loss, metrics


class GumbelDistanceLoss(nn.Module):
    """
    使用 Gumbel-Softmax 的可微 Distance Loss

    原理：
    1. 計算 student features 到所有 codes 的距離 → logits
    2. 用 Gumbel-Softmax 採樣 (forward: hard, backward: soft)
    3. 計算 hard codes 與 teacher codes 之間的距離

    相比 STEDistanceLoss:
    - STE: forward 用 argmax (確定性)，backward 用 softmax
    - Gumbel: forward 用 Gumbel-argmax (隨機性)，backward 用 Gumbel-softmax

    Gumbel-Softmax 的優勢：
    - 引入隨機性，幫助探索 codebook 空間
    - 更接近真實的離散採樣過程
    """

    def __init__(self, temperature=1.0, hard=True):
        """
        Args:
            temperature: Gumbel-Softmax temperature (τ)
            hard: 如果 True，forward 用 hard codes，backward 用 soft gradients
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, student_features, teacher_codes, codebook, distance_matrix):
        """
        Args:
            student_features: (B, 512, T) - Student encoder 輸出
            teacher_codes: (n_q, B, T) - Teacher 的離散 codes
            codebook: (4096, 512) - VQ codebook
            distance_matrix: (4096, 4096) - 預計算的 code 間距離

        Returns:
            gumbel_distance_loss: scalar
            metrics: dict with debugging info
        """
        B, C, T_feat = student_features.shape

        # 處理 teacher_codes 格式 (n_q, B, T) → (B, T)
        if teacher_codes.dim() == 3:
            teacher_codes_2d = teacher_codes[0]
        else:
            teacher_codes_2d = teacher_codes.squeeze(1)

        T_code = teacher_codes_2d.shape[1]
        T = min(T_feat, T_code)

        # 1. Reshape features: (B, 512, T) -> (B*T, 512)
        features_truncated = student_features[:, :, :T]
        features_flat = features_truncated.permute(0, 2, 1).reshape(-1, C)

        # 2. 計算到所有 codes 的距離: (B*T, 4096)
        distances = torch.cdist(
            features_flat.unsqueeze(0),
            codebook.unsqueeze(0)
        ).squeeze(0)

        # 3. 用 Gumbel-Softmax 採樣 (負距離作為 logits)
        logits = -distances
        gumbel_codes = gumbel_softmax_sample(logits, self.temperature, self.hard)  # (B*T, 4096)

        # 4. 獲取 teacher codes: (B*T,)
        teacher_truncated = teacher_codes_2d[:, :T]
        teacher_flat = teacher_truncated.reshape(-1).long()

        # 5. 計算距離: gumbel_codes @ distance_matrix[teacher_codes]
        # gumbel_codes: (B*T, 4096) - one-hot (或 soft) 向量
        # distance_matrix[teacher_flat]: (B*T, 4096)
        teacher_distances = distance_matrix[teacher_flat]  # (B*T, 4096)

        # 6. 計算加權距離 (如果 hard=True，這等於 hard code 的距離)
        gumbel_distance = (gumbel_codes * teacher_distances).sum(dim=-1)  # (B*T,)
        gumbel_distance_loss = gumbel_distance.mean()

        # Metrics
        hard_codes = distances.argmin(dim=-1)
        code_match_rate = (hard_codes == teacher_flat).float().mean()

        # Entropy (如果用 soft 模式才有意義)
        if not self.hard:
            entropy = -(gumbel_codes * torch.log(gumbel_codes + 1e-10)).sum(dim=-1).mean()
        else:
            soft_probs = F.softmax(-distances / self.temperature, dim=-1)
            entropy = -(soft_probs * torch.log(soft_probs + 1e-10)).sum(dim=-1).mean()

        metrics = {
            'gumbel_distance_loss': gumbel_distance_loss.item(),
            'code_match_rate': code_match_rate.item(),
            'gumbel_code_entropy': entropy.item(),
            'gumbel_distance_mean': gumbel_distance.mean().item(),
            'gumbel_distance_std': gumbel_distance.std().item(),
        }

        return gumbel_distance_loss, metrics


class EncoderDistillationLoss(nn.Module):
    """
    Complete Loss Function for LoRA Encoder Denoising

    組成:
    - Feature MSE Loss: ||student_features - teacher_features||²
    - Token Loss: 可微的 code alignment loss (多種模式)
    - VQ Loss: commitment loss (通常設為 0，凍結 codebook)

    支持多種 Token Loss 模式:
    - 'soft': SoftDistanceLoss (期望距離)
    - 'gumbel': GumbelDistanceLoss (ST + Gumbel-Softmax)
    - 'ste': STEDistanceLoss (Straight-Through Estimator)
    - 'ce': TokenClassificationLoss (Cross-Entropy 直接監督 token)
    - 'margin': MarginLoss (Margin-based 決策邊界優化)
    """

    def __init__(
        self,
        feature_loss_weight=1.0,
        soft_dist_loss_weight=0.1,
        vq_loss_weight=0.0,
        temperature=1.0,
        distance_loss_mode='soft',  # 'soft', 'gumbel', 'ste', 'ce', 'margin'
        gumbel_hard=True,           # 只在 gumbel 模式有效
        margin=1.0,                 # 只在 margin 模式有效
        label_smoothing=0.0,        # 只在 ce 模式有效
    ):
        super().__init__()
        self.feature_loss_weight = feature_loss_weight
        self.soft_dist_loss_weight = soft_dist_loss_weight
        self.vq_loss_weight = vq_loss_weight
        self.distance_loss_mode = distance_loss_mode

        # 選擇 Distance Loss 模式
        if distance_loss_mode == 'gumbel':
            self.distance_loss_fn = GumbelDistanceLoss(
                temperature=temperature,
                hard=gumbel_hard
            )
        elif distance_loss_mode == 'ste':
            self.distance_loss_fn = STEDistanceLoss(temperature=temperature)
        elif distance_loss_mode == 'ce':
            self.distance_loss_fn = TokenClassificationLoss(
                temperature=temperature,
                label_smoothing=label_smoothing
            )
        elif distance_loss_mode == 'margin':
            self.distance_loss_fn = MarginLoss(margin=margin)
        else:  # 'soft' (default)
            self.distance_loss_fn = SoftDistanceLoss(temperature=temperature)

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

        # 2. Distance Loss (可微!) - soft 或 gumbel 模式
        dist_loss, dist_metrics = self.distance_loss_fn(
            student_features, teacher_codes, codebook, distance_matrix
        )

        # 3. Hard Distance Loss (for monitoring only, 不參與梯度)
        with torch.no_grad():
            student_codes = model_output['student_codes']  # (n_q, B, T)
            # 處理 WavTokenizer 格式 (n_q, B, T)
            if student_codes.dim() == 3:
                student_codes_2d = student_codes[0]  # (B, T)
                teacher_codes_2d = teacher_codes[0]  # (B, T)
            else:
                student_codes_2d = student_codes.squeeze(1)
                teacher_codes_2d = teacher_codes.squeeze(1)

            B, T = student_codes_2d.shape
            student_flat = student_codes_2d.reshape(-1).long()
            teacher_flat = teacher_codes_2d.reshape(-1).long()
            hard_distances = distance_matrix[student_flat, teacher_flat]
            hard_distance_loss = hard_distances.mean()

        # Total loss
        total_loss = (
            self.feature_loss_weight * feature_loss +
            self.soft_dist_loss_weight * dist_loss +
            self.vq_loss_weight * vq_loss
        )

        # 從 dist_metrics 中提取 code_match_rate
        code_match_rate = dist_metrics.get('code_match_rate', 0.0)

        # 提取 entropy (soft 或 gumbel 模式的 key 不同)
        entropy = dist_metrics.get('soft_code_entropy',
                                   dist_metrics.get('gumbel_code_entropy', 0.0))

        metrics = {
            'total_loss': total_loss.item(),
            'feature_loss': feature_loss.item(),
            'distance_loss': dist_loss.item(),  # 通用名稱
            'hard_distance_loss': hard_distance_loss.item(),  # monitoring
            'vq_loss': vq_loss.item(),
            'code_match_rate': code_match_rate,
            'code_entropy': entropy,
            'distance_loss_mode': self.distance_loss_mode,
        }

        return total_loss, metrics


def test_gradient_flow():
    """測試 Soft 和 Gumbel Distance Loss 的梯度流"""
    print("=" * 60)
    print("Testing Distance Loss Gradient Flow")
    print("=" * 60)

    B, T, C = 2, 10, 512
    codebook_size = 4096

    # 創建測試數據
    codebook = torch.randn(codebook_size, C)
    distance_matrix = torch.cdist(codebook, codebook, p=2)

    # 測試 WavTokenizer 格式的 teacher_codes: (n_q, B, T)
    teacher_codes = torch.randint(0, codebook_size, (1, B, T))

    # ===== 測試 1: SoftDistanceLoss =====
    print("\n--- Testing SoftDistanceLoss ---")
    student_features = torch.randn(B, C, T, requires_grad=True)
    loss_fn = SoftDistanceLoss(temperature=1.0)

    loss, metrics = loss_fn(student_features, teacher_codes, codebook, distance_matrix)

    print(f"Soft Distance Loss: {loss.item():.4f}")
    print(f"  requires_grad: {loss.requires_grad}")
    print(f"  grad_fn: {loss.grad_fn}")

    try:
        loss.backward()
        if student_features.grad is not None:
            print(f"  ✅ SUCCESS! grad norm: {student_features.grad.norm().item():.6f}")
        else:
            print(f"  ❌ FAILED: grad is None")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    # ===== 測試 2: STEDistanceLoss =====
    print("\n--- Testing STEDistanceLoss ---")
    student_features = torch.randn(B, C, T, requires_grad=True)
    loss_fn = STEDistanceLoss(temperature=1.0)

    loss, metrics = loss_fn(student_features, teacher_codes, codebook, distance_matrix)

    print(f"STE Distance Loss: {loss.item():.4f}")
    print(f"  requires_grad: {loss.requires_grad}")
    print(f"  grad_fn: {loss.grad_fn}")

    try:
        loss.backward()
        if student_features.grad is not None:
            print(f"  ✅ SUCCESS! grad norm: {student_features.grad.norm().item():.6f}")
        else:
            print(f"  ❌ FAILED: grad is None")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    # ===== 測試 3: GumbelDistanceLoss =====
    print("\n--- Testing GumbelDistanceLoss ---")
    student_features = torch.randn(B, C, T, requires_grad=True)
    loss_fn = GumbelDistanceLoss(temperature=1.0, hard=True)

    loss, metrics = loss_fn(student_features, teacher_codes, codebook, distance_matrix)

    print(f"Gumbel Distance Loss: {loss.item():.4f}")
    print(f"  requires_grad: {loss.requires_grad}")
    print(f"  grad_fn: {loss.grad_fn}")

    try:
        loss.backward()
        if student_features.grad is not None:
            print(f"  ✅ SUCCESS! grad norm: {student_features.grad.norm().item():.6f}")
        else:
            print(f"  ❌ FAILED: grad is None")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    # ===== 測試 3: EncoderDistillationLoss (soft mode) =====
    print("\n--- Testing EncoderDistillationLoss (soft mode) ---")
    student_features = torch.randn(B, C, T, requires_grad=True)
    criterion = EncoderDistillationLoss(
        feature_loss_weight=1.0,
        soft_dist_loss_weight=0.1,
        distance_loss_mode='soft',
    )

    mock_output = {
        'student_features': student_features,
        'teacher_features': torch.randn(B, C, T),
        'teacher_codes': teacher_codes,
        'student_codes': torch.randint(0, codebook_size, (1, B, T)),
        'vq_loss': torch.tensor(0.0),
    }

    loss, loss_dict = criterion(mock_output, distance_matrix, codebook)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"  Feature Loss: {loss_dict['feature_loss']:.4f}")
    print(f"  Distance Loss: {loss_dict['distance_loss']:.4f}")

    try:
        loss.backward()
        if student_features.grad is not None:
            print(f"  ✅ SUCCESS! grad norm: {student_features.grad.norm().item():.6f}")
        else:
            print(f"  ❌ FAILED: grad is None")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    # ===== 測試 4: EncoderDistillationLoss (gumbel mode) =====
    print("\n--- Testing EncoderDistillationLoss (gumbel mode) ---")
    student_features = torch.randn(B, C, T, requires_grad=True)
    criterion = EncoderDistillationLoss(
        feature_loss_weight=1.0,
        soft_dist_loss_weight=0.1,
        distance_loss_mode='gumbel',
    )

    mock_output = {
        'student_features': student_features,
        'teacher_features': torch.randn(B, C, T),
        'teacher_codes': teacher_codes,
        'student_codes': torch.randint(0, codebook_size, (1, B, T)),
        'vq_loss': torch.tensor(0.0),
    }

    loss, loss_dict = criterion(mock_output, distance_matrix, codebook)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"  Feature Loss: {loss_dict['feature_loss']:.4f}")
    print(f"  Distance Loss: {loss_dict['distance_loss']:.4f}")

    try:
        loss.backward()
        if student_features.grad is not None:
            print(f"  ✅ SUCCESS! grad norm: {student_features.grad.norm().item():.6f}")
        else:
            print(f"  ❌ FAILED: grad is None")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_gradient_flow()
