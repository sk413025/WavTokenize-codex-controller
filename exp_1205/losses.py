"""
exp_1205: 新的 Loss 函數設計

基於 exp_1204 診斷結果，設計三種新的 Loss：
- 方案 A: Linear + CE (直接 Token 預測)
- 方案 B: Margin-based Loss
- 方案 C: Hard Negative Mining + CE

診斷發現的問題：
- MSE Loss 只優化「接近」，不保證「最近」
- 到正確 token 距離 = 3.75，到最近 token 距離 = 0.45
- Token Accuracy 只有 2.21%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPredictionHead(nn.Module):
    """
    方案 A: 直接 Token 預測

    不經過 codebook 距離計算，直接用 Linear projection 預測 token
    """
    def __init__(self, embed_dim=512, vocab_size=4096):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, student_emb):
        """
        Args:
            student_emb: (B, C, T) - Student encoder 輸出

        Returns:
            logits: (B, T, vocab_size)
        """
        # (B, C, T) -> (B, T, C) -> (B, T, vocab_size)
        logits = self.proj(student_emb.transpose(1, 2))
        return logits


class LinearCELoss(nn.Module):
    """
    方案 A: Linear + Cross-Entropy Loss

    直接優化 Token Accuracy，繞過距離計算
    """
    def __init__(self, embed_dim=512, vocab_size=4096, label_smoothing=0.0):
        super().__init__()
        self.head = TokenPredictionHead(embed_dim, vocab_size)
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size

    def forward(self, student_emb, teacher_codes, codebook=None):
        """
        Args:
            student_emb: (B, C, T) - Student encoder 輸出
            teacher_codes: (n_q, B, T) 或 (B, T) - Teacher 的 token codes
            codebook: 不使用，保留參數以保持接口一致

        Returns:
            dict with loss and metrics
        """
        # 處理 teacher_codes 維度
        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]  # (B, T)

        B, C, T_emb = student_emb.shape
        T_code = teacher_codes.shape[1]
        T = min(T_emb, T_code)

        # Truncate to same length
        student_emb = student_emb[:, :, :T]
        teacher_codes = teacher_codes[:, :T]

        # Get logits
        logits = self.head(student_emb)  # (B, T, vocab_size)

        # Flatten for CE
        logits_flat = logits.reshape(-1, self.vocab_size)  # (B*T, vocab_size)
        targets_flat = teacher_codes.reshape(-1).long()  # (B*T,)

        # CE Loss with optional label smoothing
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            label_smoothing=self.label_smoothing
        )

        # Calculate accuracy
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=-1)
            accuracy = (predictions == targets_flat).float().mean()

            # Top-k accuracy
            top5_correct = (logits_flat.topk(5, dim=-1).indices == targets_flat.unsqueeze(-1)).any(dim=-1)
            top5_acc = top5_correct.float().mean()

            top10_correct = (logits_flat.topk(10, dim=-1).indices == targets_flat.unsqueeze(-1)).any(dim=-1)
            top10_acc = top10_correct.float().mean()

        return {
            'loss': loss,
            'ce_loss': loss,
            'token_accuracy': accuracy,
            'top5_accuracy': top5_acc,
            'top10_accuracy': top10_acc,
        }

    def get_trainable_parameters(self):
        """返回需要訓練的參數（Linear head）"""
        return self.head.parameters()


class MarginLoss(nn.Module):
    """
    方案 B: Margin-based Loss

    確保到正確 token 的距離比到最近錯誤 token 的距離小一個 margin
    """
    def __init__(self, margin=0.5, distance_type='l2'):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type

    def forward(self, student_emb, teacher_codes, codebook):
        """
        Args:
            student_emb: (B, C, T) - Student encoder 輸出
            teacher_codes: (n_q, B, T) 或 (B, T) - Teacher 的 token codes
            codebook: (vocab_size, C) - Codebook embeddings

        Returns:
            dict with loss and metrics
        """
        # 處理維度
        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]  # (B, T)

        B, C, T_emb = student_emb.shape
        T_code = teacher_codes.shape[1]
        T = min(T_emb, T_code)

        # Truncate and flatten
        student_emb = student_emb[:, :, :T]  # (B, C, T)
        teacher_codes = teacher_codes[:, :T]  # (B, T)

        # Reshape: (B, C, T) -> (B*T, C)
        emb_flat = student_emb.permute(0, 2, 1).reshape(-1, C)  # (N, C)
        codes_flat = teacher_codes.reshape(-1).long()  # (N,)
        N = emb_flat.shape[0]

        # 計算到所有 token 的距離
        if self.distance_type == 'l2':
            distances = torch.cdist(emb_flat, codebook)  # (N, vocab_size)
        else:
            # Cosine distance
            emb_norm = F.normalize(emb_flat, dim=-1)
            cb_norm = F.normalize(codebook, dim=-1)
            distances = 1 - torch.mm(emb_norm, cb_norm.t())  # (N, vocab_size)

        # 到正確 token 的距離
        correct_dist = distances.gather(1, codes_flat.unsqueeze(1))  # (N, 1)

        # Mask 掉正確 token，找最近的錯誤 token
        mask = torch.ones_like(distances)
        mask.scatter_(1, codes_flat.unsqueeze(1), 0)
        masked_distances = distances + (1 - mask) * 1e9
        wrong_dist, _ = masked_distances.min(dim=1, keepdim=True)  # (N, 1)

        # Margin loss: max(0, correct_dist - wrong_dist + margin)
        margin_loss = F.relu(correct_dist - wrong_dist + self.margin).mean()

        # Calculate accuracy
        with torch.no_grad():
            predictions = distances.argmin(dim=-1)
            accuracy = (predictions == codes_flat).float().mean()

            # Margin 滿足率
            margin_satisfied = (correct_dist < wrong_dist).float().mean()

        return {
            'loss': margin_loss,
            'margin_loss': margin_loss,
            'token_accuracy': accuracy,
            'margin_satisfied': margin_satisfied,
            'mean_correct_dist': correct_dist.mean(),
            'mean_wrong_dist': wrong_dist.mean(),
        }


class HardNegativeCELoss(nn.Module):
    """
    方案 C: Hard Negative Mining + Cross-Entropy

    只在最近的 K 個 token 上計算 CE，強制模型區分「容易混淆」的 token
    """
    def __init__(self, k=100, temperature=1.0, include_correct=True):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.include_correct = include_correct

    def forward(self, student_emb, teacher_codes, codebook):
        """
        Args:
            student_emb: (B, C, T) - Student encoder 輸出
            teacher_codes: (n_q, B, T) 或 (B, T) - Teacher 的 token codes
            codebook: (vocab_size, C) - Codebook embeddings

        Returns:
            dict with loss and metrics
        """
        # 處理維度
        if teacher_codes.dim() == 3:
            teacher_codes = teacher_codes[0]  # (B, T)

        B, C, T_emb = student_emb.shape
        T_code = teacher_codes.shape[1]
        T = min(T_emb, T_code)
        vocab_size = codebook.shape[0]

        # Truncate and flatten
        student_emb = student_emb[:, :, :T]  # (B, C, T)
        teacher_codes = teacher_codes[:, :T]  # (B, T)

        # Reshape: (B, C, T) -> (B*T, C)
        emb_flat = student_emb.permute(0, 2, 1).reshape(-1, C)  # (N, C)
        codes_flat = teacher_codes.reshape(-1).long()  # (N,)
        N = emb_flat.shape[0]

        # 計算到所有 token 的距離
        distances = torch.cdist(emb_flat, codebook)  # (N, vocab_size)

        # 找到最近的 K 個 token
        _, top_k_indices = distances.topk(self.k, dim=1, largest=False)  # (N, K)

        if self.include_correct:
            # 確保正確 token 在候選中
            # 檢查正確 token 是否已在 top_k 中
            correct_in_topk = (top_k_indices == codes_flat.unsqueeze(1)).any(dim=1)  # (N,)

            # 對於不在 top_k 中的，將最後一個替換為正確 token
            replacement_mask = ~correct_in_topk
            top_k_indices[replacement_mask, -1] = codes_flat[replacement_mask]

        # 取出候選 token 的距離
        candidate_distances = distances.gather(1, top_k_indices)  # (N, K)

        # 轉換為 logits (負距離)
        logits = -candidate_distances / self.temperature  # (N, K)

        # 找到正確 token 在候選中的位置
        local_labels = (top_k_indices == codes_flat.unsqueeze(1)).long().argmax(dim=1)  # (N,)

        # CE Loss
        loss = F.cross_entropy(logits, local_labels)

        # Calculate metrics
        with torch.no_grad():
            # Local accuracy (在 K 個候選中)
            local_predictions = logits.argmax(dim=-1)
            local_accuracy = (local_predictions == local_labels).float().mean()

            # Global accuracy
            global_predictions = distances.argmin(dim=-1)
            global_accuracy = (global_predictions == codes_flat).float().mean()

            # 正確 token 在候選中的比例
            correct_in_candidates = (top_k_indices == codes_flat.unsqueeze(1)).any(dim=1).float().mean()

        return {
            'loss': loss,
            'hard_neg_ce_loss': loss,
            'local_accuracy': local_accuracy,
            'token_accuracy': global_accuracy,
            'correct_in_candidates': correct_in_candidates,
        }


class CombinedLoss(nn.Module):
    """
    組合多種 Loss 的包裝器

    可以組合：
    - Linear CE
    - Margin Loss
    - Hard Negative CE
    - MSE Loss (作為正則化)
    """
    def __init__(
        self,
        use_linear_ce=True,
        use_margin=False,
        use_hard_neg=False,
        use_mse=False,
        linear_ce_weight=1.0,
        margin_weight=0.5,
        hard_neg_weight=0.5,
        mse_weight=0.1,
        embed_dim=512,
        vocab_size=4096,
        margin=0.5,
        hard_neg_k=100,
        temperature=1.0,
        label_smoothing=0.0,
    ):
        super().__init__()

        self.use_linear_ce = use_linear_ce
        self.use_margin = use_margin
        self.use_hard_neg = use_hard_neg
        self.use_mse = use_mse

        self.linear_ce_weight = linear_ce_weight
        self.margin_weight = margin_weight
        self.hard_neg_weight = hard_neg_weight
        self.mse_weight = mse_weight

        if use_linear_ce:
            self.linear_ce_loss = LinearCELoss(embed_dim, vocab_size, label_smoothing)

        if use_margin:
            self.margin_loss = MarginLoss(margin=margin)

        if use_hard_neg:
            self.hard_neg_loss = HardNegativeCELoss(k=hard_neg_k, temperature=temperature)

    def forward(self, student_emb, teacher_codes, codebook):
        """
        Args:
            student_emb: (B, C, T)
            teacher_codes: (n_q, B, T) 或 (B, T)
            codebook: (vocab_size, C)

        Returns:
            dict with total loss and individual losses
        """
        total_loss = 0.0
        results = {}

        if self.use_linear_ce:
            ce_results = self.linear_ce_loss(student_emb, teacher_codes, codebook)
            total_loss = total_loss + self.linear_ce_weight * ce_results['loss']
            results['linear_ce_loss'] = ce_results['loss']
            results['token_accuracy'] = ce_results['token_accuracy']
            results['top5_accuracy'] = ce_results.get('top5_accuracy', 0)

        if self.use_margin:
            margin_results = self.margin_loss(student_emb, teacher_codes, codebook)
            total_loss = total_loss + self.margin_weight * margin_results['loss']
            results['margin_loss'] = margin_results['loss']
            results['margin_satisfied'] = margin_results.get('margin_satisfied', 0)
            if 'token_accuracy' not in results:
                results['token_accuracy'] = margin_results['token_accuracy']

        if self.use_hard_neg:
            hard_neg_results = self.hard_neg_loss(student_emb, teacher_codes, codebook)
            total_loss = total_loss + self.hard_neg_weight * hard_neg_results['loss']
            results['hard_neg_ce_loss'] = hard_neg_results['loss']
            results['local_accuracy'] = hard_neg_results.get('local_accuracy', 0)
            if 'token_accuracy' not in results:
                results['token_accuracy'] = hard_neg_results['token_accuracy']

        if self.use_mse:
            # MSE to target embeddings
            if teacher_codes.dim() == 3:
                teacher_codes_2d = teacher_codes[0]
            else:
                teacher_codes_2d = teacher_codes

            B, C, T_emb = student_emb.shape
            T = min(T_emb, teacher_codes_2d.shape[1])

            student_emb_t = student_emb[:, :, :T].permute(0, 2, 1).reshape(-1, C)
            target_codes = teacher_codes_2d[:, :T].reshape(-1).long()
            target_emb = codebook[target_codes]

            mse_loss = F.mse_loss(student_emb_t, target_emb)
            total_loss = total_loss + self.mse_weight * mse_loss
            results['mse_loss'] = mse_loss

        results['loss'] = total_loss
        return results

    def get_trainable_parameters(self):
        """返回 Loss 函數中需要訓練的參數"""
        params = []
        if self.use_linear_ce:
            params.extend(self.linear_ce_loss.get_trainable_parameters())
        return params
