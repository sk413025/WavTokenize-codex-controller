"""
CEWithSpeakerLoss: 結合 CrossEntropy Loss 和 Speaker Embedding L2 Loss

目標:
    1. 主任務: Token-level 去噪翻譯 (CE Loss)
    2. 輔助約束: 保持說話人身份 (Speaker L2 Loss)

設計理念:
    - 主任務負責學習 noisy→clean 的 token 映射
    - 輔助約束強制模型保持原始說話人特徵，不改變身份
    - 這種解耦有望提升對未見語者的泛化能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# 添加路徑以導入 speaker encoder
sys.path.insert(0, str(Path(__file__).parent.parent / 'exp'))


class CEWithSpeakerLoss(nn.Module):
    """
    結合 CrossEntropy Loss 和 Speaker Embedding L2 Loss

    Args:
        speaker_encoder: 預訓練的 speaker encoder (凍結)
        wavtokenizer: WavTokenizer 用於解碼 tokens
        lambda_speaker: Speaker loss 的權重 (default: 0.5)
        speaker_loss_start_epoch: 從第幾個 epoch 開始加入 speaker loss (default: 0)
        compute_speaker_every_n_steps: 每 N 步計算一次 speaker loss (default: 1)
    """

    def __init__(
        self,
        speaker_encoder,
        wavtokenizer,
        lambda_speaker=0.5,
        speaker_loss_start_epoch=0,
        compute_speaker_every_n_steps=1,
        device='cuda'
    ):
        super().__init__()

        self.device = device
        self.lambda_speaker = lambda_speaker
        self.speaker_loss_start_epoch = speaker_loss_start_epoch
        self.compute_speaker_every_n_steps = compute_speaker_every_n_steps

        # CrossEntropy Loss
        self.ce_loss = nn.CrossEntropyLoss()

        # Speaker Encoder (凍結)
        self.speaker_encoder = speaker_encoder
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False
        self.speaker_encoder.eval()

        # WavTokenizer (用於解碼)
        self.wavtokenizer = wavtokenizer
        for param in self.wavtokenizer.parameters():
            param.requires_grad = False
        self.wavtokenizer.eval()

        # 計數器
        self.step_counter = 0

    def _decode_soft_features_to_audio(self, soft_features):
        """
        將 soft features 解碼為音頻（可微分）

        Args:
            soft_features: (B, C, T) soft continuous features

        Returns:
            audio: (B, audio_len) waveform at 24kHz with gradient
        """
        # 直接解碼 features（不經過 discrete token lookup）
        bandwidth_id = torch.tensor([0], device=soft_features.device)
        audio = self.wavtokenizer.decode(soft_features, bandwidth_id=bandwidth_id)

        # 處理維度：確保是 (B, audio_len)
        while audio.dim() > 2:
            audio = audio.squeeze(1)

        # 確保在正確的設備上並返回
        return audio.to(self.device).contiguous()

    def _decode_tokens_to_audio(self, tokens, allow_grad=False):
        """
        將 discrete tokens 解碼為音頻（用於 noisy audio）

        Args:
            tokens: (B, T) token IDs
            allow_grad: 是否允許梯度流

        Returns:
            audio: (B, audio_len) waveform at 24kHz
        """
        B, T = tokens.shape

        # 不需要梯度：用於 noisy_audio
        with torch.no_grad():
            # ⭐ 關鍵：codes_to_features 需要逐個樣本處理，格式為 [1, T]
            # 參考 baseline train.py:841-847 的處理方式
            features_list = []
            for i in range(B):
                single_token = tokens[i:i+1, :]  # (1, T)
                # codes_to_features 期望 (1, T) 格式
                single_features = self.wavtokenizer.codes_to_features(single_token)  # (1, C, 1, T) or similar

                # 處理維度（參考 baseline）
                if single_features.dim() == 4:
                    single_features = single_features.squeeze(2)  # (1, C, T)
                if single_features.dim() == 2:
                    single_features = single_features.unsqueeze(0)  # ensure (1, C, T)

                features_list.append(single_features)

            # 合併 batch
            features = torch.cat(features_list, dim=0)  # (B, C, T)

            # 確保 features 在正確的設備上
            features = features.to(self.device)

            # Step 3: features → audio
            bandwidth_id = torch.tensor([0], device=features.device)
            audio = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

            # Step 4: squeeze to (B, audio_len) - 處理所有多餘維度
            while audio.dim() > 2:
                audio = audio.squeeze(1)

        # 確保在正確的設備上並返回
        return audio.to(self.device).contiguous()

    def _compute_speaker_loss_from_features(self, pred_soft_features, noisy_tokens):
        """
        計算 Speaker Embedding L2 Loss（使用 soft features，保持梯度流）

        ⚠️ 梯度流關鍵點：
        1. pred_soft_features 是 continuous，保持從 pred_logits 的梯度
        2. noisy_audio 不需要梯度（作為 target）
        3. speaker encoder 輸出的梯度會回傳到 pred_soft_features

        Args:
            pred_soft_features: (B, C, T) soft continuous features with gradient
            noisy_tokens: (B, T) 輸入的 noisy tokens

        Returns:
            loss_speaker: scalar tensor (保持梯度)
        """
        try:
            # 1. 解碼 soft features → pred_audio (保持梯度)
            pred_audio = self._decode_soft_features_to_audio(pred_soft_features)  # (B, audio_len) with grad

            # 2. 解碼 noisy tokens → noisy_audio (不需要梯度)
            noisy_audio = self._decode_tokens_to_audio(noisy_tokens, allow_grad=False)  # (B, audio_len) no grad

            # ⚠️ 確保audio在正確的設備上（CRITICAL FIX）
            pred_audio = pred_audio.to(self.device)
            noisy_audio = noisy_audio.to(self.device)

            # 3. 提取 speaker embeddings
            # ⚠️ 關鍵：speaker encoder 雖然凍結，但pred_emb 保持梯度
            pred_emb = self.speaker_encoder(pred_audio)  # (B, embed_dim) with grad

            with torch.no_grad():
                # noisy_emb 不需要梯度（作為 target）
                noisy_emb = self.speaker_encoder(noisy_audio)  # (B, embed_dim) no grad

            # 4. 計算 L2 loss
            # ⚠️ 梯度路徑：loss_speaker → pred_emb → pred_audio → pred_soft_features → pred_logits
            loss_speaker = F.mse_loss(pred_emb, noisy_emb)

            return loss_speaker

        except Exception as e:
            # 如果計算失敗，返回 0 (不影響主任務訓練)
            print(f"⚠️  Speaker loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def forward(self, pred_logits, target_tokens, noisy_tokens, current_epoch=0):
        """
        計算總損失

        Args:
            pred_logits: (B, T, 4096) 模型預測的 logits
            target_tokens: (B, T) Ground truth clean tokens
            noisy_tokens: (B, T) 輸入的 noisy tokens
            current_epoch: 當前 epoch (用於控制何時啟動 speaker loss)

        Returns:
            loss_total: 總損失
            loss_ce: CE Loss (用於記錄)
            loss_speaker: Speaker Loss (用於記錄)
        """
        B, T, vocab_size = pred_logits.shape

        # 1. 主任務: CrossEntropy Loss
        logits_flat = pred_logits.view(-1, vocab_size)  # (B*T, 4096)
        target_flat = target_tokens.view(-1).long()  # (B*T,)
        loss_ce = self.ce_loss(logits_flat, target_flat)

        # 2. 輔助約束: Speaker Loss
        loss_speaker = torch.tensor(0.0, device=self.device)

        # 判斷是否計算 speaker loss
        should_compute_speaker = (
            current_epoch >= self.speaker_loss_start_epoch and
            self.step_counter % self.compute_speaker_every_n_steps == 0 and
            self.lambda_speaker > 0
        )

        if should_compute_speaker:
            # ⚠️ 關鍵：使用 soft probabilities 繞過 discrete token lookup
            # 梯度路徑：pred_logits → soft_probs → soft_features → audio → speaker_emb → loss

            # Step 1: 從 logits 獲取 soft probabilities
            pred_probs = F.softmax(pred_logits, dim=-1)  # (B, T, 4096) with grad

            # Step 2: 使用 soft probabilities 計算 soft features
            # 方法：soft_features = pred_probs @ codebook
            # 其中 codebook 是 WavTokenizer 的 frozen codebook (4096, 512)
            codebook = self.wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook  # (4096, 512)

            # pred_probs: (B, T, 4096)
            # codebook: (4096, 512)
            # soft_features: (B, T, 512)
            soft_features = torch.matmul(pred_probs, codebook)  # (B, T, 512) with grad!

            # Step 3: 轉換為 WavTokenizer decoder 需要的格式
            # WavTokenizer decoder 期望 (B, C, T) 格式
            # 目前 soft_features 是 (B, T, 512)，需要轉置
            soft_features = soft_features.transpose(1, 2)  # (B, 512, T)

            # Step 4: 計算 speaker loss（保持梯度流）
            loss_speaker = self._compute_speaker_loss_from_features(soft_features, noisy_tokens)
            # ✅ 梯度路徑完整：loss_speaker → soft_features → pred_probs → pred_logits

        # 更新計數器
        self.step_counter += 1

        # 3. 總損失
        loss_total = loss_ce + self.lambda_speaker * loss_speaker

        return loss_total, loss_ce, loss_speaker


def create_loss_with_speaker(
    wavtokenizer,
    speaker_model_type='ecapa',
    lambda_speaker=0.5,
    speaker_loss_start_epoch=0,
    compute_speaker_every_n_steps=1,
    device='cuda'
):
    """
    工廠函數: 創建 CEWithSpeakerLoss

    Args:
        wavtokenizer: WavTokenizer 模型
        speaker_model_type: 'ecapa' 或 'resemblyzer'
        lambda_speaker: Speaker loss 權重
        speaker_loss_start_epoch: 從第幾個 epoch 開始
        compute_speaker_every_n_steps: 每 N 步計算一次
        device: 設備

    Returns:
        loss_fn: CEWithSpeakerLoss 實例
    """
    # 導入 speaker encoder
    from speaker_encoder import create_speaker_encoder

    print(f"\n{'='*60}")
    print("初始化 CEWithSpeakerLoss")
    print(f"{'='*60}")
    print(f"Speaker model type: {speaker_model_type}")
    print(f"Lambda (speaker loss weight): {lambda_speaker}")
    print(f"Speaker loss start epoch: {speaker_loss_start_epoch}")
    print(f"Compute speaker every N steps: {compute_speaker_every_n_steps}")

    # 創建 speaker encoder (凍結)
    speaker_encoder = create_speaker_encoder(
        model_type=speaker_model_type,
        freeze=True,
        output_dim=256
    ).to(device)

    # 創建損失函數
    loss_fn = CEWithSpeakerLoss(
        speaker_encoder=speaker_encoder,
        wavtokenizer=wavtokenizer,
        lambda_speaker=lambda_speaker,
        speaker_loss_start_epoch=speaker_loss_start_epoch,
        compute_speaker_every_n_steps=compute_speaker_every_n_steps,
        device=device
    )

    print(f"✅ CEWithSpeakerLoss initialized successfully\n")

    return loss_fn


# ============================================================================
#                            測試代碼
# ============================================================================

if __name__ == '__main__':
    print("測試 CEWithSpeakerLoss...")

    # 模擬輸入
    B, T, vocab_size = 4, 100, 4096
    pred_logits = torch.randn(B, T, vocab_size)
    target_tokens = torch.randint(0, vocab_size, (B, T))
    noisy_tokens = torch.randint(0, vocab_size, (B, T))

    print(f"pred_logits shape: {pred_logits.shape}")
    print(f"target_tokens shape: {target_tokens.shape}")
    print(f"noisy_tokens shape: {noisy_tokens.shape}")

    # 注意: 這裡只測試損失函數的基本結構
    # 實際使用需要真實的 wavtokenizer 和 speaker_encoder

    print("\n⚠️  完整測試需要在訓練腳本中進行（需要真實的 WavTokenizer 和 Speaker Encoder）")
