"""
exp_0305: 選擇性 LoRA 層實驗 (Selective-Layer LoRA)

基於 exp_0304 的 14-wav feature map 分析，針對以下問題：
  「到底哪些層的 LoRA 才真的有效？全層套用是否浪費容量？」

實驗設計：
─────────────────────────────────────────────────────────
  設 noise_sensitivity(L) = 1 - cos_sim(same_spk_clean, noisy)
  設 temporal_detail(L)   = temp_std_norm

  LoRA 目標選擇準則：
    train_layers  ← noise_sens > 0.10  OR  temporal > 0.15
    freeze_layers ← content_shared > 0.93  AND  noise_sens < 0.12

決策依據 (exp_0304 / conv18_14wav_role_metrics.csv):
─────────────────────────────────────────────────────────
  L#   模組                     noise   content  temporal  決策
  ────────────────────────────────────────────────────────
  L0   model[0]  stem           0.745   0.510   0.003     ■ LoRA (噪音最多)
  L1   model[1].block[1]        0.225   0.858   0.016     ■ LoRA (noise>0.10)
  L2   model[1].block[3]        0.245   0.862   0.024     ■ LoRA (noise>0.10)
  L3   model[1].shortcut        0.404   0.784   0.016     ■ LoRA (shortcut繞入噪音)
  L4   model[3]  Down1          0.106   0.949   0.095     □ Freeze (content高)
  L5   model[4].block[1]        0.063   0.974   0.255     □ Freeze (最乾淨)
  L6   model[4].block[3]        0.156   0.920   0.330     ■ LoRA (temporal高)
  L7   model[4].shortcut        0.086   0.961   0.072     □ Freeze (content高)
  L8   model[6]  Down2          0.197   0.906   0.524     ■ LoRA (temporal峰值區)
  L9   model[7].block[1]        0.139   0.929   0.687     ■ LoRA (★全網temporal最高)
  L10  model[7].block[3]        0.028   0.988   0.292     □ Freeze (幾乎無噪音)
  L11  model[7].shortcut        0.326   0.838   0.174     ■ LoRA (deep shortcut)
  L12  model[9]  Down3          0.164   0.919   0.185     ■ LoRA (plan_b新增)
  L13  model[10].block[1]       0.027   0.987   0.130     □ Freeze (最乾淨組)
  L14  model[10].block[3]       0.087   0.934   0.026     □ Freeze
  L15  model[10].shortcut       0.008   0.996   0.042     □ Freeze (★最純淨層)
  L16  model[12] Down4          0.201   0.782   0.001     □ Freeze (保留說話人)
  L17  model[15] output conv    0.147   0.815   0.001     □ Freeze (保留解碼接口)

三種 Plan：
─────────────────────────────────────────────────────────
  plan_a : adapt_top6 (exp_0304 summary.json 直接推薦)
           → L0, L2, L3, L8, L9, L11 (6 層), rank=32
           → ~245K parameters

  plan_b : adapt_top8 (加上兩個 L6/L12 橋接降採樣層)
           → L0, L2, L3, L6, L8, L9, L11, L12 (8 層), rank=32
           → ~307K parameters

  plan_c : all_18 等參數預算對照
           → 全 18 層, rank=10
           → ~288K parameters (與 plan_b 約等)
─────────────────────────────────────────────────────────

架構：
    exp_0224a No-VQ + Encoder LoRA + Decoder (WavTokenizer pretrained, frozen)
    → 差異：LoRA 只加在選定層(plan_a/b) 而非全層(plan_c)

Loss (與 exp_0224a 相同):
    λ_wav=1.0  * MSE(recon_wav, clean_wav)
  + λ_stft=1.0 * MR-STFT(recon_wav, clean_wav)
  + λ_mel=45.0 * Mel(recon_wav, clean_wav)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from families.deps.wavtokenizer_core.wavtok_lora_patch import apply_lora_patch

apply_lora_patch()

# ============================================================
# 層名稱定義（完整 PEFT 路徑）
# ============================================================
# 格式：feature_extractor.encodec.encoder.model.{X}.{sub}.conv.conv

_ENC = "feature_extractor.encodec.encoder.model"

# ── LoRA 目標層（依 exp_0304 指標選出）──────────────────────
L0_STEM = [f"{_ENC}.0.conv.conv"]                        # noise=0.745
L1_RB1C1 = [f"{_ENC}.1.block.1.conv.conv"]              # noise=0.225
L2_RB1C2 = [f"{_ENC}.1.block.3.conv.conv"]              # noise=0.245
L3_RB1SC = [f"{_ENC}.1.shortcut.conv.conv"]             # noise=0.404 ★shortcut
L6_RB2C2 = [f"{_ENC}.4.block.3.conv.conv"]              # temporal=0.330
L8_D2 = [f"{_ENC}.6.conv.conv"]                         # temporal=0.524 ★Down2
L9_RB3C1 = [f"{_ENC}.7.block.1.conv.conv"]              # temporal=0.687 ★★最高
L11_RB3SC = [f"{_ENC}.7.shortcut.conv.conv"]            # noise=0.326 ★shortcut
L12_D3 = [f"{_ENC}.9.conv.conv"]                        # noise=0.164, temporal=0.185

# ── Freeze 層（content>0.93, noise<0.12）────────────────────
# L4: model[3] Down1  content=0.949
# L5: model[4].block[1] content=0.974 ← L5 雖有 temporal 但 noise 極低，保留預訓練
# L7: model[4].shortcut content=0.961
# L10: model[7].block[3] content=0.988
# L13: model[10].block[1] content=0.987
# L14: model[10].block[3] content=0.934
# L15: model[10].shortcut content=0.996 ← ★最純淨，絕對凍結
# L16: model[12] Down4  speaker identity 開始出現
# L17: model[15] output conv

# ── ALL_18 參照 ──────────────────────────────────────────────
ALL_18_LAYERS = (
    L0_STEM + L1_RB1C1 + L2_RB1C2 + L3_RB1SC +
    [f"{_ENC}.3.conv.conv"] +                           # L4
    [f"{_ENC}.4.block.1.conv.conv"] +                   # L5
    L6_RB2C2 +
    [f"{_ENC}.4.shortcut.conv.conv"] +                  # L7
    L8_D2 + L9_RB3C1 +
    [f"{_ENC}.7.block.3.conv.conv"] +                   # L10
    L11_RB3SC + L12_D3 +
    [f"{_ENC}.10.block.1.conv.conv"] +                  # L13
    [f"{_ENC}.10.block.3.conv.conv"] +                  # L14
    [f"{_ENC}.10.shortcut.conv.conv"] +                 # L15
    [f"{_ENC}.12.conv.conv"] +                          # L16
    [f"{_ENC}.15.conv.conv"]                            # L17
)

# ============================================================
# Plan 定義
# ============================================================

PLANS = {
    # plan_a: exp_0304 adapt_top6 直接推薦
    'plan_a': {
        'layers': L0_STEM + L2_RB1C2 + L3_RB1SC + L8_D2 + L9_RB3C1 + L11_RB3SC,
        'rank': 32,
        'alpha': 64,
        'description': (
            'adapt_top6: L0(noise=0.745), L2(noise=0.245), L3(shortcut,noise=0.404), '
            'L8(Down2,temporal=0.524), L9(temporal=0.687★), L11(shortcut,noise=0.326)'
        ),
        'n_layers': 6,
    },
    # plan_b: 全 18 層 LoRA rank=32，對「穩定層」加 anchor 約束（取代凍結）
    # anchor 目標 = 原始 WavTokenizer：确保穩定層不偏離官方樓型→保護 frozen decoder 解碼品質
    # anchor 層 = [1,4,5,7,10,13,14,15,16,17]（exp_0304 content>0.93 AND noise<0.12）
    'plan_b': {
        'layers': ALL_18_LAYERS,   # 全 18 層都有 LoRA adapter
        'rank': 32,
        'alpha': 64,
        'anchor_layer_ids': [1, 4, 5, 7, 10, 13, 14, 15, 16, 17],
        'lambda_anchor': 1.0,
        'description': (
            'all_18_anchor: 全 18 層 LoRA rank=32，'
            'anchor 約束穩定層 [1,4,5,7,10,13,14,15,16,17] 至原始 WavTokenizer（取代凍結）'
        ),
        'n_layers': 18,
    },
    # plan_c: 全層作為等參數預算對照（rank 縮小使總參數量相當）
    'plan_c': {
        'layers': ALL_18_LAYERS,
        'rank': 10,
        'alpha': 20,
        'description': (
            'all_18_lowrank: 全 18 層 rank=10 ≈ plan_b 參數量，作為均勻分配的對照'
        ),
        'n_layers': 18,
    },
    # plan_d: 全 18 層 LoRA rank=32，穩定 10 層加 anchor 約束 + 3-phase LR + dynamic lambda
    # 差異於 plan_b：使用 3-phase LR 與動態 lambda（繼承自 exp_0305c 改良）
    # 8 層噪音敏感層（L0,L2,L3,L6,L8,L9,L11,L12）自由訓練（無 anchor）
    # 10 層穩定層（L1,L4,L5,L7,L10,L13,L14,L15,L16,L17）加 anchor 約束至官方 WavTokenizer
    'plan_d': {
        'layers': ALL_18_LAYERS,   # 全 18 層都有 LoRA rank=32
        'rank': 32,
        'alpha': 64,
        'anchor_layer_ids': [1, 4, 5, 7, 10, 13, 14, 15, 16, 17],
        'lambda_anchor': 0.5,      # 初始值，動態調整由訓練腳本 get_lambda_anchor() 控制
        'use_dynamic_lambda': True,
        'use_3phase_lr': True,
        'description': (
            'plan_d: 全 18 層 LoRA rank=32，'
            '8 層噪音敏感層(L0,L2,L3,L6,L8,L9,L11,L12)自由訓練，'
            '10 層穩定層(L1,L4,L5,L7,L10,L13,L14,L15,L16,L17) anchor 約束至官方 WavTokenizer，'
            '3-phase LR + dynamic lambda_anchor(ep1-50=0.5, ep51-150=1.5, ep151+=3.0)'
        ),
        'n_layers': 18,
    },
}


# ============================================================
# 模型
# ============================================================

class TeacherStudentSelectiveLoRA(nn.Module):
    """選擇性 LoRA 層去噪模型

    以 exp_0224a 為基礎，差異在於 LoRA 只施加在
    由 exp_0304 feature map 分析選出的「有效層」。

    Args:
        wavtok_config: WavTokenizer config yaml 路徑
        wavtok_ckpt: WavTokenizer checkpoint 路徑
        plan: 'plan_a' | 'plan_b' | 'plan_c'
        lora_dropout: LoRA dropout（預設 0.1，比 exp_0224a 的 0.0 略高以防 overfitting）
        device: 計算裝置
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        plan: str = 'plan_b',
        lora_dropout: float = 0.1,
        device: str = 'cuda',
    ):
        super().__init__()

        if plan not in PLANS:
            raise ValueError(f"plan 必須是 {list(PLANS.keys())}，得到 '{plan}'")

        plan_cfg = PLANS[plan]
        self.plan = plan
        self.lora_rank = plan_cfg['rank']
        self.lora_alpha = plan_cfg['alpha']

        print(f"\n{'='*65}")
        print(f"TeacherStudentSelectiveLoRA — {plan}")
        print(f"  {plan_cfg['description']}")
        print(f"  Layers: {plan_cfg['n_layers']}  rank={plan_cfg['rank']}  alpha={plan_cfg['alpha']}")
        print(f"{'='*65}")

        # ── Teacher (完全凍結) ──────────────────────────────────
        print("Loading Teacher (fully frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ── Student (選擇性 LoRA) ────────────────────────────────
        print(f"Loading Student with selective LoRA ({len(plan_cfg['layers'])} conv targets)...")
        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        lora_config = LoraConfig(
            r=plan_cfg['rank'],
            lora_alpha=plan_cfg['alpha'],
            target_modules=plan_cfg['layers'],
            lora_dropout=lora_dropout,
            bias='none',
        )
        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()

        # ── 移動到裝置 ───────────────────────────────────────────
        self.teacher = self.teacher.to(device)
        self.student = self.student.to(device)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n參數統計:")
        print(f"  可訓練: {trainable:,}  /  總計: {total:,}  ({100*trainable/total:.3f}%)")
        print(f"{'='*65}\n")

    def encode_student(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """Student encoder 前向（不量化）

        Args:
            noisy_audio: [B, 1, T] 含噪音頻

        Returns:
            student_features: [B, 512, T'] 連續特徵
        """
        enc = self.student.feature_extractor.encodec.encoder
        return enc(noisy_audio)

    def decode_continuous(self, features: torch.Tensor) -> torch.Tensor:
        """用連續特徵直接解碼（跳過 VQ）

        凍結的 WavTokenizer decoder (backbone + head)。

        Args:
            features: [B, 512, T']

        Returns:
            recon_wav: [B, 1, T]
        """
        bandwidth_id = torch.tensor([0], device=features.device)
        x = self.teacher.backbone(features, bandwidth_id=bandwidth_id)
        audio = self.teacher.head(x)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio

    def forward_wav(
        self,
        clean_audio: torch.Tensor,
        noisy_audio: torch.Tensor,
    ) -> dict:
        """No-VQ 前向傳播

        Args:
            clean_audio: [B, 1, T] 乾淨音頻（訓練時用於 loss，推論時不需要）
            noisy_audio: [B, 1, T] 含噪音頻

        Returns:
            dict with:
                recon_wav: [B, 1, T_recon]  重建音頻
                student_features: [B, 512, T']  encoder 輸出
        """
        # Student encode（含噪 → 連續特徵）
        student_features = self.encode_student(noisy_audio)  # [B, 512, T']

        # 跳過 VQ，直接 decode
        recon_wav = self.decode_continuous(student_features)  # [B, 1, T]

        return {
            'recon_wav': recon_wav,
            'student_features': student_features,
        }
