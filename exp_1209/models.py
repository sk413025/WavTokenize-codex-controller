"""
exp_1209: 模型定義

包含:
1. DenoiseAdapter: 輕量 MLP adapter，用於修正 encoder 輸出
2. TeacherStudentWithAdapter: 使用 Adapter 的 Teacher-Student 模型
3. TeacherStudentExpandedLoRA: 擴大 LoRA 的 Teacher-Student 模型
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from exp_1201.wavtok_lora_patch import apply_lora_patch

apply_lora_patch()


class DenoiseAdapter(nn.Module):
    """
    輕量 Adapter 模組，用於修正 encoder 輸出

    在 encoder 輸出後、VQ 前添加，專門學習「去噪修正」

    結構:
        encoder_out → Linear → ReLU → Linear → + encoder_out (殘差)

    Args:
        dim: encoder 輸出維度 (default: 512)
        hidden_dim: 隱藏層維度 (default: 256)
        num_layers: 層數 (default: 2)
        dropout: dropout 比例 (default: 0.1)

    Note:
        - 使用殘差連接，初始時近似恆等映射
        - 最後一層初始化為零，確保訓練初期不破壞原始特徵
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # 建構 MLP
        layers = []
        in_dim = dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # 最後一層
        layers.append(nn.Linear(hidden_dim, dim))

        self.net = nn.Sequential(*layers)

        # 初始化最後一層為零，確保初始時為恆等映射
        self._init_weights()

    def _init_weights(self):
        """
        初始化權重

        最後一層初始化為零，使得初始輸出為零，
        加上殘差連接後等於恆等映射
        """
        # 最後一層（Linear）初始化為零
        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        # 其他層使用標準初始化
        for module in self.net[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: encoder 輸出, shape (B, C, T)

        Returns:
            修正後的特徵, shape (B, C, T)
        """
        # (B, C, T) -> (B, T, C)
        x_permuted = x.permute(0, 2, 1)

        # MLP + 殘差連接
        out = x_permuted + self.net(x_permuted)

        # (B, T, C) -> (B, C, T)
        return out.permute(0, 2, 1)

    def get_num_params(self) -> int:
        """
        獲取參數數量

        Returns:
            可訓練參數數量
        """
        return sum(p.numel() for p in self.parameters())


class TeacherStudentWithAdapter(nn.Module):
    """
    使用 Adapter 的 Teacher-Student 模型

    架構:
        Noisy Audio → Encoder(凍結) → Adapter(訓練) → VQ → tokens
        Clean Audio → Encoder(凍結) → VQ → tokens (Teacher)

    Args:
        wavtok_config: WavTokenizer 配置文件路徑
        wavtok_ckpt: WavTokenizer checkpoint 路徑
        adapter_hidden: Adapter 隱藏層維度
        adapter_layers: Adapter 層數
        adapter_dropout: Adapter dropout
        device: 運算設備
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        adapter_hidden: int = 256,
        adapter_layers: int = 2,
        adapter_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Teacher: 完全凍結
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)

        # Student encoder: 凍結（不用 LoRA）
        print("Loading Student encoder (frozen)...")
        self.student_encoder = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        for param in self.student_encoder.parameters():
            param.requires_grad = False
        self.student_encoder = self.student_encoder.to(device)

        # Adapter: 可訓練
        print(f"Creating DenoiseAdapter (hidden={adapter_hidden}, layers={adapter_layers})...")
        self.adapter = DenoiseAdapter(
            dim=512,
            hidden_dim=adapter_hidden,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
        ).to(device)

        print(f"Adapter parameters: {self.adapter.get_num_params():,}")

        # 獲取 codebook（用於 loss 計算）
        self.codebook = self._get_codebook()
        print(f"Codebook shape: {self.codebook.shape}")

    def _get_codebook(self) -> torch.Tensor:
        """
        從 Teacher 獲取 codebook

        Returns:
            codebook tensor, shape (num_codes, dim)
        """
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        codebook = quantizer.vq.layers[0].codebook.detach()
        return codebook

    def forward(self, noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> dict:
        """
        前向傳播

        Args:
            noisy_audio: 帶噪音的音頻, shape (B, L) 或 (B, 1, L)
            clean_audio: 乾淨的音頻, shape (B, L) 或 (B, 1, L)

        Returns:
            dict 包含:
                - student_encoder_out: Adapter 前的 encoder 輸出
                - student_adapted_out: Adapter 後的輸出
                - teacher_encoder_out: Teacher encoder 輸出
                - student_codes: Student tokens
                - teacher_codes: Teacher tokens
                - codebook: VQ codebook
        """
        # 確保格式
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward（完全不動）
        with torch.no_grad():
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward
        # Step 1: Encoder（凍結）
        with torch.no_grad():
            student_encoder_out = self.student_encoder.feature_extractor.encodec.encoder(noisy_audio)

        # Step 2: Adapter（訓練）
        student_adapted_out = self.adapter(student_encoder_out)

        # Step 3: VQ（使用 adapted output）
        with torch.no_grad():
            student_vq = self.student_encoder.feature_extractor.encodec.quantizer(
                student_adapted_out, frame_rate=75, bandwidth=0.075
            )
            student_codes = student_vq.codes

        return {
            'student_encoder_out': student_encoder_out,
            'student_adapted_out': student_adapted_out,
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
        }


class TeacherStudentExpandedLoRA(nn.Module):
    """
    擴大 LoRA 的 Teacher-Student 模型

    架構:
        Noisy Audio → Encoder(LoRA 18層) → VQ → tokens
        Clean Audio → Encoder(凍結) → VQ → tokens (Teacher)

    Args:
        wavtok_config: WavTokenizer 配置文件路徑
        wavtok_ckpt: WavTokenizer checkpoint 路徑
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        device: 運算設備
    """

    # 全部 18 個 encoder conv 層
    ALL_ENCODER_CONV_MODULES = [
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",
        "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",
        "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",
        "feature_extractor.encodec.encoder.model.3.conv.conv",
        "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",
        "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",
        "feature_extractor.encodec.encoder.model.4.shortcut.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
        "feature_extractor.encodec.encoder.model.7.block.1.conv.conv",
        "feature_extractor.encodec.encoder.model.7.block.3.conv.conv",
        "feature_extractor.encodec.encoder.model.7.shortcut.conv.conv",
        "feature_extractor.encodec.encoder.model.9.conv.conv",
        "feature_extractor.encodec.encoder.model.10.block.1.conv.conv",
        "feature_extractor.encodec.encoder.model.10.block.3.conv.conv",
        "feature_extractor.encodec.encoder.model.10.shortcut.conv.conv",
        "feature_extractor.encodec.encoder.model.12.conv.conv",
        "feature_extractor.encodec.encoder.model.15.conv.conv",
    ]

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        lora_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Teacher: 凍結
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)

        # Student: 擴大 LoRA
        print(f"Loading Student with Expanded LoRA (rank={lora_rank}, {len(self.ALL_ENCODER_CONV_MODULES)} layers)...")
        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self.ALL_ENCODER_CONV_MODULES,
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()
        self.student = self.student.to(device)

        # 獲取 codebook
        self.codebook = self._get_codebook()
        print(f"Codebook shape: {self.codebook.shape}")

    def _get_codebook(self) -> torch.Tensor:
        """
        從 Teacher 獲取 codebook

        Returns:
            codebook tensor, shape (num_codes, dim)
        """
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        codebook = quantizer.vq.layers[0].codebook.detach()
        return codebook

    def forward(self, noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> dict:
        """
        前向傳播

        Args:
            noisy_audio: 帶噪音的音頻
            clean_audio: 乾淨的音頻

        Returns:
            dict 包含各種輸出和 codebook
        """
        # 確保格式
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward
        with torch.no_grad():
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward
        student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)

        # VQ
        quantizer = self.student.feature_extractor.encodec.quantizer
        student_vq = quantizer(student_encoder_out, frame_rate=75, bandwidth=0.075)
        student_codes = student_vq.codes

        return {
            'student_encoder_out': student_encoder_out,
            'student_adapted_out': student_encoder_out,  # 為了兼容性，和 adapter 模型一致
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
        }
