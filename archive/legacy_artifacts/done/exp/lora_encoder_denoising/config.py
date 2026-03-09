"""
配置文件 - LoRA Encoder Denoising
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 路徑配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "done" / "exp" / "data_with_distances"  # 使用真實數據
RESULTS_ROOT = Path(__file__).parent / "checkpoints"
LOGS_ROOT = Path(__file__).parent / "logs"

# WavTokenizer 配置
WAVTOK_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT = "/home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt"

# 數據路徑 - 使用 commit 927880a 的預處理數據
TRAIN_CACHE = DATA_ROOT / "train_cache_with_distances.pt"
VAL_CACHE = DATA_ROOT / "val_cache_with_distances.pt"

# HDF5 數據
HDF5_CACHE = DATA_ROOT / "cache_with_distances.h5"

# Distance Matrix (VQ codebook pairwise distances)
DISTANCE_MATRIX = Path(__file__).parent.parent / "wavtok_distance_mat.pt"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Smoke Test 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SmokeTestConfig:
    """
    快速測試配置 - 用於驗證實作正確性
    目標：2-5 分鐘內完成，所有檢查通過
    """
    # 數據
    num_samples: int = 20           # 極小數據量
    batch_size: int = 4             # 小 batch
    num_workers: int = 0            # 避免多進程問題

    # 訓練
    num_epochs: int = 3             # 只訓練幾個 epoch
    learning_rate: float = 1e-4     # 較大 LR，快速收斂
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0       # Smoke test 不用 warmup

    # LoRA
    lora_rank: int = 8              # 小 rank
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.3.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
        "feature_extractor.encodec.encoder.model.9.conv.conv",
    ])

    # Loss 權重
    feature_loss_weight: float = 1.0
    distance_loss_weight: float = 0.1
    vq_loss_weight: float = 0.01

    # 日誌
    log_every_n_batches: int = 1    # 每個 batch 都 log
    log_interval: int = 1           # Same as log_every_n_batches (for train.py compatibility)
    save_checkpoint: bool = True
    checkpoint_dir: Path = RESULTS_ROOT / "smoke_test"

    # Validation
    val_interval: int = 1           # Validate every N epochs
    save_interval: int = 1          # Save every N epochs

    # 設備
    device: str = "cuda"
    use_amp: bool = False           # Smoke test 不用混合精度


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 完整訓練配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrainConfig:
    """
    完整訓練配置
    """
    # 實驗
    exp_name: str = "lora_denoising_r16"
    seed: int = 42

    # 數據
    use_hdf5: bool = False          # 是否使用 HDF5 (commit 927880a)
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # 訓練
    num_epochs: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # 學習率調度
    warmup_epochs: int = 5
    warmup_ratio: float = 0.1       # Warmup steps = total_steps * warmup_ratio
    scheduler: str = "cosine"       # "cosine" or "step" or "none"
    min_lr: float = 1e-6

    # LoRA 配置
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.3.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
        "feature_extractor.encodec.encoder.model.9.conv.conv",
    ])

    # Loss 權重
    feature_loss_weight: float = 1.0
    distance_loss_weight: float = 0.1
    vq_loss_weight: float = 0.01

    # 驗證
    val_every_n_epochs: int = 1
    check_original_capability: bool = True  # 定期檢查原始能力

    # Checkpoint
    save_every_n_epochs: int = 5
    save_interval: int = 5          # Same as save_every_n_epochs (for train.py compatibility)
    save_top_k: int = 3             # 保存最好的 k 個模型
    checkpoint_dir: Optional[Path] = None

    # Validation
    val_interval: int = 1           # Validate every N epochs

    # 日誌
    log_every_n_batches: int = 50
    log_interval: int = 50          # Same as log_every_n_batches (for train.py compatibility)
    use_tensorboard: bool = True
    log_dir: Optional[Path] = None

    # 設備
    device: str = "cuda"
    use_amp: bool = True            # 混合精度訓練
    accumulate_grad_batches: int = 1  # Gradient accumulation

    def __post_init__(self):
        # 自動設置路徑
        if self.checkpoint_dir is None:
            self.checkpoint_dir = RESULTS_ROOT / self.exp_name
        if self.log_dir is None:
            self.log_dir = LOGS_ROOT / self.exp_name

        # 創建目錄
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EvalConfig:
    """
    評估配置
    """
    checkpoint_path: str
    output_dir: Path = RESULTS_ROOT / "evaluation"
    batch_size: int = 32
    num_workers: int = 4

    # 評估項目
    compute_feature_distance: bool = True
    compute_code_match_rate: bool = True
    check_original_capability: bool = True
    generate_visualizations: bool = True

    # 測試不同 noise 條件
    test_noise_levels: List[int] = field(default_factory=lambda: [20, 15, 10, 5, 0])  # SNR in dB

    device: str = "cuda"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 預設配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_smoke_test_config() -> SmokeTestConfig:
    """獲取 Smoke Test 配置"""
    return SmokeTestConfig()


def get_train_config(
    exp_name: str = "lora_denoising_r16",
    lora_rank: int = 16,
    **kwargs
) -> TrainConfig:
    """
    獲取訓練配置

    Args:
        exp_name: 實驗名稱
        lora_rank: LoRA rank
        **kwargs: 其他參數覆蓋

    Returns:
        TrainConfig
    """
    config = TrainConfig(exp_name=exp_name, lora_rank=lora_rank)

    # 覆蓋參數
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def get_eval_config(checkpoint_path: str, **kwargs) -> EvalConfig:
    """
    獲取評估配置

    Args:
        checkpoint_path: Checkpoint 路徑
        **kwargs: 其他參數覆蓋

    Returns:
        EvalConfig
    """
    config = EvalConfig(checkpoint_path=checkpoint_path)

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
