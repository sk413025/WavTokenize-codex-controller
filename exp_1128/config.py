"""
配置文件 - exp_1128: 增加 LoRA Rank 和 Distance Loss 權重
基於 exp_1126/1126-1 的分析結果:
- Feature MSE 改善 38%，Cosine Similarity 改善 72%
- 但 Code L2 Distance 只改善 18%
- 嘗試: 增加 LoRA Rank (32/64) 和 Distance Loss 權重 (0.05/0.1)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 路徑配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path("/home/sbplab/ruizi/c_code/done/exp/data3")
RESULTS_ROOT = Path(__file__).parent / "checkpoints"
LOGS_ROOT = Path(__file__).parent / "logs"

# WavTokenizer 配置
WAVTOK_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

# 數據路徑
TRAIN_CACHE = DATA_ROOT / "train_cache.pt"
VAL_CACHE = DATA_ROOT / "val_cache.pt"

# HDF5 數據 (可選)
HDF5_CACHE = DATA_ROOT / "cache.h5"

# Distance Matrix - 從 1126-1 複製
DISTANCE_MATRIX = Path(__file__).parent / "wavtok_distance_mat_corrected.pt"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Smoke Test 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SmokeTestConfig:
    """快速測試配置"""
    num_samples: int = 20
    batch_size: int = 4
    num_workers: int = 0

    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0

    # LoRA - 測試用小 rank
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.3.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
        "feature_extractor.encodec.encoder.model.9.conv.conv",
    ])

    feature_loss_weight: float = 1.0
    distance_loss_weight: float = 0.1
    vq_loss_weight: float = 0.0

    log_every_n_batches: int = 1
    log_interval: int = 1
    save_checkpoint: bool = True
    checkpoint_dir: Path = RESULTS_ROOT / "smoke_test"

    val_interval: int = 1
    save_interval: int = 1

    device: str = "cuda"
    use_amp: bool = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 完整訓練配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrainConfig:
    """
    完整訓練配置 - exp_1128
    重點改進:
    1. 增加 LoRA Rank: 16 -> 32/64
    2. 增加 Distance Loss 權重: 0.01 -> 0.05/0.1
    """
    # 實驗
    exp_name: str = "lora_r32_dist0.05"
    seed: int = 42

    # 數據
    use_hdf5: bool = False
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # 訓練
    num_epochs: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # 學習率調度
    warmup_epochs: int = 5
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    min_lr: float = 1e-6

    # LoRA 配置 - 增加 Rank!
    lora_rank: int = 32          # 從 16 增加到 32
    lora_alpha: int = 64         # alpha = 2 * rank
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.3.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
        "feature_extractor.encodec.encoder.model.9.conv.conv",
    ])

    # Loss 權重 - 增加 Distance Loss!
    feature_loss_weight: float = 1.0
    distance_loss_weight: float = 0.05   # 從 0.01 增加到 0.05
    vq_loss_weight: float = 0.0          # 保持 0 (VQ 已凍結)

    # 驗證
    val_every_n_epochs: int = 1
    check_original_capability: bool = True

    # Checkpoint
    save_every_n_epochs: int = 20
    save_interval: int = 20
    save_top_k: int = 3
    checkpoint_dir: Optional[Path] = None

    val_interval: int = 1

    # 日誌
    log_every_n_batches: int = 50
    log_interval: int = 50
    use_tensorboard: bool = True
    log_dir: Optional[Path] = None

    # 設備
    device: str = "cuda"
    use_amp: bool = True
    accumulate_grad_batches: int = 1

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = RESULTS_ROOT / self.exp_name
        if self.log_dir is None:
            self.log_dir = LOGS_ROOT / self.exp_name

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EvalConfig:
    """評估配置"""
    checkpoint_path: str
    output_dir: Path = RESULTS_ROOT / "evaluation"
    batch_size: int = 32
    num_workers: int = 4

    compute_feature_distance: bool = True
    compute_code_match_rate: bool = True
    check_original_capability: bool = True
    generate_visualizations: bool = True

    test_noise_levels: List[int] = field(default_factory=lambda: [20, 15, 10, 5, 0])

    device: str = "cuda"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 預設配置函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_smoke_test_config() -> SmokeTestConfig:
    """獲取 Smoke Test 配置"""
    return SmokeTestConfig()


def get_train_config(
    exp_name: str = "lora_r32_dist0.05",
    lora_rank: int = 32,
    lora_alpha: int = 64,
    distance_loss_weight: float = 0.05,
    **kwargs
) -> TrainConfig:
    """
    獲取訓練配置

    預設實驗配置:
    - exp_1: lora_r32_dist0.05 (rank=32, dist_weight=0.05)
    - exp_2: lora_r32_dist0.1  (rank=32, dist_weight=0.1)
    - exp_3: lora_r64_dist0.05 (rank=64, dist_weight=0.05)
    - exp_4: lora_r64_dist0.1  (rank=64, dist_weight=0.1)
    """
    config = TrainConfig(
        exp_name=exp_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        distance_loss_weight=distance_loss_weight
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def get_eval_config(checkpoint_path: str, **kwargs) -> EvalConfig:
    """獲取評估配置"""
    config = EvalConfig(checkpoint_path=checkpoint_path)

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
