"""
配置文件 - exp_1201: Soft Distance Loss

解決 exp_1128 發現的問題:
- Distance Loss 不可微 (argmax + indexing 切斷梯度)
- 改用 Soft Distance Loss (softmax 保持梯度)

新增參數:
- soft_dist_loss_weight: Soft distance loss 權重
- temperature: Softmax temperature (控制分布軟硬程度)
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
    完整訓練配置 - exp_1201: Soft Distance Loss

    重點改進 (相比 exp_1128):
    1. 改用 Soft Distance Loss (可微!)
    2. 新增 temperature 參數控制 softmax 分布
    3. 每 10 epoch 儲存 loss 圖、音檔、頻譜圖
    """
    # 實驗
    exp_name: str = "soft_dist_baseline"
    seed: int = 42

    # 數據
    use_hdf5: bool = False
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
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    min_lr: float = 1e-6

    # LoRA 配置 - 沿用 exp_1128 最佳配置
    lora_rank: int = 64          # 使用較大的 rank
    lora_alpha: int = 128        # alpha = 2 * rank
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "feature_extractor.encodec.encoder.model.0.conv.conv",
        "feature_extractor.encodec.encoder.model.3.conv.conv",
        "feature_extractor.encodec.encoder.model.6.conv.conv",
        "feature_extractor.encodec.encoder.model.9.conv.conv",
    ])

    # Loss 權重
    feature_loss_weight: float = 1.0
    soft_dist_loss_weight: float = 0.1   # Distance loss 權重 (可微!)
    vq_loss_weight: float = 0.0          # 保持 0 (原始 VQ commitment loss)
    correct_vq_loss_weight: float = 0.0  # 修正版 VQ loss (讓 features 接近 teacher 的 codebook embedding)
    
    # EmbDistillationLoss 參數 (當 distance_loss_mode='emb_distillation' 時使用)
    emb_to_codebook_weight: float = 1.0  # 主要 Loss: encoder 輸出接近 teacher 的 codebook
    ce_token_weight: float = 0.0         # 可選: CE Token Loss

    # Distance Loss 參數
    distance_loss_mode: str = 'gumbel'   # 'soft', 'gumbel', 'ste', 'ce', 'margin', 'emb_distillation'
                                          # - soft: 期望距離 (所有 codes 加權平均)
                                          # - gumbel: Gumbel-Softmax (隨機 + ST)
                                          # - ste: Straight-Through Estimator
                                          # - ce: Cross-Entropy (直接監督 token 選擇)
                                          # - margin: Margin Loss (決策邊界優化)
                                          # - emb_distillation: 直接監督 encoder 原始輸出 (修正版！)
    temperature: float = 1.0             # Softmax temperature (τ)
                                          # τ → 0: 接近 hard (one-hot)
                                          # τ → ∞: 接近 uniform
    gumbel_hard: bool = True             # 只在 gumbel 模式有效
    margin: float = 1.0                  # 只在 margin 模式有效
    label_smoothing: float = 0.0         # 只在 ce 模式有效

    # 驗證
    val_every_n_epochs: int = 1
    check_original_capability: bool = True

    # Checkpoint 與輸出
    save_every_n_epochs: int = 10         # 每 10 epoch 儲存
    save_interval: int = 10               # 每 10 epoch 儲存
    save_top_k: int = 3
    checkpoint_dir: Optional[Path] = None

    val_interval: int = 1

    # 音檔與頻譜圖儲存
    save_audio_samples: bool = True       # 儲存音檔
    save_spectrograms: bool = True        # 儲存頻譜圖
    num_audio_samples: int = 3            # 每次儲存的樣本數

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
    exp_name: str = "gumbel_baseline",
    lora_rank: int = 64,
    lora_alpha: int = 128,
    soft_dist_loss_weight: float = 0.1,
    temperature: float = 1.0,
    distance_loss_mode: str = 'gumbel',
    gumbel_hard: bool = True,
    margin: float = 1.0,
    label_smoothing: float = 0.0,
    **kwargs
) -> TrainConfig:
    """
    獲取訓練配置

    實驗配置 (exp_1201):
    - exp_1: gumbel_baseline  (Gumbel-Softmax, τ=1.0) 可微 + 隨機探索
    - exp_2: ste_baseline     (STE, τ=1.0) 可微 + 確定性
    - exp_3: ce_baseline      (Cross-Entropy) 直接監督 token
    - exp_4: margin_baseline  (Margin Loss) 決策邊界優化

    比較目的：
    - Gumbel: 引入隨機性，可能幫助逃離局部最優
    - STE: 確定性，訓練更穩定，但可能陷入局部最優
    - CE: 直接優化 Token Accuracy (本質問題)
    - Margin: 確保正確 token 和錯誤 token 有足夠距離
    """
    config = TrainConfig(
        exp_name=exp_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        soft_dist_loss_weight=soft_dist_loss_weight,
        temperature=temperature,
        distance_loss_mode=distance_loss_mode,
        gumbel_hard=gumbel_hard,
        margin=margin,
        label_smoothing=label_smoothing,
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
