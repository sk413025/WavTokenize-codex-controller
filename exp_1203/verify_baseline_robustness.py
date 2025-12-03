"""
exp_1203 實驗 1: 驗證原始 WavTokenizer 的 Noise Robustness

目的：
  在做任何訓練之前，先測試原始 WavTokenizer（無 LoRA）對噪音的魯棒性。
  這是重要的 baseline，幫助我們了解：
  1. 原始模型本身就有多少噪音魯棒性
  2. 訓練的目標應該是多少

方法：
  1. 載入原始 WavTokenizer（無任何修改）
  2. 對每對 (noisy, clean) 音頻：
     - 用同一個 encoder 分別處理 noisy 和 clean 音頻
     - 比較產生的 tokens 是否相同
  3. 統計 Token Match Rate

預期：
  如果 Epoch 1 的 Token Accuracy (~20-30%) 是因為 LoRA 初始權重 ≈ 0，
  那麼原始模型的 Token Match Rate 也應該接近這個數值。
"""

import torch
import sys
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime

# 添加必要路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "exp_1201"))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DATA_ROOT
from exp_1201.data import create_dataloaders


def load_original_wavtokenizer(config_path, ckpt_path, device):
    """載入原始 WavTokenizer（無任何修改）"""
    # 添加 WavTokenizer 路徑
    wavtok_path = Path("/home/sbplab/ruizi/WavTokenizer-main")
    if str(wavtok_path) not in sys.path:
        sys.path.insert(0, str(wavtok_path))

    from decoder.pretrained import WavTokenizer

    print(f"Loading original WavTokenizer...")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {ckpt_path}")

    model = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
    model = model.to(device)
    model.eval()

    # 統計參數
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return model


def compute_token_match_rate(model, noisy_audio, clean_audio, device):
    """
    計算單一樣本的 Token Match Rate

    Args:
        model: 原始 WavTokenizer
        noisy_audio: (1, T) 或 (T,)
        clean_audio: (1, T) 或 (T,)
        device: torch device

    Returns:
        match_rate: float (0-1)
        total_tokens: int
        matched_tokens: int
    """
    # 確保格式為 (1, T)
    if noisy_audio.dim() == 1:
        noisy_audio = noisy_audio.unsqueeze(0)
    if clean_audio.dim() == 1:
        clean_audio = clean_audio.unsqueeze(0)

    noisy_audio = noisy_audio.to(device)
    clean_audio = clean_audio.to(device)

    with torch.no_grad():
        # 使用 feature_extractor 取得 codes
        _, noisy_codes, _ = model.feature_extractor(noisy_audio, bandwidth_id=0)
        _, clean_codes, _ = model.feature_extractor(clean_audio, bandwidth_id=0)

        # codes 格式: (n_q, B, T) -> 取第一個 quantizer
        if noisy_codes.dim() == 3:
            noisy_codes = noisy_codes[0]  # (B, T)
            clean_codes = clean_codes[0]  # (B, T)

        # 計算 match rate
        matched = (noisy_codes == clean_codes).float()
        match_rate = matched.mean().item()
        total_tokens = noisy_codes.numel()
        matched_tokens = int(matched.sum().item())

    return match_rate, total_tokens, matched_tokens


def evaluate_baseline_robustness(model, dataloader, device, max_samples=None):
    """
    評估原始模型的噪音魯棒性

    Args:
        model: 原始 WavTokenizer
        dataloader: 包含 (noisy, clean) pairs 的 dataloader
        device: torch device
        max_samples: 最大樣本數（None = 全部）

    Returns:
        results: dict with statistics
    """
    all_match_rates = []
    total_tokens = 0
    total_matched = 0

    desc = "Evaluating baseline robustness"
    if max_samples:
        desc += f" (max {max_samples} samples)"

    sample_count = 0

    for batch in tqdm(dataloader, desc=desc):
        noisy_audio = batch['noisy_audio']
        clean_audio = batch['clean_audio']

        # 逐樣本處理
        batch_size = noisy_audio.shape[0]
        for i in range(batch_size):
            match_rate, n_tokens, n_matched = compute_token_match_rate(
                model, noisy_audio[i], clean_audio[i], device
            )
            all_match_rates.append(match_rate)
            total_tokens += n_tokens
            total_matched += n_matched

            sample_count += 1
            if max_samples and sample_count >= max_samples:
                break

        if max_samples and sample_count >= max_samples:
            break

    # 統計
    match_rates_array = np.array(all_match_rates)
    results = {
        'num_samples': len(all_match_rates),
        'total_tokens': total_tokens,
        'total_matched': total_matched,
        'overall_match_rate': total_matched / total_tokens if total_tokens > 0 else 0,
        'mean_match_rate': float(match_rates_array.mean()),
        'std_match_rate': float(match_rates_array.std()),
        'min_match_rate': float(match_rates_array.min()),
        'max_match_rate': float(match_rates_array.max()),
        'median_match_rate': float(np.median(match_rates_array)),
        'percentile_25': float(np.percentile(match_rates_array, 25)),
        'percentile_75': float(np.percentile(match_rates_array, 75)),
    }

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify baseline WavTokenizer noise robustness')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None = all)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for dataloader')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--output_dir', type=str, default='exp_1203/experiments/baseline_robustness',
                       help='Output directory for results')
    args = parser.parse_args()

    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 建立輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入模型
    model = load_original_wavtokenizer(WAVTOK_CONFIG, WAVTOK_CKPT, device)

    # 建立 dataloader (使用 exp_1201 的 data module)
    print(f"\nLoading data from {DATA_ROOT}...")

    # 建立簡單的 config
    class SimpleConfig:
        def __init__(self):
            self.use_hdf5 = False
            self.batch_size = args.batch_size
            self.num_workers = args.num_workers
            self.pin_memory = True

    config = SimpleConfig()
    train_loader, val_loader = create_dataloaders(config)

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

    # 評估訓練集
    print("\n" + "=" * 60)
    print("Evaluating on TRAIN set")
    print("=" * 60)
    train_results = evaluate_baseline_robustness(
        model, train_loader, device, max_samples=args.max_samples
    )

    print(f"\nTrain Set Results:")
    print(f"  Samples evaluated: {train_results['num_samples']}")
    print(f"  Overall Match Rate: {train_results['overall_match_rate']*100:.2f}%")
    print(f"  Mean Match Rate: {train_results['mean_match_rate']*100:.2f}%")
    print(f"  Std Match Rate: {train_results['std_match_rate']*100:.2f}%")
    print(f"  Min/Max: {train_results['min_match_rate']*100:.2f}% / {train_results['max_match_rate']*100:.2f}%")
    print(f"  Median: {train_results['median_match_rate']*100:.2f}%")
    print(f"  25th/75th percentile: {train_results['percentile_25']*100:.2f}% / {train_results['percentile_75']*100:.2f}%")

    # 評估驗證集
    print("\n" + "=" * 60)
    print("Evaluating on VAL set")
    print("=" * 60)
    val_results = evaluate_baseline_robustness(
        model, val_loader, device, max_samples=args.max_samples
    )

    print(f"\nVal Set Results:")
    print(f"  Samples evaluated: {val_results['num_samples']}")
    print(f"  Overall Match Rate: {val_results['overall_match_rate']*100:.2f}%")
    print(f"  Mean Match Rate: {val_results['mean_match_rate']*100:.2f}%")
    print(f"  Std Match Rate: {val_results['std_match_rate']*100:.2f}%")
    print(f"  Min/Max: {val_results['min_match_rate']*100:.2f}% / {val_results['max_match_rate']*100:.2f}%")
    print(f"  Median: {val_results['median_match_rate']*100:.2f}%")
    print(f"  25th/75th percentile: {val_results['percentile_25']*100:.2f}% / {val_results['percentile_75']*100:.2f}%")

    # 保存結果
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'config': {
            'max_samples': args.max_samples,
            'batch_size': args.batch_size,
            'wavtok_config': str(WAVTOK_CONFIG),
            'wavtok_ckpt': str(WAVTOK_CKPT),
        },
        'train': train_results,
        'val': val_results,
    }

    results_path = output_dir / 'baseline_robustness_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # 結論
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    avg_match_rate = (train_results['mean_match_rate'] + val_results['mean_match_rate']) / 2

    print(f"""
原始 WavTokenizer 的 Noise Robustness:
  - 訓練集平均 Token Match Rate: {train_results['mean_match_rate']*100:.2f}%
  - 驗證集平均 Token Match Rate: {val_results['mean_match_rate']*100:.2f}%
  - 總平均: {avg_match_rate*100:.2f}%

與 exp_1201 實驗結果比較:
  - Gumbel Epoch 1 Train Acc: 24.45%
  - STE Epoch 1 Train Acc: 22.46%
  - CE Epoch 1 Train Acc: 32.49%
  - Margin Epoch 1 Train Acc: 27.77%

如果原始模型的 Match Rate 接近 20-30%，則證實：
  1. Epoch 1 的高 Token Accuracy 是因為 LoRA 初始權重 ≈ 0
  2. 訓練過程反而「破壞」了原始魯棒性
  3. 需要重新思考訓練策略

如果原始模型的 Match Rate 遠低於 20-30%，則需要重新分析。
""")


if __name__ == '__main__':
    main()
