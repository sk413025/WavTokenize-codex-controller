"""
exp_1231_feature: Train vs Val 特徵分析

目的: 確認 Train 和 Val 是否看同樣的特徵
- 類似圖像分類中確認模型是否看「貓的輪廓」還是「背景沙發」

分析內容:
1. 比較 Train/Val 在各層的 feature distribution
2. 計算 Train/Val 的 feature similarity
3. 檢測是否有 distribution shift

使用模型: exp64_curriculum (best_model.pt)

作者: Claude Code
日期: 2024-12-31
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

# 路徑設置
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, str(Path(__file__).parent.parent))

from decoder.pretrained import WavTokenizer

# ============================================================
# 配置
# ============================================================

WAVTOK_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

# 訓練數據
TRAIN_CACHE = Path("/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt")
VAL_CACHE = Path("/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt")

# exp64 模型
EXP64_CKPT = Path("/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1226/runs/exp64_curriculum/best_model.pt")

# 輸出目錄
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 18 層 conv 名稱
ENCODER_CONV_LAYERS = [
    "model.0.conv.conv",
    "model.1.block.1.conv.conv",
    "model.1.block.3.conv.conv",
    "model.1.shortcut.conv.conv",
    "model.3.conv.conv",
    "model.4.block.1.conv.conv",
    "model.4.block.3.conv.conv",
    "model.4.shortcut.conv.conv",
    "model.6.conv.conv",
    "model.7.block.1.conv.conv",
    "model.7.block.3.conv.conv",
    "model.7.shortcut.conv.conv",
    "model.9.conv.conv",
    "model.10.block.1.conv.conv",
    "model.10.block.3.conv.conv",
    "model.10.shortcut.conv.conv",
    "model.12.conv.conv",
    "model.15.conv.conv",
]

LAYER_GROUPS = {
    'input': [0],
    'low_level': [1, 2, 3, 4],
    'mid_level': [5, 6, 7, 8],
    'semantic': [9, 10, 11, 12],
    'abstract': [13, 14, 15, 16],
    'output': [17],
}


class FeatureMapExtractor:
    """提取 WavTokenizer encoder 各層的 feature maps"""

    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _get_encoder(self):
        """獲取 encoder，處理不同的模型結構"""
        if hasattr(self.model, 'feature_extractor'):
            return self.model.feature_extractor.encodec.encoder
        elif hasattr(self.model, 'student'):
            # LoRA 模型
            return self.model.student.feature_extractor.encodec.encoder
        else:
            raise ValueError("Cannot find encoder in model")

    def _get_layer(self, layer_name: str) -> nn.Module:
        encoder = self._get_encoder()
        parts = layer_name.split('.')
        module = encoder
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _create_hook(self, name: str):
        def hook(module, input, output):
            self.feature_maps[name] = output.detach().cpu()
        return hook

    def register_hooks(self):
        for name in ENCODER_CONV_LAYERS:
            try:
                layer = self._get_layer(name)
                hook = layer.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
            except Exception as e:
                print(f"Warning: Could not register hook for {name}: {e}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def extract(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.feature_maps = {}
        self.register_hooks()

        try:
            audio = audio.to(self.device)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)

            bandwidth_id = torch.tensor([0]).to(self.device)

            if hasattr(self.model, 'encode'):
                _ = self.model.encode(audio, bandwidth_id=bandwidth_id)
            elif hasattr(self.model, 'student'):
                _ = self.model.student.encode(audio, bandwidth_id=bandwidth_id)

        finally:
            self.remove_hooks()

        return self.feature_maps.copy()


def load_cached_data(cache_path: Path, max_samples: int = 50) -> List[torch.Tensor]:
    """載入快取的訓練/驗證數據"""
    print(f"Loading cached data from {cache_path}...")
    data = torch.load(cache_path)

    # 根據數據結構提取音頻
    audios = []

    if isinstance(data, dict):
        # 可能的 key: 'noisy_audio', 'clean_audio', 'samples', etc.
        if 'samples' in data:
            samples = data['samples']
            for i, sample in enumerate(samples[:max_samples]):
                if 'noisy_audio' in sample:
                    audios.append(sample['noisy_audio'])
                elif 'clean_audio' in sample:
                    audios.append(sample['clean_audio'])
        elif 'noisy_audio' in data:
            for i in range(min(len(data['noisy_audio']), max_samples)):
                audios.append(data['noisy_audio'][i])
    elif isinstance(data, list):
        for i, sample in enumerate(data[:max_samples]):
            if isinstance(sample, dict):
                if 'noisy_audio' in sample:
                    audios.append(sample['noisy_audio'])
                elif 'clean_audio' in sample:
                    audios.append(sample['clean_audio'])
            elif isinstance(sample, torch.Tensor):
                audios.append(sample)

    print(f"  Loaded {len(audios)} samples")
    return audios


def compute_feature_statistics(feature_maps_list: List[Dict[str, torch.Tensor]]) -> Dict[str, Dict]:
    """計算多個樣本的特徵統計量"""
    layer_stats = {}

    for layer_name in ENCODER_CONV_LAYERS:
        means = []
        stds = []
        mins = []
        maxs = []

        for fm_dict in feature_maps_list:
            if layer_name in fm_dict:
                fm = fm_dict[layer_name]
                if fm.dim() == 3:
                    fm = fm.squeeze(0)
                # 計算每個樣本的統計量（避免合併大張量）
                means.append(fm.mean().item())
                stds.append(fm.std().item())
                mins.append(fm.min().item())
                maxs.append(fm.max().item())

        if means:
            layer_stats[layer_name] = {
                'mean': np.mean(means),
                'std': np.mean(stds),
                'min': np.min(mins),
                'max': np.max(maxs),
                'mean_std': np.std(means),  # mean 的變異性
            }

    return layer_stats


def compute_distribution_distance(stats1: Dict, stats2: Dict) -> Dict[str, float]:
    """計算兩個分布的距離（基於統計量）"""
    distances = {}

    for layer_name in ENCODER_CONV_LAYERS:
        if layer_name in stats1 and layer_name in stats2:
            s1 = stats1[layer_name]
            s2 = stats2[layer_name]

            # Mean 差異 (normalized)
            mean_diff = abs(s1['mean'] - s2['mean']) / (abs(s1['mean']) + 1e-8)

            # Std 差異 (ratio)
            std_ratio = max(s1['std'], s2['std']) / (min(s1['std'], s2['std']) + 1e-8)

            # 綜合指標
            distances[layer_name] = {
                'mean_diff': mean_diff,
                'std_ratio': std_ratio,
                'combined': mean_diff + (std_ratio - 1),  # std_ratio 理想是 1
            }

    return distances


def visualize_train_val_comparison(
    train_stats: Dict,
    val_stats: Dict,
    distances: Dict,
    output_path: Path,
):
    """視覺化 Train vs Val 特徵比較"""
    n_layers = len(ENCODER_CONV_LAYERS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean 比較
    ax = axes[0, 0]
    layer_names = [f"L{i}" for i in range(n_layers)]
    train_means = [train_stats.get(l, {}).get('mean', 0) for l in ENCODER_CONV_LAYERS]
    val_means = [val_stats.get(l, {}).get('mean', 0) for l in ENCODER_CONV_LAYERS]

    x = np.arange(n_layers)
    width = 0.35
    ax.bar(x - width/2, train_means, width, label='Train', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, val_means, width, label='Val', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, fontsize=8)
    ax.set_title('Feature Mean by Layer')
    ax.set_ylabel('Mean')
    ax.legend()

    # 2. Std 比較
    ax = axes[0, 1]
    train_stds = [train_stats.get(l, {}).get('std', 0) for l in ENCODER_CONV_LAYERS]
    val_stds = [val_stats.get(l, {}).get('std', 0) for l in ENCODER_CONV_LAYERS]

    ax.bar(x - width/2, train_stds, width, label='Train', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, val_stds, width, label='Val', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, fontsize=8)
    ax.set_title('Feature Std by Layer')
    ax.set_ylabel('Std')
    ax.legend()

    # 3. Distribution Distance
    ax = axes[1, 0]
    combined_distances = [distances.get(l, {}).get('combined', 0) for l in ENCODER_CONV_LAYERS]

    colors = []
    for d in combined_distances:
        if d < 0.1:
            colors.append('#4ECDC4')  # 綠色 - 相似
        elif d < 0.3:
            colors.append('#FFE66D')  # 黃色 - 有差異
        else:
            colors.append('#FF6B6B')  # 紅色 - 差異大

    ax.bar(x, combined_distances, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, fontsize=8)
    ax.set_title('Train-Val Distribution Distance\n(Lower = More Similar)')
    ax.set_ylabel('Distance')
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Warning')
    ax.legend()

    # 4. 診斷結論
    ax = axes[1, 1]
    ax.axis('off')

    # 計算分組統計
    group_distances = {}
    for group_name, indices in LAYER_GROUPS.items():
        group_dists = [combined_distances[i] for i in indices if i < len(combined_distances)]
        group_distances[group_name] = np.mean(group_dists) if group_dists else 0

    diagnosis = []
    diagnosis.append("=" * 45)
    diagnosis.append("Train vs Val 特徵分布診斷")
    diagnosis.append("=" * 45)

    diagnosis.append("\n各層組 Train-Val 差異:")
    for group_name, dist in group_distances.items():
        status = "✓" if dist < 0.1 else "⚠️" if dist < 0.3 else "❌"
        diagnosis.append(f"  {group_name:12s}: {dist:.4f} {status}")

    # 總體判斷
    avg_dist = np.mean(combined_distances)
    diagnosis.append(f"\n總體平均差異: {avg_dist:.4f}")

    if avg_dist < 0.1:
        diagnosis.append("\n[✓] Train/Val 看的特徵相似")
        diagnosis.append("    模型應該能泛化")
    elif avg_dist < 0.3:
        diagnosis.append("\n[⚠️] Train/Val 有些差異")
        diagnosis.append("    可能有輕微 distribution shift")
    else:
        diagnosis.append("\n[❌] Train/Val 差異較大")
        diagnosis.append("    可能存在嚴重 distribution shift")
        diagnosis.append("    這可能解釋 Val acc 停滯!")

    ax.text(0.05, 0.95, '\n'.join(diagnosis), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Train vs Val Feature Distribution Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    return {
        'group_distances': group_distances,
        'avg_distance': avg_dist,
        'layer_distances': combined_distances,
    }


def main():
    """主程式"""
    print("=" * 60)
    print("Train vs Val Feature Distribution Analysis")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 載入原始 WavTokenizer
    print("\nLoading WavTokenizer...")
    wavtok = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)
    wavtok.eval()
    wavtok = wavtok.to(device)

    extractor = FeatureMapExtractor(wavtok, device)

    # 載入訓練和驗證數據
    print("\n" + "=" * 60)
    print("Loading Train/Val Data")
    print("=" * 60)

    if not TRAIN_CACHE.exists():
        print(f"Error: Train cache not found at {TRAIN_CACHE}")
        return
    if not VAL_CACHE.exists():
        print(f"Error: Val cache not found at {VAL_CACHE}")
        return

    # 載入數據
    train_data = torch.load(TRAIN_CACHE)
    val_data = torch.load(VAL_CACHE)

    print(f"Train data type: {type(train_data)}")
    print(f"Val data type: {type(val_data)}")

    # 提取樣本
    max_samples = 30  # 限制樣本數量以加速

    def extract_audios(data, max_n, audio_base_dirs):
        """從數據中提取音檔，支援路徑載入"""
        import soundfile as sf
        import torchaudio

        audios = []
        audio_info = []  # 記錄音檔資訊

        if isinstance(data, list):
            for sample in data[:max_n * 3]:  # 多取一些以防找不到檔案
                if len(audios) >= max_n:
                    break

                if isinstance(sample, dict):
                    # 嘗試載入 noisy_audio（Student 看到的輸入）
                    if 'noisy_path' in sample:
                        noisy_path = sample['noisy_path']
                        audio_loaded = False

                        for base_dir in audio_base_dirs:
                            full_path = Path(base_dir) / noisy_path
                            if full_path.exists():
                                try:
                                    waveform, sr = sf.read(str(full_path))
                                    waveform = torch.from_numpy(waveform).float()
                                    if waveform.dim() == 1:
                                        waveform = waveform.unsqueeze(0)
                                    else:
                                        waveform = waveform.T
                                    if sr != 24000:
                                        resampler = torchaudio.transforms.Resample(sr, 24000)
                                        waveform = resampler(waveform)
                                    audios.append(waveform)
                                    audio_info.append({
                                        'path': str(full_path),
                                        'speaker': sample.get('speaker_id', 'unknown'),
                                        'content': sample.get('content_id', 'unknown'),
                                    })
                                    audio_loaded = True
                                    break
                                except Exception as e:
                                    print(f"Error loading {full_path}: {e}")

                        if not audio_loaded and len(audios) == 0:
                            print(f"  Could not find: {noisy_path}")

        return audios, audio_info

    # 可能的音檔目錄
    audio_base_dirs = [
        "/home/sbplab/ruizi/WavTokenize/data/raw/box",      # noisy (box)
        "/home/sbplab/ruizi/WavTokenize/data/raw/LDV",      # noisy (LDV)
        "/home/sbplab/ruizi/WavTokenize/data/clean/box2",   # clean
        "/home/sbplab/ruizi/WavTokenize/data/clean",
        "/home/sbplab/ruizi/WavTokenize/data/noisy",
        "/home/sbplab/ruizi/c_code/done/exp/data3",
    ]

    train_audios, train_info = extract_audios(train_data, max_samples, audio_base_dirs)
    val_audios, val_info = extract_audios(val_data, max_samples, audio_base_dirs)

    print(f"Train samples: {len(train_audios)}")
    print(f"Val samples: {len(val_audios)}")

    if len(train_audios) == 0 or len(val_audios) == 0:
        print("Error: No audio samples found!")
        print("Checking data structure...")
        if isinstance(train_data, list) and len(train_data) > 0:
            sample = train_data[0]
            print(f"First train sample keys: {sample.keys() if isinstance(sample, dict) else type(sample)}")
            if 'noisy_path' in sample:
                print(f"Looking for: {sample['noisy_path']}")
                print("Tried directories:")
                for d in audio_base_dirs:
                    print(f"  {d}")
        return

    # 提取 Train 特徵
    print("\n" + "=" * 60)
    print("Extracting Train Features")
    print("=" * 60)

    train_feature_maps = []
    for i, audio in enumerate(tqdm(train_audios, desc="Train")):
        fm = extractor.extract(audio)
        train_feature_maps.append(fm)

    # 提取 Val 特徵
    print("\n" + "=" * 60)
    print("Extracting Val Features")
    print("=" * 60)

    val_feature_maps = []
    for i, audio in enumerate(tqdm(val_audios, desc="Val")):
        fm = extractor.extract(audio)
        val_feature_maps.append(fm)

    # 計算統計量
    print("\n" + "=" * 60)
    print("Computing Statistics")
    print("=" * 60)

    train_stats = compute_feature_statistics(train_feature_maps)
    val_stats = compute_feature_statistics(val_feature_maps)

    # 計算分布距離
    distances = compute_distribution_distance(train_stats, val_stats)

    # 視覺化
    result = visualize_train_val_comparison(
        train_stats, val_stats, distances,
        OUTPUT_DIR / 'train_val_comparison.png'
    )

    # 保存結果
    save_result = {
        'train_stats': train_stats,
        'val_stats': val_stats,
        'distances': {k: dict(v) for k, v in distances.items()},
        'summary': result,
    }

    with open(OUTPUT_DIR / 'train_val_analysis.json', 'w') as f:
        json.dump(save_result, f, indent=2, default=str)

    # 打印總結
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\n各層組 Train-Val 差異:")
    for group_name, dist in result['group_distances'].items():
        status = "✓" if dist < 0.1 else "⚠️" if dist < 0.3 else "❌"
        print(f"  {group_name:12s}: {dist:.4f} {status}")

    print(f"\n總體平均差異: {result['avg_distance']:.4f}")

    if result['avg_distance'] < 0.1:
        print("\n診斷: Train/Val 特徵分布相似，模型應該能泛化")
    elif result['avg_distance'] < 0.3:
        print("\n診斷: Train/Val 有些差異，可能有輕微 distribution shift")
    else:
        print("\n診斷: Train/Val 差異較大，可能存在嚴重 distribution shift")

    print("\n" + "=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
