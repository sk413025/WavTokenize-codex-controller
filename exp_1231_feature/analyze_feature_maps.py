"""
exp_1231_feature: WavTokenizer Encoder Feature Map 分析

目標:
1. 分析原始 WavTokenizer 18層 conv 的 feature map
2. 比較不同說話者 (boy1 vs girl2) 說同一句話時，各層在看什麼
3. 分析 LoRA 訓練前後各層變化程度
4. 驗證 Train vs Val 是否關注相同特徵

作者: Claude Code
日期: 2024-12-31
"""

import sys
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

# 測試音檔
AUDIO_DIR = Path("/home/sbplab/ruizi/WavTokenize/data/clean/box2")
TEST_AUDIO_PAIRS = [
    ("nor_boy1_clean_001.wav", "nor_girl2_clean_001.wav"),
    ("nor_boy1_clean_010.wav", "nor_girl2_clean_010.wav"),
]

# 輸出目錄
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 18 層 conv 名稱
ENCODER_CONV_LAYERS = [
    "model.0.conv.conv",           # 輸入投影
    "model.1.block.1.conv.conv",   # ResBlock 1
    "model.1.block.3.conv.conv",
    "model.1.shortcut.conv.conv",
    "model.3.conv.conv",           # Downsample 1
    "model.4.block.1.conv.conv",   # ResBlock 2
    "model.4.block.3.conv.conv",
    "model.4.shortcut.conv.conv",
    "model.6.conv.conv",           # Downsample 2
    "model.7.block.1.conv.conv",   # ResBlock 3 (語義層)
    "model.7.block.3.conv.conv",
    "model.7.shortcut.conv.conv",
    "model.9.conv.conv",           # Downsample 3
    "model.10.block.1.conv.conv",  # ResBlock 4 (高階抽象)
    "model.10.block.3.conv.conv",
    "model.10.shortcut.conv.conv",
    "model.12.conv.conv",          # Downsample 4
    "model.15.conv.conv",          # 輸出投影
]

# 層分組 (用於分析)
LAYER_GROUPS = {
    'input': [0],                    # model.0
    'low_level': [1, 2, 3, 4],       # model.1, model.3
    'mid_level': [5, 6, 7, 8],       # model.4, model.6
    'semantic': [9, 10, 11, 12],     # model.7, model.9
    'abstract': [13, 14, 15, 16],    # model.10, model.12
    'output': [17],                  # model.15
}


class FeatureMapExtractor:
    """提取 WavTokenizer encoder 各層的 feature maps"""

    def __init__(self, model: WavTokenizer, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _get_layer(self, layer_name: str) -> nn.Module:
        """根據名稱獲取層"""
        encoder = self.model.feature_extractor.encodec.encoder
        parts = layer_name.split('.')
        module = encoder
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _create_hook(self, name: str):
        """創建 forward hook"""
        def hook(module, input, output):
            self.feature_maps[name] = output.detach().cpu()
        return hook

    def register_hooks(self):
        """註冊所有層的 hooks"""
        for name in ENCODER_CONV_LAYERS:
            layer = self._get_layer(name)
            hook = layer.register_forward_hook(self._create_hook(name))
            self.hooks.append(hook)

    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def extract(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取 feature maps

        Args:
            audio: (1, T) 音訊波形

        Returns:
            Dict[layer_name, feature_map]: 各層的 feature maps
        """
        self.feature_maps = {}
        self.register_hooks()

        try:
            audio = audio.to(self.device)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # (B, 1, T)

            # Forward pass
            bandwidth_id = torch.tensor([0]).to(self.device)
            _ = self.model.encode(audio, bandwidth_id=bandwidth_id)

        finally:
            self.remove_hooks()

        return self.feature_maps.copy()


def load_audio(path: Path, target_sr: int = 24000) -> torch.Tensor:
    """載入音訊並重採樣"""
    import soundfile as sf
    waveform, sr = sf.read(str(path))
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)
    else:
        waveform = waveform.T  # (C, T)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform


def compute_feature_statistics(feature_map: torch.Tensor) -> Dict[str, float]:
    """計算 feature map 的統計量

    Args:
        feature_map: (B, C, T) 或 (C, T)
    """
    if feature_map.dim() == 3:
        feature_map = feature_map.squeeze(0)  # (C, T)

    return {
        'mean': feature_map.mean().item(),
        'std': feature_map.std().item(),
        'max': feature_map.max().item(),
        'min': feature_map.min().item(),
        'l2_norm': feature_map.norm(2).item(),
        'sparsity': (feature_map.abs() < 0.01).float().mean().item(),
    }


def compute_feature_similarity(fm1: torch.Tensor, fm2: torch.Tensor) -> Dict[str, float]:
    """計算兩個 feature maps 的相似度

    用於比較:
    - 不同說話者說同一句話
    - 同一說話者的 clean vs noisy
    - Train vs Val 的特徵分布
    """
    if fm1.dim() == 3:
        fm1 = fm1.squeeze(0)
    if fm2.dim() == 3:
        fm2 = fm2.squeeze(0)

    # 對齊時間維度
    min_t = min(fm1.shape[1], fm2.shape[1])
    fm1 = fm1[:, :min_t]
    fm2 = fm2[:, :min_t]

    # Flatten
    fm1_flat = fm1.flatten()
    fm2_flat = fm2.flatten()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        fm1_flat.unsqueeze(0), fm2_flat.unsqueeze(0)
    ).item()

    # Pearson correlation
    fm1_centered = fm1_flat - fm1_flat.mean()
    fm2_centered = fm2_flat - fm2_flat.mean()
    correlation = (fm1_centered * fm2_centered).sum() / (
        fm1_centered.norm() * fm2_centered.norm() + 1e-8
    )

    # L2 distance (normalized)
    l2_dist = (fm1_flat - fm2_flat).norm() / (fm1_flat.norm() + 1e-8)

    return {
        'cosine_similarity': cos_sim,
        'correlation': correlation.item(),
        'l2_distance_normalized': l2_dist.item(),
    }


def visualize_feature_maps(
    feature_maps: Dict[str, torch.Tensor],
    title: str,
    output_path: Path,
):
    """視覺化 18 層的 feature maps

    每層顯示:
    - Feature map 熱力圖 (channels x time)
    - Channel activation 分布
    """
    n_layers = len(ENCODER_CONV_LAYERS)
    fig, axes = plt.subplots(6, 3, figsize=(18, 24))
    axes = axes.flatten()

    for i, layer_name in enumerate(ENCODER_CONV_LAYERS):
        if layer_name not in feature_maps:
            continue

        fm = feature_maps[layer_name]
        if fm.dim() == 3:
            fm = fm.squeeze(0)  # (C, T)

        # 熱力圖
        ax = axes[i]

        # 限制顯示的 channels (最多 64)
        n_channels = min(fm.shape[0], 64)
        fm_display = fm[:n_channels].numpy()

        im = ax.imshow(fm_display, aspect='auto', cmap='viridis')
        ax.set_title(f"L{i}: {layer_name.split('.')[-3]}", fontsize=8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')

        # 統計信息
        stats = compute_feature_statistics(fm)
        ax.text(0.02, 0.98, f"μ={stats['mean']:.2f}, σ={stats['std']:.2f}",
                transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def visualize_layer_comparison(
    fm1: Dict[str, torch.Tensor],
    fm2: Dict[str, torch.Tensor],
    label1: str,
    label2: str,
    output_path: Path,
):
    """比較兩個輸入在各層的特徵差異"""

    similarities = []
    layer_names_short = []

    for i, layer_name in enumerate(ENCODER_CONV_LAYERS):
        if layer_name not in fm1 or layer_name not in fm2:
            continue

        sim = compute_feature_similarity(fm1[layer_name], fm2[layer_name])
        similarities.append(sim)
        layer_names_short.append(f"L{i}")

    # 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = range(len(layer_names_short))

    # Cosine similarity
    cos_sims = [s['cosine_similarity'] for s in similarities]
    axes[0].bar(x, cos_sims, color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layer_names_short, rotation=45, fontsize=8)
    axes[0].set_title('Cosine Similarity')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=np.mean(cos_sims), color='red', linestyle='--', label=f'Mean: {np.mean(cos_sims):.3f}')
    axes[0].legend()

    # Correlation
    corrs = [s['correlation'] for s in similarities]
    axes[1].bar(x, corrs, color='forestgreen')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_names_short, rotation=45, fontsize=8)
    axes[1].set_title('Pearson Correlation')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=np.mean(corrs), color='red', linestyle='--', label=f'Mean: {np.mean(corrs):.3f}')
    axes[1].legend()

    # L2 distance
    l2_dists = [s['l2_distance_normalized'] for s in similarities]
    axes[2].bar(x, l2_dists, color='coral')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(layer_names_short, rotation=45, fontsize=8)
    axes[2].set_title('Normalized L2 Distance')
    axes[2].axhline(y=np.mean(l2_dists), color='red', linestyle='--', label=f'Mean: {np.mean(l2_dists):.3f}')
    axes[2].legend()

    plt.suptitle(f'{label1} vs {label2}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    return similarities


def analyze_speaker_invariance(
    extractor: FeatureMapExtractor,
    output_dir: Path,
):
    """
    分析: 不同說話者說同一句話，WavTokenizer 各層在看什麼？

    假設:
    - 淺層 (model.0-4): 關注聲學特徵 (音高、語調) → 說話者差異大
    - 深層 (model.7-15): 關注語義內容 → 說話者不變性高
    """
    print("\n" + "="*60)
    print("分析 1: Speaker Invariance (說話者不變性)")
    print("="*60)

    results = []

    for boy_file, girl_file in TEST_AUDIO_PAIRS:
        boy_path = AUDIO_DIR / boy_file
        girl_path = AUDIO_DIR / girl_file

        if not boy_path.exists() or not girl_path.exists():
            print(f"Skip: {boy_file} or {girl_file} not found")
            continue

        print(f"\n比較: {boy_file} vs {girl_file}")

        # 載入音訊
        boy_audio = load_audio(boy_path)
        girl_audio = load_audio(girl_path)

        # 提取 feature maps
        boy_fm = extractor.extract(boy_audio)
        girl_fm = extractor.extract(girl_audio)

        # 視覺化
        sentence_id = boy_file.split('_')[-1].replace('.wav', '')

        visualize_feature_maps(
            boy_fm,
            f"boy1 - Sentence {sentence_id}",
            output_dir / f"feature_maps_boy1_{sentence_id}.png"
        )

        visualize_feature_maps(
            girl_fm,
            f"girl2 - Sentence {sentence_id}",
            output_dir / f"feature_maps_girl2_{sentence_id}.png"
        )

        # 比較
        sims = visualize_layer_comparison(
            boy_fm, girl_fm,
            f"boy1_{sentence_id}", f"girl2_{sentence_id}",
            output_dir / f"comparison_boy1_girl2_{sentence_id}.png"
        )

        results.append({
            'sentence_id': sentence_id,
            'similarities': sims,
        })

    # 總結分析
    if results:
        summarize_speaker_invariance(results, output_dir)

    return results


def summarize_speaker_invariance(results: List[Dict], output_dir: Path):
    """總結說話者不變性分析結果"""

    # 平均各層的相似度
    n_layers = len(ENCODER_CONV_LAYERS)
    avg_cos_sim = np.zeros(n_layers)

    for r in results:
        for i, sim in enumerate(r['similarities']):
            avg_cos_sim[i] += sim['cosine_similarity']

    avg_cos_sim /= len(results)

    # 分組統計
    group_stats = {}
    for group_name, layer_indices in LAYER_GROUPS.items():
        group_sims = [avg_cos_sim[i] for i in layer_indices if i < n_layers]
        group_stats[group_name] = np.mean(group_sims)

    # 繪製總結圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 各層相似度
    x = range(n_layers)
    colors = []
    for i in range(n_layers):
        for group_name, indices in LAYER_GROUPS.items():
            if i in indices:
                if group_name == 'input':
                    colors.append('gray')
                elif group_name == 'low_level':
                    colors.append('lightblue')
                elif group_name == 'mid_level':
                    colors.append('skyblue')
                elif group_name == 'semantic':
                    colors.append('steelblue')
                elif group_name == 'abstract':
                    colors.append('navy')
                else:
                    colors.append('darkblue')
                break

    axes[0].bar(x, avg_cos_sim, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"L{i}" for i in x], rotation=45, fontsize=8)
    axes[0].set_title('Speaker Invariance by Layer\n(Cosine Similarity: boy1 vs girl2)')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_ylim(0, 1)

    # 分組統計
    group_names = list(group_stats.keys())
    group_values = [group_stats[g] for g in group_names]
    group_colors = ['gray', 'lightblue', 'skyblue', 'steelblue', 'navy', 'darkblue']

    axes[1].bar(group_names, group_values, color=group_colors)
    axes[1].set_title('Speaker Invariance by Layer Group')
    axes[1].set_ylabel('Average Cosine Similarity')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(group_values):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'speaker_invariance_summary.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'speaker_invariance_summary.png'}")

    # 文字總結
    print("\n" + "="*60)
    print("Speaker Invariance Summary")
    print("="*60)
    print("\n各層組平均相似度 (boy1 vs girl2 說同一句話):")
    for group_name, sim in group_stats.items():
        print(f"  {group_name:12s}: {sim:.4f}")

    print("\n解讀:")
    if group_stats['semantic'] > group_stats['low_level']:
        print("  ✓ 深層 (semantic/abstract) 比淺層更具說話者不變性")
        print("    → 深層在提取語義/內容特徵，淺層在處理聲學特徵")
    else:
        print("  ✗ 深層沒有展現預期的說話者不變性")
        print("    → 可能需要進一步調查")


def main():
    """主程式"""
    print("="*60)
    print("exp_1231_feature: WavTokenizer Feature Map Analysis")
    print("="*60)

    # 載入模型
    print("\nLoading WavTokenizer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wavtok = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)
    wavtok.eval()
    wavtok = wavtok.to(device)

    # 創建 extractor
    extractor = FeatureMapExtractor(wavtok, device)

    # 分析 1: 說話者不變性
    results = analyze_speaker_invariance(extractor, OUTPUT_DIR)

    # 保存結果
    with open(OUTPUT_DIR / 'analysis_results.json', 'w') as f:
        # Convert numpy to list for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
