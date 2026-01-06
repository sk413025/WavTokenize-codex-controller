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

# 所有說話者
SPEAKERS_BOY = ['boy1', 'boy3', 'boy4', 'boy5', 'boy6', 'boy7', 'boy8', 'boy9', 'boy10']
SPEAKERS_GIRL = ['girl2', 'girl3', 'girl4', 'girl6', 'girl7', 'girl8', 'girl9', 'girl10', 'girl11']
ALL_SPEAKERS = SPEAKERS_BOY + SPEAKERS_GIRL

# 測試句子 (使用多個句子增加信心度)
TEST_SENTENCES = ['001', '002', '003', '010', '020', '030']

# 舊的設定 (保留兼容)
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


def analyze_speaker_invariance_multi(
    extractor: FeatureMapExtractor,
    output_dir: Path,
    n_speakers: int = 6,
    n_sentences: int = 5,
):
    """
    擴展分析: 多位說話者比較，增加統計信心度

    比較策略:
    1. 跨性別比較 (boy vs girl)
    2. 同性別比較 (boy vs boy, girl vs girl)
    3. 統計 mean ± std

    Args:
        n_speakers: 每個性別使用的說話者數量
        n_sentences: 使用的句子數量
    """
    print("\n" + "="*60)
    print("分析 1b: Multi-Speaker Invariance Analysis")
    print(f"  使用說話者: {n_speakers} boys + {n_speakers} girls")
    print(f"  使用句子數: {n_sentences}")
    print("="*60)

    # 選擇說話者
    selected_boys = SPEAKERS_BOY[:n_speakers]
    selected_girls = SPEAKERS_GIRL[:n_speakers]
    sentences = TEST_SENTENCES[:n_sentences]

    print(f"\n選擇的說話者:")
    print(f"  Boys: {selected_boys}")
    print(f"  Girls: {selected_girls}")
    print(f"  Sentences: {sentences}")

    # 儲存所有比較結果
    cross_gender_results = []  # boy vs girl
    same_gender_boy_results = []  # boy vs boy
    same_gender_girl_results = []  # girl vs girl

    n_layers = len(ENCODER_CONV_LAYERS)

    # 1. 跨性別比較
    print("\n--- 跨性別比較 (Boy vs Girl) ---")
    for sentence in sentences:
        for boy in selected_boys:
            for girl in selected_girls:
                boy_file = f"nor_{boy}_clean_{sentence}.wav"
                girl_file = f"nor_{girl}_clean_{sentence}.wav"
                boy_path = AUDIO_DIR / boy_file
                girl_path = AUDIO_DIR / girl_file

                if not boy_path.exists() or not girl_path.exists():
                    continue

                boy_audio = load_audio(boy_path)
                girl_audio = load_audio(girl_path)
                boy_fm = extractor.extract(boy_audio)
                girl_fm = extractor.extract(girl_audio)

                sims = []
                for layer_name in ENCODER_CONV_LAYERS:
                    if layer_name in boy_fm and layer_name in girl_fm:
                        sim = compute_feature_similarity(boy_fm[layer_name], girl_fm[layer_name])
                        sims.append(sim['cosine_similarity'])

                if len(sims) == n_layers:
                    cross_gender_results.append(sims)

    print(f"  完成 {len(cross_gender_results)} 組比較")

    # 2. 同性別比較 - Boys
    print("\n--- 同性別比較 (Boy vs Boy) ---")
    for sentence in sentences:
        for i, boy1 in enumerate(selected_boys):
            for boy2 in selected_boys[i+1:]:
                boy1_file = f"nor_{boy1}_clean_{sentence}.wav"
                boy2_file = f"nor_{boy2}_clean_{sentence}.wav"
                boy1_path = AUDIO_DIR / boy1_file
                boy2_path = AUDIO_DIR / boy2_file

                if not boy1_path.exists() or not boy2_path.exists():
                    continue

                boy1_audio = load_audio(boy1_path)
                boy2_audio = load_audio(boy2_path)
                boy1_fm = extractor.extract(boy1_audio)
                boy2_fm = extractor.extract(boy2_audio)

                sims = []
                for layer_name in ENCODER_CONV_LAYERS:
                    if layer_name in boy1_fm and layer_name in boy2_fm:
                        sim = compute_feature_similarity(boy1_fm[layer_name], boy2_fm[layer_name])
                        sims.append(sim['cosine_similarity'])

                if len(sims) == n_layers:
                    same_gender_boy_results.append(sims)

    print(f"  完成 {len(same_gender_boy_results)} 組比較")

    # 3. 同性別比較 - Girls
    print("\n--- 同性別比較 (Girl vs Girl) ---")
    for sentence in sentences:
        for i, girl1 in enumerate(selected_girls):
            for girl2 in selected_girls[i+1:]:
                girl1_file = f"nor_{girl1}_clean_{sentence}.wav"
                girl2_file = f"nor_{girl2}_clean_{sentence}.wav"
                girl1_path = AUDIO_DIR / girl1_file
                girl2_path = AUDIO_DIR / girl2_file

                if not girl1_path.exists() or not girl2_path.exists():
                    continue

                girl1_audio = load_audio(girl1_path)
                girl2_audio = load_audio(girl2_path)
                girl1_fm = extractor.extract(girl1_audio)
                girl2_fm = extractor.extract(girl2_audio)

                sims = []
                for layer_name in ENCODER_CONV_LAYERS:
                    if layer_name in girl1_fm and layer_name in girl2_fm:
                        sim = compute_feature_similarity(girl1_fm[layer_name], girl2_fm[layer_name])
                        sims.append(sim['cosine_similarity'])

                if len(sims) == n_layers:
                    same_gender_girl_results.append(sims)

    print(f"  完成 {len(same_gender_girl_results)} 組比較")

    # 統計分析
    cross_gender = np.array(cross_gender_results)
    same_boy = np.array(same_gender_boy_results)
    same_girl = np.array(same_gender_girl_results)

    # 繪製結果
    visualize_multi_speaker_results(
        cross_gender, same_boy, same_girl,
        output_dir / 'multi_speaker_analysis.png'
    )

    # 儲存詳細結果
    results = {
        'n_speakers_per_gender': n_speakers,
        'n_sentences': n_sentences,
        'n_cross_gender_comparisons': len(cross_gender_results),
        'n_same_boy_comparisons': len(same_gender_boy_results),
        'n_same_girl_comparisons': len(same_gender_girl_results),
        'cross_gender_mean': cross_gender.mean(axis=0).tolist() if len(cross_gender) > 0 else [],
        'cross_gender_std': cross_gender.std(axis=0).tolist() if len(cross_gender) > 0 else [],
        'same_boy_mean': same_boy.mean(axis=0).tolist() if len(same_boy) > 0 else [],
        'same_boy_std': same_boy.std(axis=0).tolist() if len(same_boy) > 0 else [],
        'same_girl_mean': same_girl.mean(axis=0).tolist() if len(same_girl) > 0 else [],
        'same_girl_std': same_girl.std(axis=0).tolist() if len(same_girl) > 0 else [],
    }

    # 分組統計
    print("\n" + "="*60)
    print("Multi-Speaker Analysis Summary")
    print("="*60)

    print(f"\n比較組數:")
    print(f"  跨性別 (Boy vs Girl): {len(cross_gender_results)}")
    print(f"  同性別 (Boy vs Boy):  {len(same_gender_boy_results)}")
    print(f"  同性別 (Girl vs Girl): {len(same_gender_girl_results)}")

    if len(cross_gender) > 0:
        print("\n各層組平均相似度 ± 標準差:")
        for group_name, layer_indices in LAYER_GROUPS.items():
            indices = [i for i in layer_indices if i < n_layers]
            cross_mean = cross_gender[:, indices].mean()
            cross_std = cross_gender[:, indices].std()
            same_mean = np.concatenate([same_boy[:, indices], same_girl[:, indices]]).mean() if len(same_boy) > 0 and len(same_girl) > 0 else 0
            same_std = np.concatenate([same_boy[:, indices], same_girl[:, indices]]).std() if len(same_boy) > 0 and len(same_girl) > 0 else 0

            print(f"  {group_name:12s}: Cross={cross_mean:.4f}±{cross_std:.4f}, Same={same_mean:.4f}±{same_std:.4f}")

    return results


def visualize_multi_speaker_results(
    cross_gender: np.ndarray,
    same_boy: np.ndarray,
    same_girl: np.ndarray,
    output_path: Path,
):
    """視覺化多說話者分析結果"""
    n_layers = cross_gender.shape[1] if len(cross_gender) > 0 else 18

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    x = np.arange(n_layers)
    layer_labels = [f"L{i}" for i in range(n_layers)]

    # 1. 各層相似度 (mean ± std)
    ax = axes[0, 0]
    if len(cross_gender) > 0:
        cross_mean = cross_gender.mean(axis=0)
        cross_std = cross_gender.std(axis=0)
        ax.fill_between(x, cross_mean - cross_std, cross_mean + cross_std, alpha=0.3, color='red')
        ax.plot(x, cross_mean, 'r-o', label=f'Cross-Gender (n={len(cross_gender)})', markersize=4)

    if len(same_boy) > 0:
        same_boy_mean = same_boy.mean(axis=0)
        same_boy_std = same_boy.std(axis=0)
        ax.fill_between(x, same_boy_mean - same_boy_std, same_boy_mean + same_boy_std, alpha=0.2, color='blue')
        ax.plot(x, same_boy_mean, 'b-s', label=f'Boy vs Boy (n={len(same_boy)})', markersize=4)

    if len(same_girl) > 0:
        same_girl_mean = same_girl.mean(axis=0)
        same_girl_std = same_girl.std(axis=0)
        ax.fill_between(x, same_girl_mean - same_girl_std, same_girl_mean + same_girl_std, alpha=0.2, color='green')
        ax.plot(x, same_girl_mean, 'g-^', label=f'Girl vs Girl (n={len(same_girl)})', markersize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, fontsize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Speaker Similarity by Layer (mean ± std)')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 2. 分組統計 (bar chart)
    ax = axes[0, 1]
    group_names = list(LAYER_GROUPS.keys())
    bar_width = 0.25
    x_groups = np.arange(len(group_names))

    for i, (data, label, color) in enumerate([
        (cross_gender, 'Cross-Gender', 'red'),
        (same_boy, 'Boy vs Boy', 'blue'),
        (same_girl, 'Girl vs Girl', 'green'),
    ]):
        if len(data) > 0:
            group_means = []
            group_stds = []
            for group_name, indices in LAYER_GROUPS.items():
                valid_indices = [idx for idx in indices if idx < n_layers]
                group_means.append(data[:, valid_indices].mean())
                group_stds.append(data[:, valid_indices].std())

            ax.bar(x_groups + i * bar_width, group_means, bar_width,
                   yerr=group_stds, capsize=3, label=label, color=color, alpha=0.7)

    ax.set_xticks(x_groups + bar_width)
    ax.set_xticklabels(group_names, rotation=45, fontsize=9)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Layer Group Statistics')
    ax.legend()
    ax.set_ylim(0, 1)

    # 3. 信心區間 (boxplot for cross-gender)
    ax = axes[1, 0]
    if len(cross_gender) > 0:
        bp = ax.boxplot([cross_gender[:, i] for i in range(n_layers)],
                        labels=layer_labels, patch_artist=True)
        colors = []
        for i in range(n_layers):
            for group_name, indices in LAYER_GROUPS.items():
                if i in indices:
                    if group_name == 'input':
                        colors.append('lightgray')
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
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cross-Gender Similarity Distribution')
        ax.tick_params(axis='x', rotation=45)

    # 4. 統計摘要
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "Statistical Summary\n" + "="*40 + "\n\n"

    if len(cross_gender) > 0:
        summary_text += f"Cross-Gender Comparisons: {len(cross_gender)}\n"
        summary_text += f"Same-Gender Boy: {len(same_boy)}\n"
        summary_text += f"Same-Gender Girl: {len(same_girl)}\n\n"

        summary_text += "Layer Group Analysis:\n"
        summary_text += "-"*40 + "\n"

        for group_name, indices in LAYER_GROUPS.items():
            valid_indices = [i for i in indices if i < n_layers]
            cross_val = cross_gender[:, valid_indices].mean()
            same_val = np.concatenate([same_boy[:, valid_indices], same_girl[:, valid_indices]]).mean() if len(same_boy) > 0 and len(same_girl) > 0 else 0

            summary_text += f"{group_name:12s}: Cross={cross_val:.3f}, Same={same_val:.3f}\n"

        # 結論
        summary_text += "\n" + "="*40 + "\n"
        summary_text += "Interpretation:\n"

        # 比較 semantic 和 low_level
        semantic_idx = [i for i in LAYER_GROUPS['semantic'] if i < n_layers]
        low_level_idx = [i for i in LAYER_GROUPS['low_level'] if i < n_layers]

        semantic_sim = cross_gender[:, semantic_idx].mean()
        low_level_sim = cross_gender[:, low_level_idx].mean()

        if semantic_sim > low_level_sim:
            summary_text += "✓ Deep layers show HIGHER speaker invariance\n"
            summary_text += "  → Confirms semantic/content extraction\n"
        else:
            summary_text += "✗ Deep layers show LOWER speaker invariance\n"
            summary_text += "  → Unexpected pattern\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved: {output_path}")


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

    # 分析 1a: 說話者不變性 (原始 2 對)
    results = analyze_speaker_invariance(extractor, OUTPUT_DIR)

    # 分析 1b: 多說話者分析 (增加統計信心度)
    # 使用 6 位說話者 x 5 句話 = 更多比較組合
    multi_results = analyze_speaker_invariance_multi(
        extractor, OUTPUT_DIR,
        n_speakers=6,  # 每個性別 6 位
        n_sentences=5,  # 5 句話
    )

    # 保存結果
    all_results = {
        'original_analysis': results,
        'multi_speaker_analysis': multi_results,
    }

    with open(OUTPUT_DIR / 'analysis_results.json', 'w') as f:
        # Convert numpy to list for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        json.dump(all_results, f, indent=2, default=convert)

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
