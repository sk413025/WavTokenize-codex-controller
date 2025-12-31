"""
exp_1231_feature: 模型健康檢測 (Model Health Check)

目的: 診斷 LoRA 訓練後模型是否「學歪了」

核心問題:
- Train accuracy 上升但 Val accuracy 沒有
- 模型是否關注了錯誤的特徵？

分析內容:
1. Noisy vs Clean: 原始 WavTokenizer 各層對噪音的敏感度
2. Teacher vs Student: LoRA 訓練後各層的變化
3. Train vs Val: 兩個 split 的特徵分布是否一致

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

# 測試音檔目錄
CLEAN_DIR = Path("/home/sbplab/ruizi/WavTokenize/data/clean/box2")
NOISY_DIR = Path("/home/sbplab/ruizi/WavTokenize/data/noisy")

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

# 層分組
LAYER_GROUPS = {
    'input': [0],
    'low_level': [1, 2, 3, 4],
    'mid_level': [5, 6, 7, 8],
    'semantic': [9, 10, 11, 12],
    'abstract': [13, 14, 15, 16],
    'output': [17],
}

# 層的角色定義（降噪任務視角）
LAYER_ROLES = {
    'input': '接收原始波形，對噪音最敏感',
    'low_level': '提取聲學特徵（頻譜、音色），噪音影響大',
    'mid_level': '整合局部特徵，開始抽象化',
    'semantic': '語義編碼，對噪音較魯棒',
    'abstract': '高階語義表示，理論上對噪音不敏感',
    'output': '投影到 VQ 空間',
}


class FeatureMapExtractor:
    """提取 WavTokenizer encoder 各層的 feature maps"""

    def __init__(self, model: WavTokenizer, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _get_layer(self, layer_name: str) -> nn.Module:
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
        def hook(module, input, output):
            self.feature_maps[name] = output.detach().cpu()
        return hook

    def register_hooks(self):
        for name in ENCODER_CONV_LAYERS:
            layer = self._get_layer(name)
            hook = layer.register_forward_hook(self._create_hook(name))
            self.hooks.append(hook)

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
            _ = self.model.encode(audio, bandwidth_id=bandwidth_id)

        finally:
            self.remove_hooks()

        return self.feature_maps.copy()


def load_audio(path: Path, target_sr: int = 24000) -> torch.Tensor:
    """載入音訊"""
    import soundfile as sf
    import torchaudio
    waveform, sr = sf.read(str(path))
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform


def compute_feature_similarity(fm1: torch.Tensor, fm2: torch.Tensor) -> Dict[str, float]:
    """計算兩個 feature maps 的相似度"""
    if fm1.dim() == 3:
        fm1 = fm1.squeeze(0)
    if fm2.dim() == 3:
        fm2 = fm2.squeeze(0)

    min_t = min(fm1.shape[1], fm2.shape[1])
    fm1 = fm1[:, :min_t]
    fm2 = fm2[:, :min_t]

    fm1_flat = fm1.flatten()
    fm2_flat = fm2.flatten()

    cos_sim = torch.nn.functional.cosine_similarity(
        fm1_flat.unsqueeze(0), fm2_flat.unsqueeze(0)
    ).item()

    fm1_centered = fm1_flat - fm1_flat.mean()
    fm2_centered = fm2_flat - fm2_flat.mean()
    correlation = (fm1_centered * fm2_centered).sum() / (
        fm1_centered.norm() * fm2_centered.norm() + 1e-8
    )

    l2_dist = (fm1_flat - fm2_flat).norm() / (fm1_flat.norm() + 1e-8)

    return {
        'cosine_similarity': cos_sim,
        'correlation': correlation.item(),
        'l2_distance_normalized': l2_dist.item(),
    }


def find_audio_pairs() -> List[Tuple[Path, Path]]:
    """找到 clean/noisy 配對的音檔"""
    pairs = []

    # 嘗試不同的 noisy 目錄結構
    noisy_dirs = [
        NOISY_DIR,
        NOISY_DIR / "box2",
        Path("/home/sbplab/ruizi/WavTokenize/data/noisy_audio"),
    ]

    for clean_file in CLEAN_DIR.glob("*.wav"):
        # 從 clean 檔名推測 noisy 檔名
        # nor_boy1_clean_001.wav -> nor_boy1_noisy_001.wav 或類似
        stem = clean_file.stem

        for noisy_dir in noisy_dirs:
            if not noisy_dir.exists():
                continue

            # 嘗試不同的命名模式
            possible_noisy = [
                noisy_dir / clean_file.name.replace('_clean_', '_noisy_'),
                noisy_dir / clean_file.name.replace('clean', 'noisy'),
                noisy_dir / f"{stem}_noisy.wav",
            ]

            for noisy_path in possible_noisy:
                if noisy_path.exists():
                    pairs.append((clean_file, noisy_path))
                    break

    return pairs


def analyze_noise_sensitivity(
    extractor: FeatureMapExtractor,
    clean_audio: torch.Tensor,
    noisy_audio: torch.Tensor,
    sample_name: str,
) -> Dict:
    """
    分析 1: 各層對噪音的敏感度

    問題: 噪音主要影響哪些層？

    預期（健康模型）:
    - 淺層 (L0-L4): 對噪音敏感 → similarity 低
    - 深層 (L13-L17): 對噪音魯棒 → similarity 高

    如果反過來，表示模型「學歪了」
    """
    clean_fm = extractor.extract(clean_audio)
    noisy_fm = extractor.extract(noisy_audio)

    layer_sims = []
    for i, layer_name in enumerate(ENCODER_CONV_LAYERS):
        sim = compute_feature_similarity(clean_fm[layer_name], noisy_fm[layer_name])
        sim['layer_idx'] = i
        sim['layer_name'] = layer_name
        layer_sims.append(sim)

    return {
        'sample_name': sample_name,
        'layer_similarities': layer_sims,
    }


def analyze_teacher_student_divergence(
    teacher_extractor: FeatureMapExtractor,
    student_extractor: FeatureMapExtractor,
    noisy_audio: torch.Tensor,
    clean_audio: torch.Tensor,
    sample_name: str,
) -> Dict:
    """
    分析 2: Teacher vs Student 的特徵差異

    設置:
    - Teacher: 處理 clean audio
    - Student: 處理 noisy audio（訓練目標是輸出和 Teacher 一樣）

    問題: Student 在哪些層偏離了 Teacher？

    預期（健康模型）:
    - 深層 similarity 高 → Student 學會產生接近 Teacher 的語義表示
    - 淺層 similarity 可能較低 → 輸入不同，但這是可接受的

    如果深層 similarity 也很低，表示 Student「沒學到」
    """
    teacher_fm = teacher_extractor.extract(clean_audio)
    student_fm = student_extractor.extract(noisy_audio)

    layer_sims = []
    for i, layer_name in enumerate(ENCODER_CONV_LAYERS):
        sim = compute_feature_similarity(teacher_fm[layer_name], student_fm[layer_name])
        sim['layer_idx'] = i
        sim['layer_name'] = layer_name
        layer_sims.append(sim)

    return {
        'sample_name': sample_name,
        'layer_similarities': layer_sims,
    }


def visualize_noise_sensitivity(results: List[Dict], output_path: Path):
    """視覺化噪音敏感度分析結果"""
    n_layers = len(ENCODER_CONV_LAYERS)

    # 平均各樣本的結果
    avg_cos_sim = np.zeros(n_layers)
    for r in results:
        for sim in r['layer_similarities']:
            avg_cos_sim[sim['layer_idx']] += sim['cosine_similarity']
    avg_cos_sim /= len(results)

    # 計算分組平均
    group_stats = {}
    for group_name, indices in LAYER_GROUPS.items():
        group_sims = [avg_cos_sim[i] for i in indices]
        group_stats[group_name] = np.mean(group_sims)

    # 繪圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 各層 cosine similarity
    ax = axes[0, 0]
    colors = []
    for i in range(n_layers):
        for gname, indices in LAYER_GROUPS.items():
            if i in indices:
                color_map = {
                    'input': '#808080',
                    'low_level': '#ADD8E6',
                    'mid_level': '#87CEEB',
                    'semantic': '#4682B4',
                    'abstract': '#000080',
                    'output': '#191970',
                }
                colors.append(color_map[gname])
                break

    bars = ax.bar(range(n_layers), avg_cos_sim, color=colors)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    ax.set_title('Noise Sensitivity by Layer\n(Clean vs Noisy Cosine Similarity)', fontsize=12)
    ax.set_ylabel('Cosine Similarity\n(高 = 對噪音魯棒)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='中間值')

    # 添加趨勢線
    z = np.polyfit(range(n_layers), avg_cos_sim, 1)
    p = np.poly1d(z)
    ax.plot(range(n_layers), p(range(n_layers)), "r--", alpha=0.8, label=f'趨勢 (斜率={z[0]:.4f})')
    ax.legend()

    # 2. 分組統計
    ax = axes[0, 1]
    group_names = list(group_stats.keys())
    group_values = [group_stats[g] for g in group_names]
    group_colors = ['#808080', '#ADD8E6', '#87CEEB', '#4682B4', '#000080', '#191970']

    bars = ax.bar(group_names, group_values, color=group_colors)
    ax.set_title('Noise Sensitivity by Layer Group', fontsize=12)
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_ylim(0, 1)
    for i, v in enumerate(group_values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    # 3. 健康指標
    ax = axes[1, 0]

    # 計算健康指標
    shallow_sim = np.mean([avg_cos_sim[i] for i in LAYER_GROUPS['low_level']])
    deep_sim = np.mean([avg_cos_sim[i] for i in LAYER_GROUPS['abstract']])

    health_metrics = {
        '淺層噪音敏感度\n(期望: 低)': 1 - shallow_sim,  # 反轉，敏感度高 = similarity 低
        '深層魯棒性\n(期望: 高)': deep_sim,
        '深淺差異\n(期望: 正)': deep_sim - shallow_sim,
    }

    colors = ['#FF6B6B' if v < 0.3 else '#4ECDC4' if v > 0.5 else '#FFE66D'
              for v in health_metrics.values()]

    bars = ax.barh(list(health_metrics.keys()), list(health_metrics.values()), color=colors)
    ax.set_xlim(-0.5, 1)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title('Model Health Indicators (Noise Handling)', fontsize=12)

    for i, (k, v) in enumerate(health_metrics.items()):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)

    # 4. 診斷結論
    ax = axes[1, 1]
    ax.axis('off')

    diagnosis = []
    diagnosis.append("=" * 40)
    diagnosis.append("模型健康檢測報告 (噪音處理)")
    diagnosis.append("=" * 40)

    # 判斷 1: 淺層是否對噪音敏感
    if shallow_sim < 0.5:
        diagnosis.append("\n[✓] 淺層對噪音敏感 (正常)")
        diagnosis.append(f"    淺層 similarity = {shallow_sim:.3f}")
    else:
        diagnosis.append("\n[!] 淺層對噪音不敏感 (異常)")
        diagnosis.append(f"    淺層 similarity = {shallow_sim:.3f}")
        diagnosis.append("    → 淺層可能沒有正確處理聲學特徵")

    # 判斷 2: 深層是否對噪音魯棒
    if deep_sim > 0.5:
        diagnosis.append("\n[✓] 深層對噪音魯棒 (正常)")
        diagnosis.append(f"    深層 similarity = {deep_sim:.3f}")
    else:
        diagnosis.append("\n[!] 深層對噪音敏感 (異常)")
        diagnosis.append(f"    深層 similarity = {deep_sim:.3f}")
        diagnosis.append("    → 深層語義表示可能受噪音污染")

    # 判斷 3: 深淺差異
    diff = deep_sim - shallow_sim
    if diff > 0.1:
        diagnosis.append(f"\n[✓] 深淺層差異正常 (diff = {diff:.3f})")
        diagnosis.append("    → 模型有正確的層級結構")
    else:
        diagnosis.append(f"\n[!] 深淺層差異異常 (diff = {diff:.3f})")
        diagnosis.append("    → 可能所有層都在處理相同特徵")

    ax.text(0.05, 0.95, '\n'.join(diagnosis), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    return {
        'avg_layer_similarities': avg_cos_sim.tolist(),
        'group_stats': group_stats,
        'health_metrics': health_metrics,
    }


def main():
    """主程式"""
    print("=" * 60)
    print("Model Health Check: 模型健康檢測")
    print("=" * 60)

    # 載入模型
    print("\nLoading WavTokenizer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wavtok = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)
    wavtok.eval()
    wavtok = wavtok.to(device)

    extractor = FeatureMapExtractor(wavtok, device)

    # 分析 1: 噪音敏感度
    print("\n" + "=" * 60)
    print("分析 1: Noise Sensitivity (Clean vs Noisy)")
    print("=" * 60)

    # 找配對音檔
    pairs = find_audio_pairs()

    if not pairs:
        print("找不到 clean/noisy 配對，嘗試使用合成噪音...")
        # 使用合成噪音進行測試
        clean_files = list(CLEAN_DIR.glob("*.wav"))[:5]
        noise_sensitivity_results = []

        for clean_path in clean_files:
            clean_audio = load_audio(clean_path)

            # 合成噪音
            noise_level = 0.1
            noise = torch.randn_like(clean_audio) * noise_level
            noisy_audio = clean_audio + noise

            result = analyze_noise_sensitivity(
                extractor, clean_audio, noisy_audio,
                f"{clean_path.stem}_synthetic_noise"
            )
            noise_sensitivity_results.append(result)
            print(f"  Analyzed: {clean_path.name}")
    else:
        noise_sensitivity_results = []
        for clean_path, noisy_path in pairs[:10]:  # 限制數量
            clean_audio = load_audio(clean_path)
            noisy_audio = load_audio(noisy_path)

            result = analyze_noise_sensitivity(
                extractor, clean_audio, noisy_audio,
                f"{clean_path.stem}_vs_{noisy_path.stem}"
            )
            noise_sensitivity_results.append(result)
            print(f"  Analyzed: {clean_path.name} vs {noisy_path.name}")

    # 視覺化
    if noise_sensitivity_results:
        health_report = visualize_noise_sensitivity(
            noise_sensitivity_results,
            OUTPUT_DIR / 'noise_sensitivity_analysis.png'
        )

        # 保存結果
        with open(OUTPUT_DIR / 'noise_sensitivity_results.json', 'w') as f:
            json.dump({
                'results': noise_sensitivity_results,
                'health_report': {k: (v.tolist() if isinstance(v, np.ndarray) else
                                     (dict(v) if isinstance(v, dict) else v))
                                 for k, v in health_report.items()}
            }, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
