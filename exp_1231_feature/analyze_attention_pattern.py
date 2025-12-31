"""
exp_1231_feature: Train/Val Attention Pattern 分析

真正的問題：Train 和 Val 是否「關注同樣類型的特徵」？

類比：
- 圖像分類：Train 看貓輪廓，Val 也應該看貓輪廓（只是不同的貓）
- 降噪任務：Train 看噪音相關特徵，Val 也應該看噪音相關特徵

分析方法：
1. 用 LoRA Student 分別處理 Train/Val 的 noisy audio
2. 比較 feature activation 的「pattern」而非絕對值
3. 使用 Centered Kernel Alignment (CKA) 或相關指標

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

sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, str(Path(__file__).parent.parent))

from decoder.pretrained import WavTokenizer

# ============================================================
# 配置
# ============================================================

WAVTOK_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

TRAIN_CACHE = Path("/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt")
VAL_CACHE = Path("/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt")

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _get_encoder(self):
        if hasattr(self.model, 'feature_extractor'):
            return self.model.feature_extractor.encodec.encoder
        elif hasattr(self.model, 'student'):
            return self.model.student.feature_extractor.encodec.encoder
        else:
            raise ValueError("Cannot find encoder")

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
            except:
                pass

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


def compute_activation_pattern(fm: torch.Tensor) -> torch.Tensor:
    """
    計算 activation pattern（關注模式）

    不是比較絕對值，而是比較「哪些 channel 被激活」
    """
    if fm.dim() == 3:
        fm = fm.squeeze(0)  # (C, T)

    # 計算每個 channel 的平均激活程度
    channel_activation = fm.abs().mean(dim=1)  # (C,)

    # 標準化為比例（哪些 channel 相對更重要）
    pattern = channel_activation / (channel_activation.sum() + 1e-8)

    return pattern


def compute_pattern_similarity(pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
    """
    計算兩個 activation pattern 的相似度

    使用 cosine similarity - 比較「關注的相對分布」
    """
    # 確保維度相同
    min_len = min(len(pattern1), len(pattern2))
    p1 = pattern1[:min_len]
    p2 = pattern2[:min_len]

    cos_sim = torch.nn.functional.cosine_similarity(
        p1.unsqueeze(0), p2.unsqueeze(0)
    ).item()

    return cos_sim


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    計算 Centered Kernel Alignment (CKA)

    CKA 是一種比較兩組表示的相似性的方法
    - 不受縮放影響
    - 可以比較不同維度的表示
    - 廣泛用於比較神經網路表示
    """
    def centering(K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return H @ K @ H

    def rbf(X, sigma=None):
        GX = X @ X.T
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = np.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def hsic(K, L):
        return np.sum(centering(K) * centering(L))

    K = rbf(X)
    L = rbf(Y)

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    return hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-8)


def analyze_attention_consistency(
    train_feature_maps: List[Dict[str, torch.Tensor]],
    val_feature_maps: List[Dict[str, torch.Tensor]],
) -> Dict:
    """
    分析 Train/Val 是否關注同樣的特徵

    核心問題：模型是用「同樣的方式」處理 Train 和 Val 嗎？
    """
    results = {
        'layer_pattern_similarity': {},
        'layer_cka': {},
    }

    for layer_name in ENCODER_CONV_LAYERS:
        # 收集所有 Train 樣本的 activation pattern
        train_patterns = []
        for fm_dict in train_feature_maps:
            if layer_name in fm_dict:
                pattern = compute_activation_pattern(fm_dict[layer_name])
                train_patterns.append(pattern)

        # 收集所有 Val 樣本的 activation pattern
        val_patterns = []
        for fm_dict in val_feature_maps:
            if layer_name in fm_dict:
                pattern = compute_activation_pattern(fm_dict[layer_name])
                val_patterns.append(pattern)

        if not train_patterns or not val_patterns:
            continue

        # 計算 Train 內部的平均 pattern
        train_avg_pattern = torch.stack(train_patterns).mean(dim=0)
        val_avg_pattern = torch.stack(val_patterns).mean(dim=0)

        # 方法 1: Pattern Similarity (Train avg vs Val avg)
        pattern_sim = compute_pattern_similarity(train_avg_pattern, val_avg_pattern)
        results['layer_pattern_similarity'][layer_name] = pattern_sim

        # 方法 2: CKA (更嚴格的比較)
        # 將 patterns 堆疊成矩陣
        train_matrix = torch.stack(train_patterns).numpy()
        val_matrix = torch.stack(val_patterns).numpy()

        # 採樣使數量相同
        min_samples = min(len(train_matrix), len(val_matrix))
        train_sample = train_matrix[:min_samples]
        val_sample = val_matrix[:min_samples]

        try:
            cka = compute_cka(train_sample, val_sample)
            results['layer_cka'][layer_name] = cka
        except:
            results['layer_cka'][layer_name] = 0.0

    return results


def visualize_attention_consistency(
    results: Dict,
    output_path: Path,
):
    """視覺化 Train/Val attention 一致性"""
    n_layers = len(ENCODER_CONV_LAYERS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Pattern Similarity
    ax = axes[0, 0]
    sims = [results['layer_pattern_similarity'].get(l, 0) for l in ENCODER_CONV_LAYERS]

    colors = []
    for sim in sims:
        if sim > 0.9:
            colors.append('#4ECDC4')  # 綠色 - 高度一致
        elif sim > 0.7:
            colors.append('#FFE66D')  # 黃色 - 中等
        else:
            colors.append('#FF6B6B')  # 紅色 - 不一致

    ax.bar(range(n_layers), sims, color=colors)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    ax.set_title('Train-Val Pattern Similarity\n(Do they focus on same features?)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Good (>0.9)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='OK (>0.7)')
    ax.legend()

    # 2. CKA
    ax = axes[0, 1]
    ckas = [results['layer_cka'].get(l, 0) for l in ENCODER_CONV_LAYERS]

    colors = []
    for cka in ckas:
        if cka > 0.8:
            colors.append('#4ECDC4')
        elif cka > 0.5:
            colors.append('#FFE66D')
        else:
            colors.append('#FF6B6B')

    ax.bar(range(n_layers), ckas, color=colors)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    ax.set_title('Train-Val CKA\n(Representation Similarity)')
    ax.set_ylabel('CKA Score')
    ax.set_ylim(0, 1)

    # 3. 分組統計
    ax = axes[1, 0]
    group_pattern_sims = {}
    group_ckas = {}

    for group_name, indices in LAYER_GROUPS.items():
        group_sims = [sims[i] for i in indices if i < len(sims)]
        group_cka = [ckas[i] for i in indices if i < len(ckas)]
        group_pattern_sims[group_name] = np.mean(group_sims) if group_sims else 0
        group_ckas[group_name] = np.mean(group_cka) if group_cka else 0

    group_names = list(group_pattern_sims.keys())
    x = np.arange(len(group_names))
    width = 0.35

    ax.bar(x - width/2, [group_pattern_sims[g] for g in group_names], width,
           label='Pattern Sim', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, [group_ckas[g] for g in group_names], width,
           label='CKA', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45)
    ax.set_title('Group-wise Consistency')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend()

    # 4. 診斷結論
    ax = axes[1, 1]
    ax.axis('off')

    avg_pattern_sim = np.mean(sims)
    avg_cka = np.mean(ckas)

    diagnosis = []
    diagnosis.append("=" * 45)
    diagnosis.append("Train/Val Attention Consistency")
    diagnosis.append("=" * 45)
    diagnosis.append(f"\nAvg Pattern Similarity: {avg_pattern_sim:.4f}")
    diagnosis.append(f"Avg CKA: {avg_cka:.4f}")

    diagnosis.append("\nLayer Group Analysis:")
    for group_name in group_names:
        ps = group_pattern_sims[group_name]
        ck = group_ckas[group_name]
        status = "OK" if ps > 0.8 and ck > 0.5 else "WARN"
        diagnosis.append(f"  {group_name:12s}: PS={ps:.3f}, CKA={ck:.3f} [{status}]")

    if avg_pattern_sim > 0.85:
        diagnosis.append("\n[OK] Train/Val focus on similar features")
        diagnosis.append("     (Like both looking at cat outline)")
    elif avg_pattern_sim > 0.7:
        diagnosis.append("\n[WARN] Some inconsistency detected")
        diagnosis.append("     (Some layers focus differently)")
    else:
        diagnosis.append("\n[BAD] Train/Val focus on DIFFERENT features!")
        diagnosis.append("     (Like one sees cat, one sees background)")
        diagnosis.append("     This explains poor generalization!")

    ax.text(0.05, 0.95, '\n'.join(diagnosis), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Do Train and Val Focus on the Same Features?', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    return {
        'avg_pattern_similarity': avg_pattern_sim,
        'avg_cka': avg_cka,
        'group_pattern_sims': group_pattern_sims,
        'group_ckas': group_ckas,
    }


def main():
    print("=" * 60)
    print("Train/Val Attention Pattern Analysis")
    print("=" * 60)
    print("\nQuestion: Do Train and Val focus on the SAME features?")
    print("(Like: both should look at 'cat outline', not one at 'background')")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 載入模型
    print("\nLoading WavTokenizer...")
    wavtok = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)
    wavtok.eval()
    wavtok = wavtok.to(device)

    extractor = FeatureMapExtractor(wavtok, device)

    # 載入數據
    print("\nLoading data...")
    train_data = torch.load(TRAIN_CACHE)
    val_data = torch.load(VAL_CACHE)

    # 音檔目錄
    audio_base_dirs = [
        "/home/sbplab/ruizi/WavTokenize/data/raw/box",
        "/home/sbplab/ruizi/WavTokenize/data/raw/LDV",
        "/home/sbplab/ruizi/WavTokenize/data/clean/box2",
    ]

    max_samples = 20

    def load_audios(data, max_n):
        import soundfile as sf
        import torchaudio
        audios = []
        for sample in data[:max_n * 3]:
            if len(audios) >= max_n:
                break
            if 'noisy_path' in sample:
                for base_dir in audio_base_dirs:
                    full_path = Path(base_dir) / sample['noisy_path']
                    if full_path.exists():
                        try:
                            waveform, sr = sf.read(str(full_path))
                            waveform = torch.from_numpy(waveform).float()
                            if waveform.dim() == 1:
                                waveform = waveform.unsqueeze(0)
                            if sr != 24000:
                                resampler = torchaudio.transforms.Resample(sr, 24000)
                                waveform = resampler(waveform)
                            audios.append(waveform)
                            break
                        except:
                            pass
        return audios

    train_audios = load_audios(train_data, max_samples)
    val_audios = load_audios(val_data, max_samples)

    print(f"Train samples: {len(train_audios)}")
    print(f"Val samples: {len(val_audios)}")

    # 提取特徵
    print("\nExtracting Train features...")
    train_fms = [extractor.extract(a) for a in tqdm(train_audios)]

    print("\nExtracting Val features...")
    val_fms = [extractor.extract(a) for a in tqdm(val_audios)]

    # 分析 attention 一致性
    print("\nAnalyzing attention consistency...")
    results = analyze_attention_consistency(train_fms, val_fms)

    # 視覺化
    summary = visualize_attention_consistency(
        results,
        OUTPUT_DIR / 'attention_consistency.png'
    )

    # 保存結果
    with open(OUTPUT_DIR / 'attention_consistency.json', 'w') as f:
        json.dump({
            'layer_pattern_similarity': results['layer_pattern_similarity'],
            'layer_cka': results['layer_cka'],
            'summary': summary,
        }, f, indent=2, default=str)

    # 打印總結
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nAverage Pattern Similarity: {summary['avg_pattern_similarity']:.4f}")
    print(f"Average CKA: {summary['avg_cka']:.4f}")

    if summary['avg_pattern_similarity'] > 0.85:
        print("\n[OK] Train and Val focus on SIMILAR features")
    else:
        print("\n[WARN] Train and Val may focus on DIFFERENT features!")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
