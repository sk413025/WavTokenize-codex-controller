"""
exp_1231_feature: LoRA 訓練前後各層變化分析

目標:
1. 比較原始 WavTokenizer 與 LoRA 訓練後的各層 feature map 差異
2. 找出哪些層變化最大（模型在哪裡學習去噪）
3. 分析 Train vs Val 的特徵是否一致

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
import json

# 路徑設置
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'WavTokenize-self-supervised'))

from decoder.pretrained import WavTokenizer
from peft import get_peft_model, LoraConfig

# ============================================================
# 配置
# ============================================================

WAVTOK_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

# LoRA 訓練後的模型 (Exp67 - 最佳結果)
LORA_MODEL_PATH = Path("/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1226/runs/exp67_curriculum_vq/best_model.pt")

# 測試音檔
AUDIO_DIR = Path("/home/sbplab/ruizi/WavTokenize/data/clean/box2")
BOX_AUDIO_DIR = Path("/home/sbplab/ruizi/WavTokenize/data/raw/box")

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

# LoRA target modules (全部 18 層)
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

LAYER_GROUPS = {
    'input': [0],
    'low_level': [1, 2, 3, 4],
    'mid_level': [5, 6, 7, 8],
    'semantic': [9, 10, 11, 12],
    'abstract': [13, 14, 15, 16],
    'output': [17],
}


def load_audio(path: Path, target_sr: int = 24000) -> torch.Tensor:
    """載入音訊"""
    import soundfile as sf
    waveform, sr = sf.read(str(path))
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    if sr != target_sr:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform


def compute_lora_weight_change(model_with_lora: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    計算 LoRA 對各層權重的實際影響

    LoRA: W' = W + BA (其中 B, A 是低秩矩陣)
    這裡計算 ||BA|| / ||W|| 來量化變化幅度
    """
    changes = {}

    for name, module in model_with_lora.named_modules():
        # 檢查是否有 LoRA 層
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # 獲取原始權重
            if hasattr(module, 'base_layer'):
                base_weight = module.base_layer.weight
            elif hasattr(module, 'weight'):
                base_weight = module.weight
            else:
                continue

            # 計算 LoRA 貢獻: BA
            # lora_A: (r, in_features), lora_B: (out_features, r)
            try:
                lora_A = module.lora_A['default'].weight  # (r, in)
                lora_B = module.lora_B['default'].weight  # (out, r)
                scaling = module.scaling['default']

                # BA 矩陣
                ba = lora_B @ lora_A * scaling  # (out, in)

                # 對於 Conv1d，權重形狀是 (out, in, kernel)
                # LoRA 只影響 (out, in) 部分
                if base_weight.dim() == 3:
                    # 擴展到 kernel 維度
                    ba = ba.unsqueeze(-1)

                # 計算變化比例
                ba_norm = ba.norm().item()
                base_norm = base_weight.norm().item()
                relative_change = ba_norm / (base_norm + 1e-8)

                # 簡化名稱
                short_name = name.replace('base_model.model.', '').replace('feature_extractor.encodec.encoder.', '')

                changes[short_name] = {
                    'ba_norm': ba_norm,
                    'base_norm': base_norm,
                    'relative_change': relative_change,
                    'lora_A_norm': lora_A.norm().item(),
                    'lora_B_norm': lora_B.norm().item(),
                }

            except Exception as e:
                print(f"Warning: Could not compute LoRA change for {name}: {e}")
                continue

    return changes


def analyze_lora_weight_changes(model_path: Path, output_dir: Path):
    """分析 LoRA 權重變化"""
    print("\n" + "="*60)
    print("分析 2: LoRA 權重變化 (各層學習程度)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 載入原始模型
    print("\nLoading original WavTokenizer...")
    wavtok = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)

    # 應用 LoRA
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        target_modules=ALL_ENCODER_CONV_MODULES,
        lora_dropout=0.2,
        bias="none",
    )
    wavtok_lora = get_peft_model(wavtok, lora_config)

    # 載入訓練後的權重
    print(f"Loading trained weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # 只載入 student 的權重
    state_dict = checkpoint['model_state_dict']
    student_state_dict = {k.replace('student.', ''): v for k, v in state_dict.items() if k.startswith('student.')}

    # 載入到 LoRA 模型
    wavtok_lora.load_state_dict(student_state_dict, strict=False)
    wavtok_lora = wavtok_lora.to(device)
    wavtok_lora.eval()

    # 計算各層 LoRA 變化
    print("\nComputing LoRA weight changes...")
    changes = compute_lora_weight_change(wavtok_lora)

    # 整理結果
    layer_changes = []
    for i, layer_name in enumerate(ENCODER_CONV_LAYERS):
        # 找到對應的 LoRA 層
        short_name = layer_name
        for full_name, data in changes.items():
            if short_name in full_name or full_name.endswith(short_name.replace('model.', '')):
                layer_changes.append({
                    'layer_idx': i,
                    'layer_name': layer_name,
                    **data
                })
                break
        else:
            # 沒找到，可能是沒有 LoRA 的層
            layer_changes.append({
                'layer_idx': i,
                'layer_name': layer_name,
                'relative_change': 0,
                'ba_norm': 0,
                'base_norm': 0,
            })

    # 視覺化
    visualize_lora_changes(layer_changes, output_dir)

    return layer_changes


def visualize_lora_changes(layer_changes: List[Dict], output_dir: Path):
    """視覺化 LoRA 變化"""

    n_layers = len(layer_changes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 各層相對變化
    relative_changes = [lc.get('relative_change', 0) for lc in layer_changes]
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

    axes[0, 0].bar(range(n_layers), relative_changes, color=colors)
    axes[0, 0].set_xticks(range(n_layers))
    axes[0, 0].set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    axes[0, 0].set_title('LoRA Relative Change by Layer\n||BA|| / ||W||')
    axes[0, 0].set_ylabel('Relative Change')

    # BA norm
    ba_norms = [lc.get('ba_norm', 0) for lc in layer_changes]
    axes[0, 1].bar(range(n_layers), ba_norms, color=colors)
    axes[0, 1].set_xticks(range(n_layers))
    axes[0, 1].set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    axes[0, 1].set_title('LoRA Update Magnitude\n||BA||')
    axes[0, 1].set_ylabel('||BA||')

    # 分組統計
    group_stats = {}
    for group_name, indices in LAYER_GROUPS.items():
        group_changes = [relative_changes[i] for i in indices if i < n_layers]
        group_stats[group_name] = np.mean(group_changes) if group_changes else 0

    group_names = list(group_stats.keys())
    group_values = [group_stats[g] for g in group_names]
    group_colors = ['gray', 'lightblue', 'skyblue', 'steelblue', 'navy', 'darkblue']

    axes[1, 0].bar(group_names, group_values, color=group_colors)
    axes[1, 0].set_title('LoRA Change by Layer Group')
    axes[1, 0].set_ylabel('Average Relative Change')
    for i, v in enumerate(group_values):
        axes[1, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

    # 累積變化
    cumulative = np.cumsum(relative_changes)
    axes[1, 1].plot(range(n_layers), cumulative, 'b-o', markersize=4)
    axes[1, 1].fill_between(range(n_layers), cumulative, alpha=0.3)
    axes[1, 1].set_xticks(range(n_layers))
    axes[1, 1].set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    axes[1, 1].set_title('Cumulative LoRA Change')
    axes[1, 1].set_ylabel('Cumulative Relative Change')

    plt.tight_layout()
    plt.savefig(output_dir / 'lora_weight_changes.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'lora_weight_changes.png'}")

    # 文字總結
    print("\n" + "="*60)
    print("LoRA Weight Changes Summary")
    print("="*60)
    print("\n各層組平均相對變化 (||BA|| / ||W||):")
    for group_name, value in group_stats.items():
        print(f"  {group_name:12s}: {value:.6f}")

    # 找出變化最大的層
    sorted_changes = sorted(enumerate(relative_changes), key=lambda x: x[1], reverse=True)
    print("\n變化最大的 5 層:")
    for idx, change in sorted_changes[:5]:
        print(f"  L{idx} ({layer_changes[idx]['layer_name']}): {change:.6f}")

    print("\n解讀:")
    max_group = max(group_stats, key=group_stats.get)
    print(f"  LoRA 在 '{max_group}' 層組學習最多")
    if max_group in ['input', 'low_level']:
        print("    → 模型主要在淺層學習去噪 (聲學級別修正)")
    elif max_group in ['semantic', 'abstract']:
        print("    → 模型主要在深層學習去噪 (語義級別修正)")
    else:
        print("    → 模型在中間層學習去噪")


def analyze_feature_difference_on_noisy(output_dir: Path):
    """
    分析: 對 noisy 輸入，LoRA 前後各層 feature 差異

    這可以看出 LoRA 在哪些層「修正」了 noisy 的特徵
    """
    print("\n" + "="*60)
    print("分析 3: Noisy 輸入的 Feature 修正")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 找一個 noisy 音檔
    # 格式: nor_boy1_box_LDV_001.wav
    noisy_path = BOX_AUDIO_DIR / "nor_boy1_box_LDV_001.wav"
    clean_path = AUDIO_DIR / "nor_boy1_clean_001.wav"

    if not noisy_path.exists():
        # 嘗試其他路徑
        import glob
        noisy_files = glob.glob(str(BOX_AUDIO_DIR / "*boy1*001*.wav"))
        if noisy_files:
            noisy_path = Path(noisy_files[0])
        else:
            print(f"Cannot find noisy audio file")
            return None

    print(f"\nNoisy: {noisy_path}")
    print(f"Clean: {clean_path}")

    # 載入音訊
    noisy_audio = load_audio(noisy_path)
    clean_audio = load_audio(clean_path)

    # 這裡只是示例，完整實現需要提取兩個模型的 feature maps 並比較
    print("\n(完整實現需要同時載入原始和 LoRA 模型並比較 feature maps)")
    print("可以通過 analyze_feature_maps.py 中的 FeatureMapExtractor 擴展")

    return None


def main():
    """主程式"""
    print("="*60)
    print("exp_1231_feature: LoRA Training Analysis")
    print("="*60)

    if not LORA_MODEL_PATH.exists():
        print(f"Error: LoRA model not found at {LORA_MODEL_PATH}")
        return

    # 分析 LoRA 權重變化
    layer_changes = analyze_lora_weight_changes(LORA_MODEL_PATH, OUTPUT_DIR)

    # 保存結果
    with open(OUTPUT_DIR / 'lora_changes.json', 'w') as f:
        json.dump(layer_changes, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
