"""
exp_1231_feature: LoRA 訓練前後各層變化分析 (V2)

方法: 直接比較 LoRA 合併後的權重與原始權重的差異

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
from typing import Dict, List
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


def extract_encoder_weights(model) -> Dict[str, torch.Tensor]:
    """提取 encoder 各層的權重"""
    weights = {}

    encoder = model.feature_extractor.encodec.encoder

    for layer_name in ENCODER_CONV_LAYERS:
        parts = layer_name.split('.')
        module = encoder
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        # 獲取權重
        if hasattr(module, 'weight'):
            weights[layer_name] = module.weight.detach().clone()

    return weights


def compare_weights(original: Dict[str, torch.Tensor],
                   trained: Dict[str, torch.Tensor]) -> List[Dict]:
    """比較兩組權重的差異"""
    results = []

    for i, layer_name in enumerate(ENCODER_CONV_LAYERS):
        if layer_name not in original or layer_name not in trained:
            results.append({
                'layer_idx': i,
                'layer_name': layer_name,
                'relative_change': 0,
                'l2_change': 0,
                'original_norm': 0,
            })
            continue

        orig_w = original[layer_name]
        train_w = trained[layer_name]

        # 計算差異
        diff = train_w - orig_w
        l2_change = diff.norm().item()
        original_norm = orig_w.norm().item()
        relative_change = l2_change / (original_norm + 1e-8)

        # 計算 cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_w.flatten().unsqueeze(0),
            train_w.flatten().unsqueeze(0)
        ).item()

        results.append({
            'layer_idx': i,
            'layer_name': layer_name,
            'l2_change': l2_change,
            'original_norm': original_norm,
            'relative_change': relative_change,
            'cosine_similarity': cos_sim,
            'weight_shape': list(orig_w.shape),
        })

    return results


def visualize_weight_changes(results: List[Dict], output_dir: Path):
    """視覺化權重變化"""

    n_layers = len(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 顏色
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

    # 相對變化
    relative_changes = [r['relative_change'] for r in results]
    axes[0, 0].bar(range(n_layers), relative_changes, color=colors)
    axes[0, 0].set_xticks(range(n_layers))
    axes[0, 0].set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    axes[0, 0].set_title('Relative Weight Change by Layer\n||W_trained - W_original|| / ||W_original||')
    axes[0, 0].set_ylabel('Relative Change')

    # L2 變化
    l2_changes = [r['l2_change'] for r in results]
    axes[0, 1].bar(range(n_layers), l2_changes, color=colors)
    axes[0, 1].set_xticks(range(n_layers))
    axes[0, 1].set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    axes[0, 1].set_title('Absolute Weight Change\n||W_trained - W_original||')
    axes[0, 1].set_ylabel('L2 Change')

    # 分組統計
    group_stats = {}
    for group_name, indices in LAYER_GROUPS.items():
        group_changes = [relative_changes[i] for i in indices if i < n_layers]
        group_stats[group_name] = np.mean(group_changes) if group_changes else 0

    group_names = list(group_stats.keys())
    group_values = [group_stats[g] for g in group_names]
    group_colors = ['gray', 'lightblue', 'skyblue', 'steelblue', 'navy', 'darkblue']

    axes[1, 0].bar(group_names, group_values, color=group_colors)
    axes[1, 0].set_title('Weight Change by Layer Group')
    axes[1, 0].set_ylabel('Average Relative Change')
    for i, v in enumerate(group_values):
        axes[1, 0].text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=9)

    # Cosine similarity (1 - cos_sim = 方向變化)
    direction_changes = [1 - r.get('cosine_similarity', 1) for r in results]
    axes[1, 1].bar(range(n_layers), direction_changes, color=colors)
    axes[1, 1].set_xticks(range(n_layers))
    axes[1, 1].set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45, fontsize=8)
    axes[1, 1].set_title('Weight Direction Change\n1 - cosine_similarity(W_trained, W_original)')
    axes[1, 1].set_ylabel('Direction Change')

    plt.tight_layout()
    plt.savefig(output_dir / 'lora_weight_changes_v2.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'lora_weight_changes_v2.png'}")

    # 文字總結
    print("\n" + "="*60)
    print("LoRA Weight Changes Summary (V2)")
    print("="*60)
    print("\n各層組平均相對變化:")
    for group_name, value in group_stats.items():
        print(f"  {group_name:12s}: {value:.6f}")

    # 找出變化最大的層
    sorted_changes = sorted(enumerate(relative_changes), key=lambda x: x[1], reverse=True)
    print("\n變化最大的 5 層:")
    for idx, change in sorted_changes[:5]:
        print(f"  L{idx} ({results[idx]['layer_name']}): {change:.6f}")

    print("\n解讀:")
    max_group = max(group_stats, key=group_stats.get)
    print(f"  LoRA 在 '{max_group}' 層組學習最多")
    if max_group in ['input', 'low_level']:
        print("    → 模型主要在淺層學習去噪 (聲學級別修正)")
    elif max_group in ['semantic', 'abstract']:
        print("    → 模型主要在深層學習去噪 (語義級別修正)")
    else:
        print("    → 模型在中間層學習去噪")

    return group_stats


def main():
    """主程式"""
    print("="*60)
    print("exp_1231_feature: LoRA Weight Change Analysis (V2)")
    print("="*60)

    if not LORA_MODEL_PATH.exists():
        print(f"Error: LoRA model not found at {LORA_MODEL_PATH}")
        return

    device = 'cpu'  # 只需要權重，不需要 GPU

    # 1. 載入原始模型
    print("\n1. Loading original WavTokenizer...")
    original_model = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)
    original_weights = extract_encoder_weights(original_model)
    print(f"   Extracted {len(original_weights)} layers")

    # 2. 載入訓練後的模型
    print("\n2. Loading trained model...")
    trained_model = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)

    # 應用 LoRA
    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        target_modules=ALL_ENCODER_CONV_MODULES,
        lora_dropout=0.2,
        bias="none",
    )
    trained_model = get_peft_model(trained_model, lora_config)

    # 載入訓練後的權重
    checkpoint = torch.load(LORA_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # 只載入 student 的權重
    student_state_dict = {k.replace('student.', ''): v for k, v in state_dict.items() if k.startswith('student.')}
    trained_model.load_state_dict(student_state_dict, strict=False)

    # 合併 LoRA 權重
    print("   Merging LoRA weights...")
    trained_model = trained_model.merge_and_unload()

    # 提取權重
    trained_weights = extract_encoder_weights(trained_model)
    print(f"   Extracted {len(trained_weights)} layers")

    # 3. 比較權重
    print("\n3. Comparing weights...")
    results = compare_weights(original_weights, trained_weights)

    # 4. 視覺化
    group_stats = visualize_weight_changes(results, OUTPUT_DIR)

    # 5. 保存結果
    with open(OUTPUT_DIR / 'lora_changes_v2.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)

    return results, group_stats


if __name__ == '__main__':
    main()
