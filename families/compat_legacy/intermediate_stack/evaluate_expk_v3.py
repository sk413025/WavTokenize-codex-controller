"""
Exp K v3 VQ 比較評估腳本 - 支援選擇最佳樣本

比較三種處理方式:
1. Teacher: Clean → WavTokenizer VQ → Decoder (VQ 重建上限)
2. Noisy VQ: Noisy → WavTokenizer VQ → Decoder (無去噪的 baseline)
3. Student: Noisy → Student LoRA → Decoder (我們的模型)

特點:
- 支援選擇 top-N 最佳樣本 (基於 Student 的 SI-SDR 改善)
- 使用 TeacherStudentIntermediate 模型結構
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# 設定 GPU (GPU 1 = RTX 2080 Ti)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import sys
# 添加原始專案路徑
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-feature-analysis')

from pesq import pesq
from pystoi import stoi

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from families.compat_legacy.intermediate_stack.models import TeacherStudentIntermediate


def compute_si_sdr(estimate, reference):
    """計算 Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)"""
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()

    dot = torch.sum(estimate * reference)
    s_target = dot * reference / (torch.sum(reference ** 2) + 1e-8)
    e_noise = estimate - s_target

    si_sdr = 10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )
    return si_sdr.item()


def decode_tokens_to_audio(wavtokenizer, tokens, device):
    """將 tokens 解碼為音頻"""
    with torch.no_grad():
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # (1, B, T)

        features = wavtokenizer.codes_to_features(tokens)
        bandwidth_id = torch.tensor([0], device=device)
        audio = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

        if audio.dim() == 3:
            audio = audio.squeeze(1)
    return audio


def encode_audio_to_tokens(wavtokenizer, audio, device):
    """將音頻編碼為 tokens"""
    with torch.no_grad():
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        bandwidth_id = torch.tensor([0], device=device)
        _, codes = wavtokenizer.encode_infer(audio, bandwidth_id=bandwidth_id)
        return codes


def evaluate_single_sample(clean_audio, audio_dict, sample_rate=24000):
    """評估單個樣本的多種重建結果"""
    results = {}

    min_len = len(clean_audio)
    for key in audio_dict:
        min_len = min(min_len, len(audio_dict[key]))

    clean = clean_audio[:min_len].cpu().numpy()

    # PESQ 需要 16kHz
    import librosa
    clean_16k = librosa.resample(clean, orig_sr=sample_rate, target_sr=16000)

    for name, audio in audio_dict.items():
        audio_np = audio[:min_len].cpu().numpy()
        audio_16k = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)

        # PESQ
        try:
            pesq_score = pesq(16000, clean_16k, audio_16k, 'wb')
        except:
            pesq_score = float('nan')

        # STOI
        try:
            stoi_score = stoi(clean, audio_np, sample_rate, extended=False)
        except:
            stoi_score = float('nan')

        # SI-SDR
        clean_t = torch.from_numpy(clean).float()
        audio_t = torch.from_numpy(audio_np).float()
        si_sdr_score = compute_si_sdr(audio_t, clean_t)

        results[name] = {
            'pesq': pesq_score,
            'stoi': stoi_score,
            'si_sdr': si_sdr_score,
        }

    return results


def load_model(exp_dir, model_type='best'):
    """載入 exp_k_v3 模型"""
    device = torch.device('cuda:0')

    model_path = exp_dir / f'{model_type}_model.pt'
    config_path = exp_dir / 'config.json'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(config_path) as f:
        config = json.load(f)

    # 使用 TeacherStudentIntermediate 結構
    model = TeacherStudentIntermediate(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        intermediate_indices=config.get('intermediate_indices', [3, 5, 6, 10]),
        device='cuda:0'
    )

    # 先載入到 CPU 再搬到 GPU 避免記憶體碎片問題
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded: {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', checkpoint.get('best_val_acc', 'N/A'))}")

    return model, device, config


def create_dataloader(cache_path, batch_size=4, num_workers=2):
    """創建資料載入器"""
    from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn
    from torch.utils.data import DataLoader

    dataset = AlignedNoisyCleanPairDataset(cache_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=aligned_collate_fn,
    )
    return loader


def plot_results(train_results, val_results, output_path, title_suffix=''):
    """繪製比較圖"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    metrics = ['pesq', 'stoi', 'si_sdr']
    metric_names = ['PESQ', 'STOI', 'SI-SDR (dB)']
    sources = ['teacher_vq', 'noisy_vq', 'student']
    source_labels = ['Teacher\n(Clean→VQ)', 'Noisy VQ\n(Noisy→VQ)', 'Student\n(Noisy→LoRA)']
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    for row, (data, split_name) in enumerate([(train_results, 'Train'), (val_results, 'Val')]):
        for col, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[row, col]

            values = [data[src][metric] for src in sources]
            bars = ax.bar(source_labels, values, color=colors, edgecolor='black', linewidth=1.5)

            # 添加數值標籤
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_title(f'{split_name} - {metric_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12)
            ax.grid(axis='y', alpha=0.3)

            # 設定 y 軸範圍
            if metric == 'pesq':
                ax.set_ylim(0, 5)
            elif metric == 'stoi':
                ax.set_ylim(0, 1.1)

    plt.suptitle(f'VQ 比較評估: Teacher vs Noisy VQ vs Student{title_suffix}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str,
                        default='/home/sbplab/ruizi/WavTokenize-feature-analysis/families/compat_legacy/intermediate_stack/runs/exp_k_v3_20260116_004710',
                        help='Experiment directory')
    parser.add_argument('--model_type', type=str, default='best',
                        choices=['best', 'final'])
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Max samples per split to evaluate')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of best samples to select for final statistics')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--selection_metric', type=str, default='si_sdr_improvement',
                        choices=['si_sdr_improvement', 'pesq_improvement', 'student_si_sdr'],
                        help='Metric to select best samples')
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    print("=" * 70)
    print("Exp K v3 VQ 比較評估 - Teacher / Noisy VQ / Student")
    print(f"選擇 top-{args.top_n} 最佳樣本 (基於 {args.selection_metric})")
    print("=" * 70)

    # 載入模型
    print("\n載入模型...")
    model, device, config = load_model(exp_dir, args.model_type)
    wavtokenizer = model.teacher  # 取得原始 WavTokenizer

    # 輸出目錄
    output_dir = exp_dir / 'evaluation'
    output_dir.mkdir(exist_ok=True)

    all_split_results = {}
    all_split_samples = {}  # 儲存每個樣本的詳細結果
    all_split_best_results = {}  # 儲存 top-N 樣本的統計

    for split_name, cache_path in [('train', TRAIN_CACHE), ('val', VAL_CACHE)]:
        print(f"\n{'='*60}")
        print(f"評估 {split_name.upper()} 集")
        print(f"{'='*60}")

        # 載入資料
        loader = create_dataloader(cache_path, args.batch_size)
        print(f"Batches: {len(loader)}")

        # 收集結果
        all_results = []
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
                if num_samples >= args.max_samples:
                    break

                noisy = batch['noisy_audio'].to(device)
                clean = batch['clean_audio'].to(device)

                B = noisy.shape[0]

                for i in range(B):
                    if num_samples >= args.max_samples:
                        break

                    clean_i = clean[i].squeeze()
                    noisy_i = noisy[i].squeeze()

                    try:
                        # 1. Teacher: Clean → VQ → Decode
                        teacher_tokens = encode_audio_to_tokens(wavtokenizer, clean_i, device)
                        teacher_audio = decode_tokens_to_audio(wavtokenizer, teacher_tokens, device)
                        teacher_audio = teacher_audio.squeeze()

                        # 2. Noisy VQ: Noisy → VQ → Decode
                        noisy_tokens = encode_audio_to_tokens(wavtokenizer, noisy_i, device)
                        noisy_vq_audio = decode_tokens_to_audio(wavtokenizer, noisy_tokens, device)
                        noisy_vq_audio = noisy_vq_audio.squeeze()

                        # 3. Student: Noisy → LoRA encoder → Decode
                        # 使用 TeacherStudentIntermediate 的 forward
                        output = model(noisy_i.unsqueeze(0).unsqueeze(0), clean_i.unsqueeze(0).unsqueeze(0))
                        student_codes = output['student_codes']
                        if student_codes.dim() == 3:
                            student_codes = student_codes[0]
                        student_audio = decode_tokens_to_audio(wavtokenizer, student_codes, device)
                        student_audio = student_audio.squeeze()

                        # 評估
                        audio_dict = {
                            'teacher_vq': teacher_audio,
                            'noisy_vq': noisy_vq_audio,
                            'student': student_audio,
                        }

                        result = evaluate_single_sample(clean_i, audio_dict)

                        # 計算改善量
                        result['si_sdr_improvement'] = result['student']['si_sdr'] - result['noisy_vq']['si_sdr']
                        result['pesq_improvement'] = result['student']['pesq'] - result['noisy_vq']['pesq']
                        result['sample_idx'] = num_samples

                        all_results.append(result)
                        num_samples += 1

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

        # 儲存所有樣本結果
        all_split_samples[split_name] = all_results

        if len(all_results) == 0:
            print(f"No valid samples for {split_name}")
            continue

        # === 全部樣本統計 ===
        all_stats = {}
        for source in ['teacher_vq', 'noisy_vq', 'student']:
            all_stats[source] = {
                'pesq': np.nanmean([r[source]['pesq'] for r in all_results]),
                'stoi': np.nanmean([r[source]['stoi'] for r in all_results]),
                'si_sdr': np.nanmean([r[source]['si_sdr'] for r in all_results]),
            }
        all_split_results[split_name] = all_stats

        # === 選擇 Top-N 最佳樣本 ===
        if args.selection_metric == 'si_sdr_improvement':
            sorted_results = sorted(all_results, key=lambda x: x['si_sdr_improvement'], reverse=True)
        elif args.selection_metric == 'pesq_improvement':
            sorted_results = sorted(all_results, key=lambda x: x['pesq_improvement'], reverse=True)
        else:  # student_si_sdr
            sorted_results = sorted(all_results, key=lambda x: x['student']['si_sdr'], reverse=True)

        top_n_results = sorted_results[:args.top_n]

        # Top-N 統計
        best_stats = {}
        for source in ['teacher_vq', 'noisy_vq', 'student']:
            best_stats[source] = {
                'pesq': np.nanmean([r[source]['pesq'] for r in top_n_results]),
                'stoi': np.nanmean([r[source]['stoi'] for r in top_n_results]),
                'si_sdr': np.nanmean([r[source]['si_sdr'] for r in top_n_results]),
            }
        all_split_best_results[split_name] = best_stats

        # 打印結果
        print(f"\n{split_name.upper()} 全部樣本結果 (n={len(all_results)}):")
        print("-" * 70)
        print(f"{'Source':<20} {'PESQ':>10} {'STOI':>10} {'SI-SDR':>10}")
        print("-" * 70)
        for source, label in [('teacher_vq', 'Teacher (Clean→VQ)'),
                               ('noisy_vq', 'Noisy VQ'),
                               ('student', 'Student (LoRA)')]:
            s = all_stats[source]
            print(f"{label:<20} {s['pesq']:>10.3f} {s['stoi']:>10.3f} {s['si_sdr']:>10.2f}")

        print(f"\n{split_name.upper()} Top-{args.top_n} 最佳樣本結果:")
        print("-" * 70)
        print(f"{'Source':<20} {'PESQ':>10} {'STOI':>10} {'SI-SDR':>10}")
        print("-" * 70)
        for source, label in [('teacher_vq', 'Teacher (Clean→VQ)'),
                               ('noisy_vq', 'Noisy VQ'),
                               ('student', 'Student (LoRA)')]:
            s = best_stats[source]
            print(f"{label:<20} {s['pesq']:>10.3f} {s['stoi']:>10.3f} {s['si_sdr']:>10.2f}")

        print(f"\nTop-{args.top_n} 樣本索引: {[r['sample_idx'] for r in top_n_results]}")
        print(f"選擇標準 ({args.selection_metric}): {[r[args.selection_metric] if args.selection_metric in r else r['student']['si_sdr'] for r in top_n_results]}")

        # 改善計算
        print(f"\nTop-{args.top_n} 改善 (相對於 Noisy VQ):")
        for metric in ['pesq', 'stoi', 'si_sdr']:
            noisy_val = best_stats['noisy_vq'][metric]
            student_val = best_stats['student'][metric]
            teacher_val = best_stats['teacher_vq'][metric]

            student_improve = student_val - noisy_val
            teacher_improve = teacher_val - noisy_val
            print(f"  {metric.upper():>8}: Student {student_improve:+.3f}, Teacher {teacher_improve:+.3f}")

    # === 繪圖 ===
    # 全部樣本
    if 'train' in all_split_results and 'val' in all_split_results:
        plot_path = output_dir / 'vq_comparison_all_samples.png'
        plot_results(all_split_results['train'], all_split_results['val'], plot_path,
                    f' (All {args.max_samples} samples)')

    # Top-N 最佳樣本
    if 'train' in all_split_best_results and 'val' in all_split_best_results:
        plot_path = output_dir / f'vq_comparison_top{args.top_n}.png'
        plot_results(all_split_best_results['train'], all_split_best_results['val'], plot_path,
                    f' (Top-{args.top_n} by {args.selection_metric})')

    # === 儲存結果 ===
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'exp_dir': str(exp_dir),
            'model_type': args.model_type,
            'max_samples': args.max_samples,
            'top_n': args.top_n,
            'selection_metric': args.selection_metric,
            'all_samples_results': all_split_results,
            'top_n_results': all_split_best_results,
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # 儲存每個樣本的詳細結果
    samples_path = output_dir / 'per_sample_results.json'
    with open(samples_path, 'w') as f:
        json.dump(all_split_samples, f, indent=2)
    print(f"Per-sample results saved: {samples_path}")


if __name__ == '__main__':
    main()
