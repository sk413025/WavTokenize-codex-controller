"""
Baseline (exp_k_v6) 多 Epoch Token 分布趨勢分析

目標：
1. 載入 baseline 的 30 個 checkpoint (epoch 10~300, 每 10 epoch 一個)
2. 對每個 checkpoint 計算 student/teacher 的 token 分布指標
3. 產出趨勢圖，回答「collapse 從何時開始」及「隨 epoch 的變化」
4. 與 schemeB_trend (RVQ, 5 epoch) 的數據做對比

與 schemeB_trend 的核心差異：
- Baseline: 單一 VQ (K=4096), codebook 凍結, LoRA distillation
- SchemeB: RVQ (L=4, K=2048), codebook 可學習 (EMA), 從零開始訓練
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1226.data_curriculum import create_curriculum_dataloaders

# ========== 設定 ==========
BASELINE_CKPT_DIR = Path("/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613/checkpoints")
OUTPUT_DIR = Path(__file__).parent / "baseline_multi_epoch_trend"
DEVICE = "cuda:0"
CODEBOOK_SIZE = 4096

LORA_CONFIG = {
    'rank': 256,
    'alpha': 512,
    'dropout': 0.2,
    'intermediate_indices': [3, 6],
}

# SchemeB trend 數據（用於對比）
SCHEMEB_EPOCH_METRICS = Path(__file__).parent / "runs" / \
    "schemeB_trend_steps2430_eval486_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_235224" / \
    "epoch_metrics.json"


def collect_tokens(model, data_loader, device, max_batches=None, split_name=''):
    """收集 student 和 teacher 的 token IDs。

    Args:
        model: TeacherStudentIntermediate 模型。
        data_loader: DataLoader，提供 noisy/clean audio。
        device: 計算裝置。
        max_batches: 最大 batch 數，None 表示全部。
        split_name: 資料分割名稱，用於 progress bar。

    Returns:
        tuple: (student_codes, teacher_codes)，各為 1D tensor。
    """
    model.eval()
    student_list, teacher_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"  {split_name}", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            try:
                noisy = batch['noisy_audio'].to(device)
                clean = batch['clean_audio'].to(device)
                lengths = batch.get('lengths', None)

                if clean.dim() == 2:
                    clean = clean.unsqueeze(1)
                if noisy.dim() == 2:
                    noisy = noisy.unsqueeze(1)

                out = model(clean, noisy)
                sc = out['student_codes'].cpu()  # [B, T]
                tc = out['teacher_codes'].cpu()

                if lengths is not None:
                    hop = 320
                    frame_lens = (lengths.cpu() + hop - 1) // hop
                    frame_lens = torch.clamp(frame_lens, min=0, max=sc.shape[1])
                    for b in range(sc.shape[0]):
                        L = int(frame_lens[b].item())
                        if L > 0:
                            student_list.append(sc[b, :L])
                            teacher_list.append(tc[b, :L])
                else:
                    student_list.append(sc.reshape(-1))
                    teacher_list.append(tc.reshape(-1))
            except Exception as e:
                print(f"    batch {i} error: {e}")
                continue

    model.train()
    if not student_list:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    return torch.cat(student_list), torch.cat(teacher_list)


def compute_metrics(codes, codebook_size):
    """計算 token 分布指標。

    Args:
        codes: 1D tensor of token IDs。
        codebook_size: codebook 大小。

    Returns:
        dict: 包含 entropy, top_k_mass, used_codes 等指標。
    """
    if len(codes) == 0:
        return {
            'total_tokens': 0, 'entropy_bits': 0, 'top_1_mass_pct': 0,
            'top_10_mass_pct': 0, 'top_50_mass_pct': 0, 'top_100_mass_pct': 0,
            'used_codes': 0, 'usage_pct': 0,
        }

    counts = np.bincount(codes.numpy(), minlength=codebook_size)
    probs = counts / counts.sum()
    sorted_probs = np.sort(probs)[::-1]

    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))

    return {
        'total_tokens': int(len(codes)),
        'entropy_bits': float(entropy),
        'top_1_mass_pct': float(sorted_probs[:1].sum() * 100),
        'top_10_mass_pct': float(sorted_probs[:10].sum() * 100),
        'top_50_mass_pct': float(sorted_probs[:50].sum() * 100),
        'top_100_mass_pct': float(sorted_probs[:100].sum() * 100),
        'used_codes': int((counts > 0).sum()),
        'usage_pct': float((counts > 0).sum() / codebook_size * 100),
    }


def load_checkpoint_and_collect(model, ckpt_path, device, train_loader, val_loader,
                                 max_train_batches=50, max_val_batches=None):
    """載入一個 checkpoint 並收集 token 分布。

    Args:
        model: TeacherStudentIntermediate 模型。
        ckpt_path: checkpoint 路徑。
        device: 計算裝置。
        train_loader: train DataLoader。
        val_loader: val DataLoader。
        max_train_batches: 最大 train batch 數。
        max_val_batches: 最大 val batch 數。

    Returns:
        dict: 包含四個 split 的 metrics。
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 載入 LoRA 權重
    lora_state = {}
    for k, v in checkpoint['lora_state_dict'].items():
        key = k[8:] if k.startswith('student.') else k
        lora_state[key] = v
    model.student.load_state_dict(lora_state, strict=False)

    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', None)
    train_acc = checkpoint.get('train_acc', None)

    # 收集 tokens
    train_student, train_teacher = collect_tokens(
        model, train_loader, device, max_train_batches, f'epoch{epoch} train'
    )
    val_student, val_teacher = collect_tokens(
        model, val_loader, device, max_val_batches, f'epoch{epoch} val'
    )

    result = {
        'epoch': epoch,
        'val_acc': float(val_acc) if val_acc is not None else None,
        'train_acc': float(train_acc) if train_acc is not None else None,
        'train_student': compute_metrics(train_student, CODEBOOK_SIZE),
        'train_teacher': compute_metrics(train_teacher, CODEBOOK_SIZE),
        'val_student': compute_metrics(val_student, CODEBOOK_SIZE),
        'val_teacher': compute_metrics(val_teacher, CODEBOOK_SIZE),
    }

    return result


def plot_multi_epoch_trends(all_metrics, output_dir, func_name='plot_multi_epoch_trends'):
    """繪製多 epoch 趨勢圖（baseline only）。

    Args:
        all_metrics: list of dict，每個 dict 包含一個 epoch 的 metrics。
        output_dir: 輸出目錄。
        func_name: 函式名稱，用於檔名。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    epochs = [m['epoch'] for m in all_metrics]
    date_str = datetime.now().strftime('%Y%m%d')

    splits_student = ['train_student', 'val_student']
    splits_teacher = ['train_teacher', 'val_teacher']
    colors = {
        'train_student': '#e74c3c', 'val_student': '#c0392b',
        'train_teacher': '#3498db', 'val_teacher': '#2980b9',
    }
    labels = {
        'train_student': 'Student Train', 'val_student': 'Student Val',
        'train_teacher': 'Teacher Train', 'val_teacher': 'Teacher Val',
    }

    metrics_to_plot = [
        ('entropy_bits', 'Entropy (bits)', 'Baseline (exp_k_v6): Token Entropy vs Epoch'),
        ('top_10_mass_pct', 'Top-10 Mass (%)', 'Baseline (exp_k_v6): Top-10 Token Mass vs Epoch'),
        ('top_1_mass_pct', 'Top-1 Mass (%)', 'Baseline (exp_k_v6): Top-1 Token Mass vs Epoch'),
        ('used_codes', 'Used Codes', 'Baseline (exp_k_v6): Codebook Usage vs Epoch'),
        ('usage_pct', 'Usage (%)', 'Baseline (exp_k_v6): Codebook Usage % vs Epoch'),
    ]

    for metric_key, ylabel, title in metrics_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))

        for split in splits_student + splits_teacher:
            values = [m[split][metric_key] for m in all_metrics]
            ls = '-' if 'student' in split else '--'
            marker = 'o' if 'student' in split else 's'
            ax.plot(epochs, values, marker=marker, linestyle=ls,
                    color=colors[split], label=labels[split],
                    linewidth=2, markersize=5, alpha=0.85)

        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        # 加上理想值參考線
        if metric_key == 'entropy_bits':
            ax.axhline(y=np.log2(CODEBOOK_SIZE), color='green', linestyle=':', alpha=0.5, label=f'Ideal (log₂{CODEBOOK_SIZE}={np.log2(CODEBOOK_SIZE):.1f})')
            ax.legend(fontsize=11)
        elif metric_key == 'top_10_mass_pct':
            ideal = 10 / CODEBOOK_SIZE * 100
            ax.axhline(y=ideal, color='green', linestyle=':', alpha=0.5, label=f'Ideal ({ideal:.3f}%)')
            ax.legend(fontsize=11)
        elif metric_key == 'usage_pct':
            ax.axhline(y=100, color='green', linestyle=':', alpha=0.5, label='Ideal (100%)')
            ax.legend(fontsize=11)

        fname = f'baseline_v6_{metric_key}_vs_epoch_{date_str}_{func_name}.png'
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {fname}")


def plot_comparison_with_schemeb(all_metrics, schemeb_data, output_dir, func_name='plot_comparison_with_schemeb'):
    """繪製 baseline vs schemeB 對比趨勢圖。

    Args:
        all_metrics: list of dict，baseline 多 epoch metrics。
        schemeb_data: dict，schemeB_trend epoch_metrics.json 的內容。
        output_dir: 輸出目錄。
        func_name: 函式名稱。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')

    # 解析 schemeB 數據
    sb_epochs_val = []
    sb_entropy_val = []
    sb_top10_val = []
    sb_used_val = []
    sb_epochs_train = []
    sb_entropy_train = []
    sb_top10_train = []
    sb_used_train = []

    for row in schemeb_data['rows']:
        if row['split'] == 'val_student':
            sb_epochs_val.append(row['epoch_end'])
            sb_entropy_val.append(row['entropy_bits'])
            sb_top10_val.append(row['top_10_mass_pct'])
            sb_used_val.append(row['used_codes'])
        elif row['split'] == 'train_student':
            sb_epochs_train.append(row['epoch_end'])
            sb_entropy_train.append(row['entropy_bits'])
            sb_top10_train.append(row['top_10_mass_pct'])
            sb_used_train.append(row['used_codes'])

    # Baseline 數據
    bl_epochs = [m['epoch'] for m in all_metrics]
    bl_entropy_val = [m['val_student']['entropy_bits'] for m in all_metrics]
    bl_top10_val = [m['val_student']['top_10_mass_pct'] for m in all_metrics]
    bl_used_val = [m['val_student']['used_codes'] for m in all_metrics]
    bl_entropy_train = [m['train_student']['entropy_bits'] for m in all_metrics]
    bl_top10_train = [m['train_student']['top_10_mass_pct'] for m in all_metrics]
    bl_used_train = [m['train_student']['used_codes'] for m in all_metrics]

    comparisons = [
        ('entropy_bits', 'Entropy (bits)',
         bl_entropy_val, bl_entropy_train, sb_entropy_val, sb_entropy_train),
        ('top_10_mass_pct', 'Top-10 Mass (%)',
         bl_top10_val, bl_top10_train, sb_top10_val, sb_top10_train),
        ('used_codes', 'Used Codes',
         bl_used_val, bl_used_train, sb_used_val, sb_used_train),
    ]

    for metric_key, ylabel, bl_val, bl_train, sb_val, sb_train in comparisons:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Val
        ax = axes[0]
        ax.plot(bl_epochs, bl_val, 'o-', color='#e74c3c', linewidth=2, markersize=5,
                label=f'Baseline (K=4096, frozen VQ)')
        ax.plot(sb_epochs_val, sb_val, 's--', color='#2ecc71', linewidth=2, markersize=7,
                label=f'SchemeB (RVQ L=4, K=2048, learnable)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Val Student: {ylabel}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Train
        ax = axes[1]
        ax.plot(bl_epochs, bl_train, 'o-', color='#c0392b', linewidth=2, markersize=5,
                label=f'Baseline (K=4096, frozen VQ)')
        ax.plot(sb_epochs_train, sb_train, 's--', color='#27ae60', linewidth=2, markersize=7,
                label=f'SchemeB (RVQ L=4, K=2048, learnable)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Train Student: {ylabel}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Baseline vs SchemeB: {ylabel} Trend Comparison',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        fname = f'baseline_vs_schemeB_{metric_key}_{date_str}_{func_name}.png'
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {fname}")


def plot_initial_vs_trained(initial_metrics, trained_metrics, output_dir, func_name='plot_initial_vs_trained'):
    """繪製初始狀態 vs 訓練後的 token 分布對比。

    Args:
        initial_metrics: dict，epoch 10 的 metrics（最早可得 checkpoint）。
        trained_metrics: dict，epoch 300 的 metrics。
        output_dir: 輸出目錄。
        func_name: 函式名稱。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    splits = ['train_student', 'val_student', 'train_teacher', 'val_teacher']
    split_labels = ['Student Train', 'Student Val', 'Teacher Train', 'Teacher Val']
    metric_keys = ['entropy_bits', 'top_10_mass_pct', 'used_codes']
    metric_labels = ['Entropy (bits)', 'Top-10 Mass (%)', 'Used Codes']

    for col, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        # 上排: 初始 vs 訓練後 (各 split)
        ax = axes[0, col]
        x = np.arange(len(splits))
        width = 0.35

        init_vals = [initial_metrics[s][mk] for s in splits]
        trained_vals = [trained_metrics[s][mk] for s in splits]

        bars1 = ax.bar(x - width/2, init_vals, width, label=f'Epoch {initial_metrics["epoch"]}',
                       color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, trained_vals, width, label=f'Epoch {trained_metrics["epoch"]}',
                       color='#e74c3c', alpha=0.8)

        ax.set_ylabel(ml, fontsize=11)
        ax.set_title(f'{ml}: Early vs Final', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, rotation=20, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 在 bar 上標數值
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7)

    # 下排: 差異 (Δ = trained - initial)
    for col, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[1, col]
        x = np.arange(len(splits))

        deltas = [trained_metrics[s][mk] - initial_metrics[s][mk] for s in splits]
        colors = ['#27ae60' if d >= 0 else '#e74c3c' for d in deltas]

        bars = ax.bar(x, deltas, 0.6, color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel(f'Δ {ml}', fontsize=11)
        ax.set_title(f'Change: Epoch {trained_metrics["epoch"]} - Epoch {initial_metrics["epoch"]}',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, rotation=20, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, d in zip(bars, deltas):
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            ax.annotate(f'{d:+.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3 if h >= 0 else -3),
                       textcoords='offset points', ha='center', va=va, fontsize=8)

    plt.suptitle(f'Baseline Token Distribution: Epoch {initial_metrics["epoch"]} vs Epoch {trained_metrics["epoch"]}',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    fname = f'baseline_initial_vs_trained_{date_str}_{func_name}.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {fname}")


def main():
    """主函式：對 baseline 所有 checkpoint 進行多 epoch token 分布分析。"""
    parser = argparse.ArgumentParser(description='Baseline 多 Epoch Token 分布趨勢分析')
    parser.add_argument('--epochs', type=str, default='10,20,30,50,100,150,200,250,300',
                        help='要分析的 epoch 列表 (逗號分隔)')
    parser.add_argument('--all_epochs', action='store_true',
                        help='分析所有 30 個 checkpoint (10~300)')
    parser.add_argument('--max_train_batches', type=int, default=50,
                        help='每個 checkpoint 最大 train batch 數')
    parser.add_argument('--max_val_batches', type=int, default=None,
                        help='每個 checkpoint 最大 val batch 數')
    parser.add_argument('--device', type=str, default=DEVICE)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR

    if args.all_epochs:
        epoch_list = list(range(10, 310, 10))
    else:
        epoch_list = [int(e) for e in args.epochs.split(',')]

    print("=" * 80)
    print("Baseline (exp_k_v6) 多 Epoch Token 分布趨勢分析")
    print("=" * 80)
    print(f"  Epochs: {epoch_list}")
    print(f"  Max train batches: {args.max_train_batches}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_dir}")

    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 檢查 checkpoint 存在
    available = []
    for ep in epoch_list:
        ckpt_path = BASELINE_CKPT_DIR / f"checkpoint_epoch{ep:03d}.pt"
        if ckpt_path.exists():
            available.append((ep, ckpt_path))
        else:
            print(f"  ⚠ Missing: {ckpt_path}")

    print(f"\n  Available checkpoints: {len(available)}/{len(epoch_list)}")

    # 建立模型
    print("\n[1] Loading model architecture...")
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=LORA_CONFIG['rank'],
        lora_alpha=LORA_CONFIG['alpha'],
        lora_dropout=LORA_CONFIG['dropout'],
        intermediate_indices=LORA_CONFIG['intermediate_indices'],
        device=device,
    )

    # 建立 dataloaders（注意：data cache 非常大，載入需時間）
    print("[2] Creating dataloaders (loading ~80GB cache, please wait)...")
    import time
    t0 = time.time()
    train_loader, val_loader, _ = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=32,
        num_workers=0,
        filter_clean_to_clean=True,
        compute_snr=False,
    )
    print(f"  Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    print(f"  Data loading took {time.time()-t0:.1f}s")

    # 逐一分析 checkpoints
    print(f"\n[3] Analyzing {len(available)} checkpoints...")
    all_metrics = []

    for idx, (ep, ckpt_path) in enumerate(available):
        print(f"\n  [{idx+1}/{len(available)}] Epoch {ep}: {ckpt_path.name}")
        metrics = load_checkpoint_and_collect(
            model, ckpt_path, device, train_loader, val_loader,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )
        all_metrics.append(metrics)

        # 即時輸出
        vs = metrics['val_student']
        ts = metrics['train_student']
        print(f"    Val  Student: entropy={vs['entropy_bits']:.2f}, "
              f"top10={vs['top_10_mass_pct']:.1f}%, "
              f"used={vs['used_codes']}/{CODEBOOK_SIZE}")
        print(f"    Train Student: entropy={ts['entropy_bits']:.2f}, "
              f"top10={ts['top_10_mass_pct']:.1f}%, "
              f"used={ts['used_codes']}/{CODEBOOK_SIZE}")

    # 儲存原始數據
    print("\n[4] Saving results...")
    results = {
        'experiment': 'baseline_multi_epoch_trend',
        'baseline': 'exp_k_v6',
        'codebook_size': CODEBOOK_SIZE,
        'architecture': 'single_VQ_frozen_codebook_LoRA_distillation',
        'analyzed_at': datetime.now().isoformat(),
        'epochs_analyzed': [m['epoch'] for m in all_metrics],
        'max_train_batches': args.max_train_batches,
        'max_val_batches': args.max_val_batches,
        'metrics': all_metrics,
    }

    results_path = output_dir / 'baseline_multi_epoch_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ {results_path}")

    # 繪圖
    print("\n[5] Generating trend plots...")
    plot_multi_epoch_trends(all_metrics, output_dir)

    # 初始 vs 最終對比
    if len(all_metrics) >= 2:
        print("\n[6] Generating initial vs final comparison...")
        plot_initial_vs_trained(all_metrics[0], all_metrics[-1], output_dir)

    # 與 schemeB 對比
    if SCHEMEB_EPOCH_METRICS.exists():
        print("\n[7] Loading SchemeB data for comparison...")
        with open(SCHEMEB_EPOCH_METRICS) as f:
            schemeb_data = json.load(f)
        plot_comparison_with_schemeb(all_metrics, schemeb_data, output_dir)
    else:
        print(f"\n[7] SchemeB metrics not found: {SCHEMEB_EPOCH_METRICS}")

    # 輸出 summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Epoch':>6} | {'Val Entropy':>12} | {'Val Top10%':>10} | {'Val Used':>9} | "
          f"{'Train Entropy':>14} | {'Train Top10%':>11} | {'Train Used':>10} | {'Val Acc':>8}")
    print("-" * 100)

    for m in all_metrics:
        vs = m['val_student']
        ts = m['train_student']
        vacc = f"{m['val_acc']:.4f}" if m['val_acc'] is not None else "N/A"
        print(f"{m['epoch']:>6} | {vs['entropy_bits']:>12.2f} | {vs['top_10_mass_pct']:>10.2f} | "
              f"{vs['used_codes']:>9} | {ts['entropy_bits']:>14.2f} | "
              f"{ts['top_10_mass_pct']:>11.2f} | {ts['used_codes']:>10} | {vacc:>8}")

    print("\n" + "=" * 80)
    print(f"✓ 完成！所有結果在: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
