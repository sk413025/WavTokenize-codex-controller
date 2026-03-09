"""
Baseline (exp_k_v6) 自身的 Epoch 演化分析

只看 baseline 本身（不與 SchemeB 或其他模型比較）。
目標：
1. 對多個 epoch checkpoint 收集 student/teacher 的 token 分布
2. 畫出 baseline 自身的 epoch-over-epoch 趨勢（entropy, top-K, used_codes）
3. 對比初始 (epoch 10) vs 訓練後 (epoch 300) 的分布形狀差異
4. 產出類似 FINAL_metrics_comparison_all.png / FINAL_top20_all_splits.png 風格的圖

修正：
- batch_size=8 避免 OOM
- flatten codes 前先處理 variable-length (1D list → cat)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys
import json
import argparse
import gc
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1226.data_curriculum import create_curriculum_dataloaders

# ========== 設定 ==========
BASELINE_CKPT_DIR = Path(
    "/home/sbplab/ruizi/WavTokenize-feature-analysis"
    "/exp_0112_intermediate/runs"
    "/exp_k_v6_20260125_234609_20260125_234613/checkpoints"
)
OUTPUT_DIR = Path(__file__).parent / "baseline_epoch_evolution"
DEVICE = "cuda:0"
CODEBOOK_SIZE = 4096
FUNC_NAME = "analyze_baseline_epoch_evolution"

LORA_CONFIG = {
    'rank': 256,
    'alpha': 512,
    'dropout': 0.2,
    'intermediate_indices': [3, 6],
}


# =====================================================================
# Token collection (memory-safe)
# =====================================================================

def collect_tokens(model, data_loader, device, max_batches=None, split_name=''):
    """收集 student / teacher token IDs，正確處理 variable-length。

    Args:
        model: TeacherStudentIntermediate 模型。
        data_loader: DataLoader。
        device: 計算裝置。
        max_batches: 最大 batch 數。
        split_name: 用於 tqdm 標籤。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (student_codes_1d, teacher_codes_1d)
    """
    model.eval()
    student_list: list[torch.Tensor] = []
    teacher_list: list[torch.Tensor] = []

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
                            student_list.append(sc[b, :L].reshape(-1))
                            teacher_list.append(tc[b, :L].reshape(-1))
                else:
                    # 逐樣本 append 成 1-D，避免 shape 不一致
                    for b in range(sc.shape[0]):
                        student_list.append(sc[b].reshape(-1))
                        teacher_list.append(tc[b].reshape(-1))

                # 定期清理
                del noisy, clean, out, sc, tc
                if i % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                if 'CUDA out of memory' in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                continue

    model.train()
    if not student_list:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    return torch.cat(student_list), torch.cat(teacher_list)


# =====================================================================
# Metrics
# =====================================================================

def compute_metrics(codes, codebook_size):
    """計算 token 分布指標。

    Args:
        codes: 1-D tensor of token IDs。
        codebook_size: codebook 大小。

    Returns:
        dict: entropy_bits, top_K_mass_pct, used_codes, counts 等。
    """
    if len(codes) == 0:
        return {
            'total_tokens': 0, 'entropy_bits': 0,
            'top_1_mass_pct': 0, 'top_10_mass_pct': 0,
            'top_50_mass_pct': 0, 'top_100_mass_pct': 0,
            'used_codes': 0, 'usage_pct': 0,
            'counts': np.zeros(codebook_size, dtype=np.int64),
            'sorted_counts': np.zeros(codebook_size, dtype=np.int64),
            'sorted_indices': np.arange(codebook_size),
        }
    codes_np = codes.numpy()
    counts = np.bincount(codes_np, minlength=codebook_size)
    probs = counts / counts.sum()
    sorted_idx = np.argsort(-counts)
    sorted_counts = counts[sorted_idx]
    sorted_probs = probs[sorted_idx]

    nz = probs[probs > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))

    return {
        'total_tokens': int(len(codes)),
        'entropy_bits': entropy,
        'top_1_mass_pct': float(sorted_probs[:1].sum() * 100),
        'top_10_mass_pct': float(sorted_probs[:10].sum() * 100),
        'top_50_mass_pct': float(sorted_probs[:50].sum() * 100),
        'top_100_mass_pct': float(sorted_probs[:100].sum() * 100),
        'used_codes': int((counts > 0).sum()),
        'usage_pct': float((counts > 0).sum() / codebook_size * 100),
        'counts': counts,
        'sorted_counts': sorted_counts,
        'sorted_indices': sorted_idx,
    }


# =====================================================================
# Plotting helpers
# =====================================================================

def _plot_metrics_comparison_one_epoch(metrics_dict, epoch, output_dir):
    """仿 FINAL_metrics_comparison_all.png，畫一個 epoch 的 4-split 指標對比。

    Args:
        metrics_dict: {'train_student': {...}, 'val_student': {...},
                       'train_teacher': {...}, 'val_teacher': {...}}
        epoch: int
        output_dir: Path
    """
    date_str = datetime.now().strftime('%Y%m%d')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metrics_info = [
        ('entropy_bits', 'Entropy (bits)', 12.0),
        ('top_10_mass_pct', 'Top-10 Mass (%)', 10/CODEBOOK_SIZE*100),
        ('top_50_mass_pct', 'Top-50 Mass (%)', 50/CODEBOOK_SIZE*100),
        ('top_100_mass_pct', 'Top-100 Mass (%)', 100/CODEBOOK_SIZE*100),
        ('used_codes', 'Used Codes', CODEBOOK_SIZE),
        ('usage_pct', 'Codebook Usage (%)', 100.0),
    ]

    labels = ['Student\nTrain', 'Student\nVal', 'Teacher\nTrain', 'Teacher\nVal', 'Ideal']
    splits = ['train_student', 'val_student', 'train_teacher', 'val_teacher']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, (mk, mname, ideal) in enumerate(metrics_info):
        ax = axes[idx // 3, idx % 3]
        values = [metrics_dict[s][mk] for s in splits] + [ideal]

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel(mname, fontsize=11, fontweight='bold')
        ax.set_title(f'{mname} @ Epoch {epoch}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            h = bar.get_height()
            fmt = f'{val:.2f}' if val < 100 else f'{int(val)}'
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle(f'Baseline (exp_k_v6) Token Distribution Metrics — Epoch {epoch}',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fname = f'baseline_metrics_epoch{epoch:03d}_{date_str}_{FUNC_NAME}.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


def _plot_top20_one_epoch(metrics_dict, epoch, output_dir):
    """仿 FINAL_top20_all_splits.png，畫一個 epoch 的 Top-20 token 排名。

    Args:
        metrics_dict: 含 counts / sorted_indices 的 dict。
        epoch: int
        output_dir: Path
    """
    date_str = datetime.now().strftime('%Y%m%d')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    split_info = [
        ('train_student', 'Student Train', '#e74c3c', axes[0, 0]),
        ('val_student', 'Student Val', '#3498db', axes[0, 1]),
        ('train_teacher', 'Teacher Train', '#2ecc71', axes[1, 0]),
        ('val_teacher', 'Teacher Val', '#f39c12', axes[1, 1]),
    ]

    for split, title, color, ax in split_info:
        m = metrics_dict[split]
        sorted_idx = m['sorted_indices'][:20]
        sorted_cnt = m['sorted_counts'][:20]
        total = m['total_tokens']
        freqs = sorted_cnt / total * 100 if total > 0 else sorted_cnt * 0.0

        x = np.arange(20)
        bars = ax.bar(x, freqs, color=color, alpha=0.85,
                      edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Token Rank', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}: Top-20 Tokens @ Epoch {epoch}',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"#{i+1}" for i in x[::2]])
        ax.grid(True, alpha=0.3, axis='y')

        # Token IDs on top 10 bars
        for j in range(min(10, len(bars))):
            h = bars[j].get_height()
            tid = int(sorted_idx[j])
            ax.text(bars[j].get_x() + bars[j].get_width()/2., h + h*0.02,
                    f'T{tid}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        # Stats box
        textstr = (f"Entropy: {m['entropy_bits']:.2f}\n"
                   f"Top-10: {m['top_10_mass_pct']:.1f}%\n"
                   f"Used: {m['used_codes']}")
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle(f'Baseline (exp_k_v6) Top-20 Tokens — Epoch {epoch}',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fname = f'baseline_top20_epoch{epoch:03d}_{date_str}_{FUNC_NAME}.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


def plot_baseline_trend(all_results, output_dir):
    """畫 baseline 自身的多 epoch 趨勢圖（不比較其他模型）。

    Args:
        all_results: list of (epoch, metrics_dict) 按 epoch 排序。
        output_dir: Path
    """
    date_str = datetime.now().strftime('%Y%m%d')
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    epochs = [r[0] for r in all_results]

    splits = ['train_student', 'val_student', 'train_teacher', 'val_teacher']
    labels = {
        'train_student': 'Student Train', 'val_student': 'Student Val',
        'train_teacher': 'Teacher Train', 'val_teacher': 'Teacher Val',
    }
    colors = {
        'train_student': '#e74c3c', 'val_student': '#3498db',
        'train_teacher': '#2ecc71', 'val_teacher': '#f39c12',
    }
    markers = {
        'train_student': 'o', 'val_student': 's',
        'train_teacher': '^', 'val_teacher': 'D',
    }

    trend_metrics = [
        ('entropy_bits', 'Entropy (bits)', np.log2(CODEBOOK_SIZE), 'Ideal (uniform)'),
        ('top_1_mass_pct', 'Top-1 Mass (%)', 1/CODEBOOK_SIZE*100, 'Ideal (uniform)'),
        ('top_10_mass_pct', 'Top-10 Mass (%)', 10/CODEBOOK_SIZE*100, 'Ideal (uniform)'),
        ('used_codes', 'Used Codes', CODEBOOK_SIZE, f'Max ({CODEBOOK_SIZE})'),
        ('usage_pct', 'Usage (%)', 100.0, 'Full usage'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    for idx, (mk, ml, ideal, ideal_label) in enumerate(trend_metrics):
        ax = axes[idx // 3, idx % 3]
        for split in splits:
            vals = [r[1][split][mk] for r in all_results]
            ax.plot(epochs, vals, marker=markers[split], linestyle='-',
                    color=colors[split], label=labels[split],
                    linewidth=2, markersize=6, alpha=0.85)

        ax.axhline(y=ideal, color='#9b59b6', linestyle=':', linewidth=1.5,
                   alpha=0.6, label=ideal_label)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ml, fontsize=12)
        ax.set_title(f'Baseline: {ml} vs Epoch', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # 第 6 格: val_acc / train_acc 趨勢（從 checkpoint metadata）
    ax = axes[1, 2]
    val_accs = []
    train_accs = []
    for ep, md in all_results:
        val_accs.append(md.get('val_acc', None))
        train_accs.append(md.get('train_acc', None))

    if any(v is not None for v in val_accs):
        ax.plot(epochs, [v*100 if v else 0 for v in val_accs], 'o-',
                color='#e74c3c', linewidth=2, markersize=6, label='Val Accuracy')
        ax.plot(epochs, [v*100 if v else 0 for v in train_accs], 's-',
                color='#3498db', linewidth=2, markersize=6, label='Train Accuracy')
        ax.axhline(y=1/CODEBOOK_SIZE*100, color='gray', linestyle=':', alpha=0.5,
                   label=f'Random ({1/CODEBOOK_SIZE*100:.4f}%)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Token Accuracy (%)', fontsize=12)
        ax.set_title('Baseline: Token Accuracy vs Epoch', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Baseline (exp_k_v6) Self-Evolution: Token Distribution Across Epochs',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = f'baseline_self_trend_{date_str}_{FUNC_NAME}.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


def plot_frequency_comparison_epochs(all_results, output_dir):
    """把多個 epoch 的 log-log 頻率分布疊在同一張圖上。

    Args:
        all_results: list of (epoch, metrics_dict)
        output_dir: Path
    """
    date_str = datetime.now().strftime('%Y%m%d')
    cmap = plt.cm.viridis
    n = len(all_results)

    for split in ['val_student', 'train_student']:
        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
        for i, (ep, md) in enumerate(all_results):
            m = md[split]
            total = m['total_tokens']
            if total == 0:
                continue
            freqs = m['sorted_counts'] / total * 100
            ranks = np.arange(1, len(freqs) + 1)
            # 只畫有值的
            mask = freqs > 0
            color = cmap(i / max(n - 1, 1))
            ax.loglog(ranks[mask], freqs[mask], '-', color=color,
                      linewidth=1.8, alpha=0.8, label=f'Epoch {ep}')

        # Ideal
        ax.axhline(y=100/CODEBOOK_SIZE, color='purple', linestyle='--',
                   linewidth=2, alpha=0.5, label='Ideal uniform')

        title_split = 'Student Val' if split == 'val_student' else 'Student Train'
        ax.set_xlabel('Token Rank (log)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Frequency % (log)', fontsize=13, fontweight='bold')
        ax.set_title(f'Baseline {title_split}: Token Frequency Across Epochs',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        fname = f'baseline_{split}_freq_evolution_{date_str}_{FUNC_NAME}.png'
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {fname}")


def plot_top_token_evolution(all_results, output_dir):
    """追蹤特定 dominant token 在不同 epoch 的佔比變化。

    Args:
        all_results: list of (epoch, metrics_dict)
        output_dir: Path
    """
    date_str = datetime.now().strftime('%Y%m%d')

    for split in ['val_student', 'train_student']:
        # 從最後一個 epoch 取 top-5 token ID
        last_md = all_results[-1][1][split]
        top5_ids = last_md['sorted_indices'][:5]

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        epochs_arr = [r[0] for r in all_results]
        cmap = plt.cm.tab10

        for rank, tid in enumerate(top5_ids):
            freqs = []
            for ep, md in all_results:
                m = md[split]
                total = m['total_tokens']
                freq = m['counts'][tid] / total * 100 if total > 0 else 0
                freqs.append(freq)
            ax.plot(epochs_arr, freqs, 'o-', color=cmap(rank), linewidth=2,
                    markersize=7, label=f'T{tid} (rank #{rank+1}@ep{all_results[-1][0]})')

        ax.axhline(y=100/CODEBOOK_SIZE, color='gray', linestyle=':', alpha=0.5,
                   label='Ideal per-token')
        title_split = 'Student Val' if split == 'val_student' else 'Student Train'
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Frequency (%)', fontsize=12)
        ax.set_title(f'Baseline {title_split}: Top-5 Token Frequency Across Epochs',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'baseline_{split}_top5_evolution_{date_str}_{FUNC_NAME}.png'
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {fname}")


# =====================================================================
# Main
# =====================================================================

def main():
    """主函式：Baseline 自身的多 epoch 演化分析。"""
    parser = argparse.ArgumentParser(description='Baseline 自身 Epoch 演化分析')
    parser.add_argument('--epochs', type=str,
                        default='10,50,100,150,200,300',
                        help='要分析的 epoch (逗號分隔)')
    parser.add_argument('--max_train_batches', type=int, default=40,
                        help='每個 epoch 的 train batch 上限')
    parser.add_argument('--max_val_batches', type=int, default=None,
                        help='每個 epoch 的 val batch 上限 (None=全部)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='DataLoader batch size (小值避免 OOM)')
    parser.add_argument('--device', type=str, default=DEVICE)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    epoch_list = [int(e) for e in args.epochs.split(',')]

    print("=" * 80)
    print("Baseline (exp_k_v6) Epoch 演化分析 — 只看 baseline 自身")
    print("=" * 80)
    print(f"  Epochs: {epoch_list}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  max_train_batches: {args.max_train_batches}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_dir}")

    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 驗證 checkpoints
    available = []
    for ep in epoch_list:
        p = BASELINE_CKPT_DIR / f"checkpoint_epoch{ep:03d}.pt"
        if p.exists():
            available.append((ep, p))
        else:
            print(f"  ⚠ Missing: {p}")
    print(f"  Available: {len(available)}/{len(epoch_list)}")

    # 載入模型
    print("\n[1/5] Loading model architecture...")
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=LORA_CONFIG['rank'],
        lora_alpha=LORA_CONFIG['alpha'],
        lora_dropout=LORA_CONFIG['dropout'],
        intermediate_indices=LORA_CONFIG['intermediate_indices'],
        device=device,
    )

    # 載入資料
    print("[2/5] Loading data (this may take a few minutes for 63 GB cache)...")
    import time
    t0 = time.time()
    train_loader, val_loader, _ = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=0,
        filter_clean_to_clean=True,
        compute_snr=False,
    )
    print(f"  ✓ Train: {len(train_loader)} batches, Val: {len(val_loader)} batches  ({time.time()-t0:.0f}s)")

    # 逐 epoch 分析
    print(f"\n[3/5] Analyzing {len(available)} checkpoints...")
    all_results: list[tuple[int, dict]] = []

    for idx, (ep, ckpt_path) in enumerate(available):
        print(f"\n  [{idx+1}/{len(available)}] Epoch {ep}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        lora_state = {}
        for k, v in ckpt['lora_state_dict'].items():
            key = k[8:] if k.startswith('student.') else k
            lora_state[key] = v
        model.student.load_state_dict(lora_state, strict=False)

        val_acc = ckpt.get('val_acc', None)
        train_acc = ckpt.get('train_acc', None)

        # 收集 tokens
        train_s, train_t = collect_tokens(model, train_loader, device,
                                          args.max_train_batches, f'ep{ep} train')
        val_s, val_t = collect_tokens(model, val_loader, device,
                                      args.max_val_batches, f'ep{ep} val')

        md = {
            'train_student': compute_metrics(train_s, CODEBOOK_SIZE),
            'val_student': compute_metrics(val_s, CODEBOOK_SIZE),
            'train_teacher': compute_metrics(train_t, CODEBOOK_SIZE),
            'val_teacher': compute_metrics(val_t, CODEBOOK_SIZE),
            'val_acc': float(val_acc) if val_acc is not None else None,
            'train_acc': float(train_acc) if train_acc is not None else None,
        }
        all_results.append((ep, md))

        # 即時 print
        vs = md['val_student']
        ts = md['train_student']
        vt = md['val_teacher']
        print(f"    Val  Student: entropy={vs['entropy_bits']:.2f}, "
              f"top10={vs['top_10_mass_pct']:.1f}%, used={vs['used_codes']}")
        print(f"    Train Student: entropy={ts['entropy_bits']:.2f}, "
              f"top10={ts['top_10_mass_pct']:.1f}%, used={ts['used_codes']}")
        print(f"    Val  Teacher: entropy={vt['entropy_bits']:.2f}, "
              f"top10={vt['top_10_mass_pct']:.1f}%, used={vt['used_codes']}")

        # 清理 GPU
        del ckpt, train_s, train_t, val_s, val_t
        torch.cuda.empty_cache()
        gc.collect()

    # 排序
    all_results.sort(key=lambda x: x[0])

    # 儲存 JSON（不含 numpy arrays）
    print("\n[4/5] Saving JSON...")
    json_results = []
    for ep, md in all_results:
        entry = {'epoch': ep, 'val_acc': md.get('val_acc'), 'train_acc': md.get('train_acc')}
        for split in ['train_student', 'val_student', 'train_teacher', 'val_teacher']:
            entry[split] = {k: v for k, v in md[split].items()
                            if k not in ('counts', 'sorted_counts', 'sorted_indices')}
        json_results.append(entry)

    json_path = output_dir / 'baseline_epoch_evolution.json'
    with open(json_path, 'w') as f:
        json.dump({
            'experiment': 'baseline_epoch_evolution',
            'baseline': 'exp_k_v6',
            'codebook_size': CODEBOOK_SIZE,
            'architecture': 'single_VQ_K4096_frozen_codebook_LoRA',
            'analyzed_at': datetime.now().isoformat(),
            'results': json_results,
        }, f, indent=2)
    print(f"  ✓ {json_path}")

    # 畫圖
    print("\n[5/5] Generating plots...")

    # (a) 每個 epoch 的 FINAL-style metrics comparison & top-20
    for ep, md in all_results:
        _plot_metrics_comparison_one_epoch(md, ep, output_dir)
        _plot_top20_one_epoch(md, ep, output_dir)

    # (b) 自身趨勢
    plot_baseline_trend(all_results, output_dir)

    # (c) Log-log 頻率演化疊圖
    plot_frequency_comparison_epochs(all_results, output_dir)

    # (d) Top-5 dominant token 演化
    plot_top_token_evolution(all_results, output_dir)

    # Summary table
    print("\n" + "=" * 110)
    print("BASELINE SELF-EVOLUTION SUMMARY")
    print("=" * 110)
    header = (f"{'Epoch':>5} | {'ValAcc%':>7} | "
              f"{'S-Val Ent':>9} | {'S-Val T10%':>10} | {'S-Val Used':>10} | "
              f"{'S-Trn Ent':>9} | {'S-Trn T10%':>10} | {'S-Trn Used':>10} | "
              f"{'T-Val Ent':>9} | {'T-Val Used':>10}")
    print(header)
    print("-" * 110)

    for ep, md in all_results:
        vs = md['val_student']
        ts = md['train_student']
        vt = md['val_teacher']
        va = f"{md['val_acc']*100:.3f}" if md.get('val_acc') else 'N/A'
        print(f"{ep:>5} | {va:>7} | "
              f"{vs['entropy_bits']:>9.2f} | {vs['top_10_mass_pct']:>10.1f} | {vs['used_codes']:>10} | "
              f"{ts['entropy_bits']:>9.2f} | {ts['top_10_mass_pct']:>10.1f} | {ts['used_codes']:>10} | "
              f"{vt['entropy_bits']:>9.2f} | {vt['used_codes']:>10}")

    print("\n" + "=" * 80)
    print(f"✓ 完成！所有結果在: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
