import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    run_dir = Path('exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    out_dir = Path('exp_0112_intermediate/analysis/train_valid_gap_58a9b71')
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_path = run_dir / 'history.json'
    with open(hist_path) as f:
        hist = json.load(f)

    train = hist['train_masked_acc']
    val = hist['val_masked_acc']
    assert len(train) == len(val)

    epochs = list(range(1, len(train) + 1))
    gap = [t - v for t, v in zip(train, val)]

    best_idx = max(range(len(val)), key=lambda i: val[i])
    best_epoch = best_idx + 1
    best_val = val[best_idx]
    train_at_best = train[best_idx]

    final_epoch = len(val)
    final_val = val[-1]
    train_final = train[-1]

    best_gap = train_at_best - best_val
    final_gap = train_final - final_val

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, gap, label='train - val')
    plt.axhline(0, color='gray', linewidth=1, linestyle='--')
    plt.scatter([best_epoch], [best_gap], color='red', zorder=3, label=f'best epoch {best_epoch}')
    plt.scatter([final_epoch], [final_gap], color='black', zorder=3, label=f'final epoch {final_epoch}')
    plt.title('Masked Accuracy Gap (train - val)')
    plt.xlabel('Epoch')
    plt.ylabel('Gap')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

    out_path = out_dir / 'gap_curve.png'
    plt.savefig(out_path, dpi=150)

    summary = {
        'best_epoch': best_epoch,
        'best_val_masked_acc': best_val,
        'train_masked_acc_at_best': train_at_best,
        'gap_at_best': best_gap,
        'final_epoch': final_epoch,
        'final_val_masked_acc': final_val,
        'train_masked_acc_at_final': train_final,
        'gap_at_final': final_gap,
    }
    with open(out_dir / 'gap_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
