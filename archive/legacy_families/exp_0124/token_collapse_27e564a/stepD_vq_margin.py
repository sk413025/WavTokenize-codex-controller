import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1219.losses import create_length_mask
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum

ENCODER_STRIDE = 320
NUM_CODES = 4096


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> str:
    if requested and requested != 'auto':
        return requested
    if torch.cuda.is_available():
        try:
            for idx in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(idx)
                if major >= 7:
                    return f'cuda:{idx}'
        except Exception:
            pass
    return 'cpu'


def build_loader(cache_path, batch_size, num_workers, max_samples=None):
    dataset = CurriculumDataset(cache_path, max_samples=max_samples, compute_snr=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_curriculum,
    )
    return loader


def compute_margin_stats(model, loader, device, split_name, max_batches=None, progress_every=200):
    codebook = model.codebook.to(device)
    margins = []
    accs = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            lengths = batch['lengths'].to(device)

            output = model(noisy, clean)
            s_out = output['student_encoder_out']  # (B, D, T)
            s_codes = output['student_codes']
            t_codes = output['teacher_codes']

            if s_codes.dim() == 3:
                s_codes = s_codes[0]
            if t_codes.dim() == 3:
                t_codes = t_codes[0]

            B, D, T = s_out.shape
            max_audio_len = T * ENCODER_STRIDE
            mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=device)

            # strict acc per sample
            correct = (s_codes == t_codes).float() * mask
            acc_per = (correct.sum(dim=1) / (mask.sum(dim=1) + 1e-8)).detach().cpu().numpy().tolist()
            accs.extend(acc_per)

            # margin per frame
            z = s_out.permute(0, 2, 1).reshape(-1, D)  # (B*T, D)
            # distances to codebook
            dists = torch.cdist(z, codebook)  # (B*T, C)
            d1, _ = torch.topk(dists, k=2, largest=False, dim=1)
            margin = (d1[:, 1] - d1[:, 0]).detach().cpu().numpy()

            mask_flat = mask.reshape(-1).detach().cpu().numpy()
            margin = margin[mask_flat > 0]
            margins.extend(margin.tolist())

            if progress_every and batch_idx % progress_every == 0:
                print(f'[{split_name}] batch {batch_idx}/{len(loader)}')
            if max_batches is not None and batch_idx >= max_batches:
                break

    margins = np.array(margins)
    stats = {
        'count': int(len(margins)),
        'mean': float(np.mean(margins)),
        'std': float(np.std(margins)),
        'p10': float(np.percentile(margins, 10)),
        'p50': float(np.percentile(margins, 50)),
        'p90': float(np.percentile(margins, 90)),
    }
    return stats, margins, accs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress_every', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    checkpoint = args.checkpoint or str(run_dir / 'best_model.pt')

    device = select_device(args.device)
    print(f'Using device: {device}')

    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        lora_dropout=0.2,
    )
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)

    train_loader = build_loader(TRAIN_CACHE, args.batch_size, args.num_workers, args.max_train_samples)
    val_loader = build_loader(VAL_CACHE, args.batch_size, args.num_workers, args.max_val_samples)

    train_stats, train_margins, _ = compute_margin_stats(model, train_loader, device, 'train', progress_every=args.progress_every)
    val_stats, val_margins, _ = compute_margin_stats(model, val_loader, device, 'val', progress_every=args.progress_every)

    out_dir = Path(__file__).resolve().parent
    out_json = out_dir / 'vq_margin_stats_train_val.json'
    out_png = out_dir / 'vq_margin_hist_train_val.png'

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'seed': args.seed,
        'train': train_stats,
        'val': val_stats,
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # histogram
    plt.figure(figsize=(6, 4))
    plt.hist(train_margins, bins=50, alpha=0.6, label='train')
    plt.hist(val_margins, bins=50, alpha=0.6, label='val')
    plt.xlabel('VQ margin (d2 - d1)')
    plt.ylabel('Count')
    plt.title('VQ margin distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_png}')


if __name__ == '__main__':
    main()
