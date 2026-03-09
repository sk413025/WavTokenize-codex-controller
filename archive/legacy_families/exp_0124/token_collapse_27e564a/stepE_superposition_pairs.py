import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, VAL_CACHE
from exp_1219.losses import create_length_mask
from exp_1212.data_aligned import AlignedNoisyCleanPairDataset

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


def topk_mass_from_codes(codes, k=10):
    counts = torch.bincount(codes, minlength=NUM_CODES).float()
    total = counts.sum().item()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(torch.topk(probs, k=min(k, probs.numel())).values.sum().item())


def compute_strict_acc(s_codes, t_codes, lengths):
    if s_codes.dim() == 3:
        s_codes = s_codes[0]
    if t_codes.dim() == 3:
        t_codes = t_codes[0]

    # align length if mismatch
    T = min(s_codes.shape[1], t_codes.shape[1])
    s_codes = s_codes[:, :T]
    t_codes = t_codes[:, :T]

    max_audio_len = T * ENCODER_STRIDE
    mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=s_codes.device)
    correct = (s_codes == t_codes).float() * mask
    acc = correct.sum().item() / (mask.sum().item() + 1e-8)
    return acc, mask


def token_change_rate(codes_a, codes_b, lengths):
    if codes_a.dim() == 3:
        codes_a = codes_a[0]
    if codes_b.dim() == 3:
        codes_b = codes_b[0]
    T = min(codes_a.shape[1], codes_b.shape[1])
    codes_a = codes_a[:, :T]
    codes_b = codes_b[:, :T]
    max_audio_len = T * ENCODER_STRIDE
    mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=codes_a.device)
    diff = (codes_a != codes_b).float() * mask
    return diff.sum().item() / (mask.sum().item() + 1e-8)


def mix_with_snr(clean, noise, snr_db):
    # ensure same length
    min_len = min(clean.numel(), noise.numel())
    clean = clean[:min_len]
    noise = noise[:min_len]
    # scale noise to target SNR
    signal_power = (clean ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return clean
    target_ratio = 10 ** (-snr_db / 10.0)
    scale = torch.sqrt(signal_power * target_ratio / (noise_power + 1e-12))
    return clean + scale * noise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_clean', type=int, default=30)
    parser.add_argument('--num_noise_per_clean', type=int, default=3)
    parser.add_argument('--snr_list', type=str, default='0,5,10')
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
    model.eval()

    dataset = AlignedNoisyCleanPairDataset(VAL_CACHE, max_samples=None)
    rng = np.random.RandomState(args.seed)

    clean_indices = rng.choice(len(dataset), size=args.num_clean, replace=False)
    noise_indices = rng.choice(len(dataset), size=args.num_clean * args.num_noise_per_clean, replace=True)
    snr_list = [float(x) for x in args.snr_list.split(',')]

    results = []
    change_rates = []
    alignment_drops = []

    with torch.no_grad():
        for i, idx in enumerate(clean_indices):
            clean_item = dataset[idx]
            clean_audio = clean_item['clean_audio']
            length = torch.tensor([clean_item['length']], device=device)

            # baseline: student on clean
            noisy_in = clean_audio.unsqueeze(0).to(device)
            clean_in = clean_audio.unsqueeze(0).to(device)
            base_out = model(noisy_in, clean_in)
            base_acc, mask = compute_strict_acc(base_out['student_codes'], base_out['teacher_codes'], length)

            views = []
            view_codes = []

            for j in range(args.num_noise_per_clean):
                noise_item = dataset[noise_indices[i * args.num_noise_per_clean + j]]
                noise = (noise_item['noisy_audio'] - noise_item['clean_audio'])

                for snr_db in snr_list:
                    mixed = mix_with_snr(clean_audio, noise, snr_db)
                    noisy_in = mixed.unsqueeze(0).to(device)
                    out = model(noisy_in, clean_in)
                    acc, _ = compute_strict_acc(out['student_codes'], out['teacher_codes'], length)

                    s_codes = out['student_codes']
                    if s_codes.dim() == 3:
                        s_codes = s_codes[0]
                    topk = topk_mass_from_codes(s_codes[0].detach().cpu())

                    views.append({
                        'snr_db': snr_db,
                        'acc': acc,
                        'topk_mass': topk,
                    })
                    view_codes.append(s_codes)

            # token change rate between views (pairwise mean)
            if len(view_codes) >= 2:
                diffs = []
                for a in range(len(view_codes)):
                    for b in range(a + 1, len(view_codes)):
                        diffs.append(token_change_rate(view_codes[a], view_codes[b], length))
                change_rate = float(np.mean(diffs))
            else:
                change_rate = 0.0

            mean_acc = float(np.mean([v['acc'] for v in views])) if views else 0.0
            alignment_drop = float(base_acc - mean_acc)

            results.append({
                'clean_index': int(idx),
                'baseline_acc_clean': base_acc,
                'token_change_rate': change_rate,
                'teacher_alignment_drop': alignment_drop,
                'views': views,
            })

            change_rates.append(change_rate)
            alignment_drops.append(alignment_drop)

    # plot
    out_dir = Path(__file__).resolve().parent
    out_json = out_dir / 'superposition_pair_tests.json'
    out_png = out_dir / 'superposition_pair_plots.png'

    plt.figure(figsize=(6, 4))
    plt.scatter(change_rates, alignment_drops, s=20, alpha=0.6)
    plt.xlabel('Token change rate (view vs view)')
    plt.ylabel('Teacher alignment drop')
    plt.title('Superposition pair tests')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'seed': args.seed,
        'num_clean': args.num_clean,
        'num_noise_per_clean': args.num_noise_per_clean,
        'snr_list': snr_list,
        'results': results,
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_png}')


if __name__ == '__main__':
    main()
