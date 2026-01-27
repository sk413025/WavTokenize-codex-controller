import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import soundfile as sf
import librosa

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from pesq import pesq
from pystoi import stoi

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1226.data_curriculum import CurriculumDataset
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, VAL_CACHE

SAMPLE_RATE = 24000


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
                major, _ = torch.cuda.get_device_capability(idx)
                if major >= 6:
                    return f'cuda:{idx}'
        except Exception:
            pass
    return 'cpu'


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            return 'model_state_dict'
        if 'lora_state_dict' in ckpt:
            model.load_state_dict(ckpt['lora_state_dict'], strict=False)
            return 'lora_state_dict'
    model.load_state_dict(ckpt, strict=False)
    return 'raw_state_dict'


def decode_codes(wavtokenizer, codes, device):
    if codes.dim() == 2:
        codes = codes.unsqueeze(0)
    features = wavtokenizer.codes_to_features(codes)
    bandwidth_id = torch.tensor([0], device=device)
    audio = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    return audio.squeeze(0)


def encode_audio_to_tokens(wavtokenizer, audio, device):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(1)
    bandwidth_id = torch.tensor([0], device=device)
    _, codes = wavtokenizer.encode_infer(audio, bandwidth_id=bandwidth_id)
    return codes


def compute_si_sdr(estimate, reference):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    dot = torch.sum(estimate * reference)
    s_target = dot * reference / (torch.sum(reference ** 2) + 1e-8)
    e_noise = estimate - s_target
    si_sdr = 10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )
    return si_sdr.item()


def compute_metrics(clean, audio, sample_rate=SAMPLE_RATE):
    min_len = min(len(clean), len(audio))
    clean = clean[:min_len]
    audio = audio[:min_len]

    clean_np = clean.detach().cpu().numpy()
    audio_np = audio.detach().cpu().numpy()

    clean_16k = librosa.resample(clean_np, orig_sr=sample_rate, target_sr=16000)
    audio_16k = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)

    try:
        pesq_score = pesq(16000, clean_16k, audio_16k, 'wb')
    except Exception:
        pesq_score = float('nan')

    try:
        stoi_score = stoi(clean_np, audio_np, sample_rate, extended=False)
    except Exception:
        stoi_score = float('nan')

    si_sdr_score = compute_si_sdr(torch.from_numpy(audio_np).float(),
                                   torch.from_numpy(clean_np).float())

    return {
        'pesq': float(pesq_score),
        'stoi': float(stoi_score),
        'si_sdr': float(si_sdr_score),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--failure_set', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/failure_set.json')
    parser.add_argument('--output_dir', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/audio_quality/failure_set')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_audio', action='store_true')
    parser.add_argument('--bottom_n', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compute_teacher', action='store_true')
    parser.add_argument('--compute_noisy_vq', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    print(f'Using device: {device}')

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / 'best_model.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    failure_payload = json.loads(Path(args.failure_set).read_text())
    indices = [x['index'] for x in failure_payload.get('failure_set', [])]
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    if len(indices) == 0:
        raise ValueError('No failure indices found')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = output_dir / 'audio'
    if args.save_audio:
        for sub in ['clean', 'noisy', 'student', 'teacher_vq', 'noisy_vq']:
            (audio_dir / sub).mkdir(parents=True, exist_ok=True)

    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        lora_dropout=0.2,
        intermediate_indices=[3, 4, 6],
    )
    ckpt_type = load_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()

    dataset = CurriculumDataset(VAL_CACHE, compute_snr=False)

    results = []
    for idx in indices:
        item = dataset[idx]
        noisy = item['noisy_audio'].to(device)
        clean = item['clean_audio'].to(device)

        with torch.no_grad():
            output = model(noisy.unsqueeze(0), clean.unsqueeze(0))
            student_audio = decode_codes(model.student, output['student_codes'], device)
            teacher_audio = None
            noisy_vq_audio = None
            if args.compute_teacher:
                teacher_audio = decode_codes(model.teacher, output['teacher_codes'], device)
            if args.compute_noisy_vq:
                noisy_codes = encode_audio_to_tokens(model.teacher, noisy, device)
                noisy_vq_audio = decode_codes(model.teacher, noisy_codes, device)

        # Align lengths
        min_len = min(len(clean), len(noisy), len(student_audio))
        clean = clean[:min_len].detach().cpu()
        noisy = noisy[:min_len].detach().cpu()
        student_audio = student_audio[:min_len].detach().cpu()
        if teacher_audio is not None:
            teacher_audio = teacher_audio[:min_len].detach().cpu()
        if noisy_vq_audio is not None:
            noisy_vq_audio = noisy_vq_audio[:min_len].detach().cpu()

        metrics = {
            'student': compute_metrics(clean, student_audio),
            'noisy': compute_metrics(clean, noisy),
        }
        if teacher_audio is not None:
            metrics['teacher_vq'] = compute_metrics(clean, teacher_audio)
        if noisy_vq_audio is not None:
            metrics['noisy_vq'] = compute_metrics(clean, noisy_vq_audio)

        if args.save_audio:
            sf.write(audio_dir / 'clean' / f'{idx:05d}.wav', clean.numpy(), SAMPLE_RATE)
            sf.write(audio_dir / 'noisy' / f'{idx:05d}.wav', noisy.numpy(), SAMPLE_RATE)
            sf.write(audio_dir / 'student' / f'{idx:05d}.wav', student_audio.numpy(), SAMPLE_RATE)
            if teacher_audio is not None:
                sf.write(audio_dir / 'teacher_vq' / f'{idx:05d}.wav', teacher_audio.numpy(), SAMPLE_RATE)
            if noisy_vq_audio is not None:
                sf.write(audio_dir / 'noisy_vq' / f'{idx:05d}.wav', noisy_vq_audio.numpy(), SAMPLE_RATE)

        sample_meta = item.get('noisy_path', None)
        clean_meta = item.get('clean_path', None)
        results.append({
            'index': int(idx),
            'noisy_path': sample_meta,
            'clean_path': clean_meta,
            'metrics': metrics,
        })

        print(f'[{len(results)}/{len(indices)}] idx={idx} '
              f'PESQ={metrics["student"]["pesq"]:.3f} '
              f'STOI={metrics["student"]["stoi"]:.3f} '
              f'SI-SDR={metrics["student"]["si_sdr"]:.2f}')

    # Summary stats
    def summarize(key):
        vals = [r['metrics']['student'][key] for r in results if not np.isnan(r['metrics']['student'][key])]
        if not vals:
            return {'mean': float('nan'), 'median': float('nan')}
        return {'mean': float(np.mean(vals)), 'median': float(np.median(vals))}

    summary = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoint': str(checkpoint),
        'checkpoint_type': ckpt_type,
        'num_samples': len(results),
        'student_pesq': summarize('pesq'),
        'student_stoi': summarize('stoi'),
        'student_si_sdr': summarize('si_sdr'),
    }

    # Bottom-N subsets
    bottom_sets = {}
    for metric in ['pesq', 'stoi', 'si_sdr']:
        ranked = sorted(results, key=lambda r: r['metrics']['student'][metric])
        bottom = ranked[:min(args.bottom_n, len(ranked))]
        bottom_sets[metric] = [r['index'] for r in bottom]

    # Save outputs
    (output_dir / 'failure_set_metrics.json').write_text(json.dumps({
        'summary': summary,
        'results': results,
        'bottom_sets': bottom_sets,
    }, indent=2))

    # Write a failure_set-compatible file for bottom PESQ (default)
    bottom_key = 'pesq'
    bottom_indices = bottom_sets[bottom_key]
    bottom_failure = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'source_failure_set': args.failure_set,
        'bottom_metric': bottom_key,
        'bottom_n': len(bottom_indices),
        'failure_set': [{'index': int(i)} for i in bottom_indices],
    }
    (output_dir / 'failure_set_bottom_pesq.json').write_text(json.dumps(bottom_failure, indent=2))

    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    summary_md = [
        '# Failure Set Audio Quality Summary',
        f'- num_samples: {summary["num_samples"]}',
        f'- student PESQ mean/median: {summary["student_pesq"]["mean"]:.3f} / {summary["student_pesq"]["median"]:.3f}',
        f'- student STOI mean/median: {summary["student_stoi"]["mean"]:.3f} / {summary["student_stoi"]["median"]:.3f}',
        f'- student SI-SDR mean/median: {summary["student_si_sdr"]["mean"]:.2f} / {summary["student_si_sdr"]["median"]:.2f}',
        f'- bottom_n={len(bottom_indices)} by {bottom_key} -> failure_set_bottom_pesq.json',
    ]
    (output_dir / 'summary.md').write_text('\n'.join(summary_md) + '\n')

    print('Saved metrics to:', output_dir / 'failure_set_metrics.json')
    print('Saved summary to:', output_dir / 'summary.md')


if __name__ == '__main__':
    main()
