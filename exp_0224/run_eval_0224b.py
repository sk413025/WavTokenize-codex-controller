"""
exp_0224/run_eval_0224b.py

評估 exp_0224b 兩個 checkpoint（ep16 和 ep20）：
  - best_model.pt (epoch 16, val_mse criterion)
  - checkpoint_epoch020.pt (epoch 20, closest to val_total best ep23)

結果寫入 exp_0223/test/comparison_results.json 和 comparison_table.md

執行：
    cd /home/sbplab/ruizi/WavTokenize-feature-analysis
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \
    /home/sbplab/miniconda3/envs/test/bin/python exp_0224/run_eval_0224b.py
"""

import sys, torch, numpy as np, json
from pathlib import Path
from scipy import signal
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, VAL_CACHE
from exp_0216.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum
from torch.utils.data import DataLoader, Subset
from pesq import pesq
from pystoi import stoi as calc_stoi

DEVICE = torch.device('cuda:0')
SR = 24000
SR_PESQ = 8000
TEST_DIR = Path(__file__).parent.parent / 'exp_0223' / 'test'
RUN_DIR = Path(__file__).parent / 'runs' / 'no_vq_dec_lora_epoch_20260224_002834'

BASELINE_JSON = Path(__file__).parent.parent / 'exp_0217/FAIR_BASELINE_PESQ_STOI_n30.json'
with open(BASELINE_JSON) as f:
    baseline_data = json.load(f)
SAMPLE_INDICES = baseline_data['indices'][:3]
print(f"Val indices: {SAMPLE_INDICES}")

CHECKPOINTS = [
    {
        'name': 'exp_0224b_ep16',
        'ckpt': RUN_DIR / 'best_model.pt',
        'note': 'best_model.pt (val_mse criterion, epoch=16)',
    },
    {
        'name': 'exp_0224b_ep20',
        'ckpt': RUN_DIR / 'checkpoint_epoch020.pt',
        'note': 'epoch 20 (closest to val_total best ep23=31.84; training in progress)',
    },
]


def resample_to(wav_np, orig_sr, target_sr):
    n = int(len(wav_np) * target_sr / orig_sr)
    return signal.resample(wav_np, n).astype(np.float32)


def save_wav(arr, path, sr=SR):
    arr = np.clip(arr, -1.0, 1.0)
    wavfile.write(str(path), sr, (arr * 32767).astype(np.int16))


def compute_metrics(clean_np, recon_np, noisy_np):
    c8 = resample_to(clean_np, SR, SR_PESQ)
    r8 = resample_to(recon_np, SR, SR_PESQ)
    n8 = resample_to(noisy_np, SR, SR_PESQ)
    try:
        p_recon = pesq(SR_PESQ, c8, r8, 'nb')
        p_noisy = pesq(SR_PESQ, c8, n8, 'nb')
        s_recon = calc_stoi(clean_np, recon_np, SR, extended=False)
        s_noisy = calc_stoi(clean_np, noisy_np, SR, extended=False)
        return p_recon, p_noisy, s_recon, s_noisy
    except Exception as e:
        print(f"    metrics error: {e}")
        return None, None, None, None


def load_data():
    val_ds = AugmentedCurriculumDataset(
        VAL_CACHE, augment=False, filter_clean_to_clean=True, compute_snr=False
    )
    subset = Subset(val_ds, SAMPLE_INDICES)
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn_curriculum, num_workers=0)
    return loader


def run_eval(ckpt_info, loader):
    from exp_0224.models_no_vq_decoder_lora import TeacherStudentNoVQDecoderLoRA

    name = ckpt_info['name']
    ckpt_path = ckpt_info['ckpt']
    out_dir = TEST_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    print(f"  Note: {ckpt_info['note']}")

    model = TeacherStudentNoVQDecoderLoRA(
        wavtok_config=WAVTOK_CONFIG, wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=64, lora_alpha=128,
        device=DEVICE,
        decoder_lora_rank=32, decoder_lora_alpha=64,
    ).to(DEVICE)

    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    print(f"  Checkpoint epoch: {ckpt.get('epoch')}, metrics: {ckpt.get('metrics')}")
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    metrics_list = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            noisy = batch['noisy_audio'].to(DEVICE)
            clean = batch['clean_audio'].to(DEVICE)
            if clean.dim() == 2: clean = clean.unsqueeze(1)
            if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

            out = model.forward_wav(clean, noisy)
            recon = out['recon_wav']

            T = min(clean.shape[-1], recon.shape[-1])
            c = clean[0, 0, :T].cpu().numpy().astype(np.float32)
            r = recon[0, 0, :T].cpu().numpy().astype(np.float32)
            n = noisy[0, 0, :T].cpu().numpy().astype(np.float32)

            idx = i + 1
            save_wav(n, out_dir / f'sample{idx:02d}_noisy.wav')
            save_wav(r, out_dir / f'sample{idx:02d}_recon.wav')
            save_wav(c, out_dir / f'sample{idx:02d}_clean.wav')

            p_r, p_n, s_r, s_n = compute_metrics(c, r, n)
            metrics_list.append({'pesq_recon': p_r, 'pesq_noisy': p_n,
                                  'stoi_recon': s_r, 'stoi_noisy': s_n})
            print(f"    sample{idx}: PESQ={p_r:.4f} (noisy={p_n:.4f}), STOI={s_r:.4f}")

    del model
    torch.cuda.empty_cache()

    valid = [m for m in metrics_list if m['pesq_recon'] is not None]
    result = {
        'checkpoint': str(ckpt_path.name),
        'note': ckpt_info['note'],
        'pesq_mean': float(np.mean([m['pesq_recon'] for m in valid])),
        'pesq_noisy_mean': float(np.mean([m['pesq_noisy'] for m in valid])),
        'stoi_mean': float(np.mean([m['stoi_recon'] for m in valid])),
        'stoi_noisy_mean': float(np.mean([m['stoi_noisy'] for m in valid])),
        'per_sample': metrics_list,
    }
    print(f"  → PESQ={result['pesq_mean']:.4f}, STOI={result['stoi_mean']:.4f}")
    return name, result


def update_comparison_files(new_results: dict):
    """更新 comparison_results.json 和 comparison_table.md"""
    json_path = TEST_DIR / 'comparison_results.json'
    with open(json_path) as f:
        existing = json.load(f)

    existing.update(new_results)
    json_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"\nJSON updated: {json_path}")

    # 更新 comparison_table.md 的表格部分（append 新行）
    table_path = TEST_DIR / 'comparison_table.md'
    content = table_path.read_text(encoding='utf-8')

    # 找 baseline PESQ
    baseline_pesq = existing.get('noisy_through_teacher', {}).get('pesq_mean', 1.6765)
    baseline_stoi = existing.get('noisy_through_teacher', {}).get('stoi_mean', 0.5266)

    print("\n新增結果摘要：")
    print(f"  基準 (noisy_through_teacher): PESQ={baseline_pesq:.4f}, STOI={baseline_stoi:.4f}")
    for name, r in new_results.items():
        if 'error' not in r:
            dp = r['pesq_mean'] - baseline_pesq
            ds = r['stoi_mean'] - baseline_stoi
            print(f"  {name}: PESQ={r['pesq_mean']:.4f} (Δ{dp:+.4f}), STOI={r['stoi_mean']:.4f} (Δ{ds:+.4f})")

    print(f"\n請手動更新 comparison_table.md 或重新執行 generate_test_samples.py")


def main():
    loader = load_data()
    new_results = {}

    for ckpt_info in CHECKPOINTS:
        try:
            name, result = run_eval(ckpt_info, loader)
            new_results[name] = result
        except Exception as e:
            print(f"  ERROR in {ckpt_info['name']}: {e}")
            import traceback; traceback.print_exc()
            new_results[ckpt_info['name']] = {'error': str(e)}

    update_comparison_files(new_results)


if __name__ == '__main__':
    main()
