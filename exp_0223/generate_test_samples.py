"""
exp_0223/generate_test_samples.py

建立 exp_0223/test/ 對比音檔資料夾：
  - 5 組實驗各取同樣 3 個 val 樣本（固定 indices）
  - 每組儲存 noisy / clean / recon 音檔
  - 計算各組 PESQ/STOI 並輸出比較表

資料夾結構：
  exp_0223/test/
  ├── V2/
  │   ├── sample01_noisy.wav
  │   ├── sample01_clean.wav
  │   ├── sample01_recon.wav
  │   ├── sample02_*.wav
  │   └── sample03_*.wav
  ├── Plan_Ori/
  ├── exp_0216/
  ├── exp_0217/
  ├── exp_0223_v2/
  └── comparison_table.md

執行：
    cd /home/sbplab/ruizi/WavTokenize-feature-analysis
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \
    python exp_0223/generate_test_samples.py
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

# ── 設定 ─────────────────────────────────────────────────────
DEVICE = torch.device('cuda:0')
SR = 24000
SR_PESQ = 8000
TEST_DIR = Path(__file__).parent / 'test'

# 固定使用 FAIR_BASELINE 的前 3 個 indices（確保與之前評估一致）
BASELINE_JSON = Path(__file__).parent.parent / 'exp_0217/FAIR_BASELINE_PESQ_STOI_n30.json'
with open(BASELINE_JSON) as f:
    baseline_data = json.load(f)
SAMPLE_INDICES = baseline_data['indices'][:3]
print(f"使用 val indices: {SAMPLE_INDICES}")

# ── 實驗設定 ─────────────────────────────────────────────────
EXPERIMENTS = [
    {
        'name': 'V2',
        'folder': 'V2',
        'type': 'encoder_vq',
        'ckpt': 'exp_0206/runs/longterm_v2_20260215_120316/best_model.pt',
        'lora_rank': 256, 'lora_alpha': 512,
        'rvq_layers': 4,
        'rvq_codebook_size': 2048,
    },
    {
        'name': 'Plan_Ori',
        'folder': 'Plan_Ori',
        'type': 'encoder_vq',
        'ckpt': 'exp_0206/runs/plan_ori_long_20260211/best_model.pt',
        'lora_rank': 256, 'lora_alpha': 512,
        'rvq_layers': 1,
    },
    {
        'name': 'exp_0216',
        'folder': 'exp_0216',
        'type': 'encoder_vq',
        'ckpt': 'exp_0216/runs/augmented_long_20260216/best_model.pt',
        'lora_rank': 64, 'lora_alpha': 128,
        'rvq_layers': 1,
    },
    {
        'name': 'exp_0217',
        'folder': 'exp_0217',
        'type': 'encoder_vq',
        'ckpt': 'exp_0217/runs/t453_weighted_epoch_20260217_104843/best_model.pt',
        'lora_rank': 64, 'lora_alpha': 128,
        'rvq_layers': 1,
    },
    {
        'name': 'exp_0223_v2',
        'folder': 'exp_0223_v2',
        'type': 'decoder_lora',
        'ckpt': 'exp_0223/runs/decoder_lora_v2_epoch_20260223_042124/best_model.pt',
        'lora_rank': 64, 'lora_alpha': 128,
        'decoder_lora_rank': 32, 'decoder_lora_alpha': 64,
    },
]


# ── 工具函數 ─────────────────────────────────────────────────
def resample_to(wav_np, orig_sr, target_sr):
    n = int(len(wav_np) * target_sr / orig_sr)
    return signal.resample(wav_np, n).astype(np.float32)


def save_wav(arr, path, sr=SR):
    arr = np.clip(arr, -1.0, 1.0)
    wavfile.write(str(path), sr, (arr * 32767).astype(np.int16))


def compute_metrics(clean_np, recon_np, noisy_np):
    """計算 PESQ (nb/8kHz) 和 STOI (24kHz)"""
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


def run_encoder_vq_experiment(exp, loader, out_dir):
    """Encoder LoRA + VQ 類型實驗（V2, Plan Ori, exp_0216, exp_0217）"""
    from exp_0206.plan_ori.models_single_vq_ema import TeacherStudentSingleVQ
    from exp_0128.phase3.residual_vq.models_rvq import TeacherStudentRVQ

    rvq_layers = exp.get('rvq_layers', 1)
    ckpt_path = Path(exp['ckpt'])

    print(f"  Loading {exp['name']} (rvq_layers={rvq_layers})...")
    if rvq_layers > 1:
        model = TeacherStudentRVQ(
            wavtok_config=WAVTOK_CONFIG, wavtok_ckpt=WAVTOK_CKPT,
            lora_rank=exp['lora_rank'], lora_alpha=exp['lora_alpha'],
            device=DEVICE, n_rvq_layers=rvq_layers,
            rvq_codebook_size=exp.get('rvq_codebook_size', 1024),
        ).to(DEVICE)
    else:
        model = TeacherStudentSingleVQ(
            wavtok_config=WAVTOK_CONFIG, wavtok_ckpt=WAVTOK_CKPT,
            lora_rank=exp['lora_rank'], lora_alpha=exp['lora_alpha'],
            device=DEVICE,
        ).to(DEVICE)

    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    if 'lora_state' in ckpt:
        model.student.load_state_dict(ckpt['lora_state'], strict=False)
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if 'vq_state_dict' in ckpt:
        model.vq.load_state_dict(ckpt['vq_state_dict'])
    model.eval()

    metrics_list = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            noisy = batch['noisy_audio'].to(DEVICE)
            clean = batch['clean_audio'].to(DEVICE)
            if clean.dim() == 2: clean = clean.unsqueeze(1)
            if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

            # forward → get student_quantized → decode
            out = model(clean, noisy)
            student_quantized = out['student_quantized']
            recon = model.decode(student_quantized)

            T = min(clean.shape[-1], recon.shape[-1])
            c = clean[0, 0, :T].cpu().numpy().astype(np.float32)
            r = recon[0, 0, :T].cpu().numpy().astype(np.float32)
            n = noisy[0, 0, :T].cpu().numpy().astype(np.float32)

            c /= (np.abs(c).max() + 1e-8)
            r /= (np.abs(r).max() + 1e-8)
            n /= (np.abs(n).max() + 1e-8)

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
    return metrics_list


def run_decoder_lora_experiment(exp, loader, out_dir):
    """Decoder LoRA 類型實驗（exp_0223 v2）"""
    from exp_0223.models_decoder_lora import TeacherStudentDecoderLoRA

    print(f"  Loading {exp['name']} (decoder LoRA)...")
    model = TeacherStudentDecoderLoRA(
        wavtok_config=WAVTOK_CONFIG, wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=exp['lora_rank'], lora_alpha=exp['lora_alpha'],
        device=DEVICE,
        decoder_lora_rank=exp['decoder_lora_rank'],
        decoder_lora_alpha=exp['decoder_lora_alpha'],
    ).to(DEVICE)

    ckpt = torch.load(str(exp['ckpt']), map_location='cpu', weights_only=False)
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

            c /= (np.abs(c).max() + 1e-8)
            r /= (np.abs(r).max() + 1e-8)
            n /= (np.abs(n).max() + 1e-8)

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
    return metrics_list


# ── 主流程 ────────────────────────────────────────────────────
def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {TEST_DIR}")
    print(f"Val indices: {SAMPLE_INDICES}\n")

    loader = load_data()

    all_results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"Processing: {exp['name']}")
        print(f"  Ckpt: {exp['ckpt']}")

        out_dir = TEST_DIR / exp['folder']
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if exp['type'] == 'decoder_lora':
                metrics_list = run_decoder_lora_experiment(exp, loader, out_dir)
            else:
                metrics_list = run_encoder_vq_experiment(exp, loader, out_dir)

            valid = [m for m in metrics_list if m['pesq_recon'] is not None]
            if valid:
                all_results[exp['name']] = {
                    'pesq_mean': float(np.mean([m['pesq_recon'] for m in valid])),
                    'pesq_noisy_mean': float(np.mean([m['pesq_noisy'] for m in valid])),
                    'stoi_mean': float(np.mean([m['stoi_recon'] for m in valid])),
                    'stoi_noisy_mean': float(np.mean([m['stoi_noisy'] for m in valid])),
                    'per_sample': metrics_list,
                }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[exp['name']] = {'error': str(e)}

    # ── 比較表 ────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("PESQ/STOI 比較表（n=3 val samples, nb/8kHz）")
    print(f"{'='*60}")

    lines = []
    lines.append("# PESQ/STOI 比較表")
    lines.append(f"\n**Val indices**: {SAMPLE_INDICES}  \n**評估條件**: PESQ nb (8kHz resample), STOI (24kHz)")
    lines.append(f"\n## 各組平均結果\n")
    lines.append(f"| 實驗 | PESQ (recon) | STOI (recon) | ΔPESQ vs noisy | ΔSTOI vs noisy |")
    lines.append(f"|------|-------------|-------------|----------------|----------------|")

    for name, r in all_results.items():
        if 'error' in r:
            lines.append(f"| {name} | ERROR | ERROR | - | - |")
        else:
            dp = r['pesq_mean'] - r['pesq_noisy_mean']
            ds = r['stoi_mean'] - r['stoi_noisy_mean']
            lines.append(
                f"| {name} | {r['pesq_mean']:.4f} | {r['stoi_mean']:.4f} | "
                f"{dp:+.4f} | {ds:+.4f} |"
            )

    lines.append(f"\n## Per-Sample 明細\n")
    for name, r in all_results.items():
        if 'error' in r:
            continue
        lines.append(f"### {name}")
        lines.append(f"| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |")
        lines.append(f"|--------|-----------|-----------|-----------|-----------|")
        for i, m in enumerate(r['per_sample']):
            if m['pesq_recon'] is None:
                lines.append(f"| {i+1} | ERR | ERR | ERR | ERR |")
            else:
                lines.append(
                    f"| {i+1} | {m['pesq_recon']:.4f} | {m['pesq_noisy']:.4f} | "
                    f"{m['stoi_recon']:.4f} | {m['stoi_noisy']:.4f} |"
                )
        lines.append("")

    lines.append("## 音檔說明\n")
    lines.append("各子目錄包含 3 組音檔（sampleNN_noisy / clean / recon）：")
    lines.append("- `noisy`: 原始帶噪輸入（LDV 感測器音訊）")
    lines.append("- `clean`: 對應乾淨音訊（ground truth）")
    lines.append("- `recon`: 各實驗重建輸出")

    md_content = "\n".join(lines)
    table_path = TEST_DIR / "comparison_table.md"
    table_path.write_text(md_content, encoding='utf-8')
    print(md_content)
    print(f"\n比較表儲存: {table_path}")

    # 儲存 JSON
    json_path = TEST_DIR / "comparison_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"JSON 儲存: {json_path}")


if __name__ == '__main__':
    main()
