"""
exp_0223/eval_0226b_0227.py

評估 exp_0226b 和 exp_0227，更新 comparison_results.json 和 comparison_table.md：

  - exp_0226b: best_model_val_total.pt (epoch=42, val_total=35.7929; EncOnly+FeatAlign+HF-Mel)
  - exp_0227:  best_model_val_total.pt (epoch=161, val_total=35.6641; EncOnly+FeatAlign+MRD-FM)

執行：
    cd /home/sbplab/ruizi/WavTokenize-feature-analysis
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \
    /home/sbplab/miniconda3/envs/test/bin/python exp_0223/eval_0226b_0227.py
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
BASE = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent / 'test'

BASELINE_JSON = BASE / 'exp_0217/FAIR_BASELINE_PESQ_STOI_n30.json'
with open(BASELINE_JSON) as f:
    baseline_data = json.load(f)
SAMPLE_INDICES = baseline_data['indices'][:3]
print(f"使用 val indices: {SAMPLE_INDICES}")

NEW_EXPERIMENTS = [
    {
        'name': 'exp_0226b',
        'folder': 'exp_0226b',
        'type': 'no_vq_enc_only',
        'ckpt': 'exp_0226/runs/enc_hf_mel_epoch_20260227_020028/best_model_val_total.pt',
        'lora_rank': 64, 'lora_alpha': 128,
        'note': 'best_model_val_total.pt (epoch=42, val_total=35.7929; EncOnly+FeatAlign+HF-Mel bin40+)',
    },
    {
        'name': 'exp_0227',
        'folder': 'exp_0227',
        'type': 'no_vq_enc_only',
        'ckpt': 'exp_0227/runs/enc_mrd_fm_epoch_20260227_024953/best_model_val_total.pt',
        'lora_rank': 64, 'lora_alpha': 128,
        'note': 'best_model_val_total.pt (epoch=161, val_total=35.6641; EncOnly+FeatAlign+MRD-FM)',
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


def run_exp_enc_only(exp, loader, out_dir):
    """Encoder-only（No-VQ）model 評估，使用 TeacherStudentNoVQ。"""
    from exp_0224.models_no_vq import TeacherStudentNoVQ

    print(f"  Loading {exp['name']} (No-VQ Encoder-only)...")
    model = TeacherStudentNoVQ(
        wavtok_config=WAVTOK_CONFIG, wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=exp['lora_rank'], lora_alpha=exp['lora_alpha'],
        intermediate_indices=[3, 4, 6],
        device=DEVICE,
    ).to(DEVICE)

    ckpt = torch.load(str(BASE / exp['ckpt']), map_location='cpu', weights_only=False)
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
    return metrics_list


def rebuild_comparison_table(all_results):
    baseline = all_results.get('noisy_through_teacher', {})
    baseline_pesq = baseline.get('pesq_mean', 1.6765)
    baseline_stoi = baseline.get('stoi_mean', 0.5266)

    lines = []
    lines.append("# PESQ/STOI 比較表")
    lines.append(f"\n**Val indices**: {SAMPLE_INDICES}  \n**評估條件**: PESQ nb (8kHz resample), STOI (24kHz)")
    lines.append(f"\n## 各組平均結果\n")
    lines.append(f"> `noisy_through_teacher` = noisy → Teacher Encoder+VQ → Frozen Decoder（公平比較基準）")
    lines.append(f"> 其餘實驗的 ΔPESQ / ΔSTOI 皆相對此基準計算")
    lines.append(f"")
    lines.append(f"| 實驗 | PESQ (recon) | STOI (recon) | ΔPESQ vs teacher_baseline | ΔSTOI vs teacher_baseline |")
    lines.append(f"|------|-------------|-------------|--------------------------|--------------------------|")

    ORDER = [
        'clean_through_teacher_no_vq',
        'clean_through_teacher',
        'noisy_through_teacher',
        'noisy_through_teacher_no_vq',
        'V2',
        'Plan_Ori',
        'exp_0216',
        'exp_0217',
        'exp_0223_v2',
        'exp_0224a',
        'exp_0224b_best',
        'exp_0225a',
        'exp_0225b',
        'exp_0225c',
        'exp_0225d',
        'exp_0226_best_total',
        'exp_0226a',
        'exp_0226b',
        'exp_0227',
    ]

    LABELS = {
        'clean_through_teacher_no_vq': 'clean_through_teacher_no_vq (上限, 無VQ)',
        'clean_through_teacher': 'clean_through_teacher (上限, 有VQ)',
        'noisy_through_teacher': '**noisy_through_teacher** (baseline)',
        'noisy_through_teacher_no_vq': 'noisy_through_teacher_no_vq',
        'exp_0224b_best': '**exp_0224b_best** (No-VQ+DecLoRA from 0224a, ep142)',
        'exp_0225a': 'exp_0225a (No-VQ scratch encoder)',
        'exp_0225b': 'exp_0225b (No-VQ+DecLoRA from 0225a, ep33)',
        'exp_0225c': 'exp_0225c (No-VQ+DecLoRA+Phase from 0225a, ep5)',
        'exp_0225d': 'exp_0225d (No-VQ+DecLoRA+FM from 0225a, ep14)',
        'exp_0226_best_total': '**exp_0226_best_total** (E2E LoRA, ep142)',
        'exp_0226a': 'exp_0226a (EncOnly+FeatAlign, ep156)',
        'exp_0226b': 'exp_0226b (EncOnly+FeatAlign+HF-Mel, ep42)',
        'exp_0227':  'exp_0227  (EncOnly+FeatAlign+MRD-FM, ep161)',
    }

    for key in ORDER:
        if key not in all_results:
            continue
        r = all_results[key]
        if 'error' in r:
            lines.append(f"| {key} | ERROR | ERROR | - | - |")
            continue
        label = LABELS.get(key, key)
        dp = r['pesq_mean'] - baseline_pesq
        ds = r['stoi_mean'] - baseline_stoi
        lines.append(
            f"| {label} | {r['pesq_mean']:.4f} | {r['stoi_mean']:.4f} | "
            f"{dp:+.4f} | {ds:+.4f} |"
        )

    lines.append(f"\n## Per-Sample 明細\n")
    for key in ORDER:
        if key not in all_results:
            continue
        r = all_results[key]
        if 'error' in r:
            continue
        lines.append(f"### {key}")
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
    lines.append("")
    lines.append("## 解讀說明")
    lines.append("")
    lines.append("- **PESQ(noisy) >> PESQ(recon)** 屬正常現象：LDV noisy 與 clean 時序高度對齊，PESQ 對時序對齊敏感")
    lines.append("- **公平基準** (`noisy_through_teacher`) = noisy 直接經過相同的 Encoder+VQ+Decoder pipeline")
    lines.append("- **clean_through_teacher_no_vq**（無VQ上限）：PESQ=2.484, STOI=0.761")
    lines.append("- **clean_through_teacher**（有VQ上限）：PESQ=2.352, STOI=0.750")
    lines.append("- **exp_0226_best_total**（E2E LoRA，ep142）：目前 PESQ 最高但音檔有機械音（phase artifact）")
    lines.append("- **exp_0226a**（EncOnly+FeatAlign）：無機械音，encoder-only ceiling")
    lines.append("- **exp_0226b**（+HF-Mel）：在 0226a 基礎上強調高頻 mel bin 40+（~1.6kHz）")
    lines.append("- **exp_0227**（+MRD-FM）：用預訓練 MRD 辨別器的特徵圖做 Feature Matching")

    return "\n".join(lines)


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    json_path = TEST_DIR / 'comparison_results.json'
    with open(json_path) as f:
        all_results = json.load(f)

    loader = load_data()

    for exp in NEW_EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"Processing: {exp['name']}")
        print(f"  Ckpt: {exp['ckpt']}")
        print(f"  Note: {exp['note']}")

        out_dir = TEST_DIR / exp['folder']
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            metrics_list = run_exp_enc_only(exp, loader, out_dir)
            valid = [m for m in metrics_list if m['pesq_recon'] is not None]
            if valid:
                all_results[exp['name']] = {
                    'checkpoint': exp['note'],
                    'pesq_mean': float(np.mean([m['pesq_recon'] for m in valid])),
                    'pesq_noisy_mean': float(np.mean([m['pesq_noisy'] for m in valid])),
                    'stoi_mean': float(np.mean([m['stoi_recon'] for m in valid])),
                    'stoi_noisy_mean': float(np.mean([m['stoi_noisy'] for m in valid])),
                    'per_sample': metrics_list,
                }
                print(f"  -> PESQ_mean={all_results[exp['name']]['pesq_mean']:.4f}, "
                      f"STOI_mean={all_results[exp['name']]['stoi_mean']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[exp['name']] = {'error': str(e)}

    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nJSON 更新: {json_path}")

    md_content = rebuild_comparison_table(all_results)
    table_path = TEST_DIR / 'comparison_table.md'
    table_path.write_text(md_content, encoding='utf-8')
    print(f"MD 更新: {table_path}")
    print()
    print(md_content)


if __name__ == '__main__':
    main()
