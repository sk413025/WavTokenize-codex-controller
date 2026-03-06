"""
exp_0223/eval_0229c.py

評估 exp_0229c，更新 comparison_results.json 和 comparison_table.md：
  - exp_0229c: best_model_val_total.pt (epoch=27; LatentBWE-v2 on frozen enc_0227,
               hidden=256, blocks=8, kernel=5, HF-emphasis loss)

執行：
    cd /home/sbplab/ruizi/WavTokenize-feature-analysis
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \
    /home/sbplab/miniconda3/envs/test/bin/python exp_0223/eval_0229c.py
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


def resample_to(wav_np, orig_sr, target_sr):
    """重採樣音訊。

    Args:
        wav_np: 輸入音訊陣列。
        orig_sr: 原始採樣率。
        target_sr: 目標採樣率。

    Returns:
        重採樣後的音訊陣列。
    """
    n = int(len(wav_np) * target_sr / orig_sr)
    return signal.resample(wav_np, n).astype(np.float32)


def save_wav(arr, path, sr=SR):
    """儲存音訊檔案。

    Args:
        arr: 音訊陣列（float32，-1~1）。
        path: 輸出路徑。
        sr: 採樣率。
    """
    arr = np.clip(arr, -1.0, 1.0)
    wavfile.write(str(path), sr, (arr * 32767).astype(np.int16))


def compute_metrics(clean_np, recon_np, noisy_np):
    """計算 PESQ 和 STOI 指標。

    Args:
        clean_np: 乾淨參考音訊。
        recon_np: 重建音訊。
        noisy_np: 帶噪音訊。

    Returns:
        pesq_recon, pesq_noisy, stoi_recon, stoi_noisy 的 tuple。
    """
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
    """載入驗證資料集。

    Returns:
        驗證 DataLoader。
    """
    val_ds = AugmentedCurriculumDataset(
        VAL_CACHE, augment=False, filter_clean_to_clean=True, compute_snr=False
    )
    subset = Subset(val_ds, SAMPLE_INDICES)
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn_curriculum, num_workers=0)
    return loader


def run_exp_0229c(loader, out_dir):
    """執行 exp_0229c 評估（LatentBWEPipeline v2，frozen enc_0227 + LatentBWE hidden=256, blocks=8, kernel=5）。

    Args:
        loader: 驗證 DataLoader。
        out_dir: 輸出音檔目錄。

    Returns:
        per-sample metrics list。
    """
    sys.path.insert(0, str(BASE / 'exp_0229c'))
    from train_bwe_latent_hf import LatentBWEPipeline

    enc_ckpt = BASE / 'exp_0227/runs/enc_mrd_fm_epoch_20260227_024953/best_model_val_total.pt'
    bwe_ckpt = BASE / 'exp_0229c/runs/bwe_latent_hf_epoch_20260302_065418/best_model_val_total.pt'
    print(f"  Loading exp_0229c")
    print(f"    enc_ckpt: {enc_ckpt}")
    print(f"    bwe_ckpt: {bwe_ckpt} (best epoch=87)")

    pipeline = LatentBWEPipeline(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=64,
        lora_alpha=128,
        bwe_hidden_dim=256,
        bwe_num_blocks=8,
        bwe_kernel_size=5,
        device=DEVICE,
    ).to(DEVICE)

    # 載入 encoder（exp_0227）
    enc_state = torch.load(str(enc_ckpt), map_location='cpu', weights_only=False)
    if 'model_state_dict' in enc_state:
        pipeline.base_model.load_state_dict(enc_state['model_state_dict'], strict=False)
        print(f"    Loaded enc_0227 model_state_dict (ep{enc_state.get('epoch','?')})")

    # 載入 LatentBWE-v2 weights
    bwe_state = torch.load(str(bwe_ckpt), map_location='cpu', weights_only=False)
    pipeline.bwe.load_state_dict(bwe_state['bwe_state_dict'], strict=True)
    print(f"    Loaded bwe_state_dict (ep{bwe_state.get('epoch','?')})")

    pipeline.eval()

    metrics_list = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            noisy = batch['noisy_audio'].to(DEVICE)
            clean = batch['clean_audio'].to(DEVICE)
            if clean.dim() == 2: clean = clean.unsqueeze(1)
            if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

            out = pipeline(clean, noisy)
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

    del pipeline
    torch.cuda.empty_cache()
    return metrics_list


def rebuild_comparison_table(all_results):
    """根據 all_results 重建 comparison_table.md 內容。

    Args:
        all_results: 包含所有實驗評估結果的 dict。

    Returns:
        更新後的 markdown 字串。
    """
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
        'exp_0228_fm',
        'exp_0229b',
        'exp_0229c',
        'exp_0305b_A_tail_lock',
        'exp_0305b_B_front_tail_lock',
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
        'exp_0228_fm': 'exp_0228_fm (EncOnly+FeatAlign+MRD-FM+HuBERT, ep77)',
        'exp_0229b': 'exp_0229b (LatentBWE on enc_0227, ep140)',
        'exp_0229c': 'exp_0229c (LatentBWE-v2+HF-emph on enc_0227, ep87)',
        'exp_0305b_A_tail_lock': 'exp_0305b_A_tail_lock (anchor tail L16/L17)',
        'exp_0305b_B_front_tail_lock': 'exp_0305b_B_front_tail_lock (anchor front+tail L0/L1/L16/L17)',
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
    lines.append("- **exp_0228_fm**（+HuBERT）：在 0227 基礎上加入 frozen HuBERT 語音學特徵監督（layers 6-8）")
    lines.append("- **exp_0229b**（LatentBWE）：凍結 enc_0227，在 latent 域插入可訓練 LatentBWE（~0.5M params）")
    lines.append("- **exp_0229c**（LatentBWE-v2+HF-emph）：擴大模型容量（hidden 128→256, kernel 3→5, blocks 6→8, ~2.8M），加入 HF-emphasis STFT Loss（4kHz cutoff, 5× 加重）")
    lines.append("- **exp_0305b_A_tail_lock**：Anchor regularization 僅鎖定尾端層（L16/L17）")
    lines.append("- **exp_0305b_B_front_tail_lock**：Anchor regularization 同時鎖定前後端層（L0/L1/L16/L17）")

    return "\n".join(lines)


def main():
    """主評估流程：執行 exp_0229c，更新 JSON 和 MD。"""
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    json_path = TEST_DIR / 'comparison_results.json'
    with open(json_path) as f:
        all_results = json.load(f)

    loader = load_data()

    experiments = [
        ('exp_0229c', run_exp_0229c, 'exp_0229c',
         'best_model_val_total.pt (epoch=87; LatentBWE-v2+HF-emph on frozen enc_0227)'),
    ]

    for name, fn, folder, note in experiments:
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"  Note: {note}")

        out_dir = TEST_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            metrics_list = fn(loader, out_dir)
            valid = [m for m in metrics_list if m['pesq_recon'] is not None]
            if valid:
                all_results[name] = {
                    'checkpoint': note,
                    'pesq_mean': float(np.mean([m['pesq_recon'] for m in valid])),
                    'pesq_noisy_mean': float(np.mean([m['pesq_noisy'] for m in valid])),
                    'stoi_mean': float(np.mean([m['stoi_recon'] for m in valid])),
                    'stoi_noisy_mean': float(np.mean([m['stoi_noisy'] for m in valid])),
                    'per_sample': metrics_list,
                }
                print(f"  -> PESQ_mean={all_results[name]['pesq_mean']:.4f}, "
                      f"STOI_mean={all_results[name]['stoi_mean']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[name] = {'error': str(e)}

    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nJSON 更新: {json_path}")

    md_content = rebuild_comparison_table(all_results)
    table_path = TEST_DIR / 'comparison_table.md'
    table_path.write_text(md_content, encoding='utf-8')
    print(f"MD 更新: {table_path}")
    print()
    # 只印摘要表格
    for line in md_content.split('\n'):
        if line.startswith('|') or line.startswith('#') or line.startswith('>'):
            print(line)
        if '## Per-Sample' in line:
            break


if __name__ == '__main__':
    main()
