#!/usr/bin/env python3
"""
統計比較：
  noisy_through_teacher (有 VQ) vs noisy_through_teacher_no_vq (無 VQ)

輸出：
  - noisy_vq_vs_novq_stats.json
  - noisy_vq_vs_novq_stats.md

指標：
  - 全頻 log-spectral error (LSE)
  - 高頻 (>=4kHz) / 低頻 (300-3kHz) LSE
  - 子音候選幀（高頻比例高 + 低能量）上的 LSE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.stats import ttest_1samp, wilcoxon


ROOT = Path(__file__).resolve().parent
DIR_VQ = ROOT / "noisy_through_teacher"
DIR_NO_VQ = ROOT / "noisy_through_teacher_no_vq"
SAMPLES = ["sample01", "sample02", "sample03"]


def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float64), sr


def _frame_rms(signal: np.ndarray, win: int, hop: int) -> np.ndarray:
    padded = np.pad(signal, (win // 2, win // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[::hop]
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def _bootstrap_ci_mean(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> Tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _paired_stats(diff: np.ndarray) -> Dict[str, float]:
    """diff = (no_vq - vq); 負值代表 no-vq 較好（誤差更低）。"""
    n = int(len(diff))
    mean = float(np.mean(diff))
    std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    ci_lo, ci_hi = _bootstrap_ci_mean(diff)
    dz = float(mean / std) if std > 0 else 0.0

    # t-test (H0: mean(diff)=0)
    if n > 1:
        t_stat, t_p = ttest_1samp(diff, popmean=0.0, nan_policy="omit")
        t_stat = float(t_stat)
        t_p = float(t_p)
    else:
        t_stat, t_p = float("nan"), float("nan")

    # Wilcoxon signed-rank (H0: median(diff)=0)
    try:
        w_stat, w_p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", correction=False)
        w_stat = float(w_stat)
        w_p = float(w_p)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "n": n,
        "mean_diff_no_minus_vq": mean,
        "std_diff": std,
        "cohens_dz": dz,
        "ci95_mean_diff_lo": ci_lo,
        "ci95_mean_diff_hi": ci_hi,
        "ttest_t": t_stat,
        "ttest_p": t_p,
        "wilcoxon_w": w_stat,
        "wilcoxon_p": w_p,
    }


def main() -> None:
    n_fft, hop, win = 1024, 256, 1024

    # 全部幀串接（frame-level paired diff）
    all_diff_all = []
    all_diff_hf = []
    all_diff_lf = []
    all_diff_hf_cons = []
    all_diff_hf_other = []
    all_diff_lf_cons = []
    all_diff_lf_other = []

    per_sample = []

    for s in SAMPLES:
        clean, sr = _load_wav(DIR_VQ / f"{s}_clean.wav")
        recon_vq, sr2 = _load_wav(DIR_VQ / f"{s}_recon.wav")
        recon_no, sr3 = _load_wav(DIR_NO_VQ / f"{s}_recon.wav")
        if not (sr == sr2 == sr3):
            raise RuntimeError(f"Sample rate mismatch in {s}: {sr}, {sr2}, {sr3}")

        L = min(len(clean), len(recon_vq), len(recon_no))
        clean = clean[:L]
        recon_vq = recon_vq[:L]
        recon_no = recon_no[:L]

        f, _, Zc = stft(clean, fs=sr, nperseg=win, noverlap=win - hop, nfft=n_fft, boundary=None)
        _, _, Zv = stft(recon_vq, fs=sr, nperseg=win, noverlap=win - hop, nfft=n_fft, boundary=None)
        _, _, Zn = stft(recon_no, fs=sr, nperseg=win, noverlap=win - hop, nfft=n_fft, boundary=None)

        mag_c = np.abs(Zc) + 1e-8
        mag_v = np.abs(Zv) + 1e-8
        mag_n = np.abs(Zn) + 1e-8

        log_c = np.log(mag_c)
        log_v = np.log(mag_v)
        log_n = np.log(mag_n)

        hf = f >= 4000
        lf = (f >= 300) & (f < 3000)

        hf_ratio = mag_c[hf].mean(axis=0) / (mag_c.mean(axis=0) + 1e-12)
        rms = _frame_rms(clean, win=win, hop=hop)

        T = min(log_c.shape[1], len(hf_ratio), len(rms))
        log_c = log_c[:, :T]
        log_v = log_v[:, :T]
        log_n = log_n[:, :T]
        hf_ratio = hf_ratio[:T]
        rms = rms[:T]

        # frame-level LSE
        err_v_all = np.mean(np.abs(log_v - log_c), axis=0)
        err_n_all = np.mean(np.abs(log_n - log_c), axis=0)
        err_v_hf = np.mean(np.abs(log_v[hf] - log_c[hf]), axis=0)
        err_n_hf = np.mean(np.abs(log_n[hf] - log_c[hf]), axis=0)
        err_v_lf = np.mean(np.abs(log_v[lf] - log_c[lf]), axis=0)
        err_n_lf = np.mean(np.abs(log_n[lf] - log_c[lf]), axis=0)

        # 子音候選幀 heuristic:
        # 高頻比例 >= 該句 P75 且 RMS <= 該句 P50
        q_hf = np.quantile(hf_ratio, 0.75)
        q_rms = np.quantile(rms, 0.50)
        cons = (hf_ratio >= q_hf) & (rms <= q_rms)
        other = ~cons

        diff_all = err_n_all - err_v_all
        diff_hf = err_n_hf - err_v_hf
        diff_lf = err_n_lf - err_v_lf
        diff_hf_cons = diff_hf[cons]
        diff_hf_other = diff_hf[other]
        diff_lf_cons = diff_lf[cons]
        diff_lf_other = diff_lf[other]

        all_diff_all.append(diff_all)
        all_diff_hf.append(diff_hf)
        all_diff_lf.append(diff_lf)
        all_diff_hf_cons.append(diff_hf_cons)
        all_diff_hf_other.append(diff_hf_other)
        all_diff_lf_cons.append(diff_lf_cons)
        all_diff_lf_other.append(diff_lf_other)

        per_sample.append(
            {
                "sample": s,
                "n_frames": int(T),
                "cons_frames": int(cons.sum()),
                "cons_ratio_pct": float(cons.mean() * 100),
                "mean_lse_all_vq": float(err_v_all.mean()),
                "mean_lse_all_no_vq": float(err_n_all.mean()),
                "mean_lse_hf_vq": float(err_v_hf.mean()),
                "mean_lse_hf_no_vq": float(err_n_hf.mean()),
                "mean_lse_lf_vq": float(err_v_lf.mean()),
                "mean_lse_lf_no_vq": float(err_n_lf.mean()),
                "mean_diff_all_no_minus_vq": float(diff_all.mean()),
                "mean_diff_hf_no_minus_vq": float(diff_hf.mean()),
                "mean_diff_lf_no_minus_vq": float(diff_lf.mean()),
                "mean_diff_hf_cons_no_minus_vq": float(diff_hf_cons.mean()) if len(diff_hf_cons) else float("nan"),
                "mean_diff_hf_other_no_minus_vq": float(diff_hf_other.mean()) if len(diff_hf_other) else float("nan"),
            }
        )

    cat = lambda xs: np.concatenate(xs) if xs else np.array([], dtype=np.float64)

    diff_all = cat(all_diff_all)
    diff_hf = cat(all_diff_hf)
    diff_lf = cat(all_diff_lf)
    diff_hf_cons = cat(all_diff_hf_cons)
    diff_hf_other = cat(all_diff_hf_other)
    diff_lf_cons = cat(all_diff_lf_cons)
    diff_lf_other = cat(all_diff_lf_other)

    results = {
        "metadata": {
            "comparison": "noisy_through_teacher (vq) vs noisy_through_teacher_no_vq (no-vq)",
            "diff_definition": "no-vq LSE - vq LSE (negative means no-vq better)",
            "stft": {"n_fft": n_fft, "win": win, "hop": hop},
            "bands_hz": {"hf": ">=4000", "lf": "300-3000"},
            "consonant_like_definition": "hf_ratio>=P75 and frame_rms<=P50 (per utterance)",
            "samples": SAMPLES,
        },
        "per_sample": per_sample,
        "frame_level_stats": {
            "all_band": _paired_stats(diff_all),
            "hf_band": _paired_stats(diff_hf),
            "lf_band": _paired_stats(diff_lf),
            "hf_consonant_like": _paired_stats(diff_hf_cons),
            "hf_other_frames": _paired_stats(diff_hf_other),
            "lf_consonant_like": _paired_stats(diff_lf_cons),
            "lf_other_frames": _paired_stats(diff_lf_other),
        },
    }

    # 補充相對改善百分比（相對 VQ 誤差）
    # 先從 per-sample 估整體均值
    m_all_vq = float(np.mean([x["mean_lse_all_vq"] for x in per_sample]))
    m_all_no = float(np.mean([x["mean_lse_all_no_vq"] for x in per_sample]))
    m_hf_vq = float(np.mean([x["mean_lse_hf_vq"] for x in per_sample]))
    m_hf_no = float(np.mean([x["mean_lse_hf_no_vq"] for x in per_sample]))
    m_lf_vq = float(np.mean([x["mean_lse_lf_vq"] for x in per_sample]))
    m_lf_no = float(np.mean([x["mean_lse_lf_no_vq"] for x in per_sample]))

    results["summary_means"] = {
        "all_lse_vq": m_all_vq,
        "all_lse_no_vq": m_all_no,
        "all_relative_improvement_pct": float((m_all_vq - m_all_no) / m_all_vq * 100.0),
        "hf_lse_vq": m_hf_vq,
        "hf_lse_no_vq": m_hf_no,
        "hf_relative_improvement_pct": float((m_hf_vq - m_hf_no) / m_hf_vq * 100.0),
        "lf_lse_vq": m_lf_vq,
        "lf_lse_no_vq": m_lf_no,
        "lf_relative_improvement_pct": float((m_lf_vq - m_lf_no) / m_lf_vq * 100.0),
    }

    out_json = ROOT / "noisy_vq_vs_novq_stats.json"
    out_md = ROOT / "noisy_vq_vs_novq_stats.md"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    all_stats = results["frame_level_stats"]["all_band"]
    hf_stats = results["frame_level_stats"]["hf_band"]
    hf_cons_stats = results["frame_level_stats"]["hf_consonant_like"]
    summ = results["summary_means"]

    md = []
    md.append("# Noisy 路徑 VQ vs No-VQ 統計")
    md.append("")
    md.append("比較：`noisy_through_teacher`（VQ） vs `noisy_through_teacher_no_vq`（No-VQ）")
    md.append("差值定義：`No-VQ LSE - VQ LSE`，負值代表 No-VQ 較好。")
    md.append("")
    md.append("## 整體平均誤差")
    md.append("")
    md.append("| 指標 | VQ | No-VQ | 相對改善 |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| 全頻 LSE | {summ['all_lse_vq']:.6f} | {summ['all_lse_no_vq']:.6f} | {summ['all_relative_improvement_pct']:.3f}% |")
    md.append(f"| 高頻 LSE (>=4kHz) | {summ['hf_lse_vq']:.6f} | {summ['hf_lse_no_vq']:.6f} | {summ['hf_relative_improvement_pct']:.3f}% |")
    md.append(f"| 低頻 LSE (300-3kHz) | {summ['lf_lse_vq']:.6f} | {summ['lf_lse_no_vq']:.6f} | {summ['lf_relative_improvement_pct']:.3f}% |")
    md.append("")
    md.append("## 幀級配對統計（No-VQ - VQ）")
    md.append("")
    md.append("| 區域 | N | 平均差 | 95% CI | Wilcoxon p | t-test p |")
    md.append("|---|---:|---:|---|---:|---:|")
    md.append(
        f"| 全頻 | {all_stats['n']} | {all_stats['mean_diff_no_minus_vq']:.6f} | "
        f"[{all_stats['ci95_mean_diff_lo']:.6f}, {all_stats['ci95_mean_diff_hi']:.6f}] | "
        f"{all_stats['wilcoxon_p']:.3e} | {all_stats['ttest_p']:.3e} |"
    )
    md.append(
        f"| 高頻 | {hf_stats['n']} | {hf_stats['mean_diff_no_minus_vq']:.6f} | "
        f"[{hf_stats['ci95_mean_diff_lo']:.6f}, {hf_stats['ci95_mean_diff_hi']:.6f}] | "
        f"{hf_stats['wilcoxon_p']:.3e} | {hf_stats['ttest_p']:.3e} |"
    )
    md.append(
        f"| 高頻-子音候選幀 | {hf_cons_stats['n']} | {hf_cons_stats['mean_diff_no_minus_vq']:.6f} | "
        f"[{hf_cons_stats['ci95_mean_diff_lo']:.6f}, {hf_cons_stats['ci95_mean_diff_hi']:.6f}] | "
        f"{hf_cons_stats['wilcoxon_p']:.3e} | {hf_cons_stats['ttest_p']:.3e} |"
    )
    md.append("")
    md.append("## 注意")
    md.append("- 子音候選幀為 heuristic（非 forced alignment 的音素標註）。")
    md.append("- 幀級樣本存在時間相關性，p-value 可能偏樂觀，建議後續加做音素對齊驗證。")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
