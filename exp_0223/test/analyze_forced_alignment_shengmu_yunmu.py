#!/usr/bin/env python3
"""
使用 torchaudio MMS_FA 進行 forced alignment，並比較：
  noisy_through_teacher (VQ) vs noisy_through_teacher_no_vq (No-VQ)
在「聲母/韻母」片段上的重建誤差差異。

輸出：
  - forced_alignment_shengmu_yunmu_stats.json
  - forced_alignment_shengmu_yunmu_stats.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import stft
from scipy.stats import ttest_1samp, wilcoxon


ROOT = Path(__file__).resolve().parent
DIR_VQ = ROOT / "noisy_through_teacher"
DIR_NO_VQ = ROOT / "noisy_through_teacher_no_vq"

# 使用者提供的逐句文字，已轉為拼音（無聲調）
TRANSCRIPTS_PINYIN: Dict[str, List[str]] = {
    "sample01": ["zhe", "ge", "an", "zi", "ming", "tian", "kai", "ting", "xuan", "pan"],
    "sample02": ["ta", "jin", "tian", "jue", "de", "you", "dian", "bu", "shu", "fu"],
    "sample03": ["yi", "da", "zao", "ta", "jiu", "zai", "wai", "mian", "sao", "di"],
}

SAMPLE_IDS = ["sample01", "sample02", "sample03"]

# 漢語拼音聲母（依長度排序，避免 zh/ch/sh 被拆成 z/h）
INITIALS = [
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
]


def split_initial_final(syllable: str) -> Tuple[str, str]:
    s = syllable.lower()
    for ini in INITIALS:
        if s.startswith(ini):
            return ini, s[len(ini):]
    return "", s


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> Tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def paired_stats(diff: np.ndarray) -> Dict[str, float]:
    """diff = no_vq - vq。負值代表 No-VQ 較好。"""
    n = int(len(diff))
    mean = float(np.mean(diff)) if n else float("nan")
    std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    ci_lo, ci_hi = bootstrap_ci_mean(diff) if n else (float("nan"), float("nan"))
    dz = float(mean / std) if (n > 1 and std > 0) else float("nan")

    if n > 1:
        t_stat, t_p = ttest_1samp(diff, popmean=0.0, nan_policy="omit")
        t_stat = float(t_stat)
        t_p = float(t_p)
    else:
        t_stat, t_p = float("nan"), float("nan")

    try:
        if n > 0:
            w_stat, w_p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", correction=False)
            w_stat = float(w_stat)
            w_p = float(w_p)
        else:
            w_stat, w_p = float("nan"), float("nan")
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


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float64), sr


def get_emission_and_spans(
    model: torch.nn.Module,
    bundle,
    wav_clean: np.ndarray,
    sr_clean: int,
    transcript_syllables: List[str],
):
    wave = torch.tensor(wav_clean, dtype=torch.float32).unsqueeze(0)  # [1, T]
    if sr_clean != bundle.sample_rate:
        wave = torchaudio.functional.resample(wave, sr_clean, bundle.sample_rate)

    with torch.inference_mode():
        emissions, _ = model(wave)  # [1, num_frames, num_labels]
    emission = emissions[0].cpu()

    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    token_ids = tokenizer(transcript_syllables)
    spans = aligner(emission, token_ids)  # list per syllable, each list has len==len(syllable)

    num_frames = emission.shape[0]
    duration_s = wave.shape[-1] / bundle.sample_rate
    frame_rate = num_frames / duration_s
    return spans, frame_rate


def analyze_one_sample(sample_id: str, model, bundle):
    # 讀音檔
    clean, sr_c = load_wav(DIR_VQ / f"{sample_id}_clean.wav")
    recon_vq, sr_v = load_wav(DIR_VQ / f"{sample_id}_recon.wav")
    recon_no, sr_n = load_wav(DIR_NO_VQ / f"{sample_id}_recon.wav")
    if not (sr_c == sr_v == sr_n):
        raise RuntimeError(f"sample rate mismatch in {sample_id}: {sr_c}, {sr_v}, {sr_n}")

    L = min(len(clean), len(recon_vq), len(recon_no))
    clean = clean[:L]
    recon_vq = recon_vq[:L]
    recon_no = recon_no[:L]

    syllables = TRANSCRIPTS_PINYIN[sample_id]
    spans, align_fps = get_emission_and_spans(model, bundle, clean, sr_c, syllables)

    # STFT frame-level errors
    n_fft, win, hop = 1024, 1024, 256
    f, t, Zc = stft(clean, fs=sr_c, nperseg=win, noverlap=win - hop, nfft=n_fft, boundary=None)
    _, _, Zv = stft(recon_vq, fs=sr_c, nperseg=win, noverlap=win - hop, nfft=n_fft, boundary=None)
    _, _, Zn = stft(recon_no, fs=sr_c, nperseg=win, noverlap=win - hop, nfft=n_fft, boundary=None)

    log_c = np.log(np.abs(Zc) + 1e-8)
    log_v = np.log(np.abs(Zv) + 1e-8)
    log_n = np.log(np.abs(Zn) + 1e-8)

    hf = f >= 4000
    full_lse_v = np.mean(np.abs(log_v - log_c), axis=0)
    full_lse_n = np.mean(np.abs(log_n - log_c), axis=0)
    hf_lse_v = np.mean(np.abs(log_v[hf] - log_c[hf]), axis=0)
    hf_lse_n = np.mean(np.abs(log_n[hf] - log_c[hf]), axis=0)
    diff_full = full_lse_n - full_lse_v
    diff_hf = hf_lse_n - hf_lse_v

    # 依照 forced alignment，把 STFT frame 指派到聲母/韻母 token
    # 注意：align span 的 start/end 是 align frame index（~50fps at 16k）
    # 先轉成秒，再用 STFT frame center time 匹配。
    tokens = []
    for syl, syl_spans in zip(syllables, spans):
        initial, final = split_initial_final(syl)
        init_len = len(initial)
        # syllable letters (與 tokenizer 的字元級一致)
        letters = list(syl)
        if len(letters) != len(syl_spans):
            raise RuntimeError(f"token length mismatch for syllable {syl} in {sample_id}")
        for i, (ch, sp) in enumerate(zip(letters, syl_spans)):
            kind = "shengmu" if i < init_len else "yunmu"
            t0 = float(sp.start / align_fps)
            t1 = float(sp.end / align_fps)
            tokens.append({"char": ch, "kind": kind, "t0": t0, "t1": t1})

    # frame centers t (seconds) from scipy stft
    token_diffs_hf = {"shengmu": [], "yunmu": []}
    token_diffs_full = {"shengmu": [], "yunmu": []}

    # 保留對齊信心（token span score）
    # 這裡使用 aligner 回傳的 span.score，作為 transcript 對齊品質參考
    all_token_scores = []
    for syl_spans in spans:
        for sp in syl_spans:
            all_token_scores.append(float(sp.score))

    for tok in tokens:
        mask = (t >= tok["t0"]) & (t < tok["t1"])
        if mask.sum() == 0:
            continue
        token_diffs_hf[tok["kind"]].append(float(np.mean(diff_hf[mask])))
        token_diffs_full[tok["kind"]].append(float(np.mean(diff_full[mask])))

    out = {
        "sample": sample_id,
        "n_stft_frames": int(len(t)),
        "n_tokens_total": int(len(tokens)),
        "n_tokens_shengmu": int(sum(1 for x in tokens if x["kind"] == "shengmu")),
        "n_tokens_yunmu": int(sum(1 for x in tokens if x["kind"] == "yunmu")),
        "align_score_mean": float(np.mean(all_token_scores)) if all_token_scores else float("nan"),
        "align_score_min": float(np.min(all_token_scores)) if all_token_scores else float("nan"),
        "align_score_max": float(np.max(all_token_scores)) if all_token_scores else float("nan"),
        "mean_diff_hf_all_tokens_no_minus_vq": float(
            np.mean(token_diffs_hf["shengmu"] + token_diffs_hf["yunmu"])
        ) if (token_diffs_hf["shengmu"] or token_diffs_hf["yunmu"]) else float("nan"),
        "mean_diff_hf_shengmu_no_minus_vq": float(np.mean(token_diffs_hf["shengmu"])) if token_diffs_hf["shengmu"] else float("nan"),
        "mean_diff_hf_yunmu_no_minus_vq": float(np.mean(token_diffs_hf["yunmu"])) if token_diffs_hf["yunmu"] else float("nan"),
        "mean_diff_full_shengmu_no_minus_vq": float(np.mean(token_diffs_full["shengmu"])) if token_diffs_full["shengmu"] else float("nan"),
        "mean_diff_full_yunmu_no_minus_vq": float(np.mean(token_diffs_full["yunmu"])) if token_diffs_full["yunmu"] else float("nan"),
        "token_diffs_hf": token_diffs_hf,
        "token_diffs_full": token_diffs_full,
    }
    return out


def main():
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().eval()

    per_sample = []
    all_hf_shengmu = []
    all_hf_yunmu = []
    all_full_shengmu = []
    all_full_yunmu = []

    for sid in SAMPLE_IDS:
        res = analyze_one_sample(sid, model, bundle)
        per_sample.append(res)
        all_hf_shengmu.extend(res["token_diffs_hf"]["shengmu"])
        all_hf_yunmu.extend(res["token_diffs_hf"]["yunmu"])
        all_full_shengmu.extend(res["token_diffs_full"]["shengmu"])
        all_full_yunmu.extend(res["token_diffs_full"]["yunmu"])

    all_hf_shengmu = np.asarray(all_hf_shengmu, dtype=np.float64)
    all_hf_yunmu = np.asarray(all_hf_yunmu, dtype=np.float64)
    all_full_shengmu = np.asarray(all_full_shengmu, dtype=np.float64)
    all_full_yunmu = np.asarray(all_full_yunmu, dtype=np.float64)

    stats = {
        "hf_shengmu": paired_stats(all_hf_shengmu),
        "hf_yunmu": paired_stats(all_hf_yunmu),
        "full_shengmu": paired_stats(all_full_shengmu),
        "full_yunmu": paired_stats(all_full_yunmu),
        "hf_shengmu_minus_yunmu_mean_diff": float(np.mean(all_hf_shengmu) - np.mean(all_hf_yunmu)),
    }

    summary = {
        "mean_hf_diff_no_minus_vq_shengmu": float(np.mean(all_hf_shengmu)),
        "mean_hf_diff_no_minus_vq_yunmu": float(np.mean(all_hf_yunmu)),
        "mean_full_diff_no_minus_vq_shengmu": float(np.mean(all_full_shengmu)),
        "mean_full_diff_no_minus_vq_yunmu": float(np.mean(all_full_yunmu)),
        "n_hf_shengmu_tokens": int(len(all_hf_shengmu)),
        "n_hf_yunmu_tokens": int(len(all_hf_yunmu)),
    }

    results = {
        "metadata": {
            "comparison": "noisy_through_teacher (vq) vs noisy_through_teacher_no_vq (no-vq)",
            "diff_definition": "no-vq LSE - vq LSE (negative means no-vq better)",
            "alignment_model": str(bundle),
            "sample_ids": SAMPLE_IDS,
            "transcripts_pinyin": TRANSCRIPTS_PINYIN,
            "notes": [
                "forced alignment on clean reference audio",
                "token class uses pinyin initial/final split (shengmu/yunmu)",
                "HF band = >=4kHz",
            ],
        },
        "per_sample": per_sample,
        "summary": summary,
        "stats": stats,
    }

    out_json = ROOT / "forced_alignment_shengmu_yunmu_stats.json"
    out_md = ROOT / "forced_alignment_shengmu_yunmu_stats.md"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    s = summary
    hf_sm = stats["hf_shengmu"]
    hf_ym = stats["hf_yunmu"]
    md_lines = [
        "# Forced Alignment：聲母/韻母統計（Noisy 路徑）",
        "",
        "比較：`noisy_through_teacher`（VQ） vs `noisy_through_teacher_no_vq`（No-VQ）",
        "差值定義：`No-VQ LSE - VQ LSE`，負值代表 No-VQ 較好。",
        "",
        "## Token-level 平均差（forced alignment）",
        "",
        "| 類別 | N tokens | HF 平均差 | 95% CI | Wilcoxon p | t-test p |",
        "|---|---:|---:|---|---:|---:|",
        f"| 聲母 (shengmu) | {hf_sm['n']} | {hf_sm['mean_diff_no_minus_vq']:.6f} | "
        f"[{hf_sm['ci95_mean_diff_lo']:.6f}, {hf_sm['ci95_mean_diff_hi']:.6f}] | "
        f"{hf_sm['wilcoxon_p']:.3e} | {hf_sm['ttest_p']:.3e} |",
        f"| 韻母 (yunmu) | {hf_ym['n']} | {hf_ym['mean_diff_no_minus_vq']:.6f} | "
        f"[{hf_ym['ci95_mean_diff_lo']:.6f}, {hf_ym['ci95_mean_diff_hi']:.6f}] | "
        f"{hf_ym['wilcoxon_p']:.3e} | {hf_ym['ttest_p']:.3e} |",
        "",
        "## 平均差摘要",
        "",
        f"- HF 聲母平均差：`{s['mean_hf_diff_no_minus_vq_shengmu']:.6f}`",
        f"- HF 韻母平均差：`{s['mean_hf_diff_no_minus_vq_yunmu']:.6f}`",
        f"- Full-band 聲母平均差：`{s['mean_full_diff_no_minus_vq_shengmu']:.6f}`",
        f"- Full-band 韻母平均差：`{s['mean_full_diff_no_minus_vq_yunmu']:.6f}`",
        "",
        "## 對齊信心（每句平均 token score）",
        "",
    ]
    for r in per_sample:
        md_lines.append(
            f"- {r['sample']}: mean={r['align_score_mean']:.3f}, "
            f"min={r['align_score_min']:.3f}, max={r['align_score_max']:.3f}"
        )
    md_lines.extend([
        "",
        "## 注意",
        "- 這裡的子音/母音以中文語音學常用的「聲母/韻母」近似，而非 IPA phone 集。",
        "- 樣本數目前僅 3 句（token 級統計），結論屬初步證據。",
    ])
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
