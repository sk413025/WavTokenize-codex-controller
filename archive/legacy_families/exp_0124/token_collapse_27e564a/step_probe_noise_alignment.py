import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, "/home/sbplab/ruizi/WavTokenizer-main")
sys.path.insert(0, "/home/sbplab/ruizi/WavTokenize-self-supervised")

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1219.losses import create_length_mask
from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum


ENCODER_STRIDE = 320


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def label_from_path(path: str):
    path = path.lower()
    if "box" in path:
        return 0
    if "papercup" in path:
        return 1
    if "plastic" in path:
        return 2
    return None


def spearman_corr(x, y):
    if len(x) == 0:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson_corr(x, y):
    if len(x) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def extract_features(model, dataset, indices, batch_size, device):
    feats = []
    labels = []
    accs = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            items = []
            batch_labels = []

            for idx in batch_indices:
                sample = dataset.samples[idx]
                label = label_from_path(sample.get("noisy_path", ""))
                if label is None:
                    continue
                item = dataset[idx]
                items.append(item)
                batch_labels.append(label)

            if not items:
                continue

            batch = collate_fn_curriculum(items)
            noisy = batch["noisy_audio"].to(device)
            clean = batch["clean_audio"].to(device)
            lengths = batch["lengths"].to(device)

            output = model(noisy, clean)
            s_feat = output["student_encoder_out"]  # (B, D, T)
            s_codes = output["student_codes"]
            t_codes = output["teacher_codes"]
            if s_codes.dim() == 3:
                s_codes = s_codes[0]
            if t_codes.dim() == 3:
                t_codes = t_codes[0]

            B, D, T = s_feat.shape
            max_audio_len = T * ENCODER_STRIDE
            mask = create_length_mask(lengths, max_audio_len, ENCODER_STRIDE, device=device)

            # mean pool features
            mask_exp = mask.unsqueeze(1)
            feat_sum = (s_feat * mask_exp).sum(dim=2)
            feat_mean = feat_sum / (mask_exp.sum(dim=2) + 1e-8)

            # per-sample strict acc
            correct = (s_codes == t_codes).float() * mask
            acc_per = (correct.sum(dim=1) / (mask.sum(dim=1) + 1e-8)).detach().cpu().numpy()

            feats.append(feat_mean.detach().cpu().numpy())
            labels.extend(batch_labels)
            accs.extend(acc_per.tolist())

    if feats:
        feats = np.concatenate(feats, axis=0)
    else:
        feats = np.zeros((0, 512), dtype=np.float32)

    return feats, np.array(labels), np.array(accs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str,
                        default="exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train_indices", type=str,
                        default="exp_0124/token_collapse_27e564a/invariance_short_run/runs/train_indices.json")
    parser.add_argument("--val_indices", type=str,
                        default="exp_0124/token_collapse_27e564a/invariance_short_run/runs/val_indices.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="exp_0124/token_collapse_27e564a/probe_noise_alignment")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    run_dir = Path(args.run_dir)
    ckpt_path = args.checkpoint or str(run_dir / "best_model.pt")

    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        lora_dropout=0.2,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    train_indices = json.loads(Path(args.train_indices).read_text())
    val_indices = json.loads(Path(args.val_indices).read_text())

    train_dataset = CurriculumDataset(TRAIN_CACHE, max_samples=None, compute_snr=False)
    val_dataset = CurriculumDataset(VAL_CACHE, max_samples=None, compute_snr=False)

    X_train, y_train, acc_train = extract_features(
        model, train_dataset, train_indices, args.batch_size, device
    )
    X_val, y_val, acc_val = extract_features(
        model, val_dataset, val_indices, args.batch_size, device
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial"),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    prob = clf.predict_proba(X_val)
    max_prob = prob.max(axis=1)
    true_prob = prob[np.arange(len(y_val)), y_val]

    acc = float(accuracy_score(y_val, y_pred))
    cm = confusion_matrix(y_val, y_pred).tolist()

    # correlations
    pearson_true = pearson_corr(true_prob, acc_val)
    spearman_true = spearman_corr(true_prob, acc_val)
    pearson_max = pearson_corr(max_prob, acc_val)
    spearman_max = spearman_corr(max_prob, acc_val)

    # top/bottom quartile
    q25 = np.percentile(max_prob, 25)
    q75 = np.percentile(max_prob, 75)
    low_mask = max_prob <= q25
    high_mask = max_prob >= q75
    acc_low = float(np.mean(acc_val[low_mask])) if low_mask.any() else 0.0
    acc_high = float(np.mean(acc_val[high_mask])) if high_mask.any() else 0.0

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt_path,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "noise_type_accuracy": acc,
        "noise_type_confusion": cm,
        "corr_true_prob_acc": {"pearson": pearson_true, "spearman": spearman_true},
        "corr_max_prob_acc": {"pearson": pearson_max, "spearman": spearman_max},
        "acc_high_conf": acc_high,
        "acc_low_conf": acc_low,
    }

    (out_dir / "probe_noise_alignment.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    plt.figure(figsize=(5, 4))
    plt.scatter(max_prob, acc_val, s=8, alpha=0.5)
    plt.xlabel("Noise-type max prob")
    plt.ylabel("Strict acc (per-sample)")
    plt.title("Noise predictability vs alignment")
    plt.tight_layout()
    plt.savefig(out_dir / "probe_noise_alignment.png")
    plt.close()

    lines = [
        "# Probe noise alignment",
        "",
        f"- timestamp: {payload['timestamp']}",
        f"- checkpoint: {ckpt_path}",
        f"- train_samples: {payload['train_samples']}",
        f"- val_samples: {payload['val_samples']}",
        "",
        f"- noise_type_accuracy: {payload['noise_type_accuracy']:.4f}",
        f"- corr(true_prob, acc): pearson {pearson_true:.4f}, spearman {spearman_true:.4f}",
        f"- corr(max_prob, acc): pearson {pearson_max:.4f}, spearman {spearman_max:.4f}",
        f"- acc_high_conf (top25%): {acc_high:.4f}",
        f"- acc_low_conf (bottom25%): {acc_low:.4f}",
        "",
        "## Confusion matrix (val)",
        str(cm),
        "",
    ]
    (out_dir / "probe_noise_alignment.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote: {out_dir / 'probe_noise_alignment.json'}")
    print(f"Wrote: {out_dir / 'probe_noise_alignment.md'}")
    print(f"Wrote: {out_dir / 'probe_noise_alignment.png'}")


if __name__ == "__main__":
    main()
