import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Ensure repo + external paths are available
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_0112_intermediate.train_v6 import IntermediateSupervisionLossV6, compute_dynamic_intermediate_weight
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1219.losses import MaskedCombinedLossV2, MaskedCrossEntropyLoss
from exp_1226.data_curriculum import CurriculumDataset
from exp_1212.data_aligned import aligned_collate_fn

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


def estimate_snr(noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> float:
    noisy = noisy_audio.squeeze()
    clean = clean_audio.squeeze()
    min_len = min(len(noisy), len(clean))
    noisy = noisy[:min_len]
    clean = clean[:min_len]
    signal_power = (clean ** 2).mean()
    noise = noisy - clean
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return 100.0
    snr = 10 * torch.log10(signal_power / noise_power + 1e-10)
    return float(snr.item())


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


def parse_epoch_from_ckpt(name: str) -> int:
    if 'checkpoint_epoch' in name:
        num = name.split('checkpoint_epoch')[-1].split('.')[0]
        try:
            return int(num)
        except Exception:
            return -1
    return -1


def flatten_grads(grads):
    flats = [g.detach().reshape(-1) for g in grads if g is not None]
    if not flats:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(flats, dim=0)


def get_item(dataset, idx, device):
    item = dataset[idx]
    noisy = item['noisy_audio'].unsqueeze(0).to(device)
    clean = item['clean_audio'].unsqueeze(0).to(device)
    lengths = torch.tensor([item['length']], device=device)
    return noisy, clean, lengths, item


def get_batch(dataset, indices, device):
    items = [dataset[i] for i in indices]
    batch = aligned_collate_fn(items)
    noisy = batch['noisy_audio'].to(device)
    clean = batch['clean_audio'].to(device)
    lengths = batch['lengths'].to(device)
    return noisy, clean, lengths


def compute_loss_train(model, loss_fn, inter_loss_fn, noisy, clean, lengths, intermediate_weight):
    output = model(noisy, clean)
    final_loss, _ = loss_fn(
        student_features=output['student_encoder_out'],
        teacher_features=output['teacher_encoder_out'],
        teacher_codes=output['teacher_codes'],
        codebook=output['codebook'],
        lengths=lengths,
    )
    inter_loss, _ = inter_loss_fn(
        student_features=output['student_intermediates'],
        teacher_features=output['teacher_intermediates'],
    )
    total_loss = final_loss + intermediate_weight * inter_loss
    return total_loss


def compute_loss_anchor(model, anchor_loss_fn, noisy, clean, lengths):
    output = model(noisy, clean)
    logits = model.compute_ce_logits(output['student_encoder_out'])
    t_codes = output['teacher_codes']
    if t_codes.dim() == 3:
        t_codes = t_codes[0]
    loss = anchor_loss_fn(logits, t_codes.long(), lengths)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613')
    parser.add_argument('--failure_set', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/failure_set.json')
    parser.add_argument('--train_candidates', type=int, default=2000)
    parser.add_argument('--val_failures', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoints', type=str,
                        default='checkpoints/checkpoint_epoch010.pt,checkpoints/checkpoint_epoch100.pt,checkpoints/checkpoint_epoch300.pt')
    parser.add_argument('--output_csv', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/tracin_scores.csv')
    parser.add_argument('--meta_out', type=str,
                        default='exp_0125/tracin_token_collapse_589e6d/tracin_indices.json')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Batch size for train-side gradient approximation')
    parser.add_argument('--val_grad_dtype', type=str, default='float16',
                        choices=['float16', 'float32'], help='Store val gradients in this dtype')
    parser.add_argument('--val_grad_device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to store val gradients for dot product')
    parser.add_argument('--loss_types', type=str, default='train,anchor',
                        help='Comma-separated list: train,anchor')
    parser.add_argument('--val_aggregate', action='store_true',
                        help='Aggregate val gradients into a single mean vector to speed up TracIn')
    parser.add_argument('--approx_train_loss', action='store_true',
                        help='Use feature+intermediate loss only (triplet_weight=0) for faster TracIn')
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    print(f'Using device: {device}')

    run_dir = Path(args.run_dir)
    ckpt_paths = [run_dir / p.strip() for p in args.checkpoints.split(',')]
    ckpt_paths = [p for p in ckpt_paths if p.exists()]
    if not ckpt_paths:
        raise FileNotFoundError('No checkpoints found for TracIn')

    # Load config for loss weights
    config = json.loads((run_dir / 'config.json').read_text())
    triplet_weight = 0.0 if args.approx_train_loss else config.get('triplet_weight', 1.0)
    loss_fn = MaskedCombinedLossV2(
        feature_weight=config.get('feature_weight', 1.0),
        cosine_weight=config.get('cosine_weight', 0.0),
        triplet_weight=triplet_weight,
        triplet_margin=config.get('triplet_margin', 0.2),
        ce_weight=config.get('ce_weight', 0.0),
        encoder_stride=ENCODER_STRIDE,
    )
    inter_loss_fn = IntermediateSupervisionLossV6(
        layer_weights={
            3: config.get('intermediate_L3_weight', 0.3),
            4: config.get('intermediate_L4_weight', 0.5),
            6: config.get('intermediate_L6_weight', 0.5),
        },
        target_scale=config.get('target_scale', 1.0),
    )
    anchor_loss_fn = MaskedCrossEntropyLoss(encoder_stride=ENCODER_STRIDE)

    # Data + indices
    train_dataset = CurriculumDataset(TRAIN_CACHE, compute_snr=False)
    val_dataset = CurriculumDataset(VAL_CACHE, compute_snr=False)

    rng = np.random.RandomState(args.seed)
    train_indices = rng.choice(len(train_dataset), size=args.train_candidates, replace=False).tolist()

    failure_payload = json.loads(Path(args.failure_set).read_text())
    failure_indices_all = [x['index'] for x in failure_payload['failure_set']]
    if len(failure_indices_all) < args.val_failures:
        val_indices = failure_indices_all
    else:
        val_indices = rng.choice(failure_indices_all, size=args.val_failures, replace=False).tolist()

    # Precompute meta for indices
    train_meta = {}
    for idx in train_indices:
        item = train_dataset.samples[idx]
        noisy_path = item.get('noisy_path', None)
        clean_path = item.get('clean_path', None)
        noisy_audio, clean_audio, _, _item = get_item(train_dataset, idx, device='cpu')
        snr = estimate_snr(noisy_audio, clean_audio)
        energy = float((noisy_audio.squeeze() ** 2).mean().item())
        train_meta[idx] = {
            'snr_db': snr,
            'noisy_energy': energy,
            'noisy_path': noisy_path,
            'clean_path': clean_path,
        }

    val_meta = {}
    for idx in val_indices:
        item = val_dataset.samples[idx]
        noisy_path = item.get('noisy_path', None)
        clean_path = item.get('clean_path', None)
        noisy_audio, clean_audio, _, _item = get_item(val_dataset, idx, device='cpu')
        snr = estimate_snr(noisy_audio, clean_audio)
        energy = float((noisy_audio.squeeze() ** 2).mean().item())
        val_meta[idx] = {
            'snr_db': snr,
            'noisy_energy': energy,
            'noisy_path': noisy_path,
            'clean_path': clean_path,
        }

    # Load history for lr per epoch
    history = json.loads((run_dir / 'history.json').read_text())
    lr_list = history.get('lr', [])

    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=config.get('lora_rank', 256),
        lora_alpha=config.get('lora_alpha', 512),
        lora_dropout=config.get('lora_dropout', 0.2),
        intermediate_indices=[3, 4, 6],
    )
    model.to(device)
    model.eval()

    params = [p for p in model.parameters() if p.requires_grad]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['val_index', 'train_index', 'checkpoint', 'loss_type', 'lr', 'score'])

        loss_types = [x.strip() for x in args.loss_types.split(',') if x.strip()]

        for ckpt_path in ckpt_paths:
            ckpt_type = load_checkpoint(model, ckpt_path)
            # cuDNN RNN backward requires training mode
            model.train()
            model.teacher.eval()
            epoch = parse_epoch_from_ckpt(ckpt_path.name)
            lr = lr_list[epoch - 1] if epoch > 0 and epoch - 1 < len(lr_list) else config.get('lr', 1e-4)

            intermediate_weight = compute_dynamic_intermediate_weight(
                epoch=epoch if epoch > 0 else 1,
                curriculum_epochs=config.get('curriculum_epochs', 200),
                base_weight=config.get('intermediate_weight', 0.5),
                min_weight=config.get('intermediate_weight_min', 0.25),
                warmdown_epochs=config.get('warmdown_epochs', 50),
            )

            print(f'Checkpoint: {ckpt_path.name} ({ckpt_type}), epoch={epoch}, lr={lr:.6f}, inter_w={intermediate_weight:.3f}')

            for loss_type in loss_types:
                # Precompute val gradients
                val_grads = []
                for v_idx in val_indices:
                    noisy, clean, lengths, _item = get_item(val_dataset, v_idx, device)
                    model.zero_grad(set_to_none=True)
                    if loss_type == 'train':
                        loss = compute_loss_train(model, loss_fn, inter_loss_fn, noisy, clean, lengths, intermediate_weight)
                    else:
                        loss = compute_loss_anchor(model, anchor_loss_fn, noisy, clean, lengths)
                    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                    grad_vec = flatten_grads(grads).detach()
                    val_grads.append(grad_vec)

                if len(val_grads) == 0:
                    continue

                if args.val_aggregate:
                    val_grads_mat = torch.stack(val_grads, dim=0).mean(dim=0, keepdim=True)
                else:
                    val_grads_mat = torch.stack(val_grads, dim=0)

                if args.val_grad_dtype == 'float16':
                    val_grads_mat = val_grads_mat.half()
                else:
                    val_grads_mat = val_grads_mat.float()
                if args.val_grad_device == 'cpu':
                    val_grads_mat = val_grads_mat.cpu()

                # Train gradients and dot products (batched approximation)
                for i in range(0, len(train_indices), args.train_batch_size):
                    batch_indices = train_indices[i:i + args.train_batch_size]
                    noisy, clean, lengths = get_batch(train_dataset, batch_indices, device)
                    model.zero_grad(set_to_none=True)
                    if loss_type == 'train':
                        loss = compute_loss_train(model, loss_fn, inter_loss_fn, noisy, clean, lengths, intermediate_weight)
                    else:
                        loss = compute_loss_anchor(model, anchor_loss_fn, noisy, clean, lengths)
                    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                    train_vec = flatten_grads(grads).detach()
                    if args.val_grad_dtype == 'float16':
                        train_vec = train_vec.half()
                    else:
                        train_vec = train_vec.float()
                    if args.val_grad_device == 'cpu':
                        train_vec = train_vec.cpu()

                    scores = torch.mv(val_grads_mat, train_vec) * float(lr)
                    scores = scores.detach().cpu().numpy().tolist()
                    for t_idx in batch_indices:
                        if args.val_aggregate:
                            writer.writerow(['VAL_AGG', t_idx, ckpt_path.name, loss_type, lr, float(scores[0])])
                        else:
                            for v_idx, score in zip(val_indices, scores):
                                writer.writerow([v_idx, t_idx, ckpt_path.name, loss_type, lr, float(score)])

                # free memory
                del val_grads, val_grads_mat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    meta_out = Path(args.meta_out)
    meta_out.write_text(json.dumps({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'checkpoints': [str(p) for p in ckpt_paths],
        'train_candidates': train_indices,
        'val_failures': val_indices,
        'train_meta': train_meta,
        'val_meta': val_meta,
        'approx_train_loss': bool(args.approx_train_loss),
        'triplet_weight_used': float(triplet_weight),
        'train_batch_size': int(args.train_batch_size),
        'train_grad_approx': 'batch_grad shared across samples in batch',
        'val_grad_dtype': args.val_grad_dtype,
        'val_grad_device': args.val_grad_device,
        'loss_types': loss_types,
        'val_aggregate': bool(args.val_aggregate),
    }, indent=2, ensure_ascii=False))

    print(f'Wrote: {output_csv}')
    print(f'Wrote: {meta_out}')


if __name__ == '__main__':
    main()
