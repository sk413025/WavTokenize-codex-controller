"""
Exp61: 重新產出音檔和 loss 圖

修復問題:
1. 音檔使用錯誤的 decode 方式 (直接 decoder 而非 teacher.decode)
2. 缺少 loss 圖
"""

import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1223.data_speaker import SpeakerAwareDataset


def plot_metrics(history, exp_dir):
    """繪製訓練曲線 (包含 speaker weight)"""
    # Extract data from history structure (list of dicts per epoch)
    train_data = history.get('train', [])
    val_data = history.get('val', [])

    if not train_data or not val_data:
        print("Warning: Empty history data")
        return

    # Helper function to extract metric
    def get_metric(data, key, default=0):
        return [d.get(key, default) for d in data]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Total Loss
    ax = axes[0, 0]
    ax.plot(get_metric(train_data, 'total_loss'), label='Train')
    ax.plot(get_metric(val_data, 'total_loss'), label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    # Masked Accuracy
    ax = axes[0, 1]
    train_acc = [x * 100 for x in get_metric(train_data, 'masked_acc')]
    val_acc = [x * 100 for x in get_metric(val_data, 'masked_acc')]
    ax.plot(train_acc, label='Train Masked')
    ax.plot(val_acc, label='Val Masked')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Masked Token Accuracy')
    ax.legend()
    ax.grid(True)

    # Speaker Weight
    ax = axes[0, 2]
    train_spk = get_metric(train_data, 'speaker_weight_mean')
    val_spk = get_metric(val_data, 'speaker_weight_mean')
    if any(train_spk):
        ax.plot(train_spk, label='Train')
        ax.plot(val_spk, label='Val')
        ax.axhline(y=0.5, color='r', linestyle='--', label='Min Weight')
        ax.axhline(y=1.0, color='g', linestyle='--', label='Max Weight')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Speaker Weight')
    ax.set_title('Speaker Weight (Mean)')
    ax.legend()
    ax.grid(True)

    # Cosine Similarity (if available)
    ax = axes[0, 3]
    train_cos = get_metric(train_data, 'cos_sim_mean')
    val_cos = get_metric(val_data, 'cos_sim_mean')
    if any(train_cos):
        ax.plot(train_cos, label='Train')
        ax.plot(val_cos, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cosine Similarity')
    ax.legend()
    ax.grid(True)

    # Feature Loss
    ax = axes[1, 0]
    ax.plot(get_metric(train_data, 'feature_loss'), label='Train')
    ax.plot(get_metric(val_data, 'feature_loss'), label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature Loss')
    ax.legend()
    ax.grid(True)

    # Triplet Loss
    ax = axes[1, 1]
    ax.plot(get_metric(train_data, 'triplet_loss'), label='Train')
    ax.plot(get_metric(val_data, 'triplet_loss'), label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Triplet Loss')
    ax.set_title('Triplet Loss')
    ax.legend()
    ax.grid(True)

    # Distance
    ax = axes[1, 2]
    ax.plot(get_metric(train_data, 'distance_loss'), label='Train')
    ax.plot(get_metric(val_data, 'distance_loss'), label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Average Distance')
    ax.legend()
    ax.grid(True)

    # Val Accuracy Zoomed
    ax = axes[1, 3]
    ax.plot(val_acc, label='Val Masked Acc', color='orange')
    if val_acc:
        ax.axhline(y=max(val_acc), color='g', linestyle='--', label=f'Best: {max(val_acc):.2f}%')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Val Masked Accuracy (Zoomed)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"Saved training curves to {exp_dir / 'training_curves.png'}")


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, num_samples=5):
    """保存音檔樣本 (使用正確的 WavTokenizer decode 流程)"""
    import torchaudio

    model.eval()
    audio_dir = exp_dir / 'audio_samples_regenerated'
    audio_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    data_iter = iter(dataloader)

    torch.cuda.empty_cache()

    for i in range(num_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        noisy_audio = batch['noisy_audio'][:1].to(device)
        clean_audio = batch['clean_audio'][:1].to(device)

        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)

        # Save original audio
        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'), noisy_audio.cpu(), sample_rate)
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'), clean_audio.cpu(), sample_rate)

        try:
            # Student reconstruction: encode with student, decode with teacher
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_student_recon.wav'), student_recon.cpu(), sample_rate)
            del student_features, student_recon
            torch.cuda.empty_cache()

            # Teacher reconstruction: encode and decode with teacher (for reference)
            teacher_features, _, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(device))
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_teacher_recon.wav'), teacher_recon.cpu(), sample_rate)
            del teacher_features, teacher_recon
            torch.cuda.empty_cache()

            print(f"  Saved sample {i+1}")

        except Exception as e:
            print(f"  Warning: Failed to save audio sample {i+1}: {e}")
            continue

    print(f"Saved {num_samples} audio samples to {audio_dir}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    exp_dir = Path('/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1223/runs/exp61_speaker_weighted')

    # 1. 繪製 loss 圖
    print("\n=== Generating Training Curves ===")
    history_path = exp_dir / 'history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_metrics(history, exp_dir)
    else:
        print(f"Warning: {history_path} not found")

    # 2. 載入模型並產出音檔
    print("\n=== Generating Audio Samples ===")

    # Load config
    config_path = exp_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    # Create model (使用與訓練相同的配置)
    wavtok_config = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    wavtok_ckpt = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

    print("Loading model...")
    model = TeacherStudentConfigurableLoRA(
        wavtok_config=wavtok_config,
        wavtok_ckpt=wavtok_ckpt,
        lora_rank=config.get('lora_rank', 256),
        lora_alpha=config.get('lora_alpha', 512),
        lora_dropout=config.get('lora_dropout', 0.2),
        lora_layers=config.get('lora_layers', 'all_18'),
        device=device,
    )

    # Load best checkpoint
    best_ckpt = exp_dir / 'best_model.pt'
    if best_ckpt.exists():
        print(f"Loading checkpoint: {best_ckpt}")
        state_dict = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded")
    else:
        print(f"Warning: {best_ckpt} not found, using initial model")

    model.eval()

    # Create dataloader (使用與訓練相同的資料)
    print("Loading validation data...")
    from exp_1201.config import VAL_CACHE
    from exp_1223.data_speaker import collate_fn_speaker

    val_dataset = SpeakerAwareDataset(
        cache_path=VAL_CACHE,
        max_samples=100,
        filter_clean_to_clean=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_speaker,
    )

    # Generate audio
    save_audio_samples(model, val_loader, device, exp_dir, num_samples=5)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
