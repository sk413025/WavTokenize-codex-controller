#!/usr/bin/env python3
"""
生成 exp3 的音頻樣本，用於評估音質
"""
import torch
import torchaudio
import sys
from pathlib import Path

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 載入模型
    print("\n[1] 載入模型...")
    from model import TeacherStudentModel
    from config import TrainConfig, WAVTOK_CONFIG, WAVTOK_CKPT

    config = TrainConfig()
    config.lora_rank = 64
    config.lora_alpha = 128

    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        device=str(device)
    ).to(device)

    # 載入 checkpoint
    ckpt_path = Path(__file__).parent / "experiments/lora_r64_dist0.05/checkpoints/latest.pt"
    print(f"    Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"    Loaded epoch {ckpt.get('epoch', 'unknown')}")

    # 載入測試數據
    print("\n[2] 載入測試數據...")
    from data import NoisyCleanPairDataset, collate_fn
    from torch.utils.data import DataLoader

    val_dataset = NoisyCleanPairDataset(
        cache_path="/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt",
        max_samples=10
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 輸出目錄
    output_dir = Path(__file__).parent / "experiments/lora_r64_dist0.05/audio_samples/generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000

    print(f"\n[3] 生成音頻樣本到 {output_dir}")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # 只生成 5 個樣本
                break

            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            print(f"\n  樣本 {i+1}:")
            print(f"    noisy shape: {noisy_audio.shape}")
            print(f"    clean shape: {clean_audio.shape}")

            # 1. 保存 noisy
            noisy_path = output_dir / f"sample_{i+1}_noisy.wav"
            noisy_save = noisy_audio.squeeze(0) if noisy_audio.dim() > 2 else noisy_audio
            if noisy_save.dim() == 1:
                noisy_save = noisy_save.unsqueeze(0)
            torchaudio.save(str(noisy_path), noisy_save.cpu(), sample_rate)
            print(f"    Saved: {noisy_path.name}")

            # 2. 保存 clean
            clean_path = output_dir / f"sample_{i+1}_clean.wav"
            clean_save = clean_audio.squeeze(0) if clean_audio.dim() > 2 else clean_audio
            if clean_save.dim() == 1:
                clean_save = clean_save.unsqueeze(0)
            torchaudio.save(str(clean_path), clean_save.cpu(), sample_rate)
            print(f"    Saved: {clean_path.name}")

            # 3. 使用 model forward 獲取 codes
            output = model(noisy_audio, clean_audio)
            student_codes = output['student_codes']  # (B, 1, T)
            teacher_codes = output['teacher_codes']  # (B, 1, T)

            # Helper function: codes -> audio using WavTokenizer decode
            wavtok = model.teacher

            def codes_to_audio(codes):
                """將 codes 轉換為音頻"""
                # codes: (B, 1, T)
                # 使用 WavTokenizer 的標準流程: codes -> features -> decode
                features = wavtok.codes_to_features(codes)  # (B, 512, T)
                audio = wavtok.decode(features, bandwidth_id=torch.tensor([0]).to(device))  # (B, audio_len)
                return audio

            def save_audio(audio, path):
                """確保 audio 是 2D (C, T) 後保存"""
                # decoder 輸出: (B, C, T) where B=1, C=1
                audio = audio.squeeze()  # 移除所有 size=1 的維度
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # (T,) -> (1, T)
                elif audio.dim() > 2:
                    audio = audio.squeeze(0)  # (1, 1, T) -> (1, T)
                torchaudio.save(str(path), audio.cpu(), sample_rate)

            # Student prediction: student codes -> decode
            student_pred = codes_to_audio(student_codes)
            student_pred_path = output_dir / f"sample_{i+1}_student_pred.wav"
            save_audio(student_pred, student_pred_path)
            print(f"    Saved: {student_pred_path.name}")

            # 4. Teacher reconstruction: teacher codes -> decode
            teacher_recon = codes_to_audio(teacher_codes)
            teacher_path = output_dir / f"sample_{i+1}_teacher_recon.wav"
            save_audio(teacher_recon, teacher_path)
            print(f"    Saved: {teacher_path.name}")

            # 5. Baseline: teacher 處理 noisy (無 LoRA 效果)
            # 使用 model forward 但用 noisy 當作 clean 來獲取 teacher codes
            with torch.no_grad():
                baseline_output = model(noisy_audio, noisy_audio)  # 用 noisy 作為 clean
                teacher_noisy_codes = baseline_output['teacher_codes']  # teacher 處理 noisy 的結果
            baseline_recon = codes_to_audio(teacher_noisy_codes)
            baseline_path = output_dir / f"sample_{i+1}_baseline_noisy.wav"
            save_audio(baseline_recon, baseline_path)
            print(f"    Saved: {baseline_path.name}")

    print(f"\n✅ 完成！音頻樣本保存在: {output_dir}")
    print("\n每個樣本包含 5 個音檔:")
    print("  - noisy: 原始噪音音頻")
    print("  - clean: 目標乾淨音頻 (ground truth)")
    print("  - student_pred: Student 預測 (noisy → student encoder → decoder)")
    print("  - teacher_recon: Teacher 重建 (clean → teacher encoder → decoder)")
    print("  - baseline_noisy: Baseline (noisy → teacher encoder → decoder, 無 LoRA)")
    print("\n比較方式:")
    print("  - student_pred vs clean: 評估去噪效果")
    print("  - student_pred vs baseline_noisy: 評估 LoRA 改善程度")

if __name__ == "__main__":
    main()
