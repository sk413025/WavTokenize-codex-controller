"""
音頻質量評估腳本 - PESQ, STOI, SI-SDR

評估 Exp48 模型的去噪效果：
1. Noisy → Student Encoder → Student Tokens → Decoder → Denoised Audio
2. 比較 Denoised Audio 與 Clean Audio 的 PESQ/STOI
3. 同時計算 Noisy Audio 的基準分數
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

# 設定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pesq import pesq
from pystoi import stoi
import torchaudio

from exp_1212.data_aligned import create_aligned_dataloaders
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT
from exp_1217.models import TeacherStudentConfigurableLoRA


def compute_si_sdr(estimate, reference):
    """
    計算 Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        estimate: (T,) 估計信號
        reference: (T,) 參考信號

    Returns:
        si_sdr: float, 單位 dB
    """
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()

    # <s', s> / ||s||^2 * s
    dot = torch.sum(estimate * reference)
    s_target = dot * reference / (torch.sum(reference ** 2) + 1e-8)

    # e_noise = s' - s_target
    e_noise = estimate - s_target

    # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    si_sdr = 10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )

    return si_sdr.item()


def decode_tokens_to_audio(model, tokens, device):
    """
    將 tokens 解碼為音頻

    使用 WavTokenizer 的官方 API:
    1. codes_to_features: tokens -> features
    2. decode: features -> audio

    Args:
        model: TeacherStudentConfigurableLoRA
        tokens: (B, T) token indices
        device: torch device

    Returns:
        audio: (B, samples) 解碼後的音頻
    """
    with torch.no_grad():
        # 確保 tokens 維度正確: (K, B, T) where K=1 for single codebook
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # (1, B, T)

        # 使用 Teacher 的 WavTokenizer (因為 decoder 相同)
        wavtokenizer = model.teacher

        # Step 1: codes -> features
        # codes_to_features 期望 (K, L) or (K, B, L)
        features = wavtokenizer.codes_to_features(tokens)  # (B, C, L)

        # Step 2: features -> audio
        bandwidth_id = torch.tensor([0], device=device)
        audio = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)  # (B, 1, samples)

        if audio.dim() == 3:
            audio = audio.squeeze(1)  # (B, samples)

    return audio


def evaluate_single_sample(clean_audio, noisy_audio, denoised_audio, sample_rate=24000):
    """
    評估單個樣本的音頻質量

    Args:
        clean_audio: (T,) clean reference
        noisy_audio: (T,) noisy input
        denoised_audio: (T,) model output
        sample_rate: 採樣率

    Returns:
        dict: 包含各項指標
    """
    # 確保長度一致
    min_len = min(len(clean_audio), len(noisy_audio), len(denoised_audio))
    clean = clean_audio[:min_len].cpu().numpy()
    noisy = noisy_audio[:min_len].cpu().numpy()
    denoised = denoised_audio[:min_len].cpu().numpy()

    # PESQ (需要 16kHz)
    # 如果是 24kHz，需要 resample
    if sample_rate != 16000:
        import librosa
        clean_16k = librosa.resample(clean, orig_sr=sample_rate, target_sr=16000)
        noisy_16k = librosa.resample(noisy, orig_sr=sample_rate, target_sr=16000)
        denoised_16k = librosa.resample(denoised, orig_sr=sample_rate, target_sr=16000)
    else:
        clean_16k, noisy_16k, denoised_16k = clean, noisy, denoised

    try:
        pesq_noisy = pesq(16000, clean_16k, noisy_16k, 'wb')
        pesq_denoised = pesq(16000, clean_16k, denoised_16k, 'wb')
    except Exception as e:
        pesq_noisy = float('nan')
        pesq_denoised = float('nan')

    # STOI
    try:
        stoi_noisy = stoi(clean, noisy, sample_rate, extended=False)
        stoi_denoised = stoi(clean, denoised, sample_rate, extended=False)
    except Exception as e:
        stoi_noisy = float('nan')
        stoi_denoised = float('nan')

    # SI-SDR
    clean_t = torch.from_numpy(clean).float()
    noisy_t = torch.from_numpy(noisy).float()
    denoised_t = torch.from_numpy(denoised).float()

    si_sdr_noisy = compute_si_sdr(noisy_t, clean_t)
    si_sdr_denoised = compute_si_sdr(denoised_t, clean_t)

    return {
        'pesq_noisy': pesq_noisy,
        'pesq_denoised': pesq_denoised,
        'stoi_noisy': stoi_noisy,
        'stoi_denoised': stoi_denoised,
        'si_sdr_noisy': si_sdr_noisy,
        'si_sdr_denoised': si_sdr_denoised,
    }


def load_model():
    """載入 Exp48 模型"""
    from exp_1217.models import TeacherStudentConfigurableLoRA

    device = torch.device('cuda:0')
    model_path = Path(__file__).parent.parent / 'exp_1217/runs/exp48_best_config/best_model.pt'
    config_path = model_path.parent / 'config.json'

    with open(config_path) as f:
        config = json.load(f)

    model = TeacherStudentConfigurableLoRA(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        lora_layers=config['lora_layers'],
        device='cuda:0'
    )

    checkpoint = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, device


def main():
    print("="*60)
    print("音頻質量評估 - PESQ / STOI / SI-SDR")
    print("="*60)

    # 載入模型
    print("\n載入模型...")
    model, device = load_model()
    print("模型載入完成")

    # 載入資料
    print("\n載入驗證資料...")
    from dataclasses import dataclass

    @dataclass
    class EvalConfig:
        batch_size: int = 4  # 較小 batch 以節省記憶體
        num_workers: int = 2

    _, val_loader = create_aligned_dataloaders(EvalConfig())
    print(f"Val batches: {len(val_loader)}")

    # 評估
    print("\n開始評估...")

    all_results = []
    num_samples = 0
    max_samples = 100  # 限制評估樣本數（PESQ 計算較慢）

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if num_samples >= max_samples:
                break

            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)

            # Forward
            output = model(noisy, clean)
            student_codes = output['student_codes']

            # 確保維度正確
            if student_codes.dim() == 3:
                student_codes = student_codes[0]  # (B, T)

            # 解碼 student tokens 為音頻
            try:
                denoised = decode_tokens_to_audio(model, student_codes, device)
            except Exception as e:
                print(f"Decode error: {e}")
                continue

            # 逐樣本評估
            B = noisy.shape[0]
            for i in range(B):
                if num_samples >= max_samples:
                    break

                # 獲取單個樣本
                clean_i = clean[i].squeeze()
                noisy_i = noisy[i].squeeze()
                denoised_i = denoised[i] if denoised.dim() == 2 else denoised[i].squeeze()

                # 評估
                try:
                    result = evaluate_single_sample(clean_i, noisy_i, denoised_i)
                    all_results.append(result)
                    num_samples += 1
                except Exception as e:
                    print(f"Eval error: {e}")
                    continue

    # 統計結果
    print("\n" + "="*60)
    print("評估結果")
    print("="*60)

    if len(all_results) == 0:
        print("沒有成功評估的樣本")
        return

    # 計算平均值
    pesq_noisy = np.nanmean([r['pesq_noisy'] for r in all_results])
    pesq_denoised = np.nanmean([r['pesq_denoised'] for r in all_results])
    stoi_noisy = np.nanmean([r['stoi_noisy'] for r in all_results])
    stoi_denoised = np.nanmean([r['stoi_denoised'] for r in all_results])
    si_sdr_noisy = np.nanmean([r['si_sdr_noisy'] for r in all_results])
    si_sdr_denoised = np.nanmean([r['si_sdr_denoised'] for r in all_results])

    print(f"\n評估樣本數: {len(all_results)}")

    print("\n### PESQ (越高越好, 範圍 -0.5 ~ 4.5) ###")
    print(f"  Noisy:    {pesq_noisy:.3f}")
    print(f"  Denoised: {pesq_denoised:.3f}")
    print(f"  改善:     {pesq_denoised - pesq_noisy:+.3f}")

    print("\n### STOI (越高越好, 範圍 0 ~ 1) ###")
    print(f"  Noisy:    {stoi_noisy:.3f}")
    print(f"  Denoised: {stoi_denoised:.3f}")
    print(f"  改善:     {stoi_denoised - stoi_noisy:+.3f}")

    print("\n### SI-SDR (越高越好, 單位 dB) ###")
    print(f"  Noisy:    {si_sdr_noisy:.2f} dB")
    print(f"  Denoised: {si_sdr_denoised:.2f} dB")
    print(f"  改善:     {si_sdr_denoised - si_sdr_noisy:+.2f} dB")

    # 儲存結果
    results = {
        'num_samples': len(all_results),
        'pesq': {'noisy': pesq_noisy, 'denoised': pesq_denoised, 'improvement': pesq_denoised - pesq_noisy},
        'stoi': {'noisy': stoi_noisy, 'denoised': stoi_denoised, 'improvement': stoi_denoised - stoi_noisy},
        'si_sdr': {'noisy': si_sdr_noisy, 'denoised': si_sdr_denoised, 'improvement': si_sdr_denoised - si_sdr_noisy},
    }

    output_path = Path(__file__).parent / 'exp48_audio_quality_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n結果已儲存至: {output_path}")


if __name__ == '__main__':
    main()
