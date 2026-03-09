"""
Token 453 遮蔽聽感實驗

Teacher (WavTokenizer) val 資料中，Token 453 佔比 18.43% (第一名)。
本腳本:
1. 選幾段 val 音檔
2. 用 WavTokenizer encode → codes
3. 產出三種版本:
   a) 原始重建 (所有 token 保留)
   b) Token 453 替換為第二熱門 token (1145)
   c) Token 453 替換為隨機 non-453 token
4. 儲存為 wav 比較

執行:
    CUDA_VISIBLE_DEVICES=2 python exp_0128/baseline_token_analysis/token_masking_listen.py
"""

import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

FUNC_NAME = 'token_masking_listen'
DATE_STR = datetime.now().strftime('%Y%m%d')


def load_wavtokenizer(device='cuda:0'):
    """載入 WavTokenizer 模型

    Args:
        device: 計算裝置

    Returns:
        WavTokenizer 模型實例
    """
    from decoder.pretrained import WavTokenizer
    from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT
    model = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)
    model = model.to(device).eval()
    print(f"WavTokenizer loaded on {device}")
    return model


def encode_audio(model, audio, device='cuda:0'):
    """用 WavTokenizer encode 音檔取得 codes

    Args:
        model: WavTokenizer 模型
        audio: [1, 1, T] 音檔 tensor
        device: 計算裝置

    Returns:
        codes: [1, 1, T_frames] token indices
    """
    with torch.no_grad():
        audio = audio.to(device)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Encode
        features = model.feature_extractor.encodec.encoder(audio)
        vq_result = model.feature_extractor.encodec.quantizer(
            features, frame_rate=75, bandwidth=0.075
        )
        codes = vq_result.codes  # [n_q, B, T]
        return codes


def decode_from_codes(model, codes, device='cuda:0'):
    """用 WavTokenizer 從 codes decode 重建音檔

    Args:
        model: WavTokenizer 模型
        codes: [n_q, B, T_frames] token indices
        device: 計算裝置

    Returns:
        audio: [1, T_audio] 重建音檔
    """
    with torch.no_grad():
        codes = codes.to(device)
        # 取得 codebook embeddings
        quantizer = model.feature_extractor.encodec.quantizer
        # 從 codes 還原 quantized features
        # quantizer.vq.layers[0].codebook: [K, D]
        codebook = quantizer.vq.layers[0].codebook  # [K, D]

        if codes.dim() == 3:
            # codes: [n_q, B, T] → 取第一層
            token_ids = codes[0, 0]  # [T]
        elif codes.dim() == 2:
            token_ids = codes[0]
        else:
            token_ids = codes

        # lookup
        features = codebook[token_ids]  # [T, D]
        features = features.T.unsqueeze(0)  # [1, D, T]

        # Decode
        bandwidth_id = torch.tensor([0], device=device)
        audio = model.decode(features, bandwidth_id=bandwidth_id)
        if audio.dim() == 3:
            audio = audio.squeeze(0)  # [1, T_audio]
        return audio


def run_masking_experiment(model, audio_path, output_dir, sample_id,
                          target_token=453, replace_token=1145, device='cuda:0'):
    """對一段音檔執行 token 遮蔽實驗

    Args:
        model: WavTokenizer 模型
        audio_path: 原始音檔路徑
        output_dir: 輸出目錄
        sample_id: 樣本編號
        target_token: 要遮蔽的 token ID
        replace_token: 替代 token ID
        device: 計算裝置
    """
    sr = 24000

    # 載入音檔
    waveform, orig_sr = torchaudio.load(str(audio_path))
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform[:1]  # mono
    waveform = waveform.unsqueeze(0)  # [1, 1, T]

    # Encode
    codes = encode_audio(model, waveform, device)
    # codes: [n_q, B, T_frames]
    original_codes = codes.clone()

    n_q, B, T = codes.shape
    layer0_codes = codes[0, 0]  # [T]
    total_frames = T
    n_target = (layer0_codes == target_token).sum().item()
    pct_target = 100.0 * n_target / total_frames

    print(f"\n  Sample {sample_id}: {Path(audio_path).name}")
    print(f"    Frames: {total_frames}, Token {target_token} count: {n_target} ({pct_target:.1f}%)")

    # (a) 原始重建
    audio_original = decode_from_codes(model, original_codes, device)
    torchaudio.save(
        str(output_dir / f'sample{sample_id}_original.wav'),
        audio_original.cpu(), sr
    )

    # (b) 替換為第二熱門 token
    codes_replace_2nd = original_codes.clone()
    mask = codes_replace_2nd[0, 0] == target_token
    codes_replace_2nd[0, 0, mask] = replace_token
    audio_replace_2nd = decode_from_codes(model, codes_replace_2nd, device)
    torchaudio.save(
        str(output_dir / f'sample{sample_id}_replace_T{replace_token}.wav'),
        audio_replace_2nd.cpu(), sr
    )

    # (c) 替換為隨機 non-target token
    codes_replace_rand = original_codes.clone()
    mask = codes_replace_rand[0, 0] == target_token
    n_masked = mask.sum().item()
    if n_masked > 0:
        # 從非 target 的 token 中隨機選
        non_target = layer0_codes[layer0_codes != target_token]
        if len(non_target) > 0:
            rand_tokens = non_target[torch.randint(0, len(non_target), (n_masked,))]
        else:
            rand_tokens = torch.randint(0, 4096, (n_masked,))
        codes_replace_rand[0, 0, mask] = rand_tokens.to(device)
    audio_replace_rand = decode_from_codes(model, codes_replace_rand, device)
    torchaudio.save(
        str(output_dir / f'sample{sample_id}_replace_random.wav'),
        audio_replace_rand.cpu(), sr
    )

    # (d) 靜音版：Token 453 的位置直接 zero-out embeddings
    # (用全 0 embedding 取代，模擬完全移除)

    # 也儲存原始（未經 WavTokenizer 的）音檔作為參考
    torchaudio.save(
        str(output_dir / f'sample{sample_id}_raw_input.wav'),
        waveform.squeeze(0).cpu(), sr
    )

    return {
        'sample_id': sample_id,
        'audio_path': str(audio_path),
        'total_frames': total_frames,
        'target_token_count': n_target,
        'target_token_pct': pct_target,
    }


def main():
    """主函式：執行 Token 453 遮蔽聽感實驗"""
    device = 'cuda:0'
    target_token = 453
    replace_token = 1145  # 第二熱門

    output_dir = Path(__file__).parent / 'token_masking_listen'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Token 453 遮蔽聽感實驗")
    print(f"  Target: T{target_token} (val 佔比 18.43%, 第一名)")
    print(f"  Replace with: T{replace_token} (第二名, 1.11%)")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # 載入 WavTokenizer
    model = load_wavtokenizer(device)

    # 從 val cache 選幾段音檔
    from exp_1201.config import VAL_CACHE
    val_data = torch.load(str(VAL_CACHE), weights_only=False)

    # 選 Token 453 佔比高的樣本（效果最明顯）
    print("\n搜尋 Token 453 佔比高的樣本...")
    token_stats = []
    for i, s in enumerate(val_data):
        codes = s['clean_tokens']
        n453 = (codes == target_token).sum().item()
        pct = 100.0 * n453 / len(codes)
        token_stats.append((i, pct, n453, len(codes)))

    # 排序: 選佔比最高的 3 個 + 佔比中等的 1 個 + 佔比低的 1 個
    token_stats.sort(key=lambda x: -x[1])

    selected = []
    # Top 3 (高佔比)
    selected.extend(token_stats[:3])
    # 中等 (約 18% 附近)
    mid_idx = len(token_stats) // 2
    selected.append(token_stats[mid_idx])
    # 低佔比但非 0
    for ts in reversed(token_stats):
        if ts[2] > 0:  # 至少有 1 個 T453
            selected.append(ts)
            break

    print(f"\n選定 {len(selected)} 個樣本:")
    for idx, pct, cnt, total in selected:
        print(f"  val[{idx}]: T453={cnt}/{total} ({pct:.1f}%)")

    # 執行實驗
    results = []
    from exp_1212.data_aligned import AlignedNoisyCleanPairDataset
    for rank, (idx, pct, cnt, total) in enumerate(selected):
        s = val_data[idx]
        # 找到 clean audio path
        clean_path = s.get('clean_path', '')
        if not Path(clean_path).is_absolute():
            # 解析相對路徑
            base_dirs = [
                Path("/home/sbplab/ruizi/WavTokenize/data"),
                Path("/home/sbplab/ruizi/c_code/data"),
                Path("/home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/data_with_distances"),
            ]
            resolved = None
            for bd in base_dirs:
                candidate = bd / clean_path.lstrip('./')
                if candidate.exists():
                    resolved = candidate
                    break
                # 嘗試直接用檔名
                fname = Path(clean_path).name
                for sub in ['clean/box2', 'clean']:
                    c2 = bd / sub / fname
                    if c2.exists():
                        resolved = c2
                        break
                if resolved:
                    break
            if resolved:
                clean_path = resolved
            else:
                print(f"  ⚠️ 找不到音檔: {clean_path}, 跳過")
                continue

        r = run_masking_experiment(
            model, clean_path, output_dir, rank + 1,
            target_token=target_token, replace_token=replace_token,
            device=device,
        )
        results.append(r)

    # 儲存結果
    import json
    summary = {
        'experiment': 'token_masking_listen',
        'date': DATE_STR,
        'target_token': target_token,
        'target_token_val_pct': 18.43,
        'replace_token': replace_token,
        'description': '遮蔽 Teacher Top-1 Token 453，觀察對音檔品質的影響',
        'outputs_per_sample': [
            'sampleN_raw_input.wav - 原始音檔 (未經 codec)',
            'sampleN_original.wav - WavTokenizer 正常重建',
            f'sampleN_replace_T{replace_token}.wav - T453→T{replace_token} (第2名)',
            'sampleN_replace_random.wav - T453→隨機其他 token',
        ],
        'results': results,
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("完成！")
    print(f"輸出目錄: {output_dir}")
    print("每個樣本有 4 個音檔可比較:")
    print("  1. raw_input - 原始音檔")
    print("  2. original - WavTokenizer 正常 encode→decode")
    print(f"  3. replace_T{replace_token} - T453 全部替換為 T{replace_token}")
    print("  4. replace_random - T453 替換為隨機 token")
    print("=" * 60)


if __name__ == '__main__':
    main()
