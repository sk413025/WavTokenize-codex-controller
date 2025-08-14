#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WavTokenizer 能力驗證腳本

此腳本用於驗證 WavTokenizer 模型的編碼-解碼能力，檢查還原音頻的質量，
並提供基本的評估指標。

實驗編號: EXP_Verify
日期: 2025-08-03
作者: GitHub Copilot
"""

import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import re
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer
from datetime import datetime

# 評估指標依賴
try:
    from pesq import pesq
    from pystoi import stoi
    import librosa
    evaluation_available = True
except ImportError:
    print("警告: 無法導入評估指標庫 (pesq/pystoi/librosa)，將跳過評估部分")
    evaluation_available = False

# 配置信息
CONFIG_PATH = "/home/sbplab/ruizi/WavTokenize/config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
MODEL_PATH = "/home/sbplab/ruizi/WavTokenize/wavtokenizer_large_speech_320_24k.ckpt"
RESULTS_DIR = "/home/sbplab/ruizi/WavTokenize/results/verify"
DATE = datetime.now().strftime("%Y%m%d")

# 確保結果目錄存在
os.makedirs(RESULTS_DIR, exist_ok=True)

def verify_wavtokenizer(audio_paths):
    """
    驗證 WavTokenizer 的編碼-解碼能力
    
    Args:
        audio_paths: 待測試音頻文件路徑列表
    
    Returns:
        dict: 包含評估結果的字典
    """
    # 加載模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    wavtokenizer = WavTokenizer.from_pretrained0802(CONFIG_PATH, MODEL_PATH)
    wavtokenizer = wavtokenizer.to(device)
    print("模型加載成功")
    
    results = []
    
    for audio_path in audio_paths:
        print(f"\n處理音頻: {os.path.basename(audio_path)}")
        
        # 加載音頻
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, 24000, 1)
        wav = wav.to(device)
        
        # 步驟 1: 使用 Part2 將音頻編碼為離散 token
        bandwidth_id = torch.zeros(1, dtype=torch.long, device=device)
        features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        
        # 輸出離散編碼的形狀
        print(f"離散編碼形狀: {discrete_code.shape}")
        
        # 步驟 2: 使用 Part3 將離散 token 解碼回音頻
        # 從離散編碼轉換回特徵
        features_from_codes = wavtokenizer.codes_to_features(discrete_code)
        
        # 檢查兩種方式獲得的特徵是否一致
        features_diff = torch.abs(features - features_from_codes).mean().item()
        print(f"直接特徵與從編碼重建特徵的平均差異: {features_diff:.6f}")
        
        # 解碼為音頻
        audio_out = wavtokenizer.decode(features_from_codes, bandwidth_id=bandwidth_id)
        
        # 儲存還原後的音頻
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(RESULTS_DIR, f"{base_name}_restored_{DATE}.wav")
        torchaudio.save(output_path, audio_out.cpu(), 24000)
        
        # 輸出原始音頻和還原音頻的形狀，以確認
        print(f"原始音頻形狀: {wav.shape}")
        print(f"還原音頻形狀: {audio_out.shape}")
        
        # 步驟 3: 評估還原效果
        result = {
            "音頻文件": audio_path,
            "還原文件": output_path,
            "離散編碼形狀": discrete_code.shape,
            "特徵差異": features_diff,
        }
        
        if evaluation_available:
            # 確保長度一致
            min_len = min(wav.shape[1], audio_out.shape[1])
            wav_eval = wav[:, :min_len].cpu().numpy().flatten()
            audio_out_eval = audio_out[:, :min_len].cpu().numpy().flatten()
            
            # 計算 SNR
            def calculate_snr(original, reconstructed):
                noise = original - reconstructed
                snr = 10 * np.log10(np.sum(original**2) / (np.sum(noise**2) + 1e-10))
                return snr
            
            snr = calculate_snr(wav_eval, audio_out_eval)
            print(f"SNR: {snr:.4f} dB")
            result["SNR"] = snr
            
            # 計算 PESQ (需要降採樣至 16kHz)
            try:
                # 确保音频长度至少为0.5秒（PESQ的最小要求）
                min_samples = int(16000 * 0.5)  # 16kHz采样率下的最小样本数
                
                wav_16k = librosa.resample(wav_eval, orig_sr=24000, target_sr=16000)
                audio_out_16k = librosa.resample(audio_out_eval, orig_sr=24000, target_sr=16000)
                
                # 如果音频太短，将其填充至最小长度
                if len(wav_16k) < min_samples:
                    wav_16k = np.pad(wav_16k, (0, min_samples - len(wav_16k)))
                if len(audio_out_16k) < min_samples:
                    audio_out_16k = np.pad(audio_out_16k, (0, min_samples - len(audio_out_16k)))
                
                # 确保两个音频长度相同
                min_len = min(len(wav_16k), len(audio_out_16k))
                wav_16k = wav_16k[:min_len]
                audio_out_16k = audio_out_16k[:min_len]
                
                # 确保音频在正常范围内
                if np.max(np.abs(wav_16k)) > 0 and np.max(np.abs(audio_out_16k)) > 0:
                    pesq_score = pesq(16000, wav_16k, audio_out_16k, 'wb')
                    print(f"PESQ: {pesq_score:.4f}")
                    result["PESQ"] = pesq_score
                else:
                    print("PESQ計算錯誤: 音频信号太弱")
                    result["PESQ"] = None
            except Exception as e:
                print(f"PESQ計算錯誤: {e}")
                result["PESQ"] = None
            
            # 計算 STOI
            try:
                # 确保已经进行了16kHz采样
                if 'wav_16k' not in locals() or 'audio_out_16k' not in locals():
                    wav_16k = librosa.resample(wav_eval, orig_sr=24000, target_sr=16000)
                    audio_out_16k = librosa.resample(audio_out_eval, orig_sr=24000, target_sr=16000)
                
                # 确保两个音频长度相同
                min_len = min(len(wav_16k), len(audio_out_16k))
                wav_16k = wav_16k[:min_len]
                audio_out_16k = audio_out_16k[:min_len]
                
                # STOI需要足够长的信号（建议至少为1秒）
                if min_len >= 16000:
                    stoi_score = stoi(wav_16k, audio_out_16k, 16000, extended=False)
                    print(f"STOI: {stoi_score:.4f}")
                    result["STOI"] = stoi_score
                else:
                    print("STOI計算錯誤: 音频太短")
                    result["STOI"] = None
            except Exception as e:
                print(f"STOI計算錯誤: {e}")
                result["STOI"] = None
        
        # 計算每秒 token 數
        audio_length_seconds = wav.shape[1] / 24000
        
        # 修正 token 數量計算 - 注意 discrete_code 的形狀是 [batch_size, n_q, seq_len]
        # 或 [batch_size, seq_len]，實際 token 數量應該取最後一維
        tokens = discrete_code.shape[-1]  # 取最後一維作為時間維度上的 token 數量
        tokens_per_second = tokens / audio_length_seconds
        
        print(f"音頻長度: {audio_length_seconds:.2f} 秒")
        print(f"總 token 數: {tokens}")
        print(f"每秒 token 數: {tokens_per_second:.2f}")
        result["每秒token數"] = tokens_per_second
        
        results.append(result)
        print(f"音頻 {os.path.basename(audio_path)} 處理完成，結果保存至 {output_path}")
    
    # 生成總體報告
    report = f"""# WavTokenizer 能力驗證報告
日期: {DATE}

## 概述
- 使用設備: {device}
- 模型路徑: {MODEL_PATH}
- 測試音頻數量: {len(audio_paths)}

## 詳細結果

| 音頻文件 | 每秒token數 | SNR (dB) | PESQ | STOI | 特徵差異 |
|---------|------------|----------|------|------|---------|
"""
    
    for r in results:
        base_name = os.path.basename(r["音頻文件"])
        snr_str = f"{r.get('SNR', 'N/A'):.4f}" if r.get('SNR') is not None else "N/A"
        pesq_str = f"{r.get('PESQ', 'N/A'):.4f}" if r.get('PESQ') is not None else "N/A"
        stoi_str = f"{r.get('STOI', 'N/A'):.4f}" if r.get('STOI') is not None else "N/A"
        
        report += f"| {base_name} | {r['每秒token數']:.2f} | {snr_str} | {pesq_str} | {stoi_str} | {r['特徵差異']:.6f} |\n"
    
    report_path = os.path.join(RESULTS_DIR, f"WavTokenizer_verification_report_{DATE}.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\n驗證報告已保存至: {report_path}")
    
    # 直接更新 REPORT.md 文件
    try:
        report_file = "/home/sbplab/ruizi/WavTokenize/REPORT.md"
        with open(report_file, 'r', encoding='utf-8') as f:
            report_content = f.read()
            
        # 創建新的報告條目
        experiment_id = f"EXP_Verify_{DATE}"
        new_entry = f"\n## {datetime.now().strftime('%Y-%m-%d')} WavTokenizer 能力驗證實驗 ({experiment_id})\n"
        new_entry += f"- **實驗描述**: 驗證 WavTokenizer 的編碼-解碼能力，測試了 {len(audio_paths)} 個音頻文件\n"
        new_entry += f"- **驗證結果**: [詳細報告]({os.path.relpath(report_path, os.path.dirname(report_file))})\n"
        
        # 添加平均指標
        avg_snr = np.mean([r.get('SNR', 0) for r in results if r.get('SNR') is not None])
        if not np.isnan(avg_snr):
            new_entry += f"- **平均 SNR**: {avg_snr:.4f} dB\n"
        
        # 查找插入位置 - 在第一個章節標題後
        match = re.search(r'^## ', report_content, re.MULTILINE)
        if match:
            insert_pos = match.start()
            updated_content = report_content[:insert_pos] + new_entry + report_content[insert_pos:]
        else:
            # 如果找不到章節標題，則附加到文件末尾
            updated_content = report_content + new_entry
        
        # 寫入更新後的報告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"已更新中央報告文件: {report_file}")
    except Exception as e:
        print(f"更新中央報告時發生錯誤: {e}")
    
    return results

if __name__ == "__main__":
    # 選擇不同類型的音頻進行測試
    audio_paths = [
        "/home/sbplab/ruizi/WavTokenize/1c/nor_boy1_clean_001.wav",  # 男孩清晰語音
        "/home/sbplab/ruizi/WavTokenize/1c/nor_girl1_clean_001.wav",  # 女孩清晰語音
        "/home/sbplab/ruizi/WavTokenize/1b/nor_boy1_box_LDV_001.wav"  # 男孩帶噪聲語音
    ]
    
    results = verify_wavtokenizer(audio_paths)
    print("\n驗證完成！")
