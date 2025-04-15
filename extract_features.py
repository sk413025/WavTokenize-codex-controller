#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特徵提取工具

此腳本用於從音頻文件中提取 WavTokenizer encoder 的特徵向量，
並將其保存為 .pt 或 .npy 文件，用於後續分析和訓練。
"""

import os
import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio


def process_audio(audio_path, target_sr=24000, normalize=True):
    """處理音頻文件，包括讀取、重採樣和正規化"""
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, target_sr, 1)  # [1, T]

    if normalize:
        wav = wav / (wav.abs().max() + 1e-8)
    return wav  # [1, T]


def extract_features(
    model, 
    audio_path, 
    output_path=None, 
    device="cpu", 
    output_format="pt"
):
    """從單個音頻文件提取特徵"""
    # 處理音頻
    wav = process_audio(audio_path)
    wav = wav.to(device)
    
    # 提取特徵
    with torch.no_grad():
        features = model.feature_extractor.encodec.encoder(wav)
    
    # 確定輸出路徑
    if output_path is None:
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.dirname(audio_path)
        output_path = os.path.join(output_dir, f"{basename}_encoder")
        if output_format == "pt":
            output_path += ".pt"
        else:
            output_path += ".npy"
    
    # 保存特徵
    if output_format == "pt":
        torch.save(features.cpu(), output_path)
    else:
        np.save(output_path, features.cpu().numpy())
    
    return output_path, features


def batch_extract_features(
    model,
    input_dir,
    output_dir=None,
    device="cpu",
    output_format="pt",
    extensions=[".wav"]
):
    """批次處理整個資料夾的音頻文件提取特徵"""
    # 查找所有音頻文件
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(Path(input_dir).rglob(f"*{ext}")))
    
    if not audio_files:
        print(f"在 {input_dir} 中未找到音頻文件")
        return
    
    print(f"找到 {len(audio_files)} 個音頻文件")
    
    # 確保輸出目錄存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 批次處理
    results = []
    for audio_file in tqdm(audio_files, desc="提取特徵"):
        relative_path = audio_file.relative_to(input_dir)
        
        # 確定輸出路徑
        if output_dir:
            output_subdir = os.path.dirname(os.path.join(output_dir, relative_path))
            os.makedirs(output_subdir, exist_ok=True)
            
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            if output_format == "pt":
                output_path = os.path.join(output_subdir, f"{basename}_encoder.pt")
            else:
                output_path = os.path.join(output_subdir, f"{basename}_encoder.npy")
        else:
            output_path = None
        
        try:
            saved_path, _ = extract_features(
                model, 
                str(audio_file), 
                output_path, 
                device, 
                output_format
            )
            results.append((str(audio_file), saved_path))
        except Exception as e:
            print(f"處理 {audio_file} 時出錯: {str(e)}")
    
    return results


def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="提取 WavTokenizer encoder 特徵向量")
    parser.add_argument("--input", type=str, required=True, help="輸入音頻文件或目錄")
    parser.add_argument("--output_dir", type=str, default=None, help="輸出目錄")
    parser.add_argument("--config_path", type=str, 
                      default="./config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml", 
                      help="WavTokenizer 配置文件路徑")
    parser.add_argument("--model_path", type=str, 
                      default="./wavtokenizer_large_speech_320_24k.ckpt", 
                      help="WavTokenizer 模型路徑")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="使用的設備")
    parser.add_argument("--format", type=str, choices=["pt", "npy"], default="pt", 
                      help="輸出格式：pt (PyTorch) 或 npy (NumPy)")
    
    args = parser.parse_args()
    
    # 載入模型
    print(f"使用設備: {args.device}")
    print(f"載入 WavTokenizer 模型...")
    
    device = torch.device(args.device)
    model = WavTokenizer.from_pretrained0802(args.config_path, args.model_path).to(device)
    model.eval()
    
    print("模型已載入")
    
    # 處理路徑
    input_path = args.input
    output_dir = args.output_dir
    
    # 提取特徵
    if os.path.isdir(input_path):
        print(f"批次處理目錄: {input_path}")
        results = batch_extract_features(
            model, 
            input_path, 
            output_dir, 
            device, 
            args.format
        )
        
        print(f"\n處理完成. 已提取 {len(results)} 個特徵文件.")
        print(f"特徵儲存在: {output_dir or '各音頻文件相同目錄'}")
    else:
        print(f"處理單個文件: {input_path}")
        output_path, _ = extract_features(
            model, 
            input_path, 
            output_dir, 
            device, 
            args.format
        )
        print(f"特徵已保存到: {output_path}")


if __name__ == "__main__":
    main()