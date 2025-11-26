
import os
import sys
from pathlib import Path
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 添加必要的路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import torch.nn as nn
import traceback
import datetime


def process_audio(audio_path, target_sr=24000, normalize=True, dynamic_duration=False):
    """音頻處理函數，負責讀取、轉換採樣率和正規化音頻
    dynamic_duration參數已被停用，保留參數僅為向後兼容"""
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, target_sr, 1)  # [1, T]

    # 不再進行動態裁剪，忽略dynamic_duration參數

    if normalize:
        wav = wav / (wav.abs().max() + 1e-8)
    return wav  # 保持原始長度 [1, T]


class AudioDataset(Dataset):
    """
    自定義音頻資料集類別，可處理多個輸入目錄。
    """
    def __init__(self, input_dirs, target_dir, max_files_per_dir=None, handle_speed_diff=False, speech_rate_threshold=10, max_sentences_per_speaker=100, allowed_speakers=None):
        self.input_dirs = input_dirs if isinstance(input_dirs, list) else [input_dirs]
        self.target_dir = target_dir
        self.paired_files = []
        self.handle_speed_diff = False  # 取消處理語速差異
        self.speech_rate_threshold = speech_rate_threshold  # 語速閾值，暫時保留但不使用
        self.max_sentences_per_speaker = max_sentences_per_speaker  # 每位與者最大句子數
        self.allowed_speakers = allowed_speakers  # 允許的語者列表，None表示允許所有語者
        
        # 檢查目錄是否存在
        for input_dir in self.input_dirs:
            if not os.path.exists(input_dir):
                print(f"Error: Input directory not found: {input_dir}")
                return
        if not os.path.exists(target_dir):
            print(f"Error: Target directory not found: {target_dir}")
            return

        print("\nSearching for files in:")
        for input_dir in self.input_dirs:
            print(f"Input directory: {input_dir}")
        print(f"Target directory: {target_dir}")
        
        # 統計語速資訊，用於後續分析
        speaker_rates = {}
        
        # 收集所有輸入文件
        for input_dir in self.input_dirs:
            print(f"\nProcessing directory: {input_dir}")
            
            # 從目錄路徑獲取材質名稱（例如：box, mac, papercup, plastic）
            # 取得目錄名稱作為材質
            material = os.path.basename(input_dir)
            print(f"Directory material type: {material}")
            
            # 獲取目錄中的所有WAV文件
            input_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
            
            # 如果設定了max_files_per_dir，則限制每個目錄的文件數量
            if max_files_per_dir is not None:
                if len(input_files) > max_files_per_dir:
                    print(f"Limiting files from {input_dir} to {max_files_per_dir} (from {len(input_files)} total files)")
                    # 隨機選擇指定數量的文件
                    input_files = random.sample(input_files, max_files_per_dir)
            
            for input_file in input_files:
                if input_file.endswith('.wav'):
                    try:
                        # 解析文件名
                        parts = input_file.split('_')
                        if len(parts) >= 2:
                            # 使用目錄名稱作為材質，而不是從文件名解析
                            # material = parts[0]  # 舊的方法，從文件名取得材質
                            speaker = parts[1] if len(parts) > 1 else "unknown"  # 如 boy1, girl6 等
                            number = parts[-1]     # 136.wav or 137.wav
                            
                            # 語者過濾：如果指定了allowed_speakers，只處理允許的語者
                            if self.allowed_speakers is not None and speaker not in self.allowed_speakers:
                                continue  # 跳過不在允許列表中的語者
                            
                            # 提取內容ID (從文件名中的數字部分，去除.wav)
                            content_id = number.split('.')[0] if '.' in number else number
                            
                            print(f"\nTrying to match: {input_file}")
                            print(f"Material: {material}, Speaker: {speaker}, Number: {number}, Content ID: {content_id}")
                            
                            # 構建target文件名模式
                            target_patterns = [
                                f"nor_{speaker}_LDV_{number}",       # 標準格式
                                f"nor_{speaker}_{number}",           # 替代格式
                                f"nor_{speaker}_clean_{number[:-4]}" # 新格式: nor_boy1_clean_137
                            ]
                            
                            # 尋找匹配的target文件
                            target_file = None
                            for pattern in target_patterns:
                                matching_files = [
                                    f for f in os.listdir(target_dir)
                                    if f.startswith(pattern)
                                ]
                                if matching_files:
                                    target_file = matching_files[0]  # 使用第一個匹配的文件
                                    break
                            
                            if target_file:
                                print(f"Found matching target: {target_file}")
                                  # 不再計算語速
                                speech_rate = None
                                
                                # 添加配對檔案，移除語速相關信息
                                self.paired_files.append({
                                    'input_dir': input_dir,
                                    'input': input_file,
                                    'target': target_file,
                                    'material': material,  # 使用目錄名稱作為材質
                                    'speaker': speaker,
                                    'content_id': content_id,  # 添加內容ID
                                    'speech_rate': None  # 語速信息設為None
                                })
                            else:
                                print(f"No match found for {input_file}")
                    
                    except Exception as e:
                        print(f"Error processing {input_file}: {str(e)}")
                        continue
          # 已取消語速處理，不再進行統計和計算
        # 為了保持代碼兼容性，為每個樣本添加默認的speaker_type
        for pair in self.paired_files:
            pair['speaker_type'] = "normal"  # 所有樣本均視為正常語速
        
        # 限制每位與者的句子數量
        if self.max_sentences_per_speaker is not None:
            print(f"\n限制每位與者最多使用 {self.max_sentences_per_speaker} 句話")
            
            # 按說話者分組
            speaker_groups = {}
            for pair in self.paired_files:
                speaker = pair['speaker']
                material = pair['material']
                key = f"{speaker}_{material}"  # 使用說話者+材質作為唯一標識
                
                if key not in speaker_groups:
                    speaker_groups[key] = []
                speaker_groups[key].append(pair)
            
            # 限制每個說話者+材質組合的句子數量
            limited_paired_files = []
            for key, pairs in speaker_groups.items():
                if len(pairs) > self.max_sentences_per_speaker:
                    print(f"限制 {key} 從 {len(pairs)} 句話減少到 {self.max_sentences_per_speaker} 句話")
                    # 按內容ID排序，選擇編號最小的句子（相當於選擇1-100編號的句子）
                    pairs_sorted = sorted(pairs, key=lambda x: int(x['content_id']))
                    selected_pairs = pairs_sorted[:self.max_sentences_per_speaker]
                    print(f"選擇內容ID範圍：{selected_pairs[0]['content_id']} 到 {selected_pairs[-1]['content_id']}")
                    limited_paired_files.extend(selected_pairs)
                else:
                    print(f"保留 {key} 的全部 {len(pairs)} 句話")
                    limited_paired_files.extend(pairs)
            
            self.paired_files = limited_paired_files
            print(f"限制後的總文件數：{len(self.paired_files)}")
        
        # 打印配對結果摘要
        print(f"\nTotal paired files: {len(self.paired_files)}")
        if self.paired_files:
            print("\nFiles paired by material:")
            materials = {}
            for pair in self.paired_files:
                mat = pair['material']
                if mat not in materials:
                    materials[mat] = []
                  # 添加語速類型信息到輸出
                speed_info = f", Speed: {pair.get('speaker_type', 'unknown')}"
                materials[mat].append(
                    f"{pair['input']} -> {pair['target']} (ID: {pair['content_id']}, Speaker: {pair['speaker']}{speed_info})"
                )
            
            for mat, pairs in materials.items():
                print(f"\n{mat.upper()} ({len(pairs)} files):")
                for pair in pairs:
                    print(f"  {pair}")
        else:
            print("\nWARNING: No valid file pairs found!")
            print("Please check:")
            print("1. Input and target directory paths")
            print("2. File naming conventions")
            print("3. File extensions (.wav)")
            
    def __len__(self):
        return len(self.paired_files)
        
    def __getitem__(self, idx):
        pair = self.paired_files[idx]
        input_path = os.path.join(pair['input_dir'], pair['input'])
        target_path = os.path.join(self.target_dir, pair['target'])

        # 取消語速相關的動態處理
        dynamic_mode = False

        # ⭐ 重要修正：兩者都不正規化，保持原始音量（與 debug_single_sample.py 一致）
        # 原因：WavTokenizer 對不同音量會產生不同的 tokens，必須保持一致性
        input_wav = process_audio(input_path, normalize=False, dynamic_duration=False)
        target_wav = process_audio(target_path, normalize=False, dynamic_duration=False)

        # 返回音頻數據和內容ID
        return input_wav, target_wav, pair['content_id']
