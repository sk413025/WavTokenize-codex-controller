"""
簡單腳本：批次比較 ./test 資料夾下所有音檔的 WavTokenizer token 序列
20250904 by Copilot

輸出：
- 每個音檔的 token 序列 shape、獨特 token 種類、token 分布
- token 序列儲存到 ./test/out/{音檔名}_tokens.npy
"""
import os
import torch
import numpy as np
import torchaudio
from ttt2 import WavTokenizer  # 根據你的模型定義修改
from encoder.utils import convert_audio

config_path = "/home/sbplab/ruizi/c_code/config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"
        
TEST_DIR = './test'
OUT_DIR = './test/out'
os.makedirs(OUT_DIR, exist_ok=True)

# 載入模型
model = WavTokenizer.from_pretrained0802(config_path, model_path)
model.eval()

for fname in os.listdir(TEST_DIR):
    if not fname.endswith('.wav'):
        continue
    wav_path = os.path.join(TEST_DIR, fname)
    print(f"\n=== {fname} ===")
    # 載入音檔
    wav, sr = torchaudio.load(wav_path)
    wav = convert_audio(wav, sr, 24000, 1)  # 轉換為 24kHz, 單聲道
    with torch.no_grad():
        features, discrete_code = model.encode_infer(wav, bandwidth_id=0)
    print("Token shape:", discrete_code.shape)
    print("Token 序列:", discrete_code)
    unique_tokens = torch.unique(discrete_code)
    print("獨特 token 種類:", unique_tokens)
    bincount = torch.bincount(discrete_code.flatten())
    print("Token 分布:", bincount)
    # 儲存為 .npy
    out_path_npy = os.path.join(OUT_DIR, f"{fname}_tokens.npy")
    np.save(out_path_npy, discrete_code.cpu().numpy())
    print(f"已儲存 token 序列到 {out_path_npy}")
    # 儲存為 .txt（逗號分隔）
    out_path_txt = os.path.join(OUT_DIR, f"{fname}_tokens.txt")
    np.savetxt(out_path_txt, discrete_code.cpu().numpy().reshape(-1), fmt='%d', delimiter=',')
    print(f"已儲存 token 序列到 {out_path_txt}")
