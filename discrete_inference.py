#!/usr/bin/env python3
"""
離散 Token 降噪推理腳本
使用訓練好的 Transformer 模型進行 token 序列降噪推理
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import logging
import torchaudio

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder.pretrained import WavTokenizer

# 使用官方推薦的 convert_audio 函數
def convert_audio(wav, from_rate, to_rate, to_channels):
    """Convert audio to target format"""
    if from_rate != to_rate:
        wav = torchaudio.transforms.Resample(from_rate, to_rate)(wav)
    if wav.size(0) != to_channels:
        if to_channels == 1:
            wav = wav.mean(0, keepdim=True)
        elif to_channels == 2:
            wav = wav.expand(2, -1)
    return wav
from discrete_token_denoising import TokenToTokenTransformer, set_seed
from encoder.utils import save_audio

class TokenDenoiser:
    """Token 降噪推理器"""
    
    def __init__(self, model_path, config_path, wavtokenizer_path, device='cuda'):
        """
        初始化推理器
        
        Args:
            model_path: 訓練好的 Transformer 模型路徑
            config_path: WavTokenizer 配置文件路徑
            wavtokenizer_path: WavTokenizer 預訓練模型路徑
            device: 計算設備
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 載入 WavTokenizer
        self.wavtokenizer = WavTokenizer.from_pretrained0802(config_path, wavtokenizer_path)
        self.wavtokenizer = self.wavtokenizer.to(self.device)
        self.wavtokenizer.eval()
        
        # 載入 Transformer 模型
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 從檢查點讀取模型配置（如果可用）
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            logging.info(f"從檢查點載入模型配置: {config}")
            
            # 保存模型配置以供後續使用
            self.vocab_size = config['vocab_size']
            self.max_length = config['max_length']
            
            self.model = TokenToTokenTransformer(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_encoder_layers=config['num_encoder_layers'],
                num_decoder_layers=config['num_decoder_layers'],
                dim_feedforward=config['dim_feedforward'],
                max_length=config['max_length'],
                dropout=config['dropout']
            ).to(self.device)
            logging.info(f"從檢查點載入模型配置: {config}")
        else:
            # 使用預設配置（向下兼容）
            logging.warning("檢查點中沒有模型配置，使用預設參數")
            self.model = TokenToTokenTransformer(
                vocab_size=4098,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=2048,
                max_length=512,
                dropout=0.1
            ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 特殊 tokens
        self.pad_token = 0
        self.sos_token = 4096
        self.eos_token = 4097
        
        logging.info(f"模型載入完成，使用設備: {self.device}")
    
    def audio_to_tokens(self, audio_path, sample_rate=24000):
        """將音頻轉換為 token 序列"""
        # 如果輸入是路徑，先載入音頻
        if isinstance(audio_path, str):
            wav, sr = torchaudio.load(audio_path)
            audio = convert_audio(wav, sr, 24000, 1)
        else:
            audio = audio_path
            
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # 添加 batch 維度
        
        audio = audio.to(self.device)
        
        with torch.no_grad():
            # 使用官方推薦的方法：encode 得到 discrete codes  
            bandwidth_id = torch.tensor([0], device=self.device)  # 最高質量
            discrete_code, _ = self.wavtokenizer.encode(audio, bandwidth_id=bandwidth_id)
            # discrete_code 的格式是 [n_q, batch, time]，我們只使用第一個量化層
            tokens = discrete_code[0, 0]  # [time]
        
        return tokens.cpu()
    
    def tokens_to_audio(self, tokens, sample_rate=24000):
        """將 token 序列轉換回音頻"""
        if tokens.dim() == 1:
            # 需要轉換為 [n_q, batch, time] 格式，這裡我們只使用一個量化層
            tokens = tokens.unsqueeze(0).unsqueeze(0)  # [1, 1, time]
        elif tokens.dim() == 2 and tokens.size(0) == 1:
            tokens = tokens.unsqueeze(0)  # [1, 1, time]
        
        # 確保數據類型為 long (整數類型)
        tokens = tokens.long().to(self.device)
        
        print(f"解碼前 tokens 形狀: {tokens.shape}, 類型: {tokens.dtype}, 設備: {tokens.device}")
        
        with torch.no_grad():
            # 先將 codes 轉換為 features
            features = self.wavtokenizer.codes_to_features(tokens)
            print(f"特徵形狀: {features.shape}, 類型: {features.dtype}")
            
            # 然後解碼 features 為音頻
            bandwidth_id = torch.tensor([0], device=self.device)  # 最高質量解碼
            audio = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
        
        return audio.squeeze(0).cpu()  # [1, length]
    
    def denoise_tokens(self, noisy_tokens, max_length=512, beam_size=1):
        """
        使用 Transformer 對 token 序列進行降噪
        
        Args:
            noisy_tokens: [seq_len] - 輸入的帶噪 token 序列
            max_length: int - 最大生成長度
            beam_size: int - beam search 大小 (1 表示貪婪搜尋)
        
        Returns:
            clean_tokens: [seq_len] - 降噪後的 token 序列
        """
        self.model.eval()
        
        # 準備輸入序列
        noisy_tokens = noisy_tokens.to(self.device).long()  # 確保在正確設備上且為 long 類型
        if len(noisy_tokens.flatten()) > max_length - 1:
            noisy_tokens = noisy_tokens.flatten()[:max_length - 1]
        else:
            noisy_tokens = noisy_tokens.flatten()
        
        # 檢查 token 範圍和模型配置的一致性
        print(f"input_seq min: {noisy_tokens.min().item()}, max: {noisy_tokens.max().item()}")
        print(f"vocab_size: {self.vocab_size}")
        
        if noisy_tokens.max().item() >= self.vocab_size:
            print(f"警告: token 值超出詞彙表範圍 ({noisy_tokens.max().item()} >= {self.vocab_size})")
            # 裁剪超出範圍的token
            noisy_tokens = torch.clamp(noisy_tokens, 0, self.vocab_size - 1)

        input_seq = torch.cat([noisy_tokens, torch.tensor([self.eos_token], device=self.device, dtype=torch.long)])
        
        # Padding
        if len(input_seq) < max_length:
            input_seq = torch.cat([input_seq, torch.zeros(max_length - len(input_seq), dtype=torch.long, device=self.device)])
        
        input_seq = input_seq.unsqueeze(0).to(self.device)  # [1, seq_len]
        
        with torch.no_grad():
            if beam_size == 1:
                # 貪婪搜尋
                clean_tokens = self._greedy_decode(input_seq, max_length)
            else:
                # Beam search
                clean_tokens = self._beam_search_decode(input_seq, max_length, beam_size)
        
        return clean_tokens
    
    def _greedy_decode(self, input_seq, max_length):
        """貪婪解碼"""
        batch_size = input_seq.size(0)
        
        # 初始化解碼器輸入
        decoder_input = torch.tensor([[self.sos_token]], device=self.device)  # [1, 1]
        
        generated = []
        
        for _ in range(max_length - 1):
            # 準備解碼器輸入 (需要 padding 到固定長度)
            decoder_input_padded = torch.cat([
                decoder_input,
                torch.zeros(batch_size, max_length - decoder_input.size(1), 
                           dtype=torch.long, device=self.device)
            ], dim=1)
            
            # 前向傳播
            logits = self.model(input_seq, decoder_input_padded)
            
            # 取當前時間步的輸出
            next_token_logits = logits[:, decoder_input.size(1) - 1, :]  # [batch_size, vocab_size]
            next_token = torch.argmax(next_token_logits, dim=-1)  # [batch_size]
            
            # 如果生成了 EOS token，停止生成
            if next_token.item() == self.eos_token:
                break
            
            generated.append(next_token.item())
            
            # 更新解碼器輸入
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
        
        return torch.tensor(generated, dtype=torch.long, device=self.device)
    
    def _beam_search_decode(self, input_seq, max_length, beam_size):
        """Beam search 解碼 (簡化版本)"""
        # 為了簡化，這裡實現貪婪搜尋
        # 完整的 beam search 實現較為複雜，可以後續擴展
        return self._greedy_decode(input_seq, max_length)
    
    def denoise_audio(self, input_audio_path, output_audio_path, 
                     compare_with_original=True, save_tokens=False):
        """
        完整的音頻降噪流程
        
        Args:
            input_audio_path: 輸入音頻路徑
            output_audio_path: 輸出音頻路徑
            compare_with_original: 是否保存原始音頻用於比較
            save_tokens: 是否保存 token 序列
        """
        logging.info(f"處理音頻: {input_audio_path}")
        
        # Step 1: 音頻 -> Token
        noisy_tokens = self.audio_to_tokens(input_audio_path)
        logging.info(f"原始 token 序列長度: {noisy_tokens.shape}")
        
        # Step 2: Token 降噪 (使用較小的 max_length 以適應訓練時的配置)
        clean_tokens = self.denoise_tokens(noisy_tokens, max_length=128)
        logging.info(f"降噪後 token 序列長度: {clean_tokens.shape}")
        
        # Step 3: Token -> 音頻
        clean_audio = self.tokens_to_audio(clean_tokens)
        
        # 確保音頻有正確的形狀 [channels, samples]
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)  # 添加通道維度
        print(f"最終音頻形狀: {clean_audio.shape}")
        
        # 保存降噪後的音頻
        save_audio(clean_audio, output_audio_path, sample_rate=24000, rescale=True)
        logging.info(f"降噪音頻已保存到: {output_audio_path}")
        
        # 可選：保存原始音頻用於比較
        if compare_with_original:
            original_output_path = output_audio_path.replace('.wav', '_original.wav')
            original_audio = self.tokens_to_audio(noisy_tokens)
            save_audio(original_audio, original_output_path, sample_rate=24000, rescale=True)
            logging.info(f"重建原始音頻已保存到: {original_output_path}")
        
        # 可選：保存 token 序列
        if save_tokens:
            tokens_dir = os.path.dirname(output_audio_path)
            base_name = os.path.splitext(os.path.basename(output_audio_path))[0]
            
            np.save(os.path.join(tokens_dir, f"{base_name}_noisy_tokens.npy"), 
                   noisy_tokens.cpu().numpy())
            np.save(os.path.join(tokens_dir, f"{base_name}_clean_tokens.npy"), 
                   clean_tokens.cpu().numpy())
            
            logging.info(f"Token 序列已保存到: {tokens_dir}")

def main():
    parser = argparse.ArgumentParser(description='離散 Token 降噪推理')
    
    # 模型參數
    parser.add_argument('--model_path', type=str, required=True,
                        help='訓練好的 Transformer 模型路徑')
    parser.add_argument('--config', type=str, 
                        default='config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                        help='WavTokenizer 配置文件路徑')
    parser.add_argument('--wavtokenizer_path', type=str,
                        default='models/wavtokenizer_large_speech_320_24k.ckpt',
                        help='WavTokenizer 預訓練模型路徑')
    
    # 輸入輸出參數
    parser.add_argument('--input_audio', type=str, help='單個輸入音頻文件路徑')
    parser.add_argument('--input_dir', type=str, help='輸入音頻目錄路徑')
    parser.add_argument('--output_dir', type=str, required=True, help='輸出目錄路徑')
    
    # 推理參數
    parser.add_argument('--beam_size', type=int, default=1, help='Beam search 大小')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列長度')
    parser.add_argument('--save_tokens', action='store_true', help='是否保存 token 序列')
    parser.add_argument('--compare_with_original', action='store_true', default=True,
                        help='是否保存重建的原始音頻用於比較')
    
    args = parser.parse_args()
    
    # 設定隨機種子
    set_seed(42)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'inference.log')),
            logging.StreamHandler()
        ]
    )
    
    # 初始化降噪器
    denoiser = TokenDenoiser(
        model_path=args.model_path,
        config_path=args.config,
        wavtokenizer_path=args.wavtokenizer_path
    )
    
    # 處理音頻文件
    if args.input_audio:
        # 單個文件
        output_path = os.path.join(args.output_dir, 
                                  f"denoised_{os.path.basename(args.input_audio)}")
        denoiser.denoise_audio(
            args.input_audio, output_path,
            compare_with_original=False,  # 關閉原始音頻保存功能
            save_tokens=args.save_tokens
        )
    
    elif args.input_dir:
        # 批次處理目錄
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend([
                f for f in os.listdir(args.input_dir) 
                if f.lower().endswith(ext)
            ])
        
        if not audio_files:
            logging.error(f"在目錄 {args.input_dir} 中未找到音頻文件")
            return
        
        logging.info(f"找到 {len(audio_files)} 個音頻文件")
        
        for audio_file in tqdm(audio_files, desc="處理音頻文件"):
            input_path = os.path.join(args.input_dir, audio_file)
            output_path = os.path.join(args.output_dir, f"denoised_{audio_file}")
            
            # 確保輸出文件是 .wav 格式
            output_path = os.path.splitext(output_path)[0] + '.wav'
            
            try:
                denoiser.denoise_audio(
                    input_path, output_path,
                    compare_with_original=args.compare_with_original,
                    save_tokens=args.save_tokens
                )
            except Exception as e:
                logging.error(f"處理文件 {audio_file} 時出錯: {e}")
                continue
    
    else:
        logging.error("請指定 --input_audio 或 --input_dir 參數")
        return
    
    logging.info("推理完成！")

if __name__ == "__main__":
    main()
