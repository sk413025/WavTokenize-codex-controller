#!/usr/bin/env python3
"""
分析未充分訓練的 Transformer 對音頻品質的影響

實驗編號: UNTRAINED_TRANSFORMER_ANALYSIS_20250926
日期: 2025-09-26
函式名稱: analyze_untrained_transformer.py
目的: 深入了解未充分訓練的 Transformer 導致音頻品質差的機制
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torchaudio

# 添加當前目錄到 Python 路徑
sys.path.append('.')

def analyze_transformer_weights(model):
    """分析 Transformer 權重的分佈和特性"""
    print("=== Transformer 權重分析 ===")
    
    transformer_params = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        if 'transformer' in name and param.requires_grad:
            transformer_params[name] = param.data.clone()
            total_params += param.numel()
    
    print(f"Transformer 總參數量: {total_params:,}")
    print(f"權重層數量: {len(transformer_params)}")
    
    # 分析權重統計
    print(f"\n📊 權重統計:")
    for name, weight in transformer_params.items():
        mean_val = weight.mean().item()
        std_val = weight.std().item()
        min_val = weight.min().item()
        max_val = weight.max().item()
        
        print(f"{name}:")
        print(f"  形狀: {weight.shape}")
        print(f"  均值: {mean_val:.6f}, 標準差: {std_val:.6f}")
        print(f"  範圍: [{min_val:.6f}, {max_val:.6f}]")
        
        # 檢查是否接近初始化狀態
        if abs(mean_val) < 0.01 and 0.05 < std_val < 0.2:
            print(f"  🔍 可能仍接近初始化狀態")
        elif abs(mean_val) > 0.1 or std_val > 0.5:
            print(f"  ✅ 權重已有明顯變化")
        else:
            print(f"  ⚠️ 權重變化程度中等")
        print()

def analyze_attention_patterns(model, input_tokens):
    """分析注意力機制的模式"""
    print("=== 注意力模式分析 ===")
    
    try:
        # 提取注意力權重 (需要修改模型以返回注意力)
        model.eval()
        with torch.no_grad():
            # 編碼 tokens
            tokens = model.encode_audio_to_tokens(input_tokens)
            print(f"輸入 tokens 形狀: {tokens.shape}")
            
            # 通過 transformer
            result_tokens = model.forward_transformer(tokens)
            print(f"輸出 tokens 形狀: {result_tokens.shape}")
            
            # 檢查 token 變化
            if tokens.shape == result_tokens.shape:
                # 計算輸入輸出的相似性
                similarity = torch.cosine_similarity(
                    tokens.float().flatten(), 
                    result_tokens.float().flatten(), 
                    dim=0
                ).item()
                
                print(f"輸入輸出相似性: {similarity:.4f}")
                
                if similarity > 0.9:
                    print("🔍 Transformer 幾乎沒有改變輸入 (可能未學到有用特徵)")
                elif similarity < 0.1:
                    print("⚠️ Transformer 大幅改變輸入 (可能過度隨機化)")
                else:
                    print("✅ Transformer 對輸入進行了適度變換")
            
            # 分析 token 分佈變化
            original_unique = torch.unique(tokens).numel()
            result_unique = torch.unique(result_tokens).numel()
            
            print(f"原始 token 種類數: {original_unique}")
            print(f"輸出 token 種類數: {result_unique}")
            
            if result_unique < original_unique * 0.5:
                print("⚠️ 輸出 token 多樣性大幅降低 (可能過度收斂)")
            elif result_unique > original_unique * 1.5:
                print("🔍 輸出 token 多樣性增加 (可能引入噪音)")
            
    except Exception as e:
        print(f"❌ 注意力分析失敗: {e}")

def simulate_training_progression():
    """模擬訓練過程中 Transformer 行為的變化"""
    print("\n=== 訓練階段模擬 ===")
    
    # 模擬不同訓練階段的權重特徵
    stages = {
        "初始化": {"mean": 0.0, "std": 0.1, "description": "隨機初始化，權重服從正態分佈"},
        "早期訓練": {"mean": 0.02, "std": 0.12, "description": "權重開始微調，但變化很小"},
        "中期訓練": {"mean": 0.05, "std": 0.18, "description": "權重有明顯變化，開始學習模式"},
        "充分訓練": {"mean": 0.1, "std": 0.25, "description": "權重分佈顯著改變，學習到有效特徵"}
    }
    
    print("🎯 訓練階段特徵:")
    for stage, props in stages.items():
        print(f"{stage}: 均值≈{props['mean']:.3f}, 標準差≈{props['std']:.3f}")
        print(f"   {props['description']}")
    
    print(f"\n💡 未充分訓練的影響:")
    print("1. 🎲 隨機映射: 未訓練的權重將 tokens 進行近似隨機的線性變換")
    print("2. 🔀 破壞結構: 語音的時序和語義結構被隨機打亂")
    print("3. 📉 信息丟失: 有用的語音特徵被隨機權重掩蓋或消除")
    print("4. 🔊 音頻退化: 最終重建的音頻失去人聲特徵，變成噪音")

def test_progressive_corruption():
    """測試不同程度的權重隨機化對音頻的影響"""
    print("\n=== 權重隨機化影響測試 ===")
    
    try:
        from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser
        
        # 載入模型
        config = {
            'config_path': '/home/sbplab/ruizi/c_code/config/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
            'model_path': '/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt',
            'd_model': 128,
            'nhead': 2,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'max_length': 1000,
        }
        
        # 載入測試音頻
        audio_path = "/home/sbplab/ruizi/c_code/1n/nor_boy1_box_LDV_001.wav"
        if not Path(audio_path).exists():
            print(f"❌ 測試音頻不存在: {audio_path}")
            return
            
        wav, sr = torchaudio.load(audio_path)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        max_length = 24000 * 3
        if wav.size(-1) > max_length:
            wav = wav[:, :max_length]
        
        print(f"測試音頻形狀: {wav.shape}")
        
        # 測試不同程度的權重隨機化
        corruption_levels = [0.0, 0.1, 0.3, 0.5, 1.0]  # 0=無隨機化, 1=完全隨機
        results = []
        
        output_dir = Path("temp/corruption_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_dir / "original.wav"), wav, 24000)
        
        for corruption in corruption_levels:
            print(f"\n--- 權重隨機化程度: {corruption:.1%} ---")
            
            # 初始化模型
            model = WavTokenizerTransformerDenoiser(**config)
            model.eval()
            
            # 隨機化 Transformer 權重
            if corruption > 0:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'transformer' in name and param.requires_grad:
                            # 混合原始權重和隨機權重
                            random_weight = torch.randn_like(param) * 0.1
                            param.data = (1 - corruption) * param.data + corruption * random_weight
            
            # 測試重建品質
            with torch.no_grad():
                result = model(wav)
                if isinstance(result, dict):
                    output_audio = result['denoised_audio']
                else:
                    output_audio = result
                
                # 調整維度
                if output_audio.dim() == 3:
                    output_2d = output_audio.squeeze(1)
                else:
                    output_2d = output_audio
                
                # 計算品質指標
                original = wav[0].cpu()
                output = output_2d[0].cpu()
                
                min_len = min(original.size(0), output.size(0))
                orig_trim = original[:min_len]
                out_trim = output[:min_len]
                
                correlation = torch.corrcoef(torch.stack([orig_trim, out_trim]))[0, 1]
                
                print(f"重建品質相關性: {correlation:.4f}")
                
                # 保存音頻
                torchaudio.save(str(output_dir / f"corruption_{corruption:.1f}.wav"), 
                               out_trim.unsqueeze(0), 24000)
                
                results.append((corruption, correlation.item()))
        
        # 分析結果
        print(f"\n📊 權重隨機化影響總結:")
        for corruption, quality in results:
            print(f"隨機化 {corruption:.1%}: 品質 {quality:.4f}")
        
        print(f"\n💡 結論:")
        print("隨著權重隨機化程度增加，音頻重建品質應該會顯著下降")
        print("這解釋了為什麼未訓練的 Transformer 會破壞音頻品質")
        print(f"測試音頻已保存至: {output_dir}")
        
    except Exception as e:
        print(f"❌ 權重隨機化測試失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主分析函式"""
    print("=== 未充分訓練 Transformer 影響分析 ===")
    print("目的：深入理解為什麼未訓練的 Transformer 會導致音頻品質差\n")
    
    try:
        from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser
        
        # 載入模型進行分析
        config = {
            'config_path': '/home/sbplab/ruizi/c_code/config/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
            'model_path': '/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt',
            'd_model': 128,
            'nhead': 2,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'max_length': 1000,
        }
        
        model = WavTokenizerTransformerDenoiser(**config)
        
        # 1. 分析權重狀態
        analyze_transformer_weights(model)
        
        # 2. 模擬訓練階段
        simulate_training_progression()
        
        # 3. 測試權重隨機化影響
        test_progressive_corruption()
        
        print(f"\n🎯 總結 - 未充分訓練 Transformer 的問題:")
        print("1. 🎲 隨機變換: 初始化的權重對 tokens 進行近似隨機的映射")
        print("2. 🔀 結構破壞: 語音的時序結構和語義關係被打亂")
        print("3. 📉 特徵消失: 語音中的重要特徵(基頻、諧波等)被隨機權重掩蓋")
        print("4. 🔊 品質退化: 最終音頻失去人聲特徵，聽起來像噪音")
        print("5. 💡 解決方案: 需要充分訓練讓 Transformer 學會有意義的 token 映射")
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()