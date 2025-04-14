import torch
import torchaudio
import os
from try2 import EnhancedModel, process_audio
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.manifold import TSNE
from datetime import datetime
from tqdm import tqdm

def ensure_shape(wav):
    """確保張量形狀為 [B, C, T]"""
    if (wav.dim() == 1):  # [T]
        wav = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif (wav.dim() == 2):  # [C, T]
        wav = wav.unsqueeze(0)  # [1, C, T]
    elif (wav.dim() == 3):  # [B, T, C]
        if (wav.size(1) != 1):
            wav = wav.transpose(1, 2)  # 將形狀變為 [B, C, T]
    return wav

def calculate_average_tsne(features, n_runs=10):
    """計算多次T-SNE的平均結果"""
    tsne_results = []
    features_np = features.detach().cpu().numpy()
    
    for i in range(n_runs):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=i)
        result = tsne.fit_transform(features_np)
        tsne_results.append(result)
    
    return np.mean(tsne_results, axis=0)

def visualize_tsne(features, save_path, filename):
    """生成T-SNE視覺化圖"""
    plt.figure(figsize=(10, 10))
    tsne_avg = calculate_average_tsne(features)
    
    plt.scatter(tsne_avg[:, 0], tsne_avg[:, 1], alpha=0.6)
    plt.title(f'T-SNE Visualization for {filename}\n(Averaged over 10 runs)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.savefig(save_path)
    plt.close()

def plot_spectrograms(original_wav, enhanced_wav, save_path):
    """繪製頻譜圖比較"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    for wav, ax, title in zip(
        [original_wav, enhanced_wav],
        [ax1, ax2],
        ['Original', 'Enhanced']
    ):
        wav_numpy = wav.numpy().squeeze()
        D = librosa.stft(wav_numpy)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(D_db, y_axis='log', x_axis='time', ax=ax)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_feature_stats(features, name):
    """打印特徵統計信息"""
    print(f"\n{name} Statistics:")
    print(f"Shape: {features.shape}")
    print(f"Range: [{features.min().item():.4f}, {features.max().item():.4f}]")
    print(f"Mean: {features.mean().item():.4f}")
    print(f"Std: {features.std().item():.4f}")

def test_model(input_path, output_path, model_config):
    """確保與 try2.py 完全一致的處理流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 1. 初始化模型 - 使用 try2 的模型
        model = EnhancedModel(
            config_path=model_config['config_path'],
            model_path=model_config['model_path']
        ).to(device)
        
        # 2. 載入 try2 訓練的權重
        checkpoint = torch.load(model_config['checkpoint_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from: {model_config['checkpoint_path']}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Checkpoint loss: {checkpoint['loss']:.6f}")
        
        # 3. 評估模式
        model.eval()
        
        print("\nProcessing input audio...")
        
        # 4. 輸入處理 - 確保與訓練時完全一致
        input_wav = process_audio(input_path, normalize=True)  # [1, T]
        input_wav = ensure_shape(input_wav)  # [1, 1, T]
        input_wav = input_wav.to(device)
        input_wav = (input_wav - input_wav.mean()) / (input_wav.std() + 1e-6)
        
        print("Input wav shape:", input_wav.shape)
        print("Input wav range:", input_wav.min().item(), "to", input_wav.max().item())
        
        # 修改特徵處理流程，與train_with_tsne.py保持一致
        with torch.no_grad():
            # 1. 特徵提取，與訓練時保持一致
            input_features = model.feature_extractor.encodec.encoder(input_wav)
            print_feature_stats(input_features, "Initial encoder features")
            
            # 2. 特徵增強，明確使用各層
            enhanced_features = input_features
            layer_names = ['adapter_conv', 'adapter_bn', 'residual_blocks', 'out_conv']
            for name, layer in zip(layer_names, 
                                 [model.feature_extractor.adapter_conv,
                                  model.feature_extractor.adapter_bn,
                                  model.feature_extractor.residual_blocks,
                                  model.feature_extractor.out_conv]):
                enhanced_features = layer(enhanced_features)
                print_feature_stats(enhanced_features, f"After {name}")
            
            # 3. 直接解碼，不需要額外的scaling和tanh
            output = model.feature_extractor.encodec.decoder(enhanced_features)
            print_feature_stats(output, "Decoder output")

            # 形狀處理
            min_length = min(output.size(-1), input_wav.size(-1))
            output = output[..., :min_length]
            output = output.squeeze(1)
            
            # 正規化輸出
            output = output / (output.abs().max() + 1e-8)
            print_feature_stats(output, "Final output")
        

        # 6. 儲存結果
        print("\nSaving results...")
        output = output.cpu()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 確保輸出張量維度正確 [C, T]
        if output.dim() == 1:  # [T]
            output = output.unsqueeze(0)  # [1, T]
        elif output.dim() == 3:  # [B, C, T]
            output = output.squeeze(0)  # [C, T]
            
        # 保存處理後的音頻
        torchaudio.save(output_path, output, 24000)

        # 7. 繪製頻譜圖
        plot_path = os.path.join(os.path.dirname(output_path), 
                                f"spec_{Path(input_path).stem}.png")
        plot_spectrograms(
            input_wav.cpu().squeeze(0),
            output.cpu().squeeze(0),
            plot_path
        )
        
        print(f"Successfully saved to {output_path}")
        print(f"Spectrogram saved to {plot_path}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("Full error trace:")
        import traceback
        traceback.print_exc()
        raise

def process_directory(input_dir, output_dir, model_config):
    """批次處理整個資料夾的音檔"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tsne_plots'), exist_ok=True)
    
    # 載入模型
    model = EnhancedModel(
        config_path=model_config['config_path'],
        model_path=model_config['model_path']
    ).to(device)
    
    checkpoint = torch.load(model_config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint loss: {checkpoint['loss']:.6f}")
    model.eval()
    
    # 取得所有 .wav 檔案
    wav_files = list(Path(input_dir).rglob("*.wav"))
    total_files = len(wav_files)
    print(f"\nFound {total_files} WAV files to process")
     
    # 批次處理
    for idx, wav_path in enumerate(wav_files, 1):
        try:
            print(f"\nProcessing [{idx}/{total_files}]: {wav_path}")
            
            # 保持相對路徑結構
            rel_path = wav_path.relative_to(input_dir)
            output_path = Path(output_dir) / f"enhanced_{rel_path}"
            os.makedirs(output_path.parent, exist_ok=True)
           
            # 4. 輸入處理 - 確保與訓練時完全一致
            input_wav = process_audio(str(wav_path), normalize=True)  # [1, T]
            input_wav = ensure_shape(input_wav)  # [1, 1, T]
            input_wav = input_wav.to(device)
            input_wav = (input_wav - input_wav.mean()) / (input_wav.std() + 1e-6)
            
            with torch.no_grad():
                # 修改特徵處理流程
                # 1. 特徵提取
                input_features = model.feature_extractor.encodec.encoder(input_wav)
                
                # 2. 特徵增強
                enhanced_features = input_features
                for layer in [model.feature_extractor.adapter_conv,
                            model.feature_extractor.adapter_bn,
                            model.feature_extractor.residual_blocks,
                            model.feature_extractor.out_conv]:
                    enhanced_features = layer(enhanced_features)
                
                # 3. 直接解碼
                output = model.feature_extractor.encodec.decoder(enhanced_features)
                
                # 正規化和保存
                output = output.cpu()
                output = output.squeeze(1)  # Remove channel dimension if present
                output = output / (output.abs().max() + 1e-8)
                
                # 確保輸出維度正確 [C, T]
                if output.dim() == 1:  # [T]
                    output = output.unsqueeze(0)  # [1, T]
                elif output.dim() == 3:  # [B, C, T]
                    output = output.squeeze(0)  # [C, T]
                
            
            # 6. 儲存結果
            output_path = os.path.join(output_dir, f"enhanced_{wav_path.name}")
            torchaudio.save(output_path, output.cpu(), 24000)
            
            # 7. 繪製頻譜圖
            spec_path = os.path.join(output_dir, 'spectrograms', 
                                   f"spec_{wav_path.stem}.png")
            plot_spectrograms(
                input_wav.cpu().squeeze(0),
                output.cpu().squeeze(0),
                spec_path
            )
            
        except Exception as e:
            print(f"\nError processing {wav_path}: {str(e)}")
            continue
    
    print("\nProcessing completed!")

def plot_spectrograms(input_wav, output_wav, save_path):
    """繪製輸入和輸出的頻譜圖比較"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for wav, ax, title in [(input_wav, ax1, 'Input'), (output_wav, ax2, 'Enhanced')]:
        wav_numpy = wav.numpy().squeeze()
        D = librosa.stft(wav_numpy)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(D_db, y_axis='log', x_axis='time', ax=ax)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    config = {
        'config_path': "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        'model_path': "./wavtokenizer_large_speech_320_24k.ckpt",
        'checkpoint_path': "./tsne_output/best_model.pth"
    }
    
    # 添加詳細的模型信息輸出
    print("Model configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # 批次處理整個資料夾
    process_directory(
        input_dir="./tte",  # 輸入資料夾
        output_dir="./ttout",  # 輸出資料夾
        model_config=config
    )
    
    """
    # 單檔處理（已註解）
    test_model(
        input_path="./tte/nor_boy1_box_LDV_002.wav",
        output_path="./enhanced_outputs_tsne/test_nor_boy1_box_LDV_002.wav",
        model_config=config
    )
"""