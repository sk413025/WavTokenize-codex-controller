import os
from pathlib import Path
import torch
import torchaudio
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer
from try2 import EnhancedModel  # 新增這行
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

def load_and_process_audio(audio_path, device):
    """載入並處理音訊檔案，確保輸出形狀正確"""
    wav, sr = torchaudio.load(str(audio_path))
    wav = convert_audio(wav, sr, 24000, 1)  # 轉換為 [1, T]
    
    # 計算音訊長度（秒）
    duration = wav.size(-1) / 24000
    print(f"\nAudio duration: {duration:.2f} seconds")
    print(f"Audio samples: {wav.size(-1)}")
    
    wav = wav.to(device)
    return wav

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

def extract_features(wavtokenizer, wav, device, enhance=False, enhanced_model=None):
    """提取特徵，使用與 try2.py 相同的方法
    Args:
        wavtokenizer: 原始 WavTokenizer 模型
        wav: 輸入音訊
        device: 運算設備
        enhance: 是否為非乾淨音檔
        enhanced_model: EnhancedModel
    """
    # 確保輸入是 [B, C, T] 形狀
    wav = ensure_shape(wav)
    wav = (wav - wav.mean()) / (wav.std() + 1e-6)  # 正規化
    print(f"Input shape after ensuring shape: {wav.shape}")
    
    if enhance and enhanced_model is not None:
        # 非乾淨音檔：使用 EnhancedModel
        print("Using EnhancedModel for feature extraction...")
        with torch.no_grad():
            # 使用與 try2.py 相同的特徵提取過程
            features = enhanced_model.feature_extractor(wav)
            features = features * 1.5  # 特徵增強
            features = torch.tanh(features)  # 限制範圍
    else:
        # 乾淨音檔：使用原始 WavTokenizer
        print("Using original WavTokenizer for feature extraction...")
        with torch.no_grad():
            bandwidth_id = torch.tensor([0], device=device)
            # 使用 feature_extractor.encodec.encoder 而不是 encode_infer
            features = wavtokenizer.feature_extractor.encodec.encoder(wav)
    
    # 檢查特徵形狀
    print(f"Feature shape: {features.shape}")
    return features

def ensure_same_length(features_batch):
    """確保batch中所有特徵長度一致"""
    if not isinstance(features_batch, torch.Tensor):
        return features_batch
    
    # 如果是單個特徵，直接返回
    if features_batch.dim() <= 2:
        return features_batch
        
    # 找出最短長度
    min_length = min(features_batch.size(-1) for features in features_batch)
    
    # 截斷到相同長度
    features_batch = features_batch[..., :min_length]
    
    print(f"Aligned features shape: {features_batch.shape}")
    return features_batch

def calculate_mean_features(features):
    """計算特徵平均值，將 [B, 512, T] 轉換為 [B, 512]"""
    print(f"Input features shape: {features.shape}")
    
    # 確保長度一致
    features = ensure_same_length(features)
    '''
    # 計算平均值
    mean_features = torch.mean(features, dim=2)  # 在時間維度上平均
    print(f"Output features shape after mean: {mean_features.shape}")
    return mean_features
    '''

     # 計算最大值
    max_features, _ = torch.max(features, dim=2)  # 在時間維度上取最大值
    print(f"Output features shape after max: {max_features.shape}")
    return max_features
    

def process_features_for_tsne(features_list):
    """處理特徵以準備進行t-SNE分析"""
    processed_features = []
    
    for features in features_list:
        # 檢查並打印每組特徵的形狀
        print(f"Processing features with shape: {features.shape}")
        
        if features.ndim > 2:
            # 如果是3D特徵，展平為2D
            features = features.reshape(features.shape[0], -1)
            print(f"Reshaped to: {features.shape}")
        
        processed_features.append(features)
    
    # 找出最小維度
    min_dim = min(f.shape[1] for f in processed_features)
    print(f"Minimum feature dimension across all groups: {min_dim}")
    
    # 截斷所有特徵到相同維度
    aligned_features = []
    for features in processed_features:
        if features.shape[1] > min_dim:
            features = features[:, :min_dim]
        aligned_features.append(features)
        print(f"Aligned feature shape: {features.shape}")
    
    return aligned_features

def calculate_tsne(features_list, labels_list, perplexity=30):
    """計算t-SNE，確保特徵維度一致並動態調整 perplexity"""
    # 預處理特徵
    processed_features = process_features_for_tsne(features_list)
    
    # 合併所有特徵
    try:
        all_features = np.vstack(processed_features)
        print(f"Combined features shape: {all_features.shape}")
    except Exception as e:
        print("Error stacking features:", str(e))
        print("Feature shapes:")
        for i, f in enumerate(processed_features):
            print(f"Group {i}: {f.shape}")
        raise
    
    all_labels = np.concatenate(labels_list)
    print(f"Combined labels shape: {all_labels.shape}")
    
    # 動態調整 perplexity
    n_samples = all_features.shape[0]
    adjusted_perplexity = min(perplexity, n_samples - 1)
    # 確保 perplexity 不小於 5
    adjusted_perplexity = max(5, adjusted_perplexity)
    
    print(f"Number of samples: {n_samples}")
    print(f"Adjusted perplexity from {perplexity} to {adjusted_perplexity}")
    
    # 計算t-SNE
    tsne = TSNE(
        n_components=2, 
        perplexity=adjusted_perplexity,
        n_iter=2000,
        random_state=42,
        verbose=1  # 添加進度輸出
    )
    tsne_results = tsne.fit_transform(all_features)
    
    return tsne_results, all_labels

def extract_speaker_name(filename):
    """從檔案名稱中提取語者名稱和編號
    例如：
    'nor_boy1_plastic_LDV_014.wav' -> 'boy1014'
    'nor_girl2_box_LDV_001.wav' -> 'girl2001'
    """
    try:
        parts = filename.split('_')
        # 尋找包含 'boy' 或 'girl' 的部分
        speaker = next(part for part in parts if 'boy' in part or 'girl' in part)
        
        # 獲取編號（移除 .wav 後綴）
        if 'clean' in filename:
            # 處理 clean 檔案格式
            number = parts[-1].replace('.wav', '')
        else:
            # 處理 LDV 檔案格式
            number = parts[-2] if 'LDV' in parts[-1] else parts[-1].replace('.wav', '')
        
        # 組合語者名稱和編號
        name = f"{speaker}{number}"
        print(f"Parsed {filename} -> {name}")  # 新增除錯輸出
        return name
        
    except Exception as e:
        print(f"Error parsing filename {filename}: {str(e)}")
        # 如果解析失敗，返回移除副檔名的檔名
        return filename.replace('.wav', '')

def should_label_file(filename):
    """判斷是否要標註該檔案名稱（只標註001-005的檔案）"""
    try:
        # 從檔案名中取得編號
        parts = filename.split('_')
        number = parts[-1].replace('.wav', '')  # 移除 .wav
        if 'LDV' in number:
            number = parts[-2]  # 如果最後是LDV，取倒數第二個
        
        # 轉換為數字並檢查範圍
        number = int(number)
        return 1 <= number <= 5
    except:
        return False

def plot_combined_tsne(tsne_results, labels, filenames, save_path):
    """繪製合併的t-SNE圖，只標註特定編號的檔案"""
    plt.figure(figsize=(15, 10))
    
    colors = {
        'wav_re': 'blue',
        'box': 'red',
        'plastic': 'green',
        'papercup': 'purple',
        #'no': 'green'  
    }
    
    markers = {
        'wav_re': '*',
        'box': 'o',
        'plastic': 'o',
        'papercup': 'o',
        #'no': 'o'
    }
    
    # 為每個點添加標籤
    for label in np.unique(labels):
        mask = labels == label
        scatter = plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=colors.get(label, 'gray'),
            marker=markers.get(label, 'o'),
            s=100 if label == 'wav_re' else 70,
            label=label,
            alpha=0.6
        )
        
        # 只為編號001-005的檔案添加標註
        for idx, (x, y, filename) in enumerate(zip(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            np.array(filenames)[mask]
        )):
            if should_label_file(filename):
                speaker_name = extract_speaker_name(filename)
                plt.annotate(
                    speaker_name,
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
    
    plt.title('Combined t-SNE Visualization')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_clean_audio_tsne(features, filenames, save_path):
    """繪製只包含乾淨音檔的t-SNE圖"""
    plt.figure(figsize=(12, 12))
    
    # 如果特徵是2D以上，展平為2D
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    
    # 計算乾淨音檔的t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), n_iter=2000, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # 繪製散點圖
    plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c='blue',
        marker='*',
        s=150,  # 更大的星形
        alpha=0.7,
        label='Clean Audio'
    )
    
    # 為每個點添加語者名稱
    for idx, (x, y, filename) in enumerate(zip(
        tsne_results[:, 0],
        tsne_results[:, 1],
        filenames
    )):
        speaker_name = extract_speaker_name(filename)
        plt.annotate(
            speaker_name,
            (x, y),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=10,
            alpha=0.8
        )
    
    plt.title('T-SNE Visualization of Clean Audio Files')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def ensure_feature_length(features_list):
    """確保所有特徵的長度一致，使用最長的長度作為基準"""
    if not features_list:
        return features_list
    
    # 獲取所有特徵的長度
    lengths = [f.size(-1) for f in features_list]
    min_length = min(lengths)
    max_length = max(lengths)
    
    print(f"\nFeature lengths analysis:")
    print(f"Minimum length: {min_length} (approximately {min_length/512:.2f} seconds)")
    print(f"Maximum length: {max_length} (approximately {max_length/512:.2f} seconds)")
    print(f"Using maximum length {max_length} as target length")
    
    # 填充所有特徵到最長長度
    aligned_features = []
    for feat in features_list:
        if (feat.size(-1) < max_length):
            # 如果特徵太短，填充零到目標長度
            print(f"Padding feature from {feat.size(-1)} to {max_length}")
            padding = torch.zeros(1, 512, max_length - feat.size(-1), device=feat.device)
            aligned_features.append(torch.cat([feat, padding], dim=-1))
        else:
            aligned_features.append(feat)
    
    return aligned_features

def main():
    # 設定參數
    config_path = "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "./wavtokenizer_large_speech_320_24k.ckpt"
   #checkpoint_path = "./self_output/best_model.pth"  # 新增：您訓練好的模型權重路徑
    checkpoint_path = "./tout2/best_model.pth"  # 更新為最新模型路徑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化原始 WavTokenizer (用於乾淨音檔)
    print("Loading original WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    
    # 載入增強模型 (用於非乾淨音檔)
    print("Loading enhanced model...")
    enhanced_model = EnhancedModel(config_path, model_path).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    enhanced_model.load_state_dict(checkpoint['model_state_dict'])
    enhanced_model.eval()
    
    # 設定目錄
    wav_re_dir = "./1c"
    test2_dirs = {
        'box': "./1b",
        'plastic': "./1pl",
        'papercup': "./1pa",
        #'no': "./1n"
    }
    output_dir = "./tsne_comparison"
    os.makedirs(output_dir, exist_ok=True)

    features_list = []
    labels_list = []
    filenames_list = []  # 新增檔案名稱列表
    max_files = 2000  # 設定每個資料夾最大檔案數
    
    # 處理 wav_re 的音檔 (乾淨音檔，不需要特徵增強)
    print("\nProcessing wav_re files...")
    wav_re_files = sorted(list(Path(wav_re_dir).rglob("*.wav")))[:max_files]  # 只取前10個檔案
    wav_re_features = []
    wav_re_filenames = []  # 記錄檔案名稱
    
    print("\nAnalyzing file durations...")
    for audio_path in wav_re_files:
        wav, sr = torchaudio.load(str(audio_path))
        duration = wav.size(-1) / sr
        print(f"{audio_path.name}: {duration:.2f} seconds")
    
    for audio_path in tqdm(wav_re_files):
        print(f"\nProcessing file: {audio_path}")
        wav = load_and_process_audio(audio_path, device)
        print(f"Feature input shape: {wav.shape}")
        features = extract_features(wavtokenizer, wav, device, enhance=False)  # 不進行特徵增強
        print(f"Feature output shape: {features.shape}")
        wav_re_features.append(features)
        wav_re_filenames.append(audio_path.name)  # 添加檔案名稱
    
    # 確保特徵長度一致
    wav_re_features = ensure_feature_length(wav_re_features)
    
    if wav_re_features:
        wav_re_features = torch.stack(wav_re_features)  # [B, 512, T]
        print(f"Stacked wav_re features shape: {wav_re_features.shape}")
        wav_re_means = calculate_mean_features(wav_re_features)  # [B, 512]
        print(f"wav_re means shape: {wav_re_means.shape}")
        features_list.append(wav_re_means.cpu().numpy())
        labels_list.append(np.array(['wav_re'] * len(wav_re_means)))
        filenames_list.append(wav_re_filenames)
    
    # 處理 test2 的各個材質音檔 (需要特徵增強)
    print("\nProcessing test2 files...")
    for material, dir_path in test2_dirs.items():
        material_files = sorted(list(Path(dir_path).rglob("*.wav")))[:max_files]  # 只取前10個檔案
        material_features = []
        material_filenames = []  # 記錄檔案名稱
        
        for audio_path in tqdm(material_files, desc=f"Processing {material}"):
            print(f"\nProcessing file: {audio_path}")
            wav = load_and_process_audio(audio_path, device)
            print(f"Feature input shape: {wav.shape}")
            features = extract_features(wavtokenizer, wav, device, enhance=True, enhanced_model=enhanced_model)  # 進行特徵增強
            print(f"Feature output shape: {features.shape}")
            material_features.append(features)
            material_filenames.append(audio_path.name)  # 添加檔案名稱
        
        # 確保特徵長度一致    
        material_features = ensure_feature_length(material_features)
            
        if material_features:
            material_features = torch.stack(material_features)  # [B, 512, T]
            print(f"Stacked {material} features shape: {material_features.shape}")
            material_means = calculate_mean_features(material_features)  # [B, 512]
            print(f"{material} means shape: {material_means.shape}")
            features_list.append(material_means.cpu().numpy())
            labels_list.append(np.array([material] * len(material_means)))
            filenames_list.append(material_filenames)
    
    # 合併所有檔案名稱
    all_filenames = [name for sublist in filenames_list for name in sublist]
    
    # 在計算t-SNE之前檢查特徵
    print("\nFeature list summary:")
    for i, features in enumerate(features_list):
        print(f"Group {i} shape: {features.shape}")
    
    # 計算並繪製t-SNE
    print("\nCalculating t-SNE...")
    tsne_results, all_labels = calculate_tsne(features_list, labels_list)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 保存組合圖
    combined_path = os.path.join(output_dir, f"combined_tsne_{timestamp}.png")
    plot_combined_tsne(tsne_results, all_labels, all_filenames, combined_path)
    print(f"\nSaved combined t-SNE plot to: {combined_path}")
    '''
    # 2. 保存乾淨音檔的單獨圖
    if len(features_list) > 0:  # 確保有特徵數據
        clean_path = os.path.join(output_dir, f"clean_audio_tsne_{timestamp}.png")
        # 只取乾淨音檔的特徵（第一個特徵組）
        clean_features = features_list[0]  
        print(f"\nGenerating clean audio t-SNE plot with shape: {clean_features.shape}")
        plot_clean_audio_tsne(clean_features, wav_re_filenames, clean_path)
        print(f"Saved clean audio t-SNE plot to: {clean_path}")
    '''
if __name__ == "__main__":
    main()
