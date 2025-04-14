import torch
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import EncodecModel, AutoProcessor
import os
from pathlib import Path
import torchaudio
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
import torch.cuda

class AudioFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading EncodecModel...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.target_sample_rate = 24000
        
    def forward(self, audio_path):
        with torch.no_grad():
            try:
                # Load audio using torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample if necessary
                if sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, 
                        new_freq=self.target_sample_rate
                    )
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Process audio input
                inputs = self.processor(
                    raw_audio=waveform.squeeze().numpy(), 
                    sampling_rate=self.target_sample_rate, 
                    return_tensors="pt"
                )
                
                # Move inputs to the same device as the model
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # Get encoder outputs
                encoder_outputs = self.model.encode(
                    inputs["input_values"], 
                    inputs["padding_mask"]
                )
                
                # Use audio codes as features
                features = encoder_outputs.audio_codes.flatten()
                return features
                
            except Exception as e:
                print(f"Error in forward pass: {e}")
                raise e

def parse_uttid(uttid):
    """解析uttid以獲取說話者和材質信息"""
    # 例如: boy1_box_LDV_008 -> speaker: boy1, material: box
    parts = uttid.split('_')
    speaker = parts[0]
    material = parts[1]
    return speaker, material

def read_scp_file(scp_path):
    """讀取 wav.scp 檔案，取用所有nor目錄下的音檔"""
    uttid_to_path = {}
    speaker_count = {}
    materials_found = set()
    
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            uttid, path = line.strip().split(maxsplit=1)
            
            # 檢查路徑是否包含 "nor" 目錄
            if "nor" not in Path(path).parts:    # 取用所有自訂檔案目錄下的音檔
                continue
                
            parts = uttid.split('_')
            speaker = parts[0]
            
            # 計算每個說話者的音檔數量（用於統計）
            if speaker not in speaker_count:
                speaker_count[speaker] = 0
            speaker_count[speaker] += 1
            
            uttid_to_path[uttid] = path
            
            # 收集路徑信息（用於調試）
            materials_found.add(str(Path(path).parent))
    
    # 印出找到的路徑
    print("\nFound directories:", sorted(materials_found))
    
    # 印出每個說話者的音檔數量
    print("\nSelected normal audio files per speaker:")
    for speaker, count in speaker_count.items():
        print(f"{speaker}: {count}")
    
    if not uttid_to_path:
        raise ValueError(f"No normal audio files found! Available directories are: {sorted(materials_found)}")
    
    return uttid_to_path

class AudioDataset(Dataset):
    def __init__(self, uttid_to_path):
        self.uttids = list(uttid_to_path.keys())
        self.paths = list(uttid_to_path.values())
        
    def __len__(self):
        return len(self.uttids)
        
    def __getitem__(self, idx):
        return self.uttids[idx], self.paths[idx]

def standardize_feature_length(features_list, target_length=None):
    """Standardize all features to the same length by padding or truncating"""
    if not features_list:
        return []
    
    # If target_length is not specified, use the median length
    if target_length is None:
        lengths = [len(f) for f in features_list]
        target_length = int(np.median(lengths))
    
    standardized_features = []
    for feature in features_list:
        if len(feature) > target_length:
            # Truncate
            standardized_features.append(feature[:target_length])
        elif len(feature) < target_length:
            # Pad with zeros
            padding = np.zeros(target_length - len(feature))
            standardized_features.append(np.concatenate([feature, padding]))
        else:
            standardized_features.append(feature)
    
    return standardized_features

def load_audio_features(uttid_to_path, batch_size=32, num_workers=4):
    """批次載入並平行處理音檔特徵"""
    features = []
    metadata = []
    
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    
    # Create dataset and dataloader
    dataset = AudioDataset(uttid_to_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    print("Processing audio files in batches...")
    for batch_uttids, batch_paths in dataloader:
        batch_features = []
        batch_metadata = []
        
        for uttid, path in zip(batch_uttids, batch_paths):
            try:
                features_tensor = feature_extractor(path)
                batch_features.append(features_tensor.cpu().numpy())
                
                speaker, material = parse_uttid(uttid)
                batch_metadata.append({
                    "uttid": uttid,
                    "speaker": speaker,
                    "material": material
                })
                print(f"Processed: {uttid}")
                
            except Exception as e:
                print(f"Error processing {uttid}: {e}")
                continue
        
        if batch_features:
            features.extend(batch_features)
            metadata.extend(batch_metadata)
    
    if not features:
        raise ValueError("No audio files were successfully processed!")
    
    # Standardize feature lengths before stacking
    features = standardize_feature_length(features)
    features = np.stack(features)
    
    return features, metadata

def evaluate_speaker_clustering(clusters, metadata):
    """評估聚類結果與實際語者的對應關係"""
    # 建立語者到索引的映射
    speakers = list(set(m["speaker"] for m in metadata))
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(speakers)}
    
    # 獲取真實的語者標籤
    true_labels = [speaker_to_idx[m["speaker"]] for m in metadata]
    
    # 計算輪廓分數
    sil_score = silhouette_score(features, clusters)
    
    # 統計每個聚類中的主要語者
    cluster_speaker_stats = {}
    for cluster_id in range(max(clusters) + 1):
        cluster_mask = clusters == cluster_id
        cluster_speakers = [metadata[i]["speaker"] for i in range(len(metadata)) if cluster_mask[i]]
        if cluster_speakers:
            speaker_counts = {}
            for speaker in cluster_speakers:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            main_speaker = max(speaker_counts.items(), key=lambda x: x[1])
            cluster_speaker_stats[cluster_id] = {
                "main_speaker": main_speaker[0],
                "count": main_speaker[1],
                "total": len(cluster_speakers),
                "purity": main_speaker[1] / len(cluster_speakers)
            }
    
    return {
        "num_speakers": len(speakers),
        "detected_clusters": max(clusters) + 1,
        "silhouette_score": sil_score,
        "cluster_stats": cluster_speaker_stats
    }

def preprocess_features(features):
    """特徵預處理：標準化和降維 (GPU加速版本)"""
    # 轉換為 PyTorch tensor 並移到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_tensor = torch.from_numpy(features).to(device)
    
    # 在 GPU 上進行標準化
    features_mean = features_tensor.mean(dim=0)
    features_std = features_tensor.std(dim=0)
    features_normalized = (features_tensor - features_mean) / (features_std + 1e-8)
    
    # 轉回 CPU 進行 PCA (因為 sklearn 的 PCA 不支援 GPU)
    features_normalized = features_normalized.cpu().numpy()
    
    # PCA降維，保留95%的變異性
    pca = PCA(n_components=0.95, random_state=42)
    features_reduced = pca.fit_transform(features_normalized)
    
    print(f"Features reduced from {features.shape[1]} to {features_reduced.shape[1]} dimensions")
    return features_reduced

class GPUKMeans:
    """GPU 加速版本的 KMeans"""
    def __init__(self, n_clusters, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit_predict(self, X):
        # 轉換數據到 GPU
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        # 隨機初始化聚類中心
        n_samples = X_tensor.shape[0]
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        indices = torch.randperm(n_samples)[:self.n_clusters]
        centroids = X_tensor[indices].clone()
        
        for _ in range(self.max_iter):
            # 計算距離矩陣
            distances = torch.cdist(X_tensor, centroids)
            
            # 分配最近的聚類
            labels = torch.argmin(distances, dim=1)
            
            # 更新聚類中心
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if torch.any(mask):
                    new_centroids[k] = X_tensor[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # 檢查收斂
            if torch.all(torch.abs(new_centroids - centroids) < 1e-4):
                break
                
            centroids = new_centroids
        
        self.cluster_centers_ = centroids.cpu().numpy()
        self.inertia_ = torch.sum(torch.min(distances, dim=1)[0]).item()
        
        return labels.cpu().numpy()

def find_optimal_clusters(features, metadata):
    """自動尋找最佳分群數 (GPU加速版本)"""
    features_processed = preprocess_features(features)
    
    speakers = list(set(m["speaker"] for m in metadata))
    n_speakers = len(speakers)
    
    min_k = max(2, n_speakers - 2)
    max_k = n_speakers + 3
    K = range(min_k, max_k + 1)
    
    best_score = -1
    best_k = min_k
    scores = []
    inertias = []
    
    print("\nAnalyzing optimal cluster number using GPU...")
    for k in K:
        kmeans = GPUKMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(features_processed)
        
        sil_score = silhouette_score(features_processed, clusters)
        normalized_inertia = kmeans.inertia_ / len(features_processed)
        
        scores.append(sil_score)
        inertias.append(normalized_inertia)
        
        print(f"K={k}, Silhouette Score={sil_score:.3f}, Normalized Loss={normalized_inertia:.2f}")
        
        if sil_score > best_score:
            best_score = sil_score
            best_k = k
    
    return best_k, K, scores, inertias, features_processed

def prepare_features_for_byol(features, metadata):
    """將特徵轉換為BYOL訓練所需的格式"""
    # 確保特徵維度正確 (調整到256維)
    batch_size, total_dim = features.shape
    input_dim = 256
    seq_len = total_dim // input_dim
    
    # 如果特徵總長度不是256的倍數，進行截斷
    features = features[:, :seq_len * input_dim]
    # 重塑特徵為 (batch_size, seq_len, input_dim)
    features = features.reshape(batch_size, seq_len, input_dim)
    
    processed_data = {
        'features': torch.from_numpy(features).float(),
        'labels': [m['speaker'] for m in metadata],
        'file_paths': [m['uttid'] for m in metadata],
    }
    
    print(f"Prepared features shape: {processed_data['features'].shape}")
    print(f"Number of unique speakers: {len(set(processed_data['labels']))}")
    
    return processed_data

if __name__ == "__main__":
    # 檢查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Read wav.scp file
    scp_path = "./wav.scp"
    if not os.path.exists(scp_path):
        print(f"Error: {scp_path} not found!")
        exit(1)
    
    # Load audio paths and features
    uttid_to_path = read_scp_file(scp_path)
    features, metadata = load_audio_features(uttid_to_path, batch_size=32, num_workers=4)
    
    # 準備BYOL訓練數據並保存
    print("Preparing features for BYOL training...")
    byol_features = prepare_features_for_byol(features, metadata)
    
    # 保存特徵供後續使用
    save_path = 'encodec_features_for_byol.pth'
    torch.save(byol_features, save_path)
    print(f"Features saved to {save_path}")
    print(f"Number of samples: {len(byol_features['features'])}")
    print(f"Feature shape: {byol_features['features'].shape}")
    
    # 自動尋找最佳分群數，使用預處理後的特徵
    optimal_k, K, scores, inertias, processed_features = find_optimal_clusters(features, metadata)
    
    # 使用最佳K進行最終分群
    final_kmeans = GPUKMeans(n_clusters=optimal_k, random_state=42)
    clusters = final_kmeans.fit_predict(processed_features)
    
    # 繪製評估指標圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss曲線（使用正規化後的值）
    ax1.plot(K, inertias, 'bx-')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Normalized Loss')
    ax1.set_title('Normalized Loss vs. K')
    
    # 輪廓分數曲線
    ax2.plot(K, scores, 'rx-')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs. K')  # 將 setTitle 改為 set_title
    
    plt.suptitle(f'Clustering Evaluation (Optimal K={optimal_k})')
    plt.savefig('clustering_evaluation.png')
    plt.show()
    
    # 繪製語者分布圖（使用處理後的特徵）
    pca_viz = PCA(n_components=2)
    features_2d = pca_viz.fit_transform(processed_features)
    
    plt.figure(figsize=(15, 10))
    speakers = list(set(m["speaker"] for m in metadata))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(speakers)))
    
    for i, speaker in enumerate(speakers):
        # 獲取該說話者的所有點
        speaker_indices = [idx for idx, m in enumerate(metadata) if m["speaker"] == speaker]
        speaker_features = features_2d[speaker_indices]
        
        # 繪製散點
        plt.scatter(
            speaker_features[:, 0],
            speaker_features[:, 1],
            label=speaker,
            alpha=0.6,
            color=colors[i]
        )
        
        # 為每個點添加標籤
        for idx, (x, y) in zip(speaker_indices, speaker_features):
            uttid = metadata[idx]["uttid"]
            # 從uttid中提取句子編號
            sentence_num = uttid.split('_')[-1]  # 假設格式為 speaker_material_xxx
            plt.annotate(sentence_num, (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    plt.title(f'Speaker Distribution with Sentence Numbers (K={optimal_k})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.savefig('speaker_visualization_with_numbers.png', dpi=300, bbox_inches='tight')
    plt.show()

