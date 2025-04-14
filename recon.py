import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from train_byol import BYOL, SimpleTransformer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from datetime import datetime
import yaml  # 導入 YAML 庫

# 在文件開頭加入顏色映射字典
SPEAKER_COLORS = {
    'boy1': '#9B59B6',
    'boy2': '#3498DB',
    'boy3': '#E74C3C',
    'boy4': '#F39C12',
    'boy5': '#FFA500',
    'boy6': '#4CAF50',
    'girl1': '#1ABC9C',
    'girl2': '#7F8C8D',
    'girl3': '#8E44AD',
    'girl4': '#7F8C8D',
    'girl6': '#FFB6C1',
    'girl7': '#F8E47D'
}

# 設定字體，避免中文亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 加載配置文件
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_trained_model(model_path, device):
    # 載入BYOL模型配置
    input_dim = 256
    hidden_dim = 128
    projection_dim = 128
    num_heads = 4
    num_layers = 2
    
    # 初始化模型
    model = BYOL(input_dim, hidden_dim, projection_dim, num_heads, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_new_speakers(model, features_path, device):
    """處理新語者數據並返回完整的音訊路徑"""
    data = torch.load(features_path, map_location=device)  # 確保數據在正確的設備上加載
    features = data['features']
    labels = data['labels']
    file_paths = data['file_paths']
    
    # 修改音訊路徑處理
    base_audio_path = Path('./only_box/dataset/valid')  # 更新基礎路徑
    full_paths = []
    for path in file_paths:
        if isinstance(path, str):
            # 提取文件名
            file_name = Path(path).name
            # 構建可能的路徑
            possible_paths = [
                base_audio_path / file_name,
                base_audio_path / f"{file_name}.wav",
                Path('./only_box/dataset') / file_name,
                Path('./only_box/dataset') / f"{file_name}.wav"
            ]
            # 使用第一個存在的路徑
            audio_path = next((p for p in possible_paths if p.exists()), None)
            full_paths.append(str(audio_path) if audio_path else None)
        else:
            full_paths.append(None)
    
    # 提取embeddings
    embeddings = []
    with torch.no_grad():
        for feature in features:
            feature = feature.to(device)
            embedding = model.online_encoder(feature)
            embeddings.append(embedding.cpu())
    
    return torch.stack(embeddings), labels, full_paths

def calculate_speaker_centroids(train_embeddings, train_labels):
    """計算每個訓練語者的中心點"""
    if len(train_embeddings.shape) == 3:
        train_embeddings = torch.mean(train_embeddings, dim=1)
        
    unique_speakers = set(train_labels)
    centroids = {}
    
    for speaker in unique_speakers:
        mask = [label == speaker for label in train_labels]
        speaker_embeddings = train_embeddings[mask]
        centroids[speaker] = torch.mean(speaker_embeddings, dim=0)
    
    return centroids

def calculate_distance_to_centroids(new_embeddings, centroids):
    """計算新語者與各個訓練語者中心的距離"""
    distances = {}
    for speaker, centroid in centroids.items():
        dist = F.cosine_similarity(new_embeddings, centroid.unsqueeze(0), dim=1).mean()
        distances[speaker] = dist.item()
    return distances

def calculate_normalized_similarity(emb1, emb2):
    """改進的相似度計算方法"""
    if len(emb1.shape) == 1:
        emb1 = emb1.reshape(1, -1)
    if len(emb2.shape) == 1:
        emb2.reshape(1, -1)
    
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    emb1_scaled = scaler.fit_transform(emb1)
    emb2_scaled = scaler.transform(emb2)
    
    cos_sim = cosine_similarity(emb1_scaled, emb2_scaled)[0][0]
    
    from sklearn.metrics.pairwise import euclidean_distances
    euc_dist = euclidean_distances(emb1_scaled, emb2_scaled)[0][0]
    euc_sim = 1 / (1 + euc_dist)
    
    from sklearn.metrics.pairwise import manhattan_distances
    man_dist = manhattan_distances(emb1_scaled, emb2_scaled)[0][0]
    man_sim = 1 / (1 + man_dist)
    
    weights = [0.5, 0.3, 0.2]
    combined_sim = (
        weights[0] * cos_sim + 
        weights[1] * euc_sim + 
        weights[2] * man_sim
    )
    
    return combined_sim

def calculate_similarity_details(new_emb, centroids):
    """計算詳細的相似度信息"""
    details = {}
    for spk, centroid in centroids.items():
        cos_sim = cosine_similarity(new_emb, centroid.reshape(1, -1))[0][0]
        euc_dist = euclidean_distances(new_emb, centroid.reshape(1, -1))[0][0]
        man_dist = manhattan_distances(new_emb, centroid.reshape(1, -1))[0][0]
        
        details[spk] = {
            'cosine': cos_sim,
            'euclidean': 1 / (1 + euc_dist),
            'manhattan': 1 / (1 + man_dist)
        }
    
    return details

def process_distances(distances):
    """Convert raw distances to normalized similarities"""
    values = np.array(list(distances.values()))
    max_val = np.max(values)
    min_val = np.min(values)
    
    if max_val == min_val:
        normalized_values = np.ones_like(values)
    else:
        normalized_values = 1 - (values - min_val) / (max_val - min_val)
    
    return dict(zip(distances.keys(), normalized_values))

def visualize_validation_distances(valid_embeddings, valid_labels, train_centroids):
    """視覺化驗證語者與訓練語者團的關係"""
    if len(valid_embeddings.shape) == 3:
        valid_embeddings = torch.mean(valid_embeddings, dim=1)
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 距離熱圖
    ax1 = fig.add_subplot(121)
    distance_matrix = []
    for valid_emb in valid_embeddings:
        distances = []
        valid_emb_2d = valid_emb.unsqueeze(0)
        
        for centroid in train_centroids.values():
            centroid_2d = centroid.unsqueeze(0)
            dist = calculate_normalized_similarity(valid_emb_2d, centroid_2d)
            distances.append(dist)
        distance_matrix.append(distances)
    
    sns.heatmap(
        distance_matrix,
        xticklabels=list(train_centroids.keys()),
        yticklabels=valid_labels,
        cmap='RdBu_r',
        annot=True,
        fmt='.2f',
        center=0,
        vmin=-1,
        vmax=1
    )
    ax1.set_title('Speaker Similarity Matrix')
    
    # 2. 3D空間分布圖
    ax2 = fig.add_subplot(122, projection='3d')
    
    centroids_stack = torch.stack(list(train_centroids.values()))
    all_points = torch.cat([centroids_stack, valid_embeddings], dim=0)
    
    # t-SNE降維到3維
    tsne = TSNE(n_components=3)
    points_3d = tsne.fit_transform(all_points.detach().numpy())
    
    # 繪製訓練語者中心
    n_centroids = len(train_centroids)
    speaker_names = list(train_centroids.keys())
    
    # 為每個訓練語者使用指定顏色繪製
    for i in range(n_centroids):
        speaker = speaker_names[i]
        color = SPEAKER_COLORS.get(speaker, '#000000')
        ax2.scatter(
            points_3d[i:i+1, 0],
            points_3d[i:i+1, 1],
            points_3d[i:i+1, 2],
            c=color,
            marker='o',
            s=100,
            label=f'Training {speaker}'
        )
    
    # 繪製驗證語者
    for i, label in enumerate(valid_labels):
        idx = i + n_centroids
        color = SPEAKER_COLORS.get(label, '#000000')
        ax2.scatter(
            points_3d[idx:idx+1, 0],
            points_3d[idx:idx+1, 1],
            points_3d[idx:idx+1, 2],
            c=color,
            marker='o',
            s=100,
            label=f'Valid {label}'
        )
    
    ax2.set_title('Speaker Distribution (3D)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 設置更好的3D視角
    ax2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('./only_box/output/validation_analysis_3d.png')
    plt.close()

def calculate_3d_angle(v1, v2):
    """計算兩個3D向量之間的角度（以度為單位）"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def ensure_centroids_dict(centroids, labels=None):
    """確保centroids是字典格式"""
    if isinstance(centroids, dict):
        return centroids
    
    if labels is None:
        return {f'speaker_{i}': centroid for i, centroid in enumerate(centroids)}
    
    unique_labels = sorted(set(labels))
    return {label: centroids[i] for i, label in enumerate(unique_labels)}

def create_waveform_plot(waveform, sr, title, ax):
    """繪製波形圖"""
    time_axis = np.arange(0, len(waveform)) / sr
    ax.plot(time_axis, waveform.squeeze().numpy())
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

def create_spectrogram_plot(waveform, sr, title, ax):
    """繪製語譜圖"""
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
    spec = spectrogram(waveform)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    ax.imshow(spec_db.squeeze().numpy(), origin="lower", aspect="auto",
               extent=[0, len(waveform) / sr, 0, sr/2000], cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")

def visualize_speaker_distances(new_speaker_embedding, train_centroids, train_labels, audio_path=None):
    """改進的視覺化函數，添加3D角度資訊和音訊圖表"""
    # 確保centroids是字典格式
    train_centroids = ensure_centroids_dict(train_centroids, train_labels)
    
    # 創建不對稱的圖形，使3D圖更大
    fig = plt.figure(figsize=(24, 10)) # Adjust the figure size
    gs = plt.GridSpec(2, 3, width_ratios=[1, 2, 1], height_ratios=[2, 1]) # Adjust GridSpec for 2 rows

    # 計算平均相似度
    if len(new_speaker_embedding.shape) == 4:
        new_speaker_embedding = torch.mean(new_speaker_embedding, dim=(1, 2))
    elif len(new_speaker_embedding.shape) == 3:
        new_speaker_embedding = torch.mean(new_speaker_embedding, dim=1)
    elif len(new_speaker_embedding.shape) == 1:
        new_speaker_embedding = new_speaker_embedding.unsqueeze(0)
    
    similarities = calculate_average_speaker_similarity(new_speaker_embedding, train_centroids)
    normalized_similarities = process_distances(similarities)
    sorted_items = sorted(normalized_similarities.items(), key=lambda x: x[1], reverse=True)
    speakers, values = zip(*sorted_items)
    
    # 1. 相似度條形圖
    ax1 = fig.add_subplot(gs[0])
    colors = [SPEAKER_COLORS.get(speaker, '#000000') for speaker in speakers]
    values_array = np.array(values)
    normalized_values = (values_array + 1) / 2
    bars = ax1.bar(speakers, normalized_values, color=colors)
    ax1.set_title('Speaker Similarity Scores', pad=10)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.set_ylabel('Similarity Score')
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 2. 3D空間分布圖
    ax2 = fig.add_subplot(gs[1], projection='3d')
    
    avg_embedding = torch.mean(new_speaker_embedding, dim=0, keepdim=True)
    centroids_stack = torch.stack(list(train_centroids.values()))
    all_points = torch.cat([centroids_stack, avg_embedding], dim=0)
    
    n_samples = all_points.shape[0]
    perplexity = min(n_samples - 1, 5)
    
    try:
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            n_iter=2000,
            random_state=42,
            method='exact'
        )
        points_3d = tsne.fit_transform(all_points.detach().numpy())
    except ValueError:
        pca = PCA(n_components=3, random_state=42)
        points_3d = pca.fit_transform(all_points.detach().numpy())
        print("Using PCA instead of t-SNE due to small sample size")
    
    speaker_names = list(train_centroids.keys())
    for i, speaker in enumerate(speaker_names):
        color = SPEAKER_COLORS.get(speaker, '#000000')
        ax2.scatter(
            points_3d[i:i+1, 0],
            points_3d[i:i+1, 1],
            points_3d[i:i+1, 2],
            c=color,
            marker='o',
            s=100,
            label=f'Training {speaker}'
        )
    
    ax2.scatter(
        points_3d[-1:, 0],
        points_3d[-1:, 1],
        points_3d[-1:, 2],
        c='red',
        marker='*',
        s=200,
        label='New Speaker'
    )
    
    for i, (spk, sim) in enumerate(normalized_similarities.items()):
        color = SPEAKER_COLORS.get(spk, '#000000')
        ax2.plot(
            [points_3d[-1,0], points_3d[i,0]],
            [points_3d[-1,1], points_3d[i,1]],
            [points_3d[-1,2], points_3d[i,2]],
            '--',
            alpha=sim,
            color=color
        )
    
    angles = {}
    new_point = points_3d[-1]
    for i, speaker in enumerate(speaker_names):
        vector1 = points_3d[i]
        vector2 = new_point
        angle = calculate_3d_angle(vector1, vector2)
        angles[speaker] = angle
        mid_point = (vector1 + vector2) / 2
        ax2.text(mid_point[0], mid_point[1], mid_point[2],
                f'{angle:.1f}°',
                color=SPEAKER_COLORS.get(speaker, '#000000'),
                fontsize=8)
    
    print("\n3D空間中的向量角度:")
    for speaker, angle in sorted(angles.items(), key=lambda x: x[1]):
        print(f"{speaker}: {angle:.1f}°")
    
    ax2.set_title('Speaker Distribution (3D)')
    ax2.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    ax2.view_init(elev=20, azim=45)

    # 3. 加入音訊圖表，只在有 audio_path 的情況下執行
    if audio_path:
        try:
            audio_file = Path(audio_path)
            if audio_file.exists():
                waveform, sr = torchaudio.load(str(audio_file))
            else:
                # 嘗試不同的文件擴展名
                for ext in ['.wav', '.mp3', '.flac']:
                    alt_path = audio_file.with_suffix(ext)
                    if alt_path.exists():
                        waveform, sr = torchaudio.load(str(alt_path))
                        break
                else:
                    print(f"Warning: Could not find audio file: {audio_path}")
                    return normalized_similarities, angles

            # 音訊處理成功後的操作
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val

            ax3 = fig.add_subplot(gs[3])
            create_waveform_plot(waveform, sr, 'Waveform', ax3)
            
            ax4 = fig.add_subplot(gs[4])
            create_spectrogram_plot(waveform, sr, 'Spectrogram', ax4)
            
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            print(f"Full audio path: {str(audio_file.absolute())}")
            # 繼續執行而不中斷
    
    plt.subplots_adjust(wspace=0.3, hspace=0.4) # Adjust spacing
    plt.savefig('./only_box/output/new_speaker_analysis_3d.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()
    
    return normalized_similarities, angles

def calculate_average_speaker_similarity(all_utterances, centroids):
    """計算說話者的平均相似度"""
    if len(all_utterances.shape) == 3:
        all_utterances = torch.mean(all_utterances, dim=1)
    elif len(all_utterances.shape) == 1:
        all_utterances = all_utterances.unsqueeze(0)
    
    speaker_embedding = torch.mean(all_utterances, dim=0)
    
    similarities = {}
    for spk, centroid in centroids.items():
        spk_emb = speaker_embedding.view(1, -1)
        cent_emb = centroid.view(1, -1)
        
        sim = calculate_normalized_similarity(spk_emb, cent_emb)
        similarities[spk] = sim
    
    return similarities

def calculate_unified_speaker_similarity(speaker_embeddings, train_centroids):
    """Calculate average similarity for all utterances of a speaker"""
    if len(speaker_embeddings.shape) == 3:
        speaker_mean = torch.mean(speaker_embeddings, dim=0)
    else:
        speaker_mean = speaker_embeddings
        
    if len(speaker_mean.shape) == 2:
        speaker_mean = torch.mean(speaker_mean, dim=0)
    speaker_mean = speaker_mean.unsqueeze(0)
    
    similarities = {}
    for spk, centroid in train_centroids.items():
        sim = calculate_normalized_similarity(speaker_mean, centroid.unsqueeze(0))
        similarities[spk] = sim
    
    print(f"Average similarity scores for speaker:")
    for spk, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        print(f"  To {spk}: {score:.4f}")
        
    return similarities

def calculate_speaker_weights(similarities, angles, threshold=0.3):
    """計算用於語音重建的語者權重"""
    angle_weights = {spk: max(0, 1 - angle/180) for spk, angle in angles.items()}
    
    combined_weights = {}
    for spk in similarities.keys():
        sim = similarities[spk]
        ang_w = angle_weights[spk]
        combined_weights[spk] = (sim * ang_w) ** 0.5
    
    total_weight = sum(combined_weights.values())
    if total_weight > 0:
        normalized_weights = {spk: combined_weights[spk]/total_weight 
                            for spk in combined_weights.keys()}
    else:
        normalized_weights = {spk: 1.0/len(combined_weights) 
                            for spk in combined_weights.keys()}
    
    significant_weights = {
        spk: weight 
        for spk, weight in normalized_weights.items() 
        if weight > threshold
    }

    if significant_weights:
        total_sig_weight = sum(significant_weights.values())
        significant_weights = {
            spk: weight/total_sig_weight 
            for spk, weight in significant_weights.items()
        }
    
    return significant_weights

def analyze_unknown_speaker(new_speaker_embedding, train_centroids, train_labels, audio_path=None):
    """改進的未知語者分析"""
    n_trials = 5
    all_similarities = []
    all_angles = []
    
    for _ in range(n_trials):
        similarities, angles = visualize_speaker_distances(
            new_speaker_embedding, 
            train_centroids, 
            train_labels,
            audio_path=audio_path
        )
        all_similarities.append(similarities)
        all_angles.append(angles)
    
    avg_similarities = {}
    avg_angles = {}
    for spk in train_centroids.keys():
        avg_similarities[spk] = np.mean([s[spk] for s in all_similarities])
        avg_angles[spk] = np.mean([a[spk] for a in all_angles])
    
    reconstruction_weights = calculate_speaker_weights(avg_similarities, avg_angles)
    
    analysis_data = {
        'angles': avg_angles,
        'distances': avg_similarities,
        'angles_list': all_angles,
        'similarities_list': all_similarities,
        'reconstruction_weights': reconstruction_weights,
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("\n穩定化後的分析結果:")
    for spk, weight in sorted(reconstruction_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{spk}: {weight:.3f} (平均角度: {avg_angles[spk]:.1f}°)")
    
    return analysis_data

def main():
    # 加載配置文件
    config = load_config('config.yaml')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 載入訓練好的BYOL模型
    model = load_trained_model(config['byol_model_path'], device)
    
    # 處理訓練集特徵來獲取embeddings
    train_embeddings, train_labels, _ = process_new_speakers(
        model, 
        config['train_features_path'],
        device
    )
    
    # 計算訓練語者的中心點
    centroids = calculate_speaker_centroids(train_embeddings, train_labels)
    
    # 處理驗證集特徵
    valid_embeddings, valid_labels, file_paths = process_new_speakers(
        model, 
        config['valid_features_path'],
        device
    )
    
    # 儲存所有驗證集語者的權重到同一個檔案中
    all_reconstruction_weights = {}
    for i, (emb, label) in enumerate(zip(valid_embeddings, valid_labels)):
        if len(emb.shape) == 2:
            emb = torch.mean(emb, dim=0)

        print(f"\n分析可能的未知語者 {label}:")

        #  獲取音訊路徑並檢查其有效性
        audio_path = None
        if i < len(file_paths) and file_paths[i]:
            base_path = Path('./only_box/dataset')
            file_name = Path(file_paths[i]).name
            possible_paths = [
                base_path / 'valid' / file_name,
                base_path / 'valid' / f"{file_name}.wav",
                base_path / file_name,
                base_path / f"{file_name}.wav"
            ]
            
            for path in possible_paths:
                if path.exists():
                    audio_path = str(path)
                    break
            
            if not audio_path:
                print(f"Warning: Could not find audio file for {label}")

        analysis_data = analyze_unknown_speaker(
            emb,
            centroids,
            train_labels,
            audio_path=audio_path
        )
        all_reconstruction_weights[label] = analysis_data['reconstruction_weights']

    torch.save({
        'train_embeddings': train_embeddings,  # 儲存訓練集的 embeddings
        'train_labels': train_labels,        # 儲存訓練集的 labels
        'reconstruction_weights': all_reconstruction_weights # 儲存所有語者的權重
    }, './only_box/output/all_speakers_weights.pth')

if __name__ == "__main__":
    main()