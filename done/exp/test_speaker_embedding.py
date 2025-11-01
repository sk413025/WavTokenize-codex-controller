"""
測試 ECAPA-TDNN Speaker Embedding 的區分能力

展示:
1. 不同 speaker 的 embedding 分布
2. 相同 speaker 不同音檔的相似度
3. 不同 speaker 之間的距離
4. t-SNE 視覺化
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from speaker_encoder import create_speaker_encoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_audio_samples(data_dir='../data/clean/box2', max_per_speaker=3, min_duration=1.0):
    """載入音頻樣本
    
    Args:
        data_dir: 音頻目錄
        max_per_speaker: 每個 speaker 最多取幾個音檔
        min_duration: 最小音頻長度（秒）- 過短的音頻會影響 speaker 特徵提取
    
    Returns:
        audios: list of (audio_tensor, speaker_id, filename)
    """
    data_path = Path(data_dir)
    audio_files = list(data_path.glob('*.wav'))
    
    # 按 speaker 分組
    speaker_files = {}
    for audio_file in audio_files:
        # 檔名格式: nor_boy10_clean_001.wav
        parts = audio_file.stem.split('_')
        if len(parts) >= 2:
            speaker = parts[1]  # boy10, girl9, etc.
        else:
            speaker = 'unknown'
        
        if speaker not in speaker_files:
            speaker_files[speaker] = []
        speaker_files[speaker].append(audio_file)
    
    # 每個 speaker 取前 N 個
    samples = []
    for speaker, files in speaker_files.items():
        for i, file in enumerate(files[:max_per_speaker]):
            # 載入音頻
            audio, sr = torchaudio.load(file)
            audio = audio[0]  # 取單聲道
            
            # 過濾太短的音頻
            duration = len(audio) / sr
            if duration < min_duration:
                continue
            
            samples.append({
                'audio': audio,
                'speaker': speaker,
                'filename': file.name,
                'sample_idx': i
            })
    
    return samples


def compute_similarity(emb1, emb2):
    """計算餘弦相似度"""
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).item()


def compute_embeddings(encoder, samples, device='cuda', normalize=False):
    """提取所有樣本的 embeddings
    
    Args:
        encoder: Speaker encoder
        samples: 音頻樣本列表
        device: 運算裝置
        normalize: 是否額外進行 L2 正規化
    
    Returns:
        embeddings: list of {embedding, speaker, filename}
    """
    embeddings = []
    
    print(f"提取 {len(samples)} 個音檔的 speaker embeddings...")
    
    for sample in samples:
        audio = sample['audio'].to(device).unsqueeze(0)  # (1, T)
        
        with torch.no_grad():
            emb = encoder(audio)  # (1, D)
            
            # 額外的正規化（預設關閉）
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        
        embeddings.append({
            'embedding': emb.cpu().squeeze(0),  # (D,)
            'speaker': sample['speaker'],
            'filename': sample['filename']
        })
    
    return embeddings


def analyze_embeddings(embeddings):
    """分析 embeddings 的統計特性"""
    print("\n" + "="*80)
    print("Embedding 統計分析")
    print("="*80)
    
    # 按 speaker 分組
    speaker_groups = {}
    for item in embeddings:
        speaker = item['speaker']
        if speaker not in speaker_groups:
            speaker_groups[speaker] = []
        speaker_groups[speaker].append(item['embedding'])
    
    # 1. 同一 speaker 內部相似度
    print("\n1️⃣ 同一 speaker 不同音檔之間的相似度 (應該很高):")
    within_speaker_sims = []
    
    for speaker, embs in speaker_groups.items():
        if len(embs) < 2:
            continue
        
        sims = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                sim = compute_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0))
                sims.append(sim)
        
        avg_sim = np.mean(sims)
        within_speaker_sims.extend(sims)
        print(f"  - {speaker}: 平均相似度 = {avg_sim:.4f} (範圍: {min(sims):.4f} - {max(sims):.4f})")
    
    # 2. 不同 speaker 之間相似度
    print("\n2️⃣ 不同 speaker 之間的相似度 (應該很低):")
    between_speaker_sims = []
    
    speakers = list(speaker_groups.keys())
    for i in range(len(speakers)):
        for j in range(i+1, len(speakers)):
            speaker1, speaker2 = speakers[i], speakers[j]
            embs1 = speaker_groups[speaker1]
            embs2 = speaker_groups[speaker2]
            
            sims = []
            for emb1 in embs1:
                for emb2 in embs2:
                    sim = compute_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                    sims.append(sim)
            
            avg_sim = np.mean(sims)
            between_speaker_sims.extend(sims)
            print(f"  - {speaker1} vs {speaker2}: 平均相似度 = {avg_sim:.4f}")
    
    # 3. 總結
    print("\n3️⃣ 總結:")
    print(f"  - 同一 speaker 內部平均相似度: {np.mean(within_speaker_sims):.4f} ± {np.std(within_speaker_sims):.4f}")
    print(f"  - 不同 speaker 之間平均相似度: {np.mean(between_speaker_sims):.4f} ± {np.std(between_speaker_sims):.4f}")
    print(f"  - 區分度 (差異): {np.mean(within_speaker_sims) - np.mean(between_speaker_sims):.4f}")
    
    if np.mean(within_speaker_sims) > np.mean(between_speaker_sims):
        print(f"  ✅ ECAPA-TDNN 能有效區分不同 speaker！")
    else:
        print(f"  ❌ 警告：區分能力不足")
    
    return within_speaker_sims, between_speaker_sims, speaker_groups


def plot_similarity_distribution(within_sims, between_sims, output_path):
    """Plot similarity distribution"""
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(within_sims, bins=30, alpha=0.6, label='Same Speaker', color='green', density=True)
    plt.hist(between_sims, bins=30, alpha=0.6, label='Different Speakers', color='red', density=True)
    
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Speaker Embedding Similarity Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Mark averages
    plt.axvline(np.mean(within_sims), color='green', linestyle='--', linewidth=2, 
                label=f'Same Speaker Avg: {np.mean(within_sims):.3f}')
    plt.axvline(np.mean(between_sims), color='red', linestyle='--', linewidth=2,
                label=f'Different Speakers Avg: {np.mean(between_sims):.3f}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✅ Similarity distribution plot saved: {output_path}")


def plot_tsne(embeddings, output_path):
    """Visualize embeddings using t-SNE"""
    print("\n" + "="*80)
    print("t-SNE Visualization")
    print("="*80)
    
    # Prepare data
    emb_matrix = torch.stack([item['embedding'] for item in embeddings]).numpy()
    speakers = [item['speaker'] for item in embeddings]
    
    # PCA preprocessing (speed up t-SNE)
    if emb_matrix.shape[1] > 50:
        print("Using PCA preprocessing...")
        pca = PCA(n_components=50)
        emb_matrix = pca.fit_transform(emb_matrix)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    emb_2d = tsne.fit_transform(emb_matrix)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Assign colors for each speaker
    unique_speakers = sorted(set(speakers))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_speakers)))
    speaker_colors = {speaker: colors[i] for i, speaker in enumerate(unique_speakers)}
    
    # Plot each speaker
    for speaker in unique_speakers:
        indices = [i for i, s in enumerate(speakers) if s == speaker]
        plt.scatter(
            emb_2d[indices, 0], 
            emb_2d[indices, 1],
            c=[speaker_colors[speaker]], 
            label=speaker,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Speaker Embeddings t-SNE Visualization', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ t-SNE visualization saved: {output_path}")


def plot_similarity_matrix(speaker_groups, output_path):
    """Plot speaker similarity matrix"""
    print("\n" + "="*80)
    print("Speaker Similarity Matrix")
    print("="*80)
    
    speakers = sorted(speaker_groups.keys())
    n = len(speakers)
    
    # Compute average similarity for each speaker pair
    sim_matrix = np.zeros((n, n))
    
    for i, speaker1 in enumerate(speakers):
        for j, speaker2 in enumerate(speakers):
            embs1 = speaker_groups[speaker1]
            embs2 = speaker_groups[speaker2]
            
            sims = []
            for emb1 in embs1:
                for emb2 in embs2:
                    sim = compute_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                    sims.append(sim)
            
            sim_matrix[i, j] = np.mean(sims)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim_matrix, 
        annot=True, 
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=speakers,
        yticklabels=speakers,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Speaker Embedding Similarity Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Speaker', fontsize=12)
    plt.ylabel('Speaker', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Similarity matrix saved: {output_path}")


def main():
    # 設定
    data_dir = '../../data/clean/box2'  # 使用 clean 音檔測試
    output_dir = Path('./speaker_embedding_test')
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 1. 載入音檔
    print("\n" + "="*80)
    print("載入音檔樣本")
    print("="*80)
    samples = load_audio_samples(data_dir, max_per_speaker=3)
    print(f"載入 {len(samples)} 個音檔")
    
    # 統計 speaker 分布
    speakers = {}
    for sample in samples:
        speaker = sample['speaker']
        speakers[speaker] = speakers.get(speaker, 0) + 1
    
    print(f"Speaker 分布:")
    for speaker, count in sorted(speakers.items()):
        print(f"  - {speaker}: {count} 個音檔")
    
    # 2. 創建 encoder (測試兩種)
    print("\n" + "="*80)
    print("創建 Speaker Encoder")
    print("="*80)
    
    # 使用 ECAPA-TDNN (效果更好)
    print("Using ECAPA-TDNN...")
    encoder = create_speaker_encoder(model_type='ecapa', freeze=True, output_dim=256)
    # ECAPA 保持在 CPU 以避免設備衝突
    encoder.eval()
    
    # 3. 提取 embeddings (在 CPU 上進行)
    embeddings = compute_embeddings(encoder, samples, device='cpu')
    print(f"✅ 提取完成！每個音檔的 embedding 維度: {embeddings[0]['embedding'].shape}")
    
    # 4. 分析 embeddings
    within_sims, between_sims, speaker_groups = analyze_embeddings(embeddings)
    
    # 5. 視覺化
    print("\n" + "="*80)
    print("生成視覺化圖表")
    print("="*80)
    
    # 相似度分布
    plot_similarity_distribution(
        within_sims, 
        between_sims, 
        output_dir / 'similarity_distribution.png'
    )
    
    # t-SNE
    plot_tsne(embeddings, output_dir / 'tsne_visualization.png')
    
    # 相似度矩陣
    plot_similarity_matrix(speaker_groups, output_dir / 'similarity_matrix.png')
    
    print("\n" + "="*80)
    print("✅ 所有測試完成！")
    print("="*80)
    print(f"結果保存在: {output_dir}")
    print(f"  - similarity_distribution.png: 相似度分布圖")
    print(f"  - tsne_visualization.png: t-SNE 視覺化")
    print(f"  - similarity_matrix.png: Speaker 相似度矩陣")


if __name__ == '__main__':
    main()
