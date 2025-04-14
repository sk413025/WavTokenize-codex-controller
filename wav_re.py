import os
from pathlib import Path
from tqdm import tqdm
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime


def calculate_average_tsne(features, n_runs=10):
    """Calculate average T-SNE over multiple runs"""
    tsne_results = []
    features_np = features.detach().cpu().numpy()
    
    for i in range(n_runs):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=i)
        result = tsne.fit_transform(features_np)
        tsne_results.append(result)
    
    return np.mean(tsne_results, axis=0)

def visualize_tsne(features, save_path, filename):
    """Generate T-SNE visualization for a single audio file"""
    plt.figure(figsize=(10, 10))
    tsne_avg = calculate_average_tsne(features)
    
    plt.scatter(tsne_avg[:, 0], tsne_avg[:, 1], alpha=0.6)
    plt.title(f'T-SNE Visualization for {filename}\n(Averaged over 10 runs)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.savefig(save_path)
    plt.close()

def process_folder(input_dir, output_dir, config_path, model_path):
    device = torch.device('cpu')
    
    print("Loading model...")
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    tsne_dir = os.path.join(output_dir, 'tsne_plots')
    os.makedirs(tsne_dir, exist_ok=True)
    
    input_files = list(Path(input_dir).rglob("*.wav"))
    total_files = len(input_files)
    print(f"\nFound {total_files} WAV files to process")
    
    for idx, input_path in enumerate(tqdm(input_files, desc="Processing files")):
        try:
            # Load and process audio
            wav, sr = torchaudio.load(str(input_path))
            wav = convert_audio(wav, sr, 24000, 1)
            wav = wav.to(device)
            
            # Extract features
            bandwidth_id = torch.tensor([0])
            features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
            print(f"\nFeatures shape: {features.shape}")
            print(f"Discrete code: {discrete_code.shape}")
            
            # Generate T-SNE plot before decoding
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tsne_filename = f"tsne_{input_path.stem}_{timestamp}.png"
            tsne_path = os.path.join(tsne_dir, tsne_filename)
            
            # Reshape features if needed and generate T-SNE
            features_2d = features.squeeze().view(-1, features.size(-1))
            print(f"\nFeatures shape: {features_2d.shape}")
            visualize_tsne(features_2d, tsne_path, input_path.name)
            print(f"\nSaved T-SNE plot to: {tsne_path}")
            
            # Process audio output
            rel_path = input_path.relative_to(input_dir)
            output_path = Path(output_dir) / f"processed_{rel_path}"
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Decode and save audio
            audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
            torchaudio.save(
                output_path,
                audio_out,
                sample_rate=24000,
                encoding='PCM_S',
                bits_per_sample=16
            )
            
            print(f"Processed [{idx + 1}/{total_files}]: {input_path}")
            print(f"Saved to: {output_path}")
            
        except Exception as e:
            print(f"\nError processing {input_path}: {str(e)}")
            continue
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    config_path = "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "./wavtokenizer_large_speech_320_24k.ckpt"
    input_dir = "./wav_in"
    output_dir = "./wav_out"
    
    process_folder(input_dir, output_dir, config_path, model_path)