import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio
import numpy as np
from tqdm import tqdm
import re
# 添加新的導入
import librosa
import matplotlib.pyplot as plt

def process_audio(audio_path, target_sr=24000, normalize=True, target_length=24000*3):
    """與 try.py 完全一致的音頻處理"""
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, target_sr, 1)  # [1, T]
    if normalize:
        wav = wav / wav.abs().max()
    
    # 確保音頻長度固定
    current_length = wav.size(-1)
    if (current_length > target_length):
        wav = wav[..., :target_length]
    elif (current_length < target_length):
        padding = torch.zeros(1, target_length - current_length)
        wav = torch.cat([wav, padding], dim=-1)
    
    return wav  # 保持 [1, T] 形狀

class AudioDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.paired_files = []
        
        # 收集輸入文件並找到對應的目標文件
        for input_file in os.listdir(input_dir):
            if not input_file.endswith('.wav'):
                continue
                
            # 解析輸入文件名 (例如: nor_boy1_box_LDV_001.wav)
            parts = input_file.split('_')
            if len(parts) >= 5:
                try:
                    speaker = parts[1]      # boy1
                    material = parts[2]     # box
                    file_num = parts[4]     # 001.wav
                    
                    # 構建目標文件名 (例如: nor_boy1_clean_LDV_001.wav)
                    target_file = f"{parts[0]}_{speaker}_clean_{file_num}"
                    target_path = os.path.join(target_dir, target_file)
                    
                    if os.path.exists(target_path):
                        self.paired_files.append({
                            'input': input_file,
                            'target': target_file,
                            'speaker': speaker,
                            'material': material,
                            'number': file_num
                        })
                    else:
                        print(f"Warning: No matching target file found for {input_file}")
                except Exception as e:
                    print(f"Error processing filename {input_file}: {str(e)}")
                    continue
        
        # 顯示發現的材質和配對數量
        materials = sorted(list(set(pair['material'] for pair in self.paired_files)))
        print(f"\nFound materials: {materials}")
        print(f"Total paired files: {len(self.paired_files)}")
        
    def __len__(self):
        return len(self.paired_files)
        
    def __getitem__(self, idx):
        pair = self.paired_files[idx]
        input_path = os.path.join(self.input_dir, pair['input'])
        target_path = os.path.join(self.target_dir, pair['target'])
        
        try:
            input_wav = process_audio(input_path)
            target_wav = process_audio(target_path)
            return input_wav, target_wav
        except Exception as e:
            print(f"Error loading audio files {pair['input']} -> {pair['target']}: {str(e)}")
            return torch.zeros(1, 24000*3), torch.zeros(1, 24000*3)

def train_model(model, train_loader, optimizer, device, save_dir, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_wav, target_wav) in enumerate(tqdm(train_loader)):
            # Move to device and prepare tensors
            input_wav = input_wav.to(device)
            target_wav = target_wav.to(device)
            
            # Reshape tensors
            input_wav = input_wav.view(input_wav.size(0), -1)  # [B, T]
            target_wav = target_wav.view(input_wav.size(0), -1)  # [B, T]
            
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            # bandwidth_id = torch.tensor([2], device=device)
            bandwidth_id = torch.randint(0, 3, (1,)).to(device)
            try:
                with torch.set_grad_enabled(True):
                    # Encode
                    features, _ = model.encode_infer(input_wav, bandwidth_id=bandwidth_id)
                    
                    # Decode - call decoder directly without bandwidth_id
                    output = model.feature_extractor.encodec.decoder(features)
                    
                    # Reshape output to match target dimensions
                    output = output.view(target_wav.shape)  # [B, T]
                    
                    # Compute loss
                    loss = torch.nn.functional.mse_loss(output.float(), target_wav.float())
                    
                    # Backward pass
                    loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
            except Exception as e:
                print(f"Error during training: {e}")
                continue
                
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.6f}')
        
        # Save checkpoints and samples
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        audio_dir = os.path.join(save_dir, f'epoch_{epoch+1}_samples')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        # Save model
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        # Generate and save example outputs
        model.eval()
        with torch.no_grad():
            for i, (input_wav, target_wav) in enumerate(train_loader):
                if i >= 3:  # Only save first 3 samples
                    break
                    
                input_wav = input_wav.to(device)
                input_wav = input_wav.view(input_wav.size(0), -1)
                
                bandwidth_id = torch.tensor([0]).to(device)
                features, _ = model.encode_infer(input_wav, bandwidth_id=bandwidth_id)
                output = model.feature_extractor.encodec.decoder(features)
                output = output.view(input_wav.shape)  # Reshape to [B, T]
                
                # Save audio files - ensure correct dimensions for torchaudio.save
                save_sample(input_wav, output, target_wav, epoch, i, save_dir, device)
                    
        model.train()

def plot_spectrograms(wav, save_path, device):
    """繪製頻譜圖並保存"""
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(wav.cpu().numpy().squeeze())),
        ref=np.max
    )
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(save_path)
    plt.close()

def save_sample(input_wav, output, target_wav, epoch, batch_idx, save_dir, device):
    """保存音頻樣本和頻譜圖"""
    try:
        audio_dir = os.path.join(save_dir, f'epoch_{epoch+1}_samples')
        os.makedirs(audio_dir, exist_ok=True)
        
        for j in range(output.size(0)):
            try:
                # 確保所有張量都先detach並移至CPU
                with torch.no_grad():
                    output_audio = output[j].detach().cpu().reshape(1, -1)
                    input_audio = input_wav[j].detach().cpu().reshape(1, -1)
                    target_audio = target_wav[j].detach().cpu().reshape(1, -1)
                
                # 保存音頻
                output_path = os.path.join(audio_dir, f'batch_{batch_idx}_sample_{j+1}_output.wav')
                input_path = os.path.join(audio_dir, f'batch_{batch_idx}_sample_{j+1}_input.wav')
                target_path = os.path.join(audio_dir, f'batch_{batch_idx}_sample_{j+1}_target.wav')
                
                torchaudio.save(output_path, output_audio, 24000)
                torchaudio.save(input_path, input_audio, 24000)
                torchaudio.save(target_path, target_audio, 24000)
                
                # 保存頻譜圖
                plot_spectrograms(output_audio, 
                                os.path.join(audio_dir, f'batch_{batch_idx}_sample_{j+1}_output_spec.png'),
                                device)
                plot_spectrograms(input_audio,
                                os.path.join(audio_dir, f'batch_{batch_idx}_sample_{j+1}_input_spec.png'),
                                device)
                plot_spectrograms(target_audio,
                                os.path.join(audio_dir, f'batch_{batch_idx}_sample_{j+1}_target_spec.png'),
                                device)
                
            except Exception as e:
                print(f"Error saving sample {j+1} from batch {batch_idx}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in save_sample function: {str(e)}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "./wavtokenizer_large_speech_320_24k.ckpt"
    
    # Initialize model with gradient tracking enabled
    model = WavTokenizer.from_pretrained0802(config_path, model_path)
    model = model.to(device)
    model.train()
    
    # Force enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad_(True)
    
    # Verify model is in training mode
    print("\nModel state:")
    print(f"Training mode: {model.training}")
    print(f"Device: {next(model.parameters()).device}")
    
    # Setup training
    dataset = AudioDataset("./box", "./box2")
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    
    # 設置輸出目錄
    save_dir = './finetune_output'
    os.makedirs(save_dir, exist_ok=True)
    
    # 開始訓練
    train_model(model, train_loader, optimizer, device, save_dir)

if __name__ == "__main__":
    main()