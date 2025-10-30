
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchaudio
import librosa
import librosa.display
import numpy as np
from datetime import datetime

# =============================
# MiniAudioDataset 類別 (多筆音檔批次處理)
# =============================
class MiniAudioDataset:
    """
    用於多筆音檔的批次 token 化與 padding。
    Args:
        noisy_files: 噪音音檔路徑列表
        clean_files: 乾淨音檔路徑列表
        wavtokenizer: WavTokenizer 實例
        device: torch.device
    Returns:
        noisy_tokens: (B, T) tensor
        clean_tokens: (B, T) tensor
    """
    def __init__(self, noisy_files, clean_files, wavtokenizer, device):
        assert len(noisy_files) == len(clean_files), "noisy/clean 檔案數量不一致"
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.wavtokenizer = wavtokenizer
        self.device = device

    def load_and_tokenize(self):
        noisy_tokens_list = []
        clean_tokens_list = []
        target_sr = 24000
        for noisy_path, clean_path in zip(self.noisy_files, self.clean_files):
            noisy_wav, noisy_sr = torchaudio.load(noisy_path)
            clean_wav, clean_sr = torchaudio.load(clean_path)
            noisy_wav, clean_wav = noisy_wav.to(self.device), clean_wav.to(self.device)
            if noisy_sr != target_sr:
                noisy_wav = torchaudio.functional.resample(noisy_wav, orig_freq=noisy_sr, new_freq=target_sr)
            if clean_sr != target_sr:
                clean_wav = torchaudio.functional.resample(clean_wav, orig_freq=clean_sr, new_freq=target_sr)
            with torch.no_grad():
                _, noisy_tokens = self.wavtokenizer.encode_infer(noisy_wav.unsqueeze(0), bandwidth_id=torch.tensor([0], device=self.device))
                _, clean_tokens = self.wavtokenizer.encode_infer(clean_wav.unsqueeze(0), bandwidth_id=torch.tensor([0], device=self.device))
            noisy_tokens, clean_tokens = noisy_tokens.squeeze(0), clean_tokens.squeeze(0)
            noisy_tokens_list.append(noisy_tokens)
            clean_tokens_list.append(clean_tokens)
        # 對齊到最長長度
        max_len = max(t.shape[1] for t in noisy_tokens_list + clean_tokens_list)
        noisy_batch = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=0) for t in noisy_tokens_list], dim=0)
        clean_batch = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=0) for t in clean_tokens_list], dim=0)
        return noisy_batch, clean_batch

# ===================================================================
# ⭐️ 1. 主要配置區域 - 請檢查這些路徑是否正確 ⭐️
# ===================================================================

# --- 添加項目根目錄到 Python 路徑 ---
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
print(f"項目根目錄已添加到 sys.path: {PROJECT_ROOT}")

# --- 模型和 Tokenizer 路徑 ---
WAVTOKENIZER_CONFIG = '/home/sbplab/ruizi/c_code/config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
WAVTOKENIZER_CHECKPOINT = '/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt'

# --- 選擇要過擬合的單一音檔 ---
NOISY_AUDIO_FILES = [
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_boy1_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_boy3_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_boy4_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_boy5_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_boy6_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_boy9_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_boy10_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_girl2_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_girl3_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_girl4_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_girl6_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_girl7_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_girl8_box_LDV_001.wav',
    '/home/sbplab/ruizi/c_code/data/raw/box/nor_girl11_box_LDV_001.wav'
]
CLEAN_AUDIO_FILES = [
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_boy1_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_boy3_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_boy4_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_boy5_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_boy6_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_boy9_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_boy10_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_girl2_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_girl3_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_girl4_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_girl6_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_girl7_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_girl8_clean_001.wav',
    '/home/sbplab/ruizi/c_code/data/clean/box2/nor_girl11_clean_001.wav'
]

# --- 訓練參數 ---
NUM_EPOCHS = 200
LEARNING_RATE = 3e-4
SAVE_INTERVAL = 50 # 每 50 個 epoch 保存一次音檔和頻譜圖

# --- 模型參數 ---
CODEBOOK_DIM = 512
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 4
DROPOUT = 0.0

# ===================================================================
# ⭐️ 2. 導入你的模型和工具 - 請確保這些導入有效 ⭐️
# ===================================================================
try:
    from token_denoising_transformer import TokenDenoisingTransformer
    from decoder.pretrained import WavTokenizer
except ImportError as e:
    print(f"❌ 導入錯誤: {e}")
    sys.exit(1)

# ===================================================================
# ⭐️ 3. 輔助函數 (用於儲存和繪圖) ⭐️
# ===================================================================
def plot_spectrograms(noisy_audio, pred_audio, clean_audio, save_path):
    """繪製三個音頻的頻譜圖"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    sr = 24000  # WavTokenizer 的採樣率是 24k
    
    for idx, (audio, title) in enumerate([
        (noisy_audio, 'Noisy Audio'),
        (pred_audio, 'Predicted Audio'),
        (clean_audio, 'Clean Audio (Target)')
    ]):
        D = librosa.stft(audio)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', ax=axes[idx], cmap='viridis')
        axes[idx].set_title(title)
        fig.colorbar(img, ax=axes[idx], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def save_debug_outputs(wavtokenizer, model, noisy_tokens, clean_tokens, epoch, output_dir, device):
    """在給定 epoch，預測、解碼並儲存音檔和頻譜圖"""
    model.eval() # 切換到評估模式
    
    with torch.no_grad():
        # 預測 tokens
        pred_logits = model(noisy_tokens, return_logits=True)
        pred_tokens = pred_logits.argmax(dim=-1)

    # 創建該 epoch 的輸出目錄
    epoch_dir = output_dir / f'epoch_{epoch}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    sr = 24000 # WavTokenizer sample rate
    B = noisy_tokens.shape[0]
    for i in range(B):
        # 取出每一筆 (T,)
        noisy_tok = noisy_tokens[i].unsqueeze(0)
        pred_tok = pred_tokens[i].unsqueeze(0)
        clean_tok = clean_tokens[i].unsqueeze(0)
        with torch.no_grad():
            noisy_features = wavtokenizer.codes_to_features(noisy_tok)
            pred_features = wavtokenizer.codes_to_features(pred_tok)
            clean_features = wavtokenizer.codes_to_features(clean_tok)
            if noisy_features.dim() == 4: noisy_features = noisy_features.squeeze(2)
            if pred_features.dim() == 4: pred_features = pred_features.squeeze(2)
            if clean_features.dim() == 4: clean_features = clean_features.squeeze(2)
            noisy_audio = wavtokenizer.decode(noisy_features, bandwidth_id=torch.tensor([0], device=device)).cpu()
            pred_audio = wavtokenizer.decode(pred_features, bandwidth_id=torch.tensor([0], device=device)).cpu()
            clean_audio = wavtokenizer.decode(clean_features, bandwidth_id=torch.tensor([0], device=device)).cpu()
            if noisy_audio.dim() == 1:
                noisy_audio = noisy_audio.unsqueeze(0)
            if pred_audio.dim() == 1:
                pred_audio = pred_audio.unsqueeze(0)
            if clean_audio.dim() == 1:
                clean_audio = clean_audio.unsqueeze(0)
        # 保存音檔
        torchaudio.save(epoch_dir / f'noisy_{i}.wav', noisy_audio, sr)
        torchaudio.save(epoch_dir / f'predicted_{i}.wav', pred_audio, sr)
        torchaudio.save(epoch_dir / f'clean_{i}.wav', clean_audio, sr)
        # 保存頻譜圖
        plot_spectrograms(
            noisy_audio.squeeze(0).numpy(),
            pred_audio.squeeze(0).numpy(),
            clean_audio.squeeze(0).numpy(),
            epoch_dir / f'spectrogram_{i}.png'
        )
    print(f"✅ Epoch {epoch}: 已儲存所有音檔和頻譜圖至 {epoch_dir}")
    model.train() # 切換回訓練模式

# ===================================================================


def overfit_multi_samples():
    """
    主函數：多筆音檔過擬合測試 (批次)
    """
    # 創建唯一的實驗輸出目錄
    exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'results/overfit_test_{exp_time}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"所有輸出將保存到: {output_dir}")

    # 1. 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 2. 載入 WavTokenizer
    print("\n[1/7] 載入 WavTokenizer...")
    config_path = PROJECT_ROOT / WAVTOKENIZER_CONFIG
    checkpoint_path = PROJECT_ROOT / WAVTOKENIZER_CHECKPOINT
    wavtokenizer = WavTokenizer.from_pretrained0802(str(config_path), str(checkpoint_path)).to(device).eval()
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    print(f"✅ WavTokenizer 載入成功! Codebook 形狀: {codebook.shape}")

    # 3. 準備多筆數據樣本 (批次)
    print("\n[2/7] 準備多筆數據樣本...")
    dataset = MiniAudioDataset(NOISY_AUDIO_FILES, CLEAN_AUDIO_FILES, wavtokenizer, device)
    noisy_tokens, clean_tokens = dataset.load_and_tokenize()  # (B, T)
    # 自動 squeeze 多餘維度，確保 (B, T)
    if noisy_tokens.dim() > 2:
        noisy_tokens = noisy_tokens.squeeze()
    if clean_tokens.dim() > 2:
        clean_tokens = clean_tokens.squeeze()
    print(f"✅ Token 化完成! Noisy: {noisy_tokens.shape}, Clean: {clean_tokens.shape}")

    # 4. 創建模型
    print("\n[3/7] 創建 Token Denoising Transformer...")
    model = TokenDenoisingTransformer(codebook=codebook, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型創建成功! 可訓練參數: {trainable_params:,}")

    # 5. 設置損失函數和優化器
    print("\n[4/7] 設置損失函數和優化器...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"✅ 損失函數: CrossEntropyLoss, 優化器: AdamW, 學習率: {LEARNING_RATE}")

    # 6. 開始訓練循環
    print("\n[5/7] 開始訓練循環...")
    model.train()
    losses, accuracies = [], []

    progress_bar = tqdm(range(NUM_EPOCHS), desc="Overfitting Multi Samples")
    for epoch in progress_bar:
        # 在第0個 epoch 和每個保存間隔，保存一次輸出 (整個批次)
        if epoch == 0 or (epoch + 1) % SAVE_INTERVAL == 0:
            save_debug_outputs(wavtokenizer, model, noisy_tokens, clean_tokens, epoch, output_dir, device)

        logits = model(noisy_tokens, return_logits=True)  # (B, T, C)
        # 修正 inference tensor backward 問題
        loss = criterion(logits.view(-1, logits.shape[-1]).clone(), clean_tokens.view(-1).clone())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = (logits.argmax(dim=-1) == clean_tokens).sum().item() / clean_tokens.numel() * 100

        losses.append(loss.item())
        accuracies.append(accuracy)
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Accuracy': f'{accuracy:.2f}%'})

        if accuracy > 99.9:
            print(f"\n🎉 在 Epoch {epoch+1} 完美擬合！")
            break

    print("\n[6/7] 訓練完成！")
    # 保存最後一個 epoch 的結果 (整個批次)
    save_debug_outputs(wavtokenizer, model, noisy_tokens, clean_tokens, NUM_EPOCHS, output_dir, device)

    # 7. 儲存模型權重 checkpoint
    checkpoint_path = output_dir / f'weights_overfit_multi_samples_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'accuracies': accuracies
    }, checkpoint_path)
    print(f"✅ 模型權重已儲存於 {checkpoint_path}")

    # 8. 繪製並儲存 Loss 圖
    print("\n[7/7] 繪製並儲存 Loss/Accuracy 曲線...")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color=color)
    ax1.plot(losses, color=color, label='Loss', marker='.')
    ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(accuracies, color=color, label='Accuracy', marker='.')
    ax2.tick_params(axis='y', labelcolor=color); ax2.set_ylim(0, 105)
    fig.tight_layout()
    plt.title('Multi Sample Overfitting Test Results')
    loss_curve_path = output_dir / 'loss_accuracy_curve.png'
    plt.savefig(loss_curve_path, dpi=150)
    print(f"✅ Loss 曲線圖已保存為 {loss_curve_path}")

if __name__ == '__main__':
    overfit_multi_samples()
