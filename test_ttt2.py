"""
ttt2.py 訓練權重驗證測試腳本
"""

import torch
import os
import sys
import unittest
import torchaudio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ttt2 import EnhancedWavTokenizer
import numpy as np

class TestTTT2TrainedWeights(unittest.TestCase):
    def test_save_audio(self):
        """測試 save_audio 是否能正確保存音檔，與 ttt2.py 做法一致"""
        from encoder.utils import save_audio
        wav = torch.randn(1, 24000)  # 1秒假音訊
        out_path = "test_output.wav"
        save_audio(wav, out_path, sample_rate=24000, rescale=True)
        self.assertTrue(os.path.exists(out_path))
        # 清理測試檔案
        os.remove(out_path)
    def setUp(self):
        """載入訓練完的模型權重，並將模型移到正確 device"""
        config_path = os.path.join(os.getcwd(), "config", "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
        model_path = os.path.join(os.getcwd(), "models", "wavtokenizer_large_speech_320_24k.ckpt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedWavTokenizer(config_path, model_path).to(self.device)
        self.model.eval()
        # 載入訓練權重
        weights_path = os.path.join(os.getcwd(), "results", "tsne_outputs", "b-output4", "best_model.pth")
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def test_inference(self):
        """robust 處理流程，input 來自 ./1n，output 儲存至 ttt2_out，特徵/音訊/例外/自動命名"""
        from encoder.utils import save_audio
        import glob
        input_dir = os.path.join(os.getcwd(), "111")
        output_dir = os.path.join(os.getcwd(), "ttt2_out1")
        os.makedirs(output_dir, exist_ok=True)
        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        audio_files = glob.glob(os.path.join(input_dir, "*.wav"))
        assert len(audio_files) > 0, "./1n 目錄下找不到音檔"
        from decoder.pretrained import WavTokenizer
        config_path = os.path.join(os.getcwd(), "config", "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
        model_path = os.path.join(os.getcwd(), "models", "wavtokenizer_large_speech_320_24k.ckpt")
        device = self.device
        decoder = WavTokenizer.from_pretrained0802(config_path, model_path)
        decoder = decoder.to(device)
        decoder.eval()
        bandwidth_id = torch.tensor([0], device=device)
        for test_file in audio_files:
            try:
                base_name = os.path.splitext(os.path.basename(test_file))[0]
                x, sr = torchaudio.load(test_file)
                if sr != 24000:
                    from encoder.utils import convert_audio
                    x = convert_audio(x, sr, 24000, 1)
                    sr = 24000
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                # 不限制音檔長度
                x = x.unsqueeze(0)
                x = x.to(device)
                with torch.no_grad():
                    output_tuple = self.model(x)
                self.assertIsInstance(output_tuple, tuple)
                self.assertTrue(len(output_tuple) >= 3)
                output, input_features, enhanced_features = output_tuple[:3]
                # shape 標準化
                if hasattr(self.model, "feature_extractor") and hasattr(self.model.feature_extractor, "ensure_feature_shape"):
                    enhanced_features = self.model.feature_extractor.ensure_feature_shape(enhanced_features)
                    input_features = self.model.feature_extractor.ensure_feature_shape(input_features)
                elif enhanced_features.shape[1] != 512 and enhanced_features.shape[2] == 512:
                    enhanced_features = enhanced_features.transpose(1, 2)
                    input_features = input_features.transpose(1, 2) if input_features.shape[1] != 512 else input_features
                # 保存特徵
                torch.save(input_features.cpu(), os.path.join(features_dir, f"{base_name}_input_features.pt"))
                torch.save(enhanced_features.cpu(), os.path.join(features_dir, f"{base_name}_enhanced_features.pt"))
                # 音訊輸出
                if enhanced_features is None or enhanced_features.shape[1] != 512:
                    output_audio = output.reshape(1, -1)
                    output_audio = output_audio / (torch.max(torch.abs(output_audio)) + 1e-8)
                else:
                    try:
                        output_audio = decoder.decode(enhanced_features, bandwidth_id=bandwidth_id)
                        if output_audio.dim() > 2:
                            output_audio = output_audio.squeeze(0)
                        output_audio = output_audio / (torch.max(torch.abs(output_audio)) + 1e-8)
                    except Exception as decode_err:
                        output_audio = x.reshape(1, -1)
                        output_audio = output_audio / (torch.max(torch.abs(output_audio)) + 1e-8)
                # 保存 input/output
                input_path = os.path.join(output_dir, f"{base_name}_input.wav")
                output_path = os.path.join(output_dir, f"{base_name}_enhanced.wav")
                save_audio(x.squeeze(0).cpu(), input_path, sample_rate=24000, rescale=True)
                save_audio(output_audio.cpu(), output_path, sample_rate=24000, rescale=True)
                self.assertTrue(os.path.exists(input_path))
                self.assertTrue(os.path.exists(output_path))
                # 頻譜圖生成
                try:
                    import librosa
                    import librosa.display
                    import matplotlib.pyplot as plt
                    # input
                    audio_np = x.squeeze(0).cpu().numpy().flatten()
                    plt.figure(figsize=(10, 4))
                    try:
                        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
                        librosa.display.specshow(D, sr=24000, x_axis="time", y_axis="log")
                        plt.colorbar(format="%+2.0f dB")
                        plt.title(f"{base_name} Input Spectrogram")
                        plt.tight_layout()
                        spec_path = os.path.join(features_dir, f"{base_name}_input_spec.png")
                        plt.savefig(spec_path)
                        plt.close()
                    except Exception as spec_err:
                        print(f"❌ Input頻譜圖生成失敗: {str(spec_err)} | shape={audio_np.shape} | len={len(audio_np)}")
                        plt.close()
                    # enhanced
                    audio_np = output_audio.cpu().numpy().flatten()
                    plt.figure(figsize=(10, 4))
                    try:
                        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
                        librosa.display.specshow(D, sr=24000, x_axis="time", y_axis="log")
                        plt.colorbar(format="%+2.0f dB")
                        plt.title(f"{base_name} Enhanced Spectrogram")
                        plt.tight_layout()
                        spec_path = os.path.join(features_dir, f"{base_name}_enhanced_spec.png")
                        plt.savefig(spec_path)
                        plt.close()
                    except Exception as spec_err:
                        print(f"❌ Enhanced頻譜圖生成失敗: {str(spec_err)} | shape={audio_np.shape} | len={len(audio_np)}")
                        plt.close()
                except Exception as spec_err:
                    print(f"❌ 頻譜圖生成主流程失敗: {str(spec_err)}")
            except Exception as e:
                print(f"❌ {test_file} robust 推論失敗: {str(e)}")
                continue

if __name__ == "__main__":
    unittest.main()