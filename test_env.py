import torch
import torchaudio
from decoder.pretrained import WavTokenizer

def test_environment():
    # 1. 測試 CUDA
    print("\nTesting CUDA availability:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # 2. 測試 WavTokenizer 載入
    try:
        print("\nTesting WavTokenizer loading:")
        config_path = "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        model_path = "./wavtokenizer_large_speech_320_24k.ckpt"
        
        model = WavTokenizer.from_pretrained0802(config_path, model_path)
        print("WavTokenizer loaded successfully!")
        
        # 移動到 GPU (如果可用)
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to GPU successfully!")
        
        print("\nEnvironment test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return False

if __name__ == "__main__":
    test_environment()
