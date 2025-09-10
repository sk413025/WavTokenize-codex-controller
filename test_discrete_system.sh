#!/bin/bash

echo "🧪 離散 Token 降噪系統快速測試"
echo "=================================="

# 檢查必要文件
echo "1️⃣ 檢查核心文件..."
files_to_check=(
    "discrete_token_denoising.py"
    "discrete_inference.py" 
    "run_discrete_token_experiment.sh"
    "config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    "models/wavtokenizer_large_speech_320_24k.ckpt"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (缺失)"
    fi
done

# 檢查 Python 導入
echo ""
echo "2️⃣ 檢查 Python 模組..."
python -c "
try:
    import torch
    import torchaudio
    import numpy as np
    import yaml
    import tqdm
    from sklearn.manifold import TSNE
    import matplotlib
    print('✅ 所有必要的 Python 套件都可用')
except ImportError as e:
    print(f'❌ 缺少套件: {e}')
"

# 檢查自定義模組
echo ""
echo "3️⃣ 檢查自定義模組..."
python -c "
try:
    import sys, os
    sys.path.append('.')
    from decoder.pretrained import WavTokenizer
    from ttdata import AudioDataset
    print('✅ WavTokenizer 和 AudioDataset 可正常導入')
except ImportError as e:
    print(f'❌ 自定義模組導入失敗: {e}')
"

# 檢查腳本語法
echo ""
echo "4️⃣ 檢查腳本語法..."
python -m py_compile discrete_token_denoising.py && echo "✅ discrete_token_denoising.py 語法正確" || echo "❌ discrete_token_denoising.py 語法錯誤"
python -m py_compile discrete_inference.py && echo "✅ discrete_inference.py 語法正確" || echo "❌ discrete_inference.py 語法錯誤"

# 測試 TokenSequenceDataset 類
echo ""
echo "5️⃣ 測試核心類別..."
python -c "
try:
    import torch
    import sys, os
    sys.path.append('.')
    from discrete_token_denoising import TokenSequenceDataset, TokenToTokenTransformer
    
    # 測試數據集類
    fake_noisy = [torch.randint(0, 4096, (50,)) for _ in range(10)]
    fake_clean = [torch.randint(0, 4096, (45,)) for _ in range(10)]
    dataset = TokenSequenceDataset(fake_noisy, fake_clean, max_length=128)
    print(f'✅ TokenSequenceDataset: 數據集大小 {len(dataset)}')
    
    # 測試數據載入
    sample = dataset[0]
    print(f'   - input_seq shape: {sample[\"input_seq\"].shape}')
    print(f'   - target_seq shape: {sample[\"target_seq\"].shape}')
    print(f'   - vocab_size: {dataset.vocab_size}')
    
    # 測試模型類
    model = TokenToTokenTransformer(vocab_size=4098, d_model=256, max_length=128)
    print(f'✅ TokenToTokenTransformer: 參數數量 {sum(p.numel() for p in model.parameters()):,}')
    
    # 測試前向傳播
    batch_size = 2
    src = torch.randint(0, 4096, (batch_size, 64))
    tgt = torch.randint(0, 4096, (batch_size, 60))  
    
    with torch.no_grad():
        output = model(src, tgt)
    print(f'✅ 前向傳播測試: 輸出 shape {output.shape}')
    
except Exception as e:
    print(f'❌ 核心類別測試失敗: {e}')
    import traceback
    traceback.print_exc()
"

# 檢查 WavTokenizer 載入
echo ""
echo "6️⃣ 測試 WavTokenizer 載入..."
python -c "
try:
    import torch
    import sys, os
    sys.path.append('.')
    from decoder.pretrained import WavTokenizer
    
    config_path = 'config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
    model_path = 'models/wavtokenizer_large_speech_320_24k.ckpt'
    
    if os.path.exists(config_path) and os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
        wavtokenizer = wavtokenizer.to(device)
        print(f'✅ WavTokenizer 載入成功，設備: {device}')
        
        # 測試編碼
        fake_audio = torch.randn(1, 1, 24000).to(device)  # 1秒音頻
        with torch.no_grad():
            bandwidth_id = torch.tensor([0], device=device)
            tokens, _ = wavtokenizer.encode_infer(fake_audio, bandwidth_id=bandwidth_id)
            if isinstance(tokens, tuple):
                tokens = tokens[0]
        print(f'✅ 編碼測試: tokens shape {tokens.shape}')
        
    else:
        print('❌ WavTokenizer 文件缺失，跳過測試')
        
except Exception as e:
    print(f'❌ WavTokenizer 測試失敗: {e}')
    import traceback
    traceback.print_exc()
"

# 測試音頻數據集
echo ""
echo "7️⃣ 測試音頻數據集..."
python -c "
try:
    import sys, os
    sys.path.append('.')
    from ttdata import AudioDataset
    
    # 嘗試不同的初始化方式
    try:
        dataset = AudioDataset()
        print(f'✅ AudioDataset (基本初始化): 數據集大小 {len(dataset)}')
    except TypeError:
        print('⚠️  AudioDataset 需要額外參數，跳過測試')
        dataset = None
    
    if dataset is not None and len(dataset) > 0:
        sample = dataset[0]
        print(f'   - noisy_audio shape: {sample[\"noisy_audio\"].shape}')
        print(f'   - clean_audio shape: {sample[\"clean_audio\"].shape}')
        print(f'   - speaker: {sample[\"speaker\"]}')
    elif dataset is not None:
        print('⚠️  數據集為空，請檢查數據路徑')
        
except Exception as e:
    print(f'❌ AudioDataset 測試失敗: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "8️⃣ 系統資源檢查..."
python -c "
import torch
import psutil

# GPU 信息
if torch.cuda.is_available():
    print(f'🚀 GPU: {torch.cuda.get_device_name(0)}')
    print(f'   - GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'   - CUDA 版本: {torch.version.cuda}')
else:
    print('💻 使用 CPU 計算')

# 系統記憶體
memory = psutil.virtual_memory()
print(f'💾 系統記憶體: {memory.total / 1024**3:.1f} GB (可用: {memory.available / 1024**3:.1f} GB)')

# CPU 信息
print(f'⚡ CPU 核心數: {psutil.cpu_count()} (邏輯: {psutil.cpu_count(logical=True)})')
"

echo ""
echo "=================================="
echo "✅ 快速測試完成！"
echo ""
echo "如果所有測試都通過，可以運行："
echo "  ./run_discrete_token_experiment.sh"
echo ""
echo "如果要運行小規模測試："
echo "  python discrete_token_denoising.py --max_samples 10 --num_epochs 3 --batch_size 1"
