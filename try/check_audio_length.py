import torchaudio
import os

# 檢查路徑
base_dir = '../results/overfit_test_20251029_051323'
epochs = ['epoch_0', 'epoch_200']
files = ['noisy.wav', 'predicted.wav', 'clean_target.wav']

for epoch in epochs:
    print(f'==== {epoch} ====' )
    for fname in files:
        fpath = os.path.join(base_dir, epoch, fname)
        if not os.path.exists(fpath):
            print(f'{fname}: 檔案不存在')
            continue
        info = torchaudio.info(fpath)
        print(f'{fname}: sample_rate={info.sample_rate}, num_frames={info.num_frames}')
