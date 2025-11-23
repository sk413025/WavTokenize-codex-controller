"""
檢查現有數據是否適合 LoRA Encoder Denoising

此腳本檢查：
1. 數據文件是否存在
2. 數據格式是否正確
3. 是否有 noisy/clean audio pairs
4. 數據統計信息
"""

import torch
from pathlib import Path
import sys

def check_data_file(filepath):
    """檢查單個數據文件"""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ 文件不存在: {filepath}")
        return False

    print(f"\n{'='*60}")
    print(f"檢查: {filepath.name}")
    print(f"{'='*60}")
    print(f"文件大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        data = torch.load(filepath, weights_only=False)

        print(f"數據類型: {type(data)}")

        if isinstance(data, list):
            print(f"✅ 樣本數: {len(data)}")

            if len(data) > 0:
                sample = data[0]
                print(f"\n第一個樣本結構:")

                if isinstance(sample, dict):
                    print(f"  Keys: {list(sample.keys())}")

                    # 檢查是否有必要的 keys
                    has_noisy = 'noisy_audio' in sample or 'noisy' in sample
                    has_clean = 'clean_audio' in sample or 'clean' in sample or 'audio' in sample
                    has_waveform = 'waveform' in sample

                    if has_noisy and has_clean:
                        print(f"  ✅ 有 noisy-clean pairs")
                    elif has_waveform or 'audio' in sample:
                        print(f"  ⚠️  只有單一 audio，需要生成 noisy version")
                    else:
                        print(f"  ❌ 無法識別 audio keys")

                    # 顯示詳細信息
                    for k, v in sample.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}:")
                            print(f"    shape: {v.shape}")
                            print(f"    dtype: {v.dtype}")
                            print(f"    min/max: {v.min():.4f} / {v.max():.4f}")
                        else:
                            print(f"  {k}: {type(v)}")

                    # 檢查長度
                    audio_key = None
                    if 'noisy_audio' in sample:
                        audio_key = 'noisy_audio'
                    elif 'audio' in sample:
                        audio_key = 'audio'
                    elif 'waveform' in sample:
                        audio_key = 'waveform'

                    if audio_key and isinstance(sample[audio_key], torch.Tensor):
                        length = sample[audio_key].shape[-1]
                        duration = length / 24000  # 假設 24kHz
                        print(f"\n  音訊長度: {length} samples ({duration:.2f} 秒 @ 24kHz)")

        elif isinstance(data, dict):
            print(f"數據是 dict，Keys: {list(data.keys())}")

        print(f"\n✅ 數據載入成功")
        return True

    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print(" "*20 + "數據格式檢查工具")
    print("="*80)

    # 檢查可能的數據位置
    base_dir = Path(__file__).parent.parent

    data_locations = [
        base_dir / "data3" / "train_cache.pt",
        base_dir / "data3" / "val_cache.pt",
        base_dir / "data_with_distances" / "train_cache_with_distances.pt",
        base_dir / "data_with_distances" / "val_cache_with_distances.pt",
    ]

    found_any = False
    compatible_data = []

    for loc in data_locations:
        if loc.exists():
            found_any = True
            success = check_data_file(loc)
            if success:
                compatible_data.append(loc)

    if not found_any:
        print("\n" + "="*80)
        print("❌ 未找到任何數據文件")
        print("="*80)
        print("\n請執行以下步驟準備數據：")
        print("1. 創建 data3 目錄:")
        print("   mkdir -p done/exp/data3")
        print("\n2. 準備 noisy-clean audio pairs，保存為 PyTorch cache:")
        print("   見 REPRODUCE.md 的「數據準備」章節")
        print("\n3. 或使用 smoke test 的 dummy data 進行初步測試:")
        print("   ./run_smoke_test.sh")
    else:
        print("\n" + "="*80)
        print("總結")
        print("="*80)
        if compatible_data:
            print(f"✅ 找到 {len(compatible_data)} 個可用數據文件:")
            for loc in compatible_data:
                print(f"  - {loc}")
            print("\n下一步:")
            print("1. 確認數據有 noisy-clean pairs（或準備添加噪聲）")
            print("2. 執行訓練: python train.py --exp_name my_experiment")
        else:
            print("⚠️  找到數據文件，但格式可能需要調整")
            print("請參考 REPRODUCE.md 準備正確格式的數據")


if __name__ == "__main__":
    main()
