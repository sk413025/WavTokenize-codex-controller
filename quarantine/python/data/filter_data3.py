"""
從 data3 過濾並重新分割數據
- 過濾 clean→clean 配對
- 保持與 data_with_distances 相同的語者分配
- 包含 plastic 材質
"""

import torch
from pathlib import Path
from collections import defaultdict

# 路徑
DATA3_TRAIN_PATH = Path("/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt")
DATA3_VAL_PATH = Path("/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt")
OUTPUT_DIR = Path("/home/sbplab/ruizi/WavTokenize-feature-analysis/data")

# 語者分割（與當前一致）
TRAIN_SPEAKERS = [
    'boy1', 'boy3', 'boy4', 'boy5', 'boy6', 'boy9', 'boy10',
    'girl2', 'girl3', 'girl4', 'girl7', 'girl8', 'girl10', 'girl11'
]
VAL_SPEAKERS = ['boy7', 'boy8', 'girl9']

def filter_clean_to_clean(samples, dataset_name):
    """過濾 clean→clean 配對"""
    print(f"\n過濾 {dataset_name} clean→clean 配對...")
    filtered_samples = []
    clean_to_clean_count = 0

    for sample in samples:
        noisy_path = sample['noisy_path']

        # 檢查 noisy_path 是否包含 'clean'
        if 'clean' in noisy_path.lower():
            clean_to_clean_count += 1
            continue

        filtered_samples.append(sample)

    print(f"  過濾掉 {clean_to_clean_count} 個 clean→clean 配對")
    print(f"  剩餘樣本數: {len(filtered_samples)}")

    return filtered_samples

def print_statistics(samples, dataset_name):
    """打印數據統計"""
    print("\n" + "=" * 80)
    print(f"{dataset_name} 統計")
    print("=" * 80)

    by_speaker = defaultdict(lambda: defaultdict(int))
    materials_total = defaultdict(int)

    for s in samples:
        # 從路徑提取材質
        material = 'unknown'
        if 'box' in s['noisy_path']:
            material = 'box'
        elif 'papercup' in s['noisy_path']:
            material = 'papercup'
        elif 'plastic' in s['noisy_path']:
            material = 'plastic'

        by_speaker[s['speaker_id']][material] += 1
        materials_total[material] += 1

    for spk in sorted(by_speaker.keys()):
        materials = by_speaker[spk]
        total = sum(materials.values())
        print(f"  {spk}: {total} 樣本", end="")
        if len(materials) > 1:
            print(f" ({', '.join(f'{mat}:{cnt}' for mat, cnt in sorted(materials.items()))})")
        else:
            print()

    print(f"\n{dataset_name} 材質分布:")
    for mat in sorted(materials_total.keys()):
        print(f"  {mat}: {materials_total[mat]}")

def main():
    print("=" * 80)
    print("從 data3 過濾並重新分割")
    print("=" * 80)

    # 載入 data3 train
    print(f"\n載入 data3 train: {DATA3_TRAIN_PATH}")
    data3_train = torch.load(DATA3_TRAIN_PATH)
    print(f"原始 train 樣本數: {len(data3_train)}")

    # 載入 data3 val
    print(f"\n載入 data3 val: {DATA3_VAL_PATH}")
    data3_val = torch.load(DATA3_VAL_PATH)
    print(f"原始 val 樣本數: {len(data3_val)}")

    # 過濾 clean→clean
    train_samples = filter_clean_to_clean(data3_train, "Train")
    val_samples = filter_clean_to_clean(data3_val, "Val")

    # 驗證語者分配
    print("\n" + "=" * 80)
    print("驗證語者分配")
    print("=" * 80)

    train_speakers = set(s['speaker_id'] for s in train_samples)
    val_speakers = set(s['speaker_id'] for s in val_samples)

    print(f"\nTrain 語者 ({len(train_speakers)}): {sorted(train_speakers)}")
    print(f"Val 語者 ({len(val_speakers)}): {sorted(val_speakers)}")

    # 檢查是否有重疊
    overlap = train_speakers & val_speakers
    if overlap:
        print(f"\n⚠️ 警告: Train 和 Val 有重疊語者: {overlap}")
    else:
        print(f"\n✅ Train 和 Val 無重疊語者")

    # 檢查是否缺少語者
    missing_train = set(TRAIN_SPEAKERS) - train_speakers
    missing_val = set(VAL_SPEAKERS) - val_speakers

    if missing_train:
        print(f"⚠️ 缺少 Train 語者: {missing_train}")
    if missing_val:
        print(f"⚠️ 缺少 Val 語者: {missing_val}")

    if not missing_train and not missing_val:
        print(f"✅ 所有語者都已包含")

    print(f"\n✅ Train 樣本數: {len(train_samples)}")
    print(f"✅ Val 樣本數: {len(val_samples)}")

    # 統計
    print_statistics(train_samples, "Train")
    print_statistics(val_samples, "Val")

    # 儲存
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    train_path = OUTPUT_DIR / "train_cache_filtered.pt"
    val_path = OUTPUT_DIR / "val_cache_filtered.pt"

    torch.save(train_samples, train_path)
    print(f"\n✅ Train 已儲存: {train_path} ({len(train_samples)} 樣本)")

    torch.save(val_samples, val_path)
    print(f"✅ Val 已儲存: {val_path} ({len(val_samples)} 樣本)")

    print(f"\n總計: {len(train_samples) + len(val_samples)} 樣本")
    print("\n✅ 完成!")

if __name__ == "__main__":
    main()
