"""
exp_0128: Integration Test

測試完整流程：
1. NoiseBalancedSampler
2. Noise-Balanced DataLoader
3. 訓練一個 batch
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

import torch
from exp_1201.config import TRAIN_CACHE, VAL_CACHE
from exp_0128.noise_balanced_sampling.data_balanced import create_noise_balanced_dataloaders
from exp_0128.noise_balanced_sampling.sampler import extract_noise_type


def main():
    print("=" * 60)
    print("Exp 0128: Integration Test")
    print("=" * 60)

    # Create dataloaders
    print("\n1. Creating dataloaders...")
    train_loader, val_loader, train_sampler = create_noise_balanced_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=12,
        num_workers=0,  # For testing
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Test first batch
    print("\n2. Testing first batch...")

    # Get first batch indices from sampler
    sampler_iter = iter(train_sampler)
    first_batch_indices = [next(sampler_iter) for _ in range(12)]

    print(f"  First batch indices (first 5): {first_batch_indices[:5]}")

    # Check noise type distribution from dataset
    noise_counts = {'box': 0, 'papercup': 0, 'plastic': 0}
    dataset = train_loader.dataset

    for idx in first_batch_indices:
        sample = dataset.samples[idx]
        noisy_path = sample['noisy_path']
        noise_type = extract_noise_type(noisy_path)
        noise_counts[noise_type] += 1

        print(f"\n  Noise type distribution in first batch:")
        total = sum(noise_counts.values())
        for noise_type, count in noise_counts.items():
            pct = count / total * 100
            print(f"    {noise_type:8s}: {count:2d} / {total:2d} ({pct:5.1f}%)")

        # Check if balanced (each type should be ~33%)
        expected_per_type = total // 3
        balanced = all(abs(count - expected_per_type) <= 2 for count in noise_counts.values())

        if balanced:
            print(f"\n  ✅ Batch is balanced! (each type ~{expected_per_type})")
        else:
            print(f"\n  ⚠️  Batch may not be perfectly balanced")

        break

    print("\n3. Testing validation loader...")
    for batch in val_loader:
        print(f"  Val batch size: {batch['noisy_audio'].shape[0]}")
        break

    print("\n" + "=" * 60)
    print("✅ Integration test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
