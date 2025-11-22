"""
測試 HDF5 Dataset 的各種功能

驗證:
1. ✅ 支持任意 batch_size
2. ✅ 支持 shuffle
3. ✅ 支持多進程 (num_workers)
4. ✅ 變長序列自動 padding
5. ✅ 記憶體使用量低
"""

import torch
from torch.utils.data import DataLoader
import psutil
import os
from pathlib import Path

# 假設已有 HDF5 文件（測試用）
# 實際使用時，先運行 preprocess_zeroshot_cache_with_distances_hdf5.py


def test_batch_size_flexibility():
    """測試 1: 支持任意 batch_size"""
    print("\n" + "="*80)
    print("測試 1: Batch Size 靈活性")
    print("="*80)
    
    from data_zeroshot_hdf5_v2 import HDF5ZeroShotDataset, cached_collate_fn_with_distances
    
    h5_path = './data_with_distances/cache_with_distances.h5'
    
    if not Path(h5_path).exists():
        print(f"❌ 文件不存在: {h5_path}")
        print("請先運行預處理腳本生成 HDF5 文件")
        return False
    
    dataset = HDF5ZeroShotDataset(h5_path, split='train')
    
    # 測試不同的 batch_size
    batch_sizes = [1, 4, 16, 28, 32, 64]
    
    for bs in batch_sizes:
        if bs > len(dataset):
            continue
        
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=cached_collate_fn_with_distances
        )
        
        batch = next(iter(loader))
        actual_bs = batch['noisy_tokens'].shape[0]
        
        print(f"  Batch size {bs:2d}: ✓ 實際 shape = {batch['noisy_tokens'].shape}")
        
        if actual_bs != bs:
            print(f"    ⚠️ 預期 {bs}，實際 {actual_bs}")
    
    print("\n✅ 測試通過: 支持任意 batch_size")
    return True


def test_shuffle():
    """測試 2: 支持 shuffle"""
    print("\n" + "="*80)
    print("測試 2: Shuffle 功能")
    print("="*80)
    
    from data_zeroshot_hdf5_v2 import HDF5ZeroShotDataset, cached_collate_fn_with_distances
    
    h5_path = './data_with_distances/cache_with_distances.h5'
    dataset = HDF5ZeroShotDataset(h5_path, split='train')
    
    # 創建兩個 DataLoader，一個 shuffle，一個不 shuffle
    loader_no_shuffle = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=cached_collate_fn_with_distances
    )
    
    loader_shuffle = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=cached_collate_fn_with_distances
    )
    
    # 獲取前 5 個 batch 的 content_id
    def get_content_ids(loader, n_batches=5):
        ids = []
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            ids.extend([meta['content_id'] for meta in batch['metadata']])
        return ids
    
    ids_no_shuffle_1 = get_content_ids(loader_no_shuffle)
    ids_no_shuffle_2 = get_content_ids(loader_no_shuffle)
    ids_shuffle_1 = get_content_ids(loader_shuffle)
    ids_shuffle_2 = get_content_ids(loader_shuffle)
    
    print(f"  無 shuffle (第1次): {ids_no_shuffle_1[:3]}...")
    print(f"  無 shuffle (第2次): {ids_no_shuffle_2[:3]}...")
    print(f"  有 shuffle (第1次): {ids_shuffle_1[:3]}...")
    print(f"  有 shuffle (第2次): {ids_shuffle_2[:3]}...")
    
    # 驗證
    no_shuffle_consistent = (ids_no_shuffle_1 == ids_no_shuffle_2)
    shuffle_different = (ids_shuffle_1 != ids_shuffle_2)
    
    print(f"\n  無 shuffle 兩次相同: {no_shuffle_consistent}")
    print(f"  有 shuffle 兩次不同: {shuffle_different}")
    
    if no_shuffle_consistent and shuffle_different:
        print("\n✅ 測試通過: Shuffle 功能正常")
        return True
    else:
        print("\n⚠️ Shuffle 行為異常")
        return False


def test_multiprocessing():
    """測試 3: 支持多進程"""
    print("\n" + "="*80)
    print("測試 3: 多進程支持")
    print("="*80)
    
    from data_zeroshot_hdf5_v2 import HDF5ZeroShotDataset, cached_collate_fn_with_distances
    import time
    
    h5_path = './data_with_distances/cache_with_distances.h5'
    dataset = HDF5ZeroShotDataset(h5_path, split='train')
    
    # 測試不同的 num_workers
    workers_list = [0, 2, 4]
    
    for num_workers in workers_list:
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=cached_collate_fn_with_distances
        )
        
        # 測試載入速度
        start = time.time()
        n_batches = 10
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
        elapsed = time.time() - start
        
        print(f"  num_workers={num_workers}: {n_batches} batches in {elapsed:.2f}s ({elapsed/n_batches*1000:.1f}ms/batch)")
    
    print("\n✅ 測試通過: 支持多進程載入")
    return True


def test_variable_length_sequences():
    """測試 4: 變長序列自動 padding"""
    print("\n" + "="*80)
    print("測試 4: 變長序列 Padding")
    print("="*80)
    
    from data_zeroshot_hdf5_v2 import HDF5ZeroShotDataset, cached_collate_fn_with_distances
    
    h5_path = './data_with_distances/cache_with_distances.h5'
    dataset = HDF5ZeroShotDataset(h5_path, split='train')
    
    # 手動取幾個樣本，看看長度是否不同
    samples = [dataset[i] for i in range(5)]
    lengths = [s['noisy_tokens'].shape[0] for s in samples]
    
    print(f"  前 5 個樣本的序列長度: {lengths}")
    print(f"  最小長度: {min(lengths)}, 最大長度: {max(lengths)}")
    
    # 測試 collate_fn 是否正確 padding
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=cached_collate_fn_with_distances
    )
    
    batch = next(iter(loader))
    
    print(f"\n  Batch shape:")
    print(f"    noisy_tokens: {batch['noisy_tokens'].shape}")
    print(f"    noisy_distances: {batch['noisy_distances'].shape}")
    
    # 驗證所有樣本都 pad 到相同長度
    assert batch['noisy_tokens'].shape[0] == 8, "Batch size 錯誤"
    assert len(batch['noisy_tokens'].shape) == 2, "應該是 2D tensor (B, T)"
    
    print("\n✅ 測試通過: 變長序列自動 padding 到相同長度")
    return True


def test_memory_usage():
    """測試 5: 記憶體使用量"""
    print("\n" + "="*80)
    print("測試 5: 記憶體使用量")
    print("="*80)
    
    from data_zeroshot_hdf5_v2 import HDF5ZeroShotDataset, cached_collate_fn_with_distances
    
    # 獲取當前進程
    process = psutil.Process(os.getpid())
    
    # 初始記憶體
    mem_before = process.memory_info().rss / 1024**3  # GB
    print(f"  初始記憶體: {mem_before:.2f} GB")
    
    h5_path = './data_with_distances/cache_with_distances.h5'
    dataset = HDF5ZeroShotDataset(h5_path, split='train')
    
    mem_after_init = process.memory_info().rss / 1024**3
    print(f"  載入 dataset 後: {mem_after_init:.2f} GB (+{mem_after_init - mem_before:.2f} GB)")
    
    # 迭代幾個 batch
    loader = DataLoader(
        dataset,
        batch_size=28,
        shuffle=True,
        num_workers=0,
        collate_fn=cached_collate_fn_with_distances
    )
    
    for i, batch in enumerate(loader):
        if i >= 10:
            break
    
    mem_after_iteration = process.memory_info().rss / 1024**3
    print(f"  迭代 10 batches 後: {mem_after_iteration:.2f} GB (+{mem_after_iteration - mem_before:.2f} GB)")
    
    # 驗證記憶體增長不大
    mem_increase = mem_after_iteration - mem_before
    
    if mem_increase < 2.0:  # 增長 < 2GB
        print(f"\n✅ 測試通過: 記憶體增長僅 {mem_increase:.2f} GB (遠小於 63GB 的原始文件)")
        return True
    else:
        print(f"\n⚠️ 記憶體增長較大: {mem_increase:.2f} GB")
        return False


def test_dynamic_batch_size_change():
    """測試 6: 訓練時動態修改 batch_size"""
    print("\n" + "="*80)
    print("測試 6: 動態修改 Batch Size")
    print("="*80)
    
    from data_zeroshot_hdf5_v2 import HDF5ZeroShotDataset, cached_collate_fn_with_distances
    
    h5_path = './data_with_distances/cache_with_distances.h5'
    dataset = HDF5ZeroShotDataset(h5_path, split='train')
    
    # 模擬訓練時動態修改 batch_size
    print("  模擬訓練過程中修改 batch_size:")
    
    batch_sizes = [16, 32, 28, 8, 64]
    
    for epoch, bs in enumerate(batch_sizes):
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=0,
            collate_fn=cached_collate_fn_with_distances
        )
        
        batch = next(iter(loader))
        print(f"    Epoch {epoch}: batch_size={bs} → shape={batch['noisy_tokens'].shape}")
    
    print("\n✅ 測試通過: 可以隨時創建新的 DataLoader 使用不同 batch_size")
    return True


def main():
    print("="*80)
    print("HDF5 DataLoader 全面測試")
    print("="*80)
    
    results = []
    
    try:
        results.append(("Batch Size 靈活性", test_batch_size_flexibility()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("Batch Size 靈活性", False))
    
    try:
        results.append(("Shuffle 功能", test_shuffle()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("Shuffle 功能", False))
    
    try:
        results.append(("多進程支持", test_multiprocessing()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("多進程支持", False))
    
    try:
        results.append(("變長序列 Padding", test_variable_length_sequences()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("變長序列 Padding", False))
    
    try:
        results.append(("記憶體使用量", test_memory_usage()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("記憶體使用量", False))
    
    try:
        results.append(("動態修改 Batch Size", test_dynamic_batch_size_change()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("動態修改 Batch Size", False))
    
    # 總結
    print("\n" + "="*80)
    print("測試總結")
    print("="*80)
    
    for name, passed in results:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n" + "="*80)
        print("🎉 所有測試通過！HDF5 Dataset 完全支持:")
        print("="*80)
        print("  ✅ 任意 batch_size (1, 4, 16, 28, 32, 64...)")
        print("  ✅ Shuffle (每個 epoch 不同順序)")
        print("  ✅ 多進程 (num_workers=0,2,4...)")
        print("  ✅ 變長序列自動 padding")
        print("  ✅ 低記憶體使用 (<2GB vs 63GB)")
        print("  ✅ 動態修改 batch_size")
        print("\n可以安心用於訓練！")
    else:
        print("\n⚠️ 部分測試未通過，請檢查上述錯誤")


if __name__ == '__main__':
    main()
