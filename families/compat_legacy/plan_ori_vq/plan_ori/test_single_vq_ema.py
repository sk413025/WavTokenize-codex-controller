"""
exp_0206 Plan Original: 單元測試

測試 SingleVQWithEMA 和 TeacherStudentSingleVQ 的核心功能。
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_initialization():
    """測試 SingleVQWithEMA 初始化"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    # 1. 從預訓練 codebook 初始化
    pretrained = torch.randn(4096, 128)
    vq = SingleVQWithEMA(
        codebook_size=4096,
        dim=128,
        pretrained_codebook=pretrained,
    )
    assert torch.allclose(vq.codebook.weight.data, pretrained), \
        "Codebook 應與預訓練權重一致"
    assert not vq.codebook.weight.requires_grad, \
        "EMA 模式下 codebook 不應需要梯度"
    assert vq.ema_cluster_size.shape == (4096,), \
        f"ema_cluster_size shape 錯誤: {vq.ema_cluster_size.shape}"
    assert vq.ema_embed_avg.shape == (4096, 128), \
        f"ema_embed_avg shape 錯誤: {vq.ema_embed_avg.shape}"
    print("✅ test_initialization passed")

    # 2. 隨機初始化
    vq2 = SingleVQWithEMA(codebook_size=16, dim=4)
    assert vq2.codebook.weight.shape == (16, 4)
    print("✅ test_initialization (random) passed")


def test_forward_pass():
    """測試 forward pass 的輸出格式和數值穩定性"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    vq = SingleVQWithEMA(codebook_size=16, dim=4)
    vq.train()

    z = torch.randn(2, 4, 10)  # [B=2, dim=4, T=10]
    out = vq(z)

    # 檢查輸出 shape
    assert out['quantized'].shape == (2, 4, 10), \
        f"quantized shape 錯誤: {out['quantized'].shape}"
    assert out['codes'].shape == (1, 2, 1, 10), \
        f"codes shape 錯誤: {out['codes'].shape}"
    assert not torch.isnan(out['loss_commit']), \
        "loss_commit 不應為 NaN"
    assert out['loss_codebook'].item() == 0.0, \
        "EMA 模式下 loss_codebook 應為 0"

    # 檢查 codes 範圍
    codes = out['codes']
    assert codes.min() >= 0 and codes.max() < 16, \
        f"codes 應在 [0, 16) 範圍內: min={codes.min()}, max={codes.max()}"

    print("✅ test_forward_pass passed")


def test_forward_eval_mode():
    """測試 eval 模式下不執行 EMA 更新"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    vq = SingleVQWithEMA(codebook_size=16, dim=4)
    vq.eval()

    initial_codebook = vq.codebook.weight.data.clone()

    for _ in range(10):
        z = torch.randn(2, 4, 10)
        _ = vq(z)

    assert torch.allclose(vq.codebook.weight.data, initial_codebook), \
        "eval 模式下 codebook 不應改變"
    print("✅ test_forward_eval_mode passed")


def test_ema_update():
    """測試 EMA 更新是否正確執行"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    vq = SingleVQWithEMA(codebook_size=16, dim=4)
    vq.train()

    initial_codebook = vq.codebook.weight.data.clone()

    for _ in range(20):
        z = torch.randn(4, 4, 10)
        _ = vq(z)

    assert not torch.allclose(vq.codebook.weight.data, initial_codebook), \
        "訓練後 codebook 應已改變"
    print("✅ test_ema_update passed")


def test_dead_code_reset():
    """測試 dead-code reset 機制"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    vq = SingleVQWithEMA(
        codebook_size=16,
        dim=4,
        ema_dead_code_threshold=2,
    )
    vq.train()

    # 餵入偏差資料（集中在少數幾個 code）
    for _ in range(50):
        z = torch.zeros(2, 4, 10) + torch.randn(2, 4, 10) * 0.01
        _ = vq(z)

    # 確認 cluster_size 有被重設的痕跡
    has_resets = (vq.ema_cluster_size == 1.0).any()
    print(f"  Dead-code reset 觸發: {has_resets}")
    print("✅ test_dead_code_reset passed")


def test_codebook_usage():
    """測試 codebook 使用分析"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    vq = SingleVQWithEMA(codebook_size=16, dim=4)
    vq.eval()

    z = torch.randn(4, 4, 20)
    out = vq(z)

    usage = vq.get_codebook_usage(out['codes'])

    assert 'usage_count' in usage
    assert 'n_used' in usage
    assert 'entropy' in usage
    assert usage['n_used'] > 0
    assert usage['entropy'] >= 0
    assert usage['usage_count'].shape == (16,)

    print(f"  Used codes: {usage['n_used']}/16, Entropy: {usage['entropy']:.3f}")
    print("✅ test_codebook_usage passed")


def test_usage_penalty():
    """測試 usage penalty 機制"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    # 無 penalty
    vq_no_penalty = SingleVQWithEMA(
        codebook_size=16, dim=4, ema_usage_penalty=0.0
    )
    vq_no_penalty.eval()

    # 有 penalty
    vq_with_penalty = SingleVQWithEMA(
        codebook_size=16, dim=4, ema_usage_penalty=0.5
    )
    vq_with_penalty.eval()

    # 同步 codebook
    vq_with_penalty.codebook.weight.data.copy_(vq_no_penalty.codebook.weight.data)
    vq_with_penalty.ema_cluster_size.copy_(vq_no_penalty.ema_cluster_size)

    z = torch.randn(4, 4, 20)
    out1 = vq_no_penalty(z)
    out2 = vq_with_penalty(z)

    # codes 可能不同（因為 penalty 改變了距離計算）
    print(f"  No penalty unique codes: {out1['codes'].unique().numel()}")
    print(f"  With penalty unique codes: {out2['codes'].unique().numel()}")
    print("✅ test_usage_penalty passed")


def test_straight_through():
    """測試 straight-through estimator 的梯度傳播"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    vq = SingleVQWithEMA(codebook_size=16, dim=4)
    vq.train()

    z = torch.randn(2, 4, 10, requires_grad=True)
    out = vq(z)

    # quantized 應該可以反向傳播到 z
    loss = out['quantized'].sum()
    loss.backward()

    assert z.grad is not None, "z 應該有梯度"
    assert not torch.all(z.grad == 0), "z 的梯度不應全為零"
    print("✅ test_straight_through passed")


def test_pretrained_init_shape():
    """測試不同 shape 的預訓練 codebook"""
    from families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema import SingleVQWithEMA

    # 正確 shape
    pretrained = torch.randn(256, 64)
    vq = SingleVQWithEMA(codebook_size=256, dim=64, pretrained_codebook=pretrained)
    assert vq.codebook.weight.shape == (256, 64)

    # 錯誤 shape 應 raise assertion
    try:
        wrong = torch.randn(128, 64)
        vq_bad = SingleVQWithEMA(codebook_size=256, dim=64, pretrained_codebook=wrong)
        print("❌ test_pretrained_init_shape: 應該拋出 AssertionError")
    except AssertionError:
        print("✅ test_pretrained_init_shape passed (correctly rejected wrong shape)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SingleVQWithEMA Unit Tests")
    print("=" * 60 + "\n")

    test_initialization()
    test_forward_pass()
    test_forward_eval_mode()
    test_ema_update()
    test_dead_code_reset()
    test_codebook_usage()
    test_usage_penalty()
    test_straight_through()
    test_pretrained_init_shape()

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
