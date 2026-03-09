# 演算法實驗工程師 Agent - 任務指示書

**任務編號**: EXP-0206-PLAN-ORI
**優先級**: P2 (備用方案)
**狀態**: 🟡 待執行
**預計時間**: 2-3 天 (Short-run) + 1-2 週 (Long-run, optional)

---

## 📋 任務總覽

### 目標

實現並驗證 **方案 A: Single VQ 4096 + EMA Update**，作為現有 RVQ 方案的科學對照實驗。

### 核心問題

1. **預訓練 codebook + EMA 能否避免 token collapse？**
2. **Warm start (預訓練) vs Cold start (隨機初始化) 哪個更好？**
3. **單層 vs 多層 VQ 的必要性是什麼？**

### 成功標準

**Short-run (1000 steps):**
- ✅ P1 (step 200): top10≤0.95, used≥82, mse≤0.1
- ✅ P2 (step 1000): entropy≥5.0, top10≤0.5, used≥410, mse≤0.1
- 🎯 P3 (bonus): entropy>6.5, top10<0.15, used≥2867

---

## 📚 必讀文件

### 閱讀順序

1. **README.md** (5 分鐘) - 快速理解方案背景
2. **PLAN.md** (15 分鐘) - 詳細計劃與時間線
3. **SPEC.md** (20 分鐘) - 技術規格與實現細節
4. **參考**: `exp_0128/phase3/residual_vq/models_rvq.py` - RVQ 實現（作為對照）

### 關鍵理解點

```
Baseline (失敗):
  ✅ Single VQ 4096
  ✅ 預訓練初始化
  ❌ Frozen quantizer → Token collapse (top10=19.7%)

RVQ (成功):
  ✅ 4×2048 多層
  ✅ 隨機初始化
  ✅ EMA update → No collapse (top10=15.8%)

方案 A (待驗證):
  ✅ Single VQ 4096
  ✅ 預訓練初始化
  ✅ EMA update → ？
```

---

## 🎯 Phase 1: 實現階段 (Day 1-2)

### Task 1.1: 實現 `SingleVQWithEMA` 類別

**檔案**: `families/compat_legacy/plan_ori_vq/plan_ori/models_single_vq_ema.py`

**核心功能**:
```python
class SingleVQWithEMA(nn.Module):
    """
    單層 VQ + EMA 更新機制

    關鍵特性:
    - 從 WavTokenizer 預訓練 codebook 初始化 (4096×128)
    - EMA update (decay=0.99, eps=1e-5)
    - Dead-code reset (threshold=2)
    - 可選 usage penalty
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        dim: int = 128,
        pretrained_codebook: torch.Tensor = None,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        ema_dead_code_threshold: int = 2,
        ema_usage_penalty: float = 0.0,
    ): ...

    def _ema_update(self, z_flat, indices): ...

    def forward(self, z, frame_rate=75, bandwidth=0.075): ...

    def get_codebook_usage(self, codes): ...
```

**實現要點**:
1. ✅ 參考 `SPEC.md` 中的完整程式碼範例
2. ✅ EMA buffers: `ema_cluster_size`, `ema_embed_avg`
3. ✅ Dead-code reset: 當 `cluster_size < threshold` 時隨機重設
4. ✅ 輸出格式兼容 WavTokenizer: `codes` shape 為 `[1, B, 1, T]`

**驗證清單**:
- [ ] Codebook 正確從預訓練初始化
- [ ] EMA update 邏輯正確（參考 RVQ 實現）
- [ ] Dead-code reset 機制運作
- [ ] Forward pass 無錯誤
- [ ] 輸出 shape 正確

---

### Task 1.2: 實現 `TeacherStudentSingleVQ` 模型

**檔案**: 同 `models_single_vq_ema.py`

**核心功能**:
```python
class TeacherStudentSingleVQ(TeacherStudentIntermediate):
    """
    繼承 TeacherStudentIntermediate
    替換 frozen quantizer 為 SingleVQWithEMA
    """

    def __init__(self, ...):
        super().__init__(...)

        # 提取預訓練 codebook
        original_quantizer = self.student.feature_extractor.encodec.quantizer
        pretrained_codebook = original_quantizer.vq.layers[0].codebook.detach().clone()

        # 替換為 SingleVQWithEMA
        self.vq = SingleVQWithEMA(
            codebook_size=4096,
            dim=128,
            pretrained_codebook=pretrained_codebook,
            ...
        )

    def forward(self, clean_audio, noisy_audio): ...
```

**驗證清單**:
- [ ] 正確繼承 `TeacherStudentIntermediate`
- [ ] 預訓練 codebook 成功載入
- [ ] Forward pass 返回正確的輸出字典
- [ ] 與 baseline 輸出格式相容

---

### Task 1.3: 實現訓練腳本

**檔案**: `families/compat_legacy/plan_ori_vq/plan_ori/train_single_vq_ema.py`

**基於**: 參考 `families/compat_legacy/plan_ori_vq/train_long_v2.py` 並修改

**修改重點**:
```python
# 1. 使用 TeacherStudentSingleVQ 而非 TeacherStudentRVQ
from models_single_vq_ema import TeacherStudentSingleVQ

model = TeacherStudentSingleVQ(
    wavtok_config=WAVTOK_CONFIG,
    wavtok_ckpt=WAVTOK_CKPT,
    lora_rank=256,
    lora_alpha=512,
    intermediate_indices=[3, 4, 6],
    # SingleVQ specific
    vq_ema_decay=0.99,
    vq_ema_threshold=2,
)

# 2. Loss 計算保持不變
loss_quant = masked_mse(student_quantized, teacher_encoder_out, lengths)
loss_inter = inter_loss_fn(student_inter, teacher_inter)
loss_commit = output['vq_loss_commit']

total_loss = (
    config['lambda_quant'] * loss_quant +
    config['lambda_inter'] * loss_inter +
    config['beta_commit'] * loss_commit
)

# 3. Evaluation metrics (與 RVQ 相同)
- Entropy, top-10 mass, used codes
- Feature MSE
- Codebook usage distribution
```

**驗證清單**:
- [ ] 正確載入模型與資料
- [ ] Loss 計算無誤
- [ ] Metrics logging 完整
- [ ] Checkpoint saving 正常

---

### Task 1.4: 建立執行腳本

**檔案**: `families/compat_legacy/plan_ori_vq/plan_ori/run_exp_ori_short.sh`

```bash
#!/bin/bash

# Short-run: 1000 steps
# Purpose: 驗證方案 A 可行性

GPU_ID=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="families/compat_legacy/plan_ori_vq/runs/plan_ori_short_${TIMESTAMP}"

echo "=========================================="
echo "Exp 0206 - Plan Original: Short-run"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Steps: 1000"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python families/compat_legacy/plan_ori_vq/plan_ori/train_single_vq_ema.py \
  --output_dir "${OUTPUT_DIR}" \
  --steps 1000 \
  --batch_size 8 \
  --grad_accum 2 \
  --learning_rate 1e-4 \
  --eval_interval 200 \
  --checkpoint_interval 200 \
  --lambda_quant 1.0 \
  --lambda_inter 0.5 \
  --beta_commit 1.0 \
  --vq_ema_decay 0.99 \
  --vq_ema_threshold 2 \
  --seed 42

echo ""
echo "✅ Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
```

---

## 🧪 Phase 2: 測試階段 (Day 2-3)

### Task 2.1: 單元測試

**檔案**: `families/compat_legacy/plan_ori_vq/plan_ori/test_single_vq_ema.py`

```python
import torch
from models_single_vq_ema import SingleVQWithEMA

def test_initialization():
    """測試初始化"""
    pretrained = torch.randn(4096, 128)
    vq = SingleVQWithEMA(
        codebook_size=4096,
        dim=128,
        pretrained_codebook=pretrained,
    )
    assert torch.allclose(vq.codebook.weight, pretrained)
    print("✅ Initialization test passed")

def test_forward_pass():
    """測試 forward pass"""
    vq = SingleVQWithEMA(codebook_size=16, dim=4)
    z = torch.randn(2, 4, 10)
    vq.train()

    out = vq(z)

    assert out['quantized'].shape == (2, 4, 10)
    assert out['codes'].shape == (1, 2, 1, 10)
    assert not torch.isnan(out['loss_commit'])
    print("✅ Forward pass test passed")

def test_ema_update():
    """測試 EMA 更新"""
    vq = SingleVQWithEMA(codebook_size=16, dim=4)
    vq.train()

    # Record initial codebook
    initial_codebook = vq.codebook.weight.data.clone()

    # Run several forward passes
    for _ in range(10):
        z = torch.randn(2, 4, 10)
        _ = vq(z)

    # Codebook should have changed
    assert not torch.allclose(vq.codebook.weight, initial_codebook)
    print("✅ EMA update test passed")

def test_dead_code_reset():
    """測試 dead-code reset"""
    vq = SingleVQWithEMA(
        codebook_size=16,
        dim=4,
        ema_dead_code_threshold=2,
    )
    vq.train()

    # Force some codes to be dead by feeding biased data
    for _ in range(50):
        z = torch.zeros(2, 4, 10) + torch.randn(2, 4, 10) * 0.1
        _ = vq(z)

    # Check that cluster_size has been reset for dead codes
    has_resets = (vq.ema_cluster_size == 1.0).any()
    print(f"✅ Dead-code reset test: has_resets={has_resets}")

if __name__ == "__main__":
    test_initialization()
    test_forward_pass()
    test_ema_update()
    test_dead_code_reset()
    print("\n🎉 All tests passed!")
```

**執行**:
```bash
python families/compat_legacy/plan_ori_vq/plan_ori/test_single_vq_ema.py
```

---

### Task 2.2: Smoke Test (10 steps)

**目的**: 驗證整個訓練流程可運行

```bash
python families/compat_legacy/plan_ori_vq/plan_ori/train_single_vq_ema.py \
  --output_dir test_smoke \
  --steps 10 \
  --batch_size 2 \
  --eval_interval 10
```

**檢查點**:
- [ ] 腳本無錯誤完成
- [ ] 生成 `config.json`, `summary.json`, `metrics_history.json`
- [ ] Loss 值合理（無 NaN/Inf）
- [ ] Metrics 正確記錄

---

## 🚀 Phase 3: 實驗階段 (Day 3-4)

### Task 3.1: Short-run 實驗 (1000 steps)

**執行**:
```bash
bash families/compat_legacy/plan_ori_vq/plan_ori/run_exp_ori_short.sh 0
```

**監控指標** (每 200 steps):
```
Step 200 (P1 Gate):
  - entropy: 記錄值
  - top_10_mass: ≤0.95 ✅
  - used_codes: ≥82 ✅
  - feature_mse: ≤0.1 ✅

Step 400, 600, 800:
  - 持續觀察 entropy 上升趨勢
  - 觀察 top_10_mass 下降趨勢
  - 觀察 used_codes 增加趨勢

Step 1000 (P2 Gate):
  - entropy: ≥5.0 ✅ (P2), >6.5 🎯 (P3)
  - top_10_mass: ≤0.5 ✅ (P2), <0.15 🎯 (P3)
  - used_codes: ≥410 ✅ (P2), ≥2867 🎯 (P3)
  - feature_mse: ≤0.1 ✅
```

**實時監控**:
```bash
# 監控 loss
tail -f <output_dir>/train.log | grep "loss:"

# 監控 metrics
watch -n 60 "tail -20 <output_dir>/metrics_history.json"
```

---

### Task 3.2: 結果分析與決策

**分析腳本**: `families/compat_legacy/plan_ori_vq/plan_ori/analyze_results.py`

```python
import json
import matplotlib.pyplot as plt

def analyze_short_run(output_dir):
    """分析 short-run 結果"""

    # Load metrics
    with open(f"{output_dir}/metrics_history.json") as f:
        metrics = json.load(f)

    # Extract final metrics
    final = metrics[-1]

    print("="*60)
    print("Short-run Results (Step 1000)")
    print("="*60)
    print(f"Entropy:     {final['entropy']:.3f}")
    print(f"Top-10 mass: {final['top10_mass']:.1%}")
    print(f"Used codes:  {final['used_codes']}/4096 ({final['usage_pct']:.1%})")
    print(f"Feature MSE: {final['feature_mse']:.4f}")
    print("="*60)

    # Decision
    if final['entropy'] >= 5.0 and final['top10_mass'] <= 0.5 and final['used_codes'] >= 410:
        print("✅ P2 PASSED - 可進行 long-run")
        if final['entropy'] > 6.5 and final['top10_mass'] < 0.15 and final['used_codes'] >= 2867:
            print("🎯 P3 ACHIEVED - 優秀表現！")
    else:
        print("❌ P2 FAILED - 方案 A 未通過驗證")
        print("建議: 終止方案 A，專注於 RVQ")

    # Plot curves
    plot_metrics(metrics, output_dir)

def plot_metrics(metrics, output_dir):
    """繪製 metrics 曲線"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    steps = [m['step'] for m in metrics]

    # Entropy
    axes[0, 0].plot(steps, [m['entropy'] for m in metrics])
    axes[0, 0].axhline(y=5.0, color='orange', linestyle='--', label='P2: 5.0')
    axes[0, 0].axhline(y=6.5, color='green', linestyle='--', label='P3: 6.5')
    axes[0, 0].set_title('Entropy')
    axes[0, 0].legend()

    # Top-10 mass
    axes[0, 1].plot(steps, [m['top10_mass'] for m in metrics])
    axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', label='P2: 0.5')
    axes[0, 1].axhline(y=0.15, color='green', linestyle='--', label='P3: 0.15')
    axes[0, 1].set_title('Top-10 Mass')
    axes[0, 1].legend()

    # Used codes
    axes[1, 0].plot(steps, [m['used_codes'] for m in metrics])
    axes[1, 0].axhline(y=410, color='orange', linestyle='--', label='P2: 410')
    axes[1, 0].axhline(y=2867, color='green', linestyle='--', label='P3: 2867')
    axes[1, 0].set_title('Used Codes')
    axes[1, 0].legend()

    # Feature MSE
    axes[1, 1].plot(steps, [m['feature_mse'] for m in metrics])
    axes[1, 1].axhline(y=0.1, color='orange', linestyle='--', label='P2: 0.1')
    axes[1, 1].set_title('Feature MSE')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_curves.png", dpi=150)
    print(f"✅ Metrics plot saved: {output_dir}/metrics_curves.png")

if __name__ == "__main__":
    import sys
    analyze_short_run(sys.argv[1])
```

**執行**:
```bash
python families/compat_legacy/plan_ori_vq/plan_ori/analyze_results.py <output_dir>
```

---

## 📊 Phase 4: 文檔階段 (Day 4-5)

### Task 4.1: 填寫實驗結果

**檔案**: `families/compat_legacy/plan_ori_vq/plan_ori/RESULTS.md`

```markdown
# Exp 0206 - Plan Original: Results

**日期**: <填入實驗日期>
**狀態**: <🟢 成功 | 🟡 部分成功 | 🔴 失敗>

---

## Short-run Results (1000 steps)

### Final Metrics (Step 1000)

| Metric | Value | P2 Target | P3 Target | Status |
|--------|-------|-----------|-----------|--------|
| Entropy | <填入> | ≥5.0 | >6.5 | <✅/❌> |
| Top-10 mass | <填入> | ≤50% | <15% | <✅/❌> |
| Used codes | <填入> | ≥410 | ≥2867 | <✅/❌> |
| Usage % | <填入> | ≥10% | ≥70% | <✅/❌> |
| Feature MSE | <填入> | ≤0.1 | - | <✅/❌> |

**P2 Gate**: <✅ PASSED | ❌ FAILED>
**P3 Bonus**: <🎯 ACHIEVED | ⚠️ NOT MET>

---

## Comparison with Baselines

| Method | Entropy | Top-10 | Used | Usage % |
|--------|---------|--------|------|---------|
| **Baseline** (frozen) | 6.07 | 19.7% | 740 | 18% |
| **RVQ** (random+EMA) | 9.03 | 15.8% | 1089/layer | 53% |
| **Plan Ori** (pretrain+EMA) | <填入> | <填入> | <填入> | <填入> |

---

## Analysis

### 與 Baseline 對比

<填入分析>

### 與 RVQ 對比

<填入分析>

### 科學問題回答

1. **預訓練 + EMA 能否避免 collapse？**
   - <填入結論>

2. **Warm start vs Cold start？**
   - <填入結論>

3. **單層 vs 多層必要性？**
   - <填入結論>

---

## Visualization

- Metrics curves: `<output_dir>/metrics_curves.png`
- Codebook usage: `<output_dir>/codebook_usage.png`
- Loss curves: `<output_dir>/loss_curves.png`

---

## Decision

<根據結果填寫決策>

### 如果 P2 PASSED:
- [ ] 進行 long-run 實驗 (300 epochs)
- [ ] 與 RVQ 詳細對比
- [ ] 評估作為主要方案的可能性

### 如果 P2 FAILED:
- [ ] 分析失敗原因
- [ ] 寫入 ablation study
- [ ] 終止方案 A
- [ ] 全力投入 RVQ

---

## Next Steps

<填入後續計劃>
```

---

### Task 4.2: 更新主文檔

更新以下文件的狀態:
- [ ] `families/compat_legacy/plan_ori_vq/plan_ori/README.md` - 更新狀態為 🟢/🟡/🔴
- [ ] `families/compat_legacy/plan_ori_vq/README.md` - 加入 Plan Original 結果連結
- [ ] 主 README - 更新實驗進度

---

## 🔄 進度追蹤

使用以下 checklist 追蹤進度:

### Day 1-2: 實現
- [ ] Task 1.1: `SingleVQWithEMA` 實現完成
- [ ] Task 1.2: `TeacherStudentSingleVQ` 實現完成
- [ ] Task 1.3: 訓練腳本實現完成
- [ ] Task 1.4: 執行腳本建立完成

### Day 2-3: 測試
- [ ] Task 2.1: 單元測試全部通過
- [ ] Task 2.2: Smoke test 成功

### Day 3-4: 實驗
- [ ] Task 3.1: Short-run 實驗完成
- [ ] Task 3.2: 結果分析完成

### Day 4-5: 文檔
- [ ] Task 4.1: RESULTS.md 填寫完成
- [ ] Task 4.2: 主文檔更新完成

---

## ⚠️ 注意事項

### 執行前檢查

```bash
# 1. 確認 GPU 可用
nvidia-smi

# 2. 確認資料路徑
ls $TRAIN_CACHE
ls $VAL_CACHE
ls $WAVTOK_CKPT

# 3. 確認依賴已安裝
python -c "import torch; import torchaudio; from peft import LoraConfig"
```

### 常見問題

**Q: 如何載入預訓練 codebook？**
```python
from decoder.pretrained import WavTokenizer

wavtok = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_CKPT)
quantizer = wavtok.feature_extractor.encodec.quantizer
pretrained_codebook = quantizer.vq.layers[0].codebook.detach().clone()
```

**Q: EMA 更新何時執行？**
```python
# Only during training
if self.training:
    self._ema_update(z_flat.detach(), indices)
```

**Q: Dead-code reset 如何觸發？**
```python
if self.ema_dead_code_threshold > 0:
    dead = self.ema_cluster_size < float(self.ema_dead_code_threshold)
    # Reset dead codes with random samples
```

---

## 📞 協助與支援

### 參考資源

- **RVQ 實現**: `exp_0128/phase3/residual_vq/models_rvq.py`
- **Baseline**: `families/compat_legacy/intermediate_stack/models.py`
- **Phase 3-2 文檔**: `exp_0128/phase3-2/SUMMARY.md`

### 問題回報

如遇到技術問題，請記錄:
1. 錯誤訊息
2. 重現步驟
3. 環境資訊 (GPU, Python, PyTorch 版本)

---

## ✅ 驗收標準

### 程式碼品質
- [ ] 所有單元測試通過
- [ ] Smoke test (10 steps) 成功
- [ ] Code 有適當註解
- [ ] 符合 SPEC.md 規範

### 實驗結果
- [ ] Short-run 完成 1000 steps
- [ ] Metrics 完整記錄
- [ ] P2 gate 判定明確
- [ ] 視覺化圖表清晰

### 文檔完整性
- [ ] RESULTS.md 填寫完整
- [ ] 分析結論明確
- [ ] 後續決策清楚
- [ ] 主文檔已更新

---

**建立日期**: 2026-02-11
**最後更新**: 2026-02-11
**責任人**: 演算法實驗工程師 Agent
**審核人**: TBD
