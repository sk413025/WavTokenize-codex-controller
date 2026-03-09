# Plan Original 音質提升策略

**日期**: 2026-02-15
**當前狀態**: Epoch 300 完成，P2 通過，音質待提升

---

## 當前性能分析

### ✅ 已達成
- **Token Diversity**: Entropy 9.45, Top-10 12.5% (優於 V2)
- **訓練穩定性**: 無 NaN/Inf，穩定收斂
- **P2 Gate**: 通過

### ⚠️ 待改善
- **Feature MSE**: 0.0418 (比 V2 的 0.0367 高 +13.9%)
- **Train/Val Loss**: 略高於 V2

### 🎯 目標
**將 Feature MSE 降低至接近 V2 水平 (0.037-0.038)**，同時保持 token diversity 優勢。

---

## 核心問題診斷

### Plan Original vs V2 架構差異

| 維度 | Plan Original | V2 (RVQ) | 影響 |
|------|---------------|----------|------|
| **Codebook 結構** | 1 layer × 4096 | 4 layers × 2048 | ⚠️ 表達力 |
| **Quantizer 維度** | **128** (from checkpoint) | **512** (encoder output) | ⚠️ **關鍵差異** |
| **殘差建模** | ❌ 無 | ✅ 4 層殘差 | ⚠️ 細節重建 |
| **初始化** | 預訓練 (warm start) | 隨機 (cold start) | ✅ 起點好 |

### 🔴 **關鍵發現：維度不匹配問題**

從評估日誌發現：
```
Codebook shape: torch.Size([4096, 512])  ← 顯示的維度
但實際使用: dim=128  ← train_single_vq_ema.py 中的設定
錯誤: mat1 and mat2 shapes cannot be multiplied (512x300 and 512x4096)
```

**問題**：Encoder 輸出是 512 維，但 VQ codebook 可能使用 128 維（繼承自 WavTokenizer 預訓練）

---

## 🎯 提升策略（按優先級排序）

### 策略 1: 修正維度匹配 **(最高優先級，立即實施)**

#### 問題分析
```python
# 當前狀況（推測）
encoder_out: [B, 512, T]  # Encoder 輸出
codebook:    [4096, 128]  # 預訓練 VQ（WavTokenizer 原始維度）

# 導致
- 需要 projection layer: 512 → 128
- 或者 codebook mismatch
```

#### 解決方案 A: 使用 512 維 Codebook（推薦）

**原理**: 匹配 Encoder 輸出維度，避免降維損失

```python
class TeacherStudentSingleVQ(TeacherStudentIntermediate):
    def __init__(self, ...):
        super().__init__(...)

        # 獲取 encoder 輸出維度
        encoder_dim = self.student.feature_extractor.encodec.encoder.dimension
        # encoder_dim = 512 (from WavTokenizer)

        # ❌ 舊版（使用 128 維預訓練 codebook）
        # pretrained_codebook = original_quantizer.vq.layers[0].codebook  # [4096, 128]

        # ✅ 新版（使用 512 維，隨機初始化或插值）
        self.vq = SingleVQWithEMA(
            codebook_size=4096,
            dim=512,  # ← 改為 512
            pretrained_codebook=None,  # ← 或使用插值擴展
            ema_decay=0.99,
            ema_dead_code_threshold=2,
            ema_usage_penalty=0.0,
        )
```

**優點**:
- ✅ 消除維度不匹配
- ✅ 充分利用 encoder 的 512 維表達能力
- ✅ 無需額外 projection layer

**缺點**:
- ⚠️ 失去預訓練 warm start（但 V2 證明 cold start 也有效）

#### 解決方案 B: 預訓練 Codebook 插值擴展

**原理**: 保留 warm start 優勢，同時擴展到 512 維

```python
def expand_codebook_128_to_512(pretrained_128: torch.Tensor) -> torch.Tensor:
    """
    將 128 維 codebook 插值擴展到 512 維

    Args:
        pretrained_128: [4096, 128]

    Returns:
        expanded_512: [4096, 512]
    """
    K, D_old = pretrained_128.shape  # [4096, 128]
    D_new = 512

    # 方法 1: Zero padding（簡單但可能次優）
    # expanded = F.pad(pretrained_128, (0, D_new - D_old), value=0)

    # 方法 2: 重複 + 噪聲（推薦）
    repeat_times = D_new // D_old  # 512 // 128 = 4
    expanded = pretrained_128.repeat(1, repeat_times)  # [4096, 512]

    # 添加小噪聲避免完全重複
    noise = torch.randn_like(expanded) * 0.01
    expanded = expanded + noise

    return expanded

# 使用
original_codebook_128 = ...  # [4096, 128]
expanded_codebook_512 = expand_codebook_128_to_512(original_codebook_128)

self.vq = SingleVQWithEMA(
    codebook_size=4096,
    dim=512,
    pretrained_codebook=expanded_codebook_512,  # [4096, 512]
    ...
)
```

**優點**:
- ✅ 保留 warm start 優勢
- ✅ 匹配 encoder 維度
- ✅ 初始化比隨機更穩定

**缺點**:
- ⚠️ 插值質量可能不如原生 512 維

---

### 策略 2: 引入殘差結構（提升細節重建）

#### 問題分析
V2 使用 4 層殘差 VQ，每層處理前一層的殘差：
```
residual_0 = encoder_out
quantized_0 = VQ_0(residual_0)
residual_1 = residual_0 - quantized_0  ← 捕獲細節
quantized_1 = VQ_1(residual_1)
...
final = quantized_0 + quantized_1 + quantized_2 + quantized_3
```

Plan Original 只有單層，無法建模細節。

#### 解決方案: 2-Layer Residual VQ（折衷方案）

**原理**: 增加一層殘差，在簡潔性與表達力間平衡

```python
class TwoLayerResidualVQ(nn.Module):
    """2 層殘差 VQ"""

    def __init__(
        self,
        codebook_size: int = 4096,
        dim: int = 512,
        ema_decay: float = 0.99,
        ema_dead_code_threshold: int = 2,
    ):
        super().__init__()

        # Layer 0: 主量化器
        self.vq_0 = SingleVQWithEMA(
            codebook_size=codebook_size,
            dim=dim,
            ema_decay=ema_decay,
            ema_dead_code_threshold=ema_dead_code_threshold,
        )

        # Layer 1: 殘差量化器
        self.vq_1 = SingleVQWithEMA(
            codebook_size=codebook_size,
            dim=dim,
            ema_decay=ema_decay,
            ema_dead_code_threshold=ema_dead_code_threshold,
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, dim, T]

        Returns:
            quantized: [B, dim, T]
            codes: [2, B, 1, T]
            loss_commit: scalar
        """
        # Layer 0
        out_0 = self.vq_0(z)
        quantized_0 = out_0['quantized']

        # Compute residual
        residual_1 = z - quantized_0.detach()

        # Layer 1
        out_1 = self.vq_1(residual_1)
        quantized_1 = out_1['quantized']

        # Final quantized
        quantized = quantized_0 + quantized_1

        # Stack codes
        codes = torch.cat([
            out_0['codes'],  # [1, B, 1, T]
            out_1['codes'],  # [1, B, 1, T]
        ], dim=0)  # [2, B, 1, T]

        # Total commitment loss
        loss_commit = out_0['loss_commit'] + out_1['loss_commit']

        return {
            'quantized': quantized,
            'codes': codes,
            'loss_commit': loss_commit,
            'loss_codebook': torch.tensor(0.0, device=z.device),
        }
```

**預期提升**:
- Feature MSE: 0.0418 → 0.038-0.040 (降低 5-10%)
- 訓練時間: +50% (仍比 4 層 RVQ 快)
- Token diversity: 保持或略降

---

### 策略 3: 優化 Loss 權重配置

#### 當前配置問題

```python
# Plan Original (推測)
lambda_quant = 1.0
lambda_inter = 0.03
beta_commit = 1.0

# 問題: commitment loss 可能過強
# 導致 encoder 過度約束於 codebook，犧牲 feature alignment
```

#### 調整建議

**方案 A: 降低 Commitment Loss 權重**

```python
lambda_quant = 1.0
lambda_inter = 0.03
beta_commit = 0.25  # ← 從 1.0 降至 0.25

# 原理:
# - 減少 encoder 對 codebook 的過度依賴
# - 讓 L_quant 主導訓練，優化 feature alignment
```

**方案 B: 提高 Intermediate Loss 權重**

```python
lambda_quant = 1.0
lambda_inter = 0.05  # ← 從 0.03 提升至 0.05
beta_commit = 1.0

# 原理:
# - 加強中間層監督，提升 encoder 表達能力
# - 但不能太高（V1 的 0.5 導致 collapse）
```

**方案 C: 動態權重調整（Curriculum Weight）**

```python
def get_loss_weights(epoch, total_epochs=300):
    """動態調整 loss 權重"""

    # 前期（0-100 epochs）: 強化 feature alignment
    if epoch < 100:
        lambda_quant = 1.0
        lambda_inter = 0.02
        beta_commit = 0.5

    # 中期（100-200 epochs）: 平衡
    elif epoch < 200:
        lambda_quant = 1.0
        lambda_inter = 0.03
        beta_commit = 0.75

    # 後期（200-300 epochs）: 穩定
    else:
        lambda_quant = 1.0
        lambda_inter = 0.03
        beta_commit = 1.0

    return lambda_quant, lambda_inter, beta_commit
```

---

### 策略 4: 增加 Codebook 大小（探索性）

#### 原理
V2 使用 4×2048 = 8192 codes (總容量)，Plan Original 只有 4096 codes

#### 方案

```python
# 嘗試更大的 codebook
codebook_size = 8192  # ← 從 4096 倍增

# 或保持 4096 但使用 2 層殘差（策略 2）
# 有效容量: 4096 × 4096 = 16.8M 組合（遠超 V2）
```

**權衡**:
- ✅ 更強表達力
- ⚠️ 需要更多數據才能充分訓練
- ⚠️ 可能降低 codebook 使用率

---

### 策略 5: Fine-tuning Decoder（進階）

#### 問題分析
當前 Decoder 是凍結的（與 Teacher 共享），可能不適應 Student VQ 的輸出分佈

#### 解決方案

```python
class TeacherStudentSingleVQ(TeacherStudentIntermediate):
    def __init__(self, ...):
        super().__init__(...)

        # ✅ 選項 1: 為 Student 添加獨立 Decoder（LoRA）
        lora_config_decoder = LoraConfig(
            r=128,  # 比 encoder 小
            lora_alpha=256,
            target_modules=["decoder.model.*"],
        )
        self.student_decoder = get_peft_model(
            deepcopy(self.teacher.decoder),
            lora_config_decoder
        )

        # ✅ 選項 2: 添加 Adapter Layer（輕量）
        self.adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
```

**預期提升**:
- 音質: +10-20% (PESQ/STOI)
- 增加參數: ~5-10M
- 訓練時間: +20%

---

## 🚀 推薦實施路徑

### Phase 1: 快速修復（1-2 天）

**優先級 P0: 修正維度問題**

1. 檢查當前 codebook 實際維度
2. 實施「策略 1 - 解決方案 A」: 使用 512 維 codebook
3. 重新訓練 short-run (1000 steps) 驗證

**預期結果**: Feature MSE 降至 0.040-0.042

---

### Phase 2: 性能優化（3-5 天）

**優先級 P1: 引入 2-Layer Residual VQ**

1. 實施「策略 2」: 2層殘差結構
2. 配合「策略 3 - 方案 A」: 降低 commitment loss
3. Long-run (300 epochs) 訓練

**預期結果**: Feature MSE 降至 0.037-0.039，接近 V2

---

### Phase 3: 進階提升（1-2 週，可選）

**優先級 P2: Decoder Fine-tuning**

1. 實施「策略 5」: 添加 Decoder LoRA
2. 結合前兩個 Phase 的改進
3. 完整評估 PESQ/STOI

**預期結果**: 音質達到或超越 V2

---

## 📊 預期性能對比

| 指標 | Current | Phase 1 | Phase 2 | Phase 3 | V2 |
|------|---------|---------|---------|---------|-----|
| **Feature MSE** | 0.0418 | 0.041 | 0.038 | 0.036 | 0.0367 |
| **Entropy** | 9.45 | 9.40 | 9.30 | 9.20 | 9.01 |
| **訓練時間** | 1 天 | 1 天 | 1.5 天 | 2 天 | 3 天 |
| **參數量** | +4.4% | +4.4% | +8.8% | +14% | +20% |

---

## 🔬 實驗驗證計劃

### Short-run Validation (每個 Phase)

```bash
# 1000 steps 快速驗證
python families/compat_legacy/plan_ori_vq/plan_ori/train_single_vq_ema.py \
    --mode step \
    --steps 1000 \
    --output_dir families/compat_legacy/plan_ori_vq/runs/plan_ori_v2_short \
    --eval_interval 200

# 驗收標準
P1 (step 200): top10 ≤0.95, used ≥82, mse ≤0.1
P2 (step 1000): entropy ≥5.0, top10 ≤0.5, used ≥410, mse ≤0.1
```

### Long-run Training

```bash
# 300 epochs 完整訓練
bash families/compat_legacy/plan_ori_vq/plan_ori/run_exp_ori_long_v2.sh 0
```

### 音質評估

```bash
# Three-mode evaluation (修復後)
python /tmp/evaluate_plan_ori_three_modes_v2.py

# 對比 V2
python families/compat_legacy/plan_ori_vq/compare_audio_quality.py \
    --plan_ori families/compat_legacy/plan_ori_vq/runs/plan_ori_v2_long \
    --v2 families/compat_legacy/plan_ori_vq/runs/longterm_20260208_114702
```

---

## 💡 關鍵洞察

### 為什麼 V2 音質更好？

1. **4 層殘差建模**: 每層捕獲不同粒度的細節
   - Layer 0: 主要結構
   - Layer 1-3: 逐步細化

2. **512 維一致性**: Encoder → VQ → Decoder 全程 512 維
   - Plan Original 可能存在 512→128 降維瓶頸

3. **Codebook 總容量**: 4×2048 = 8192 codes
   - 雖然 Plan Original 單層有 4096 codes
   - 但無法建模殘差，表達力受限

### Plan Original 的優勢

1. **更好的 Token Diversity**: Entropy 9.45 > 9.01
   - 預訓練初始化提供更均勻的起點

2. **訓練效率**: 1 天 vs 3 天
   - 單層量化，計算更快

3. **架構簡潔**: 易於部署和維護

---

## 📝 實施建議

### 立即行動（今天）

1. ✅ **檢查維度問題**
   ```bash
   python -c "
   import torch
   ckpt = torch.load('families/compat_legacy/plan_ori_vq/runs/plan_ori_long_20260211/checkpoints/checkpoint_epoch300.pt')
   vq_state = ckpt['vq_state_dict']
   print('Codebook shape:', vq_state['codebook.weight'].shape)
   "
   ```

2. ✅ **創建 Plan Original V2 分支**
   ```bash
   mkdir -p families/compat_legacy/plan_ori_vq/plan_ori_v2
   cp families/compat_legacy/plan_ori_vq/plan_ori/models_single_vq_ema.py families/compat_legacy/plan_ori_vq/plan_ori_v2/
   ```

3. ✅ **修改 models_single_vq_ema.py**
   - 實施「策略 1 - 解決方案 A」
   - 修正維度為 512

### 本週目標（2-3 天）

1. ✅ Short-run 驗證 (1000 steps)
2. ✅ 確認 Feature MSE 降至 <0.042
3. ✅ 開始 Long-run (300 epochs)

### 下週目標（5-7 天）

1. ✅ Long-run 完成
2. ✅ 三模式音質評估
3. ✅ 與 V2 對比分析
4. ✅ 決定是否實施 Phase 2/3

---

## 🎯 成功標準

### Minimum Viable Performance (MVP)

- Feature MSE ≤ 0.040
- Entropy ≥ 9.0
- P2 Gate: ✅ PASS
- 訓練時間 ≤ 2 天

### Target Performance

- Feature MSE ≤ 0.038 (接近 V2)
- Entropy ≥ 9.2
- PESQ improvement > +1% vs Noisy VQ
- 訓練時間 ≤ 2.5 天

### Stretch Goal

- Feature MSE ≤ 0.036 (優於 V2)
- Entropy ≥ 9.4 (保持優勢)
- PESQ improvement > +2% vs Noisy VQ
- 作為主要生產方案

---

**創建日期**: 2026-02-15
**作者**: Claude Sonnet 4.5
**狀態**: 🟢 Ready for Implementation
