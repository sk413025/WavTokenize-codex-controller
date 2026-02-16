# Plan Original 音質快速提升指南

**日期**: 2026-02-15
**當前性能**: MSE 0.0418, Entropy 9.45

---

## ✅ 好消息：維度已正確

檢查 checkpoint 確認：
- Codebook: [4096, 512] ✅ 正確
- 與 Encoder 輸出維度匹配
- 無降維瓶頸

---

## 🎯 核心問題：單層 vs 多層

### V2 (RVQ) 為什麼音質更好？

```
V2 Residual VQ (4 layers):
  encoder_out → VQ_0 → quantized_0
                  ↓ residual_1
               VQ_1 → quantized_1
                  ↓ residual_2
               VQ_2 → quantized_2
                  ↓ residual_3
               VQ_3 → quantized_3

  final = quantized_0 + quantized_1 + quantized_2 + quantized_3
  ↑ 每層捕獲不同粒度的細節

Plan Original (Single VQ):
  encoder_out → VQ_0 → quantized_0
  ↑ 只有一次量化，無法建模細節殘差
```

---

## 🚀 推薦改進方案（按效果排序）

### 方案 1: 增加到 2-Layer Residual VQ ⭐⭐⭐⭐⭐

**預期提升**: MSE 0.0418 → **0.038-0.040** (-10%)

#### 為什麼選 2 層而非 4 層？

| 層數 | Feature MSE | Training Time | Token Diversity | 推薦度 |
|------|-------------|---------------|-----------------|--------|
| 1 層 (當前) | 0.0418 | 1 天 | 9.45 ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 2 層 (推薦) | 0.038-0.040 | 1.5 天 | 9.2-9.3 | ⭐⭐⭐⭐⭐ |
| 3 層 | 0.037-0.039 | 2 天 | 9.1-9.2 | ⭐⭐⭐⭐ |
| 4 層 (V2) | 0.0367 | 3 天 | 9.01 | ⭐⭐⭐ |

**2 層是最佳平衡點**：
- ✅ 顯著提升音質（~10%）
- ✅ 訓練時間僅增加 50%
- ✅ 保持較好的 token diversity
- ✅ 實現簡單，風險低

#### 實施步驟

**Step 1: 修改 models_single_vq_ema.py**

```python
class TwoLayerResidualVQ(nn.Module):
    """2-Layer Residual Vector Quantizer with EMA"""

    def __init__(
        self,
        codebook_size: int = 4096,
        dim: int = 512,
        ema_decay: float = 0.99,
        ema_dead_code_threshold: int = 2,
        ema_usage_penalty: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.n_layers = 2

        # 創建 2 個獨立的 VQ layer
        self.layers = nn.ModuleList([
            SingleVQWithEMA(
                codebook_size=codebook_size,
                dim=dim,
                pretrained_codebook=None,  # Layer 0 可選預訓練
                ema_decay=ema_decay,
                ema_dead_code_threshold=ema_dead_code_threshold,
                ema_usage_penalty=ema_usage_penalty,
            )
            for _ in range(2)
        ])

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, dim, T]

        Returns:
            quantized: [B, dim, T]
            codes: [2, B, 1, T]
            loss_commit: scalar
        """
        B, D, T = z.shape
        all_quantized = []
        all_codes = []
        total_commit_loss = 0.0

        residual = z

        for layer_idx, layer in enumerate(self.layers):
            # 量化當前殘差
            out = layer(residual)

            all_quantized.append(out['quantized'])
            all_codes.append(out['codes'])
            total_commit_loss += out['loss_commit']

            # 計算新殘差（用於下一層）
            if layer_idx < self.n_layers - 1:
                residual = residual - out['quantized'].detach()

        # 累加所有層的量化輸出
        quantized = torch.stack(all_quantized, dim=0).sum(dim=0)

        # 堆疊 codes: [2, B, 1, T]
        codes = torch.cat(all_codes, dim=0)

        return {
            'quantized': quantized,
            'codes': codes,
            'loss_commit': total_commit_loss,
            'loss_codebook': torch.tensor(0.0, device=z.device),
            'commitment_loss': total_commit_loss,
            'bandwidth': torch.tensor([0.15], device=z.device),  # 2 layers × 75 Hz
        }

    def get_codebook_usage(self, codes: torch.Tensor):
        """分析每層 codebook 使用情況"""
        if codes.dim() == 4:
            # codes: [2, B, 1, T]
            n_layers = codes.shape[0]
            usage_stats = {}

            for layer_idx in range(n_layers):
                layer_codes = codes[layer_idx, :, 0, :]  # [B, T]
                layer_codes_flat = layer_codes.flatten()

                usage_count = torch.bincount(
                    layer_codes_flat, minlength=self.codebook_size
                )
                n_used = (usage_count > 0).sum().item()

                probs = usage_count.float() / usage_count.sum()
                probs = probs[probs > 0]
                entropy = -(probs * probs.log2()).sum().item()

                usage_stats[f'layer_{layer_idx}'] = {
                    'n_used': n_used,
                    'entropy': entropy,
                    'usage_count': usage_count,
                }

            return usage_stats

        # Fallback for single layer
        return super().get_codebook_usage(codes)


class TeacherStudentTwoLayerVQ(TeacherStudentIntermediate):
    """使用 2-Layer Residual VQ 的 Teacher-Student 模型"""

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        intermediate_indices: List[int] = [3, 4, 6],
        device: str = 'cuda',
        vq_ema_decay: float = 0.99,
        vq_ema_threshold: int = 2,
        vq_ema_usage_penalty: float = 0.0,
    ):
        super().__init__(
            wavtok_config=wavtok_config,
            wavtok_ckpt=wavtok_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            intermediate_indices=intermediate_indices,
            device=device,
        )

        # 替換為 2-Layer Residual VQ
        encoder_dim = self.student.feature_extractor.encodec.encoder.dimension

        self.vq = TwoLayerResidualVQ(
            codebook_size=4096,
            dim=encoder_dim,  # 512
            ema_decay=vq_ema_decay,
            ema_dead_code_threshold=vq_ema_threshold,
            ema_usage_penalty=vq_ema_usage_penalty,
        )

        print("="*60)
        print("TwoLayerResidualVQ Configuration:")
        print(f"  Codebook size: 4096 per layer")
        print(f"  Dimension: {encoder_dim}")
        print(f"  Layers: 2")
        print(f"  EMA decay: {vq_ema_decay}")
        print(f"  Dead-code threshold: {vq_ema_threshold}")
        print(f"  Usage penalty: {vq_ema_usage_penalty}")
        print("="*60)

    def forward(self, clean_audio, noisy_audio):
        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_enc_out, teacher_inter = self.extract_teacher_features(clean_audio)

        # Student forward
        student_enc_out, student_inter = self.extract_student_features(noisy_audio)

        # 2-Layer Residual VQ
        vq_out = self.vq(student_enc_out)

        return {
            'student_encoder_out': student_enc_out,
            'student_quantized': vq_out['quantized'],
            'student_codes': vq_out['codes'],
            'vq_loss_commit': vq_out['loss_commit'],
            'teacher_encoder_out': teacher_enc_out,
            'student_intermediates': student_inter,
            'teacher_intermediates': teacher_inter,
        }
```

**Step 2: 創建訓練腳本**

```bash
# 複製並修改
cp exp_0206/plan_ori/train_single_vq_ema.py \
   exp_0206/plan_ori/train_two_layer_vq_ema.py

# 修改 import
# from models_single_vq_ema import TeacherStudentSingleVQ
# → from models_single_vq_ema import TeacherStudentTwoLayerVQ
```

**Step 3: Short-run 驗證**

```bash
python exp_0206/plan_ori/train_two_layer_vq_ema.py \
    --mode step \
    --steps 1000 \
    --output_dir exp_0206/runs/plan_ori_2layer_short \
    --seed 42 \
    --batch_size 8 \
    --grad_accum 2 \
    --learning_rate 1e-4 \
    --eval_interval 200

# 預期結果 (step 1000):
# - entropy: 5.5-6.5
# - top10: 20-30%
# - feature_mse: 0.04-0.05
```

**Step 4: Long-run 訓練**

```bash
python exp_0206/plan_ori/train_two_layer_vq_ema.py \
    --mode epoch \
    --epochs 300 \
    --output_dir exp_0206/runs/plan_ori_2layer_long \
    --seed 42 \
    --eval_interval 10 \
    --save_checkpoint_every 10 \
    --save_audio_interval 50

# 預期結果 (epoch 300):
# - entropy: 9.2-9.3
# - top10: 13-15%
# - feature_mse: 0.038-0.040
# - 訓練時間: ~1.5 天
```

---

### 方案 2: 調整 Loss 權重 ⭐⭐⭐⭐

**預期提升**: MSE 0.0418 → **0.041-0.042** (-2%)

#### 當前問題

```python
# 當前配置
lambda_quant = 1.0
lambda_inter = 0.03
beta_commit = 1.0

# 問題: commitment loss 可能過強
# encoder 過度約束於 codebook，犧牲 feature alignment
```

#### 解決方案

```python
# 方案 2A: 降低 Commitment Loss
lambda_quant = 1.0
lambda_inter = 0.03
beta_commit = 0.25  # ← 從 1.0 降至 0.25

# 或方案 2B: 提高 Intermediate Loss
lambda_quant = 1.0
lambda_inter = 0.05  # ← 從 0.03 提升至 0.05
beta_commit = 1.0
```

**實施**:
1. 修改 train_single_vq_ema.py
2. Short-run 測試兩個方案
3. 選擇效果較好的配置

**風險**: 可能影響 token diversity（需監控）

---

### 方案 3: 增大 Codebook Size ⭐⭐⭐

**預期提升**: MSE 0.0418 → **0.040-0.042** (-5%)

#### 原理

```
V2: 4 × 2048 = 8192 總 codes
Plan Ori: 1 × 4096 = 4096 總 codes  ← 容量只有一半
```

#### 解決方案

```python
# 嘗試更大的 codebook
codebook_size = 8192  # ← 從 4096 倍增

# 注意:
# - 需要更多數據才能充分訓練
# - 可能降低 codebook 使用率
# - 推薦與方案 1 (2-Layer RVQ) 結合使用
```

**實施**:
- 單獨測試效果不佳（數據不足）
- 建議與 2-Layer RVQ 結合: 2 × 8192 = 16K codes

---

### 方案 4: 預訓練 Codebook for Layer 0 ⭐⭐

**預期提升**: MSE 0.0418 → **0.041** (-2%)

#### 原理

當前 2-Layer RVQ 使用隨機初始化，失去預訓練優勢

#### 解決方案

```python
# 為 Layer 0 使用預訓練 codebook
original_quantizer = self.student.feature_extractor.encodec.quantizer
pretrained_codebook = original_quantizer.vq.layers[0].codebook.detach().clone()

self.layers[0] = SingleVQWithEMA(
    codebook_size=4096,
    dim=512,
    pretrained_codebook=pretrained_codebook,  # ← Layer 0 warm start
    ema_decay=0.99,
    ...
)

self.layers[1] = SingleVQWithEMA(
    codebook_size=4096,
    dim=512,
    pretrained_codebook=None,  # ← Layer 1 cold start
    ema_decay=0.99,
    ...
)
```

**優點**:
- 保留 warm start 優勢
- Layer 0 更快收斂

**風險**:
- 可能限制 Layer 0 的探索空間

---

## 📊 方案對比總結

| 方案 | MSE 提升 | 實施難度 | 訓練時間 | 風險 | 推薦度 |
|------|----------|----------|----------|------|--------|
| **1. 2-Layer RVQ** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | +50% | 低 | ⭐⭐⭐⭐⭐ |
| 2. Loss 權重 | ⭐⭐ | ⭐ | 0% | 中 | ⭐⭐⭐⭐ |
| 3. 大 Codebook | ⭐⭐⭐ | ⭐ | 0% | 中 | ⭐⭐⭐ |
| 4. 預訓練 L0 | ⭐⭐ | ⭐⭐ | 0% | 低 | ⭐⭐ |

---

## 🎯 推薦實施計劃

### 本週（2-3 天）

**Day 1: 實施 2-Layer RVQ**
- ✅ 修改 models_single_vq_ema.py
- ✅ 創建 train_two_layer_vq_ema.py
- ✅ Short-run 驗證 (1000 steps)

**Day 2-3: Long-run 訓練**
- ✅ 啟動 300 epochs 訓練
- ✅ 監控 metrics

### 下週（5-7 天）

**Day 4-5: 結果分析**
- ✅ Long-run 完成
- ✅ 三模式音質評估
- ✅ 與 V2 對比

**Day 6-7: 可選優化**
- ✅ 如果 MSE 仍 >0.040，嘗試方案 2 (Loss 權重)
- ✅ 如果 MSE <0.040，考慮方案 4 (預訓練 L0) 進一步提升

---

## 🎯 預期最終性能

### 保守估計 (2-Layer RVQ)

```
Entropy: 9.2-9.3
Top-10 mass: 13-15%
Used codes: 1200-1400 / 4096 per layer
Feature MSE: 0.038-0.040
訓練時間: 1.5 天
```

### 樂觀估計 (2-Layer RVQ + Loss 調整)

```
Entropy: 9.3-9.4
Top-10 mass: 12-14%
Used codes: 1300-1500 / 4096 per layer
Feature MSE: 0.036-0.038  ← 接近或優於 V2
訓練時間: 1.5-2 天
```

---

## ✅ 行動清單

- [ ] 創建 `models_single_vq_ema.py` 備份
- [ ] 添加 `TwoLayerResidualVQ` class
- [ ] 添加 `TeacherStudentTwoLayerVQ` class
- [ ] 創建 `train_two_layer_vq_ema.py`
- [ ] 創建 `run_exp_ori_2layer_short.sh`
- [ ] 運行 short-run 驗證
- [ ] 如果 P2 通過，運行 long-run
- [ ] 評估結果並與 V2 對比
- [ ] 決定是否需要進一步優化

---

**關鍵洞察**: Plan Original 的音質差距主要源於**單層量化無法建模殘差**，而非架構設計缺陷。增加到 2 層殘差 VQ 是最有效的改進方案，預期可將 Feature MSE 降低 10%，同時保持訓練效率優勢。

**創建**: 2026-02-15
**狀態**: 🟢 Ready to Implement
