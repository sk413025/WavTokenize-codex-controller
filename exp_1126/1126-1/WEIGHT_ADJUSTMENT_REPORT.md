# 1126-1 權重調整實驗報告

## 實驗日期
2025-11-27

## 背景問題
原始實驗 (distance_loss_weight=0.1) 顯示：
1. Distance Loss (~3.5) 貢獻遠大於 Feature Loss (~0.03)
2. Token Accuracy 從 26% 崩潰至 2-4%
3. 總損失被 Distance Loss 主導

## 實驗設計

### 實驗 A: 平衡權重 (balanced)
```
feature_loss_weight: 1.0
distance_loss_weight: 0.01  # 降低 10 倍
vq_loss_weight: 0.0
```
Loss 貢獻:
- Feature: 0.03 × 1.0 = 0.03
- Distance: 3.5 × 0.01 = 0.035

### 實驗 B: 純 Feature Distillation (FD_v2)
```
feature_loss_weight: 1.0
distance_loss_weight: 0.0   # 完全關閉
vq_loss_weight: 0.0
```

## 實驗結果

### Epoch 1 比較 (兩個實驗初始值相同)
| 指標 | FD_v2 (d=0) | Balanced (d=0.01) |
|------|-------------|-------------------|
| Train Token Acc | 30.24% | 30.24% |
| Val Token Acc | 20.46% | 20.46% |
| Total Loss | 0.0430 | 0.0755 |

### FD_v2 實驗趨勢 (distance=0)
| Epoch | Feature Loss | Val Token Acc |
|-------|-------------|---------------|
| 1 | 0.0417 | 20.46% |
| 5 | 0.0342 | 2.85% |
| 9 | 0.0326 | 4.41% |

### Balanced 實驗趨勢 (distance=0.01)
| Epoch | Feature Loss | Val Token Acc |
|-------|-------------|---------------|
| 1 | 0.0417 | 20.46% |
| 25 | 0.0304 | 4.19% |
| 31 | 0.0301 | 1.63% |

## 關鍵發現

### ⚠️ 發現 1: Token Accuracy 下降是 LoRA 微調的固有問題
即使完全關閉 Distance Loss (FD_v2)，Token Accuracy 仍然從 20% 降至 ~4%。
這表明 LoRA 微調本身會破壞 encoder 原有的離散化能力。

### ✅ 發現 2: Feature Loss 持續下降
兩個實驗的 Feature Loss 都穩定下降：
- FD_v2: 0.0417 → 0.0326 (↓22%)
- Balanced: 0.0417 → 0.0301 (↓28%)

### ⚠️ 發現 3: Distance Loss 並非主要問題
原本假設 Distance Loss 權重過高導致 Token Acc 崩潰，
但關閉 Distance Loss 後問題依然存在。

## 解讀

1. **Feature 對齊 ≠ Token 對齊**: 
   - LoRA 學會產生相似的連續特徵 (Feature Loss ↓)
   - 但這些特徵通過 VQ 離散化後，token 不一定相同
   
2. **LoRA 可能改變了特徵分佈**:
   - 原始 encoder 產生的特徵落在特定的 codebook 區域
   - LoRA 微調後特徵分佈可能偏移，導致 VQ 選擇不同的 token

3. **需要更直接的 token 對齊目標**:
   - 可能需要加入 token-level 的監督 (如 Cross-Entropy)
   - 或使用 STE (Straight-Through Estimator) 讓梯度流過 VQ

## 後續建議

1. **實驗 C: Token-level Cross-Entropy Loss**
   ```
   ce_loss = CrossEntropy(student_logits, teacher_tokens)
   total_loss = feature_loss + 0.1 * ce_loss
   ```

2. **實驗 D: 更小的 LoRA rank**
   - 目前 rank=16 可能改變太多
   - 嘗試 rank=4 或 rank=8

3. **實驗 E: 凍結部分 LoRA 層**
   - 只微調最後 1-2 層
   - 保持前面層的原始能力

4. **實驗 F: 加入 Token Accuracy 作為 Loss**
   - 使用 STE 讓 token 選擇可微分
   - 直接優化 token 準確率

## 目前訓練狀態
- GPU 0: `lora_encoder_1126_1_FD_v2` (distance=0, 進行中)
- GPU 1: `lora_encoder_1126_1_balanced` (distance=0.01, 進行中)

## 重現實驗
```bash
cd exp_1126/1126-1

# 平衡權重版本
bash run_train_balanced.sh

# 純 Feature Distillation
bash run_train_FD_v2.sh
```
