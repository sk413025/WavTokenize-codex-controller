# exp_0128 One-pager：Phase 1 結果統整 + Phase 2 行動建議

**更新日期**：2026-01-29  
**涵蓋 commits**：
- `d0f9ecb7654a77b66782e1013be4b153f834e2c4`（exp_0125：5-checkpoint TracIn 診斷完成）
- `997d3577974df3998533a195c86134094bcf2b80`（exp_0128：Phase 1 完整結果報告）

---

## TL;DR（最重要結論）

- Phase 1 的兩種「採樣/權重」短期修復（TracIn-weighted、Noise-balanced）**都失敗**；訓練後 collapse 比 baseline 更嚴重。
- 兩個實驗在 **Step 0（未訓練 LoRA）指標反而優於 baseline（epoch 300）**，且約 **step ~200** 就開始快速崩壞 → **token collapse 更像是訓練動態/目標函數/優化行為導致**，不是單純資料分佈或 sampling 可以解的問題。
- Phase 2 應優先做「訓練動態修正」：**Entropy regularization**、**Codebook refresh**（已在 `exp_0128/phase2/` 準備好腳本，但目前尚無執行結果）。

---

## 背景（TracIn 診斷 → Phase 1 假設）

exp_0125（5-checkpoint TracIn）指出：導致 val failure 的高影響 train samples（proponents）在 profile 上呈現：
- **papercup 過度代表**：57%（全體 train 33%）
- **較低 SNR**：-2.24 dB（全體 train -1.88 dB）
而 opponents 偏向：
- **box**：58%
- **較高 SNR**：-0.68 dB

Phase 1 因此提出兩個短期驗證：
1. **TracIn-Weighted Soft Reweighting**：對高 influence proponents 降權（soft），避免硬刪樣本造成代表性破壞
2. **Noise-Balanced Sampling**：強制 batch 內噪音材質分佈更平均，避免被特定 noise profile 主導更新

---

## 統一指標表（Baseline / Phase 1 / Phase 2）

> 指標越「健康」：Entropy 越高越好；Top-10 Mass 越低越好；Strict Acc 越高越好。

| Phase | 實驗 | 目的/方法 | 主要設定 | Entropy | Top-10 Mass | Strict Acc | 判定 |
|---|---|---|---|---:|---:|---:|---|
| Baseline | exp_k v6 @ epoch 300 | 既有訓練配置 | LoRA rank 256, lr 1e-4 | 6.07 | 19.7% | 0.91% | - |
| Phase 1 | Exp1：TracIn weighted | 依 TracIn score 重加權抽樣 | α=0.5 | 5.63 | 29.0% | 0.60% | ❌ 失敗 |
| Phase 1 | Exp2：Noise balanced | batch 內噪音材質平衡 | sampler 平衡抽樣 | 5.56 | 28.3% | 0.52% | ❌ 失敗 |
| Phase 2 (待跑) | Exp3a/b/c：Entropy reg | loss 加入 entropy maximization | λ ∈ {0.01, 0.05, 0.1} | TBD | TBD | TBD | - |
| Phase 2 (待跑) | Exp4a/b/c/d：Codebook refresh | 週期性重置未用 code | interval ∈ {50,100}, threshold ∈ {5,10} | TBD | TBD | TBD | - |

---

## Phase 1 的關鍵觀察（為什麼推論「訓練動態」）

### 1) 兩個完全不同的 sampler，失敗型態卻極度相似

Exp1 vs Exp2 最終指標差異很小：
- Entropy：5.63 vs 5.56
- Top-10 Mass：29.0% vs 28.3%
- Strict Acc：0.60% vs 0.52%

→ 暗示「採樣策略」不是主導因素（至少在此短跑設定下）。

### 2) Step 0（未訓練）比 baseline（訓練後）更好

Phase 1 的 Step 0：
- Entropy 6.26（> 6.07）
- Top-10 Mass 11.8%（< 19.7%）
- Strict Acc 3.52%（> 0.91%）

→ **訓練本身把模型推向更差的 token 分佈與對齊品質**。

### 3) Collapse 出現得很早（step ~200）

兩個實驗在 step 200 就出現：
- Entropy 明顯下降到 ~5.5–5.7
- Top-10 Mass 明顯上升到 ~20–27%

→ 更像是「loss landscape / 優化路徑」快速把模型帶到 collapse basin。

---

## 重要比較注意事項（避免誤判）

Phase 1 的 short-run 與 baseline training loop **並非完全同構**，以下差異會影響「可比性」的嚴謹程度：

1. **總 loss 混合公式不同**
   - Phase 1 short-run：`loss = iw*inter + (1-iw)*main`（weighted average）
   - baseline：`total_loss = final_loss + iw*inter_loss`（加總）
   → 這會改變 main vs intermediate 的相對尺度與梯度比例。

2. **Intermediate weight 動態策略不同**
   - Phase 1：固定 `iw = 0.5`
   - baseline：`iw` 會隨 epoch warmdown（0.5 → 0.25）

3. **LR schedule 不同**
   - Phase 1：固定 lr（無 warmup / cosine scheduler）
   - baseline：有 warmup + cosine 類型 scheduler

> 結論不變：即使考慮上述差異，Phase 1 仍清楚顯示「光靠 sampler 調整」無法逆轉 collapse，下一步應把改動放在 loss / codebook / optimization dynamics。

---

## Phase 2 建議的最小可行跑法（先求快驗證）

優先跑 2 個最省時間、訊號最直接的設定：

```bash
# Entropy regularization（建議先跑中等強度）
bash exp_0128/phase2/entropy_regularization/run_exp3b.sh

# Codebook refresh（建議先跑保守設定）
bash exp_0128/phase2/codebook_refresh/run_exp4a.sh
```

**成功判準（沿用 Phase 1 報告）**：
- entropy > 6.07
- top_10_mass < 0.197
- strict_acc ≥ 0.82%

