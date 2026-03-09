# Exp 0206 結果總結

**日期**: 2026-02-12
**狀態**: ✅ 兩項實驗完成

---

## 快速結論

兩項實驗皆成功完成 300 epochs，且**都通過 P2 Gate**，成功避免 token collapse。

### 關鍵差異

| 維度 | V2 (RVQ) | Plan Original | 優勝 |
|------|----------|---------------|------|
| **Token Diversity** | Entropy 9.01, Top-10 18.9% | Entropy 9.45, Top-10 12.5% | 🏆 **Plan Ori** |
| **Feature Alignment** | MSE 0.0367 | MSE 0.0418 | 🏆 **V2** |
| **訓練時間** | ~3 天 | ~1 天 | 🏆 **Plan Ori** |
| **架構複雜度** | 4 layers × 2048 | 1 layer × 4096 | 🏆 **Plan Ori** (更簡單) |
| **Codebook 使用率** | 55.7% (1142/2048) | 35.1% (1437/4096) | 🏆 **V2** |

---

## 詳細指標 (Epoch 300)

### Token Diversity

```
V2 (RVQ):
  Entropy:     9.01 bits
  Top-10 mass: 18.88%
  Used codes:  1142/2048 (55.7%)

Plan Original (Single VQ + EMA):
  Entropy:     9.45 bits ⬆️ +4.9% 更好
  Top-10 mass: 12.53% ⬇️ -33.6% 更均勻
  Used codes:  1437/4096 (35.1%)
```

### Feature Alignment

```
V2 (RVQ):
  Feature MSE:    0.0367 ⬇️ 更好
  Train loss:     0.0462
  Val loss:       0.0673

Plan Original:
  Feature MSE:    0.0418
  Train loss:     0.0479
  Val loss:       0.0715
```

### Pass Gates

```
            V2 (RVQ)    Plan Original
P2 Gate:    ✅ PASS     ✅ PASS
P3 Gate:    ❌ FAIL     ❌ FAIL
```

**P2 標準**: Entropy ≥5.0, Top-10 ≤50%, Used ≥410, MSE ≤0.1 ✅
**P3 標準**: Entropy >6.5, Top-10 <15%, Used ≥70% ❌

---

## 科學問題解答

### Q1: 預訓練 codebook + EMA 能否避免 collapse？

**✅ 是的！** Plan Original 成功達到 Entropy 9.45，證明 EMA 更新是關鍵，預訓練初始化無害。

### Q2: Warm start vs Cold start 哪個更好？

**Warm start (Plan Ori) 在 diversity 上略優**，但 Cold start (V2) 在 feature alignment 上稍好。兩者皆有效。

### Q3: 單層 vs 多層差異？

**單層足夠！** Plan Original 用更簡單的架構達到更好的 token diversity，但 V2 在 feature 重建上稍優。

---

## 實用建議

### 選擇 V2 (RVQ) 的場景

- ✅ 需要最佳 feature alignment（音質優先）
- ✅ 需要高 codebook 使用率
- ✅ GPU 資源充足，可接受 3 天訓練

### 選擇 Plan Original 的場景

- ✅ 需要最佳 token diversity（分佈均勻）
- ✅ 訓練時間受限（需要快速迭代）
- ✅ 部署資源受限（模型更小更簡單）
- ✅ 與 WavTokenizer baseline 兼容性

---

## 可視化結果

查看詳細對比圖表：
- [V2_vs_PlanOri_Comparison.png](V2_vs_PlanOri_Comparison.png) - 6 項指標對比
- [V2_vs_PlanOri_Radar.png](V2_vs_PlanOri_Radar.png) - 雷達圖多維度對比

完整分析報告：
- [COMPARISON_V2_vs_PLAN_ORI.md](COMPARISON_V2_vs_PLAN_ORI.md)

---

## 後續工作

- [ ] Plan Original 三模式評估 (Teacher/Noisy VQ/Student)
- [ ] PESQ/STOI 音質對比
- [ ] 主觀聽感測試
- [ ] 推理性能對比 (latency, memory)

---

**結論**: 兩種方案皆成功，可根據實際需求選擇：
- **音質優先** → V2 (RVQ)
- **效率優先** → Plan Original (Single VQ + EMA)

---

**創建**: 2026-02-12
**實驗狀態**: ✅ 完成
