# 內容一致性損失實驗指南

本實驗設計了兩種方案來比較內容一致性損失在連續特徵與離散特徵上的表現。

## 🧪 實驗方案

### 方案一：階層式內容一致性損失
- **目標**：結合連續和離散特徵的優勢
- **特點**：使用加權方式平衡連續特徵（保留韻律）和離散特徵（強化語意）
- **腳本**：`run_experiment_hierarchical_content.sh`

### 方案二：純離散內容一致性損失
- **目標**：驗證純離散特徵的表達能力
- **特點**：僅使用離散特徵進行內容一致性約束
- **腳本**：`run_experiment_discrete_content.sh`

## 🚀 運行實驗

### 快速開始

1. **運行方案一**：
```bash
./run_experiment_hierarchical_content.sh
```

2. **運行方案二**：
```bash
./run_experiment_discrete_content.sh
```

### 實驗設定

- **數據範圍**：僅使用 box 材質
- **數據量限制**：每位與者限制前100句話
- **批次大小**：8
- **內容一致性權重**：0.01
- **階層權重**（方案一）：連續特徵 0.7，離散特徵 0.3

## 📊 實驗輸出

### 輸出目錄結構
```
results/tsne_outputs/
├── exp1-hierarchical-{timestamp}/  # 方案一輸出
│   ├── models/                     # 訓練模型
│   ├── features/                   # 特徵文件
│   ├── tsne_plots/                 # t-SNE可視化
│   └── audio_samples/              # 音頻樣本
└── exp2-discrete-{timestamp}/      # 方案二輸出
    ├── models/
    ├── features/
    ├── tsne_plots/
    └── audio_samples/
```

### 日誌文件
- 方案一：`logs/exp1_hierarchical_content_{timestamp}.log`
- 方案二：`logs/exp2_discrete_content_{timestamp}.log`

## 📈 評估指標

### 1. 損失收斂
- 觀察總損失、特徵損失、內容一致性損失的收斂情況
- 比較兩種方案的訓練穩定性

### 2. 語音品質
- 主觀評估增強後的語音品質
- 使用 MOS、PESQ 等客觀指標

### 3. 內容保留
- 使用 ASR 模型評估語音內容的保留程度
- 比較原始語音與增強語音的文字準確率

### 4. 韻律分析
- 分析音調、節奏、重音的保留情況
- 使用韻律特徵提取工具進行客觀評估

### 5. 特徵分析
- t-SNE 可視化特徵分布
- 觀察相同內容樣本的聚集程度

## ⚠️ 注意事項

### 方案一（階層式）
- 監控連續和離散損失的平衡
- 調整 `hierarchy_alpha` 參數優化性能
- 觀察是否能兼顧韻律和語意

### 方案二（純離散）
- 注意是否出現 token collapse
- 監控離散特徵的多樣性
- 評估韻律資訊的損失程度

## 🔧 參數調整

### 階層權重調整（方案一）
```bash
# 更重視連續特徵（保留更多韻律）
python ttt2.py --experiment_hierarchical_content --hierarchy_alpha 0.8

# 更重視離散特徵（強化語意一致性）
python ttt2.py --experiment_hierarchical_content --hierarchy_alpha 0.5
```

### 內容一致性權重調整
```bash
# 增強內容一致性約束
python ttt2.py --experiment_discrete_content --content_alpha 0.05

# 減弱內容一致性約束
python ttt2.py --experiment_discrete_content --content_alpha 0.005
```

## 📋 實驗檢查清單

### 運行前檢查
- [ ] 確認 conda test 環境已激活
- [ ] 確認 box 材質數據存在
- [ ] 確認 GPU 記憶體充足
- [ ] 確認輸出目錄有寫入權限

### 運行中監控
- [ ] 觀察損失收斂情況
- [ ] 監控 GPU 記憶體使用
- [ ] 檢查是否出現異常錯誤
- [ ] 定期檢查輸出樣本品質

### 運行後分析
- [ ] 比較兩種方案的損失曲線
- [ ] 評估音頻樣本品質
- [ ] 分析 t-SNE 可視化結果
- [ ] 整理實驗報告

## 🔍 故障排除

### 常見問題
1. **記憶體不足**：減少批次大小或使用梯度累積
2. **訓練不穩定**：調整學習率或損失權重
3. **特徵維度不匹配**：檢查模型輸出維度設定

### 調試工具
- 使用 `test_experiment_setup.py` 驗證損失函數
- 檢查 `logs/` 目錄中的詳細錯誤信息
- 使用 t-SNE 可視化監控特徵品質

## 📚 實驗結果分析範例

### 成功指標
- 損失穩定收斂
- 語音品質主觀評估良好
- t-SNE 顯示相同內容樣本聚集
- ASR 準確率保持高水準

### 比較分析
- 方案一是否能兼顧韻律和語意？
- 方案二是否出現韻律品質下降？
- 哪種方案的訓練更穩定？
- 在不同評估指標上的表現差異？

---

**實驗完成後，請更新 `REPORT.md` 文件記錄主要發現和結論。**
