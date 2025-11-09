單語者 boy1 — Baseline Gate 300ep（一致 gate 評估）

實驗設定（摘自 config.json）
- baseline gate（無 quantile‑gate、無 direction loss）
- d_model=512, nhead=8, num_layers=4, ff=2048, dropout=0.1, speaker_tokens=4
- batch_size=96，num_epochs=300

重現/評估（一致 gate）
- 訓練：參見本目錄 config.json 與 training.log（未提交）
- 分析指令（示意）：
  - margins_topk：`python -u done/exp/analyze_margins_topk.py --results_dir <本目錄> --cache_dir done/exp/data/subsets/boy1 --epochs 10 50 100 150 200 250 300 --k 5 --batch_size 24`
  - logit_geometry：`python -u done/exp/analyze_logit_shift_geometry.py --results_dir <本目錄> --cache_dir done/exp/data/subsets/boy1 --epochs 10 50 100 150 200 250 300 --batch_size 24`

關鍵結果（CSV 路徑）
- `analysis/epoch_summary.csv`
- `analysis/margins_topk/epoch_*/margins_bins_epoch_*.csv`
- `analysis/logit_geometry/epoch_*/geometry_epoch_*.csv`
- 其他：`analysis/{attn_entropy,gate_distribution,influence_breakdown,pertoken_mid}/...`

備註
- 最佳驗證準確率（training.log）：約 57.02%
- 本 README 僅作索引，詳見各 CSV 與 config.json。

