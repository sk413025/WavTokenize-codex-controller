單語者 boy1 — Quantile‑Gate 300ep（一致 gate 評估）

實驗設定（摘自 config.json）
- qgate_enable=true；p_low=0.2, p_mid=0.6
- gate_min=0.1, gate_max=0.9；reg_warmup=20
- gate_reg_w: low=0.02, mid=0.02, hi=0.01；gate_reg_t: low=0.2, mid=0.3, hi=0.8
- gate_smooth_weight=0.001
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
- 最佳驗證準確率（training.log）：約 58.17%
- 本 README 僅作索引，詳見各 CSV 與 config.json。

