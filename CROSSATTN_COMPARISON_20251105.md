# Cross-Attention S=1（舊）vs K=4（新）對比總結

目的
- 將舊版 S=1（退化）與新版 K=4（修正）在相同 epoch（10/20/30/40）的關鍵統計對齊比較，確認退化解除與行為差異。

資料與路徑
- 舊 run summary：`/home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/results/crossattn_100epochs_20251105_025951/analysis/summary.csv`
- 新 run summary：`/home/sbplab/ruizi/WavTokenize-self-supervised/results/crossattn_k4_100epochs_20251105_051626/analysis/summary.csv`
- 對比表（已產出）：`/home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/analysis_outputs/crossattn_compare/s1_vs_k4.csv`

重現指令
- `python -u done/exp/compare_crossattn_runs.py \
    --s1_summary done/exp/results/crossattn_100epochs_20251105_025951/analysis/summary.csv \
    --k4_summary results/crossattn_k4_100epochs_20251105_051626/analysis/summary.csv \
    --out_csv done/exp/analysis_outputs/crossattn_compare/s1_vs_k4.csv`

主要發現
- 注意力退化解除：S=1 → `attn_w_std=0.0`；K=4 → `attn_w_std≈0.12`、`token_var_proxy>0`（隨 token 變化）。
- 參數範數走勢相近：encoder/output head 持續上升；K=4 早中期有更佳的 Val Acc（見各 run 的 training.log），但 20 後趨於平臺。
- 說明：K=4 修復了結構問題，但效能停滯另有因素（決策邊界、使用位置/強度、對齊上限），需進一步機轉分析（見後續影響力/邊界/Δlogits 幾何）。

