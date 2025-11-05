# CrossAttn 門控（gated, 10–20ep 計畫）vs 基線 K=4 對比（占位）

目的
- 規劃並占位 gated 版本與基線/Deep 的三方對比；待 gated 小跑完成（10ep 或 20ep）後，補齊同一套 E1/E2/E3 指標並更新本檔。

資料與路徑（預計）
- 基線（K=4, 100ep run）：`results/crossattn_k4_100epochs_20251105_051626`
- 門控（gated，小跑）：待定，例如 `results/crossattn_k4_gate_10ep_YYYYMMDD_HHMMSS`

重現指令（待 gated 跑完填入實際目錄後執行）
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_influence_breakdown.py --results_dir <GATED_RESULTS_DIR> --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --batch_size 32

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_margins_topk.py --results_dir <GATED_RESULTS_DIR> --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --k 5 --batch_size 32

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_logit_shift_geometry.py --results_dir <GATED_RESULTS_DIR> --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --batch_size 32
```

對比要點（將與基線/Deep 並列）
- Influence：`W→C - C→W`，低/中 margin 的 ΔAcc 變化。
- Margins/Top‑k：低 margin 抑制、中 margin 提升的幅度與覆蓋率。
- 幾何：cos_mean、Δmargin 在中 margin 的方向性（期望 gated > deep ≥ baseline）。

備註
- 若 gated 10ep 成功，建議延伸至 20ep 與 Deep 20ep 對齊；必要時加入 margin‑aware gate 與方向性輔助損失的 ablation。

