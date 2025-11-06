# CrossAttn Deep（50 epoch）分析報告（草稿）

目的
- 針對 Deep‑50 小跑於 epoch 10/20/30/40/50 執行 E1/E2/E3，驗證「深層條件注入改善方向性與穩定性」之假說，並與 K=4 基線對齊。

資料
- Run 目錄：`results/crossattn_k4_deep_50ep_20251105_211848`
- 分析日誌：`results/crossattn_k4_deep_50ep_20251105_211848/analysis/analysis_gpu0_job.log`

機轉與假說（摘要）
- 使用率高但方向性不足 → 需改善 W→C 的方向性傳遞，降低低‑margin 的破壞性擾動。
- Deep 假說：多層注入（encoder 層 1,3）能加強中‑margin 的有效使用與穩定性。
- 預期：`W→C − C→W` 上升；中‑margin 的 ΔAcc、覆蓋率提高；cos_mean、Δmargin 更正向。

分析項目（E1/E2/E3）
- 與 Gated‑50 同一套指標，按 epoch 對齊。

重現指令（GPU0）
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u done/exp/analyze_influence_breakdown.py \
  --results_dir results/crossattn_k4_deep_50ep_20251105_211848 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 10 20 30 40 50 --batch_size 16

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u done/exp/analyze_margins_topk.py \
  --results_dir results/crossattn_k4_deep_50ep_20251105_211848 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 10 20 30 40 50 --k 5 --batch_size 16

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u done/exp/analyze_logit_shift_geometry.py \
  --results_dir results/crossattn_k4_deep_50ep_20251105_211848 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 10 20 30 40 50 --batch_size 16
```

輸出位置（完成後）
- `results/crossattn_k4_deep_50ep_20251105_211848/analysis/influence_breakdown/epoch_XX/breakdown_epoch_XX.csv`
- `results/crossattn_k4_deep_50ep_20251105_211848/analysis/margins_topk/epoch_XX/*.csv|*.png`
- `results/crossattn_k4_deep_50ep_20251105_211848/analysis/logit_geometry/epoch_XX/geometry_epoch_XX.csv`

---

