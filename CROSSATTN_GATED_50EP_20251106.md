# CrossAttn Gated（50 epoch）分析報告（草稿）

目的
- 針對 Gated‑50 小跑於 epoch 10/20/30/40/50 執行 E1/E2/E3，驗證「抑制低‑margin 雜訊、強化中‑margin 方向性」之假說，並與 K=4 基線對齊。

資料
- Run 目錄：`results/crossattn_k4_gate_50ep_20251105_105730`
- 分析日誌：`results/crossattn_k4_gate_50ep_20251105_105730/analysis/analysis_gpu0_job.log`

機轉與假說（摘要）
- 使用率高但方向性不足：大量翻轉但 ΔAcc 小（移除/隨機化 speaker 時 Val Acc 幾乎不降）。
- Gated 假說：在 Cross‑Attn 殘差上以 per‑token gate 抑制低‑margin 的 C→W，放大中‑margin 的 W→C。
- 預期：`W→C − C→W` 上升且為正；中‑margin 的 ΔAcc、覆蓋率提高；幾何 cos_mean、Δmargin 更正向；Val Acc 改善。

分析項目（E1/E2/E3）
- E1 影響力分解：翻轉率、acc_normal/acc_variant、W→C/C→W、net ΔAcc。
- E2 邊界/Top‑k：按 margin 分 bin 的 flip、ΔAcc、覆蓋率、Top‑k 重疊與 target‑in‑topk 變化。
- E3 ΔLogits 幾何：cos_mean、Δmargin，並切分 W→C/C→W 的 cos。

重現指令（GPU0）
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u done/exp/analyze_influence_breakdown.py \
  --results_dir results/crossattn_k4_gate_50ep_20251105_105730 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 10 20 30 40 50 --batch_size 16

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u done/exp/analyze_margins_topk.py \
  --results_dir results/crossattn_k4_gate_50ep_20251105_105730 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 10 20 30 40 50 --k 5 --batch_size 16

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u done/exp/analyze_logit_shift_geometry.py \
  --results_dir results/crossattn_k4_gate_50ep_20251105_105730 \
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data \
  --epochs 10 20 30 40 50 --batch_size 16
```

輸出位置（完成後）
- `results/crossattn_k4_gate_50ep_20251105_105730/analysis/influence_breakdown/epoch_XX/breakdown_epoch_XX.csv`
- `results/crossattn_k4_gate_50ep_20251105_105730/analysis/margins_topk/epoch_XX/*.csv|*.png`
- `results/crossattn_k4_gate_50ep_20251105_105730/analysis/logit_geometry/epoch_XX/geometry_epoch_XX.csv`

---

