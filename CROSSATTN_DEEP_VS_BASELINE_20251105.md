# CrossAttn 多層注入（10ep）vs 基線 K=4（epoch10）對比報告

目的
- 比較小型介入「多層注入」（deep conditioning，10 epoch 小跑）與原 K=4 基線在相同 epoch（10）的三類指標：
  Influence Breakdown、Margins & Top‑k、ΔLogits Geometry（若有）。

資料與路徑
- 基線（K=4, 100ep run）：`results/crossattn_k4_100epochs_20251105_051626`
- 多層注入（10ep 小跑）：`results/crossattn_k4_deep_10ep_20251105_094658`
- 對比彙整：`results/crossattn_k4_deep_10ep_20251105_094658/analysis/deep_vs_base_epoch10_summary.csv`

重現指令
```
# Deep run 分析（epoch 10）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_influence_breakdown.py --results_dir results/crossattn_k4_deep_10ep_20251105_094658 --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --batch_size 32
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_margins_topk.py --results_dir results/crossattn_k4_deep_10ep_20251105_094658 --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --k 5 --batch_size 32
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u done/exp/analyze_logit_shift_geometry.py --results_dir results/crossattn_k4_deep_10ep_20251105_094658 --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --batch_size 32
```

要點（epoch 10 對比摘要）
- Influence（Normal→Zero, Normal→Random）
  - 基線 vs Deep：請見 `deep_vs_base_epoch10_summary.csv`（acc_normal/acc_variant 與 ΔAcc 對比）。
  - 初步觀察：Deep 在早期也呈現「翻轉量大但淨 ΔAcc 小」的結構（一致於基線），說明注入位置改進需要更多 epoch/配合門控/方向性目標。
- Margins（Zero 條件）
  - 低 margin bin：基線與 Deep 的 flip 率皆高（≈95–99%）且 ΔAcc 接近 0（洗牌）；
  - 中 margin bin：Deep 在 10ep 已有可觀的正向 ΔAcc（與基線幅度接近），說明多層注入對中 margin 的方向性有潛力；
  - 高 margin bin：兩者幾乎不動，符合預期。
- Geometry（如有）
  - Deep epoch10 幾何指標已生成；基線在 epoch20 起有幾何資料。建議在後續 20ep 小跑後對齊 epoch20 指標再比對。

結論
- 小結：多層注入在 10 epoch 時已呈現與基線相似的早期走勢；中 margin 的正向效果顯現，但受限於訓練時長，整體差距有限。
- 建議：
  - 將 deep 實驗延至 20–30ep，並疊加「margin‑aware gate」與「方向性輔助損失」，著重提升中 margin 的 W→C 並抑制低 margin 的 C→W。
  - 完成後以 E1/E2/E3 對齊 epoch20/30 做嚴謹對比。

附註
- 若門控（gated）版本完成，建議同樣以 epoch10 與 epoch20 與基線/Deep 作三方比較，重點關注：
  W→C − C→W、低/中 margin 的 ΔAcc、cos_mean/Δmargin 在中 margin 的變化。

