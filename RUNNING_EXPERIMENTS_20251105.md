# CrossAttn 正在執行的實驗（2025-11-05）

目的
- 系統性記錄目前 GPU 上三個正在跑的改進實驗（Gated 100ep、Deep 50ep、Deep 100ep）。
- 說明各自的背景/動機/假設、執行與監控方式、分析檢查點（E1/E2/E3）與成功判準。

---

## Gated（門控）100 epoch（GPU2, tmux: gate100）
- 路徑
  - 輸出目錄：`results/crossattn_k4_gate_100ep_20251105_221334`
  - 即時日誌：`results/crossattn_k4_gate_100ep_20251105_221334/console.log`
  - 訓練日誌：`results/crossattn_k4_gate_100ep_20251105_221334/training.log`
- 指令（已啟動）
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python -u done/exp/train_crossattn_gated_cached.py --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --num_epochs 100 --batch_size 64 --learning_rate 1e-4 --speaker_tokens 4 --output_dir results/crossattn_k4_gate_100ep_20251105_221334`
- 模型/改動
  - `done/exp/model_zeroshot_crossattn_gated.py`
  - Cross-Attn 殘差加上 per-token learnable gate：抑制低 margin 雜訊，放大具方向性的擾動。
- 動機/假設（Plateau 修復）
  - 現象：speaker channel 使用率高但對 Val Acc 貢獻小 → 低 margin 區反而被擾動。
  - 假設：用 gate 抑制 C→W、強化 W→C（特別是中 margin 區），可提升有效使用率與泛化。
- 監控
  - attach：`tmux attach -t gate100`
  - tail：`tail -f results/crossattn_k4_gate_100ep_20251105_221334/console.log`
  - GPU：`nvidia-smi -l 2`
  - 檢查點：`results/crossattn_k4_gate_100ep_20251105_221334/checkpoint_epoch_*.pth`、`best_model.pth`
- 分析檢查點（E1/E2/E3）
  - Epochs：10/20/30/40/50/80/100（完成即跑）
  - E1 Influence：`python -u done/exp/analyze_influence_breakdown.py --results_dir results/crossattn_k4_gate_100ep_20251105_221334 --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --batch_size 32`
  - E2 Margins/Top‑k：`python -u done/exp/analyze_margins_topk.py --results_dir results/crossattn_k4_gate_100ep_20251105_221334 --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --k 5 --batch_size 32`
  - E3 ΔLogits 幾何：`python -u done/exp/analyze_logit_shift_geometry.py --results_dir results/crossattn_k4_gate_100ep_20251105_221334 --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --epochs 10 --batch_size 32`
- 成功判準
  - 中 margin：`ΔAcc > 0` 且覆蓋率上升；`W→C − C→W > 0` 且走高。
  - 幾何：中 margin 的 `cos_mean`、`Δmargin` 更正向；Val Acc 隨 epoch 穩定上升。

---

## Deep Injection 50 epoch（GPU1, tmux: deep50）
- 路徑
  - 輸出目錄：`results/crossattn_k4_deep_50ep_20251105_211848`
  - 即時日誌：`results/crossattn_k4_deep_50ep_20251105_211848/console.log`
  - 訓練日誌：`results/crossattn_k4_deep_50ep_20251105_211848/training.log`
- 指令（已啟動）
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u done/exp/train_crossattn_deep_cached.py --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --num_epochs 50 --batch_size 64 --learning_rate 1e-4 --speaker_tokens 4 --inject_layers 1,3 --output_dir results/crossattn_k4_deep_50ep_20251105_211848`
- 模型/改動
  - `done/exp/model_zeroshot_crossattn_deep.py`
  - 在 encoder 多層注入 cross‑attn 融合（`inject_layers=1,3`）。
- 動機/假設
  - 增加深層對 speaker 的條件化路徑，改善方向性傳遞與穩定性，特別是中 margin。
- 監控
  - attach：`tmux attach -t deep50`
  - tail：`tail -f results/crossattn_k4_deep_50ep_20251105_211848/console.log`
  - GPU：`nvidia-smi -l 2`
  - 檢查點：`results/crossattn_k4_deep_50ep_20251105_211848/checkpoint_epoch_*.pth`、`best_model.pth`
- 分析檢查點（E1/E2/E3）
  - Epochs：10/20/30/40/50
  - 指令範例：將 `--results_dir` 換成上列 deep50 目錄，與 Gated 相同模板執行。
- 成功判準
  - 相對基線與 gated：中 margin 的 `ΔAcc`、`W→C − C→W`、幾何指標更優；Val Acc 更快/更高。

---

## Deep Injection 100 epoch（GPU0, tmux: deep100）
- 路徑
  - 輸出目錄：`results/crossattn_k4_deep_100ep_20251105_221426`
  - 即時日誌：`results/crossattn_k4_deep_100ep_20251105_221426/console.log`
  - 訓練日誌：`results/crossattn_k4_deep_100ep_20251105_221426/training.log`
- 指令（已啟動）
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -u done/exp/train_crossattn_deep_cached.py --cache_dir /home/sbplab/ruizi/c_code/done/exp/data --num_epochs 100 --batch_size 64 --learning_rate 1e-4 --speaker_tokens 4 --inject_layers 1,3 --output_dir results/crossattn_k4_deep_100ep_20251105_221426`
- 模型/改動/動機：同上 Deep（50ep），延長訓練以觀察長期趨勢與平臺。
- 監控/分析：同上（Epochs：10/20/30/40/50/80/100）。

---

## 共通監控與疑難排解
- 初期長時間無 epoch 訊息屬正常（WavTokenizer 大模型與 codebook 載入較慢）。
- GPU 監控：`watch -n 1 nvidia-smi`，觀察顯存與利用率跳動以確認訓練進行。
- 若日誌停滯：
  - 連線會話查看 traceback：`tmux attach -t <session>`；
  - 檢查 `training.log` 與 `console.log` 的最後 200 行；
  - 若 OOM：降低 `--batch_size`（如 64 → 48/32）。
- 建議（效能/穩定）
  - 代碼尚未加入 codebook 快取檔：之後可在訓練腳本加入 `--codebook_path`（存在則直接載入 Tensor）以縮短啟動時間。

---

## 分析產物與報告
- 既有分析報告：
  - `CROSSATTN_DEEP_VS_BASELINE_20251105.md`（Deep‑10 vs 基線）
  - `CROSSATTN_GATED_VS_BASELINE_20251105.md`（Gated 對比占位，待新 run 數據補齊）
- 待新增（完成各 epoch 檢查點後）：
  - Gated 100ep：E1/E2/E3 at 10/20/30/40/50/80/100 → 更新 `CROSSATTN_GATED_VS_BASELINE_20251105.md`
  - Deep 50/100ep：對齊相同 epoch，補入 Deep vs Gated vs Baseline 三方彙整。

---

## 成功條件（綜合）
- Val Acc 曲線：無明顯平臺，或最終高於基線相同 epoch 指標。
- Influence：`W→C − C→W` 為正且隨訓練增加（尤其中 margin）。
- Margins：低 margin flip 減少，中 margin 的正向 `ΔAcc` 與覆蓋率上升；高 margin 幾乎不動。
- 幾何：中 margin 的 `cos_mean`、`Δmargin` 明顯更正向，顯示擾動方向更貼近目標。

