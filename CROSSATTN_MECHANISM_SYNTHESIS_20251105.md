# CrossAttn 機轉綜述 + 改進計畫（K=4 修正之後）

**目標**
- 將三份分析（Influence Breakdown、Margins & Top‑k、ΔLogits Geometry）的結論收斂為單一機轉模型，說明為何 Train/Val Acc 在中期出現平臺。
- 規劃兩個可落地的小型介入實驗（10–20 epoch）：margin‑aware gate 與多層注入，並定義評估指標與成功準則。

**一、機轉綜述**
- 共同因子（高層）：
  - 低可辨識性：離散量化與時間對齊噪聲導致大量低 margin token（近決策邊界），「容易翻轉但難以朝目標方向穩定提升」。
  - 低互資訊：I(Y; S | Xnoisy) 偏低，speaker 對 clean token 的附加可預測性有限，多數擾動屬近鄰重排。
  - 注入位置/強度/方向性不足：前置一次 cross‑attn + LayerNorm 稀釋；Δlogits 與目標方向 cosine 近 0 或負；encoder/output 主導學習。
- 證據對照：
  - 非退化（H0 修正）：`results/crossattn_k4_100epochs_20251105_051626/analysis/summary.csv` 中 `attn_w_std≈0.12`、token 變異>0。
  - 翻轉大、淨增益小（H1/H3）：`analysis/influence_breakdown/epoch_40/breakdown_epoch_40.csv`，Normal→Zero: ΔAcc≈+0.364pp；Normal→Random: ΔAcc≈−0.174pp（W→C 與 C→W 互抵）。
  - 邊界洗牌（H2）：`analysis/margins_topk/epoch_40/margins_bins_epoch_40.csv`，低 margin bin flip≈99% 但 ΔAcc≈0；中 margin 少量翻轉 ΔAcc≈+19pp；高 margin 穩定。
  - 方向性不足（H4）：`analysis/logit_geometry/epoch_{20,40}/geometry_epoch_E.csv`，cos_mean 與 Δmargin 在低/中 margin 區多為 0 至負；雖 cos_w2c_mean > cos_c2w_mean（方向性有意義），但整體方向性不足。
- 綜合結論：
  - 「退化」已消除；瓶頸來自「低互資訊 × 低 margin 長尾 × 注入方向/位置/強度不足」的交互，導致大量無方向洗牌翻轉，W→C 與 C→W 長期互抵，平臺形成。

**二、改進設計（小型實驗 10–20 epoch）**
- 介入 A：Margin‑Aware Gate（殘差門控）
  - 直覺：低 margin 區抑制破壞性擾動，高 margin 區保守，中 margin 區放大具方向性的擾動。
  - 門控定義：對 cross‑attn 殘差 r，加入 per‑token gate \(g\in(0,1)\)，\(\tilde h = h + g\cdot r\)。
    - 選項 1（輕量）：用 logits 的局部 margin 作 proxy：\(g = \sigma(w_0 + w_1 \cdot m)\)，\(m = p_1 - p_2\)（可 detatch 以穩定）。
    - 選項 2（學習式）：\(g = \sigma(\text{MLP}(\text{LN}(h)))\)，或 \(\sigma(\text{MLP}([h; r]))\)。
  - 方向性輔助（可選）：最大化 \(\Delta m = (z_t - z_{c2})^N - (z_t - z_{c2})^0\) 或 \(\langle \Delta z, d\rangle\) 的期望；加入小權重正則以免主任務失真。
  - 預期：低 margin bin 的 `delta_target_in_topk` 由負轉正或趨近 0；中 margin W→C↑、C→W↓；Val Acc 輕度抬升。
- 介入 B：多層注入（或 Prepend Speaker Tokens）
  - 直覺：避免「單次前置注入 + LayerNorm 稀釋」，讓條件訊號在 encoder 多層整合，提升方向性。
  - 作法：
    - 在 encoder 第 2、4 層加入 cross‑attn（可參數共享或獨立），仍用 K=4 speaker tokens；或改為在序列前 prepend K 個 speaker tokens，使用原生 self‑attn。
    - 可與 A 的門控疊加：每個注入點皆用門控殘差。
  - 預期：中 margin bin 的 `ΔAcc` 曲線上升；cos_mean 與 Δmargin 整體偏正；Val Acc 超越 ~41–42% 平臺。

**三、實驗設定與步驟（10–20 epoch）**
- 共同設定：
  - 環境：GPU2（2080 Ti）`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`
  - 資料快取：`/home/sbplab/ruizi/c_code/done/exp/data`
  - batch_size=64、lr=1e-4、epochs=10（先小跑驗證）
  - 指標：Val Acc；E1/E2/E3 指標（W→C−C→W、margin‑binned ΔAcc/overlap、cos/Δmargin）
- A：Gate 版本（草案）
  - 代碼草擬：新增 `model_zeroshot_crossattn_gated.py`，在 `CrossAttentionFusion.forward` 或其後加 gate：
    - `margin = softmax(logits).top2_gap().detach()`（或就地以 `hidden` 計算 `g=σ(MLP(LN(hidden)))`）
    - `fused = h + g * attn_output`
  - 訓練：複製 `train_crossattn_cached.py` 為 `train_crossattn_gated_cached.py`，新增 `--use_gate`, `--gate_type`, `--aux_directional_loss_weight`。
  - 命令（示例）：
    - `python -u done/exp/train_crossattn_gated_cached.py --cache_dir ./data --num_epochs 10 --batch_size 64 --learning_rate 1e-4 --speaker_tokens 4 --use_gate margin --aux_directional_loss_weight 0.1`
- B：多層注入（草案）
  - 代碼草擬：新增 `model_zeroshot_crossattn_deep.py`，在 encoder 第 2、4 層插入 cross‑attn（或 prepend tokens 版本 `model_zeroshot_crossattn_prepend.py`）。
  - 訓練：`train_crossattn_deep_cached.py` 新增參數 `--inject_layers 2,4` 或 `--prepend_tokens`。
  - 命令（示例）：
    - `python -u done/exp/train_crossattn_deep_cached.py --cache_dir ./data --num_epochs 10 --batch_size 64 --learning_rate 1e-4 --speaker_tokens 4 --inject_layers 2,4`

**四、評估與成功準則**
- 量化指標（改進即視為成功）：
  - Influence：`W→C − C→W` ↑（至少 +0.5–1.0pp）
  - Margins：中 margin bin 的 `ΔAcc` 顯著上升；低 margin 的 `delta_target_in_topk` 由負轉正或趨近 0
  - Geometry：cos_mean 與 Δmargin 在低/中 margin 區轉為正或接近 0（不再負）
  - 整體：Val Acc > 41.5% 平臺（+0.5–1.0%），Train Acc 輕度上升且 entropy 合理
- 重現分析：
  - `done/exp/analyze_influence_breakdown.py`（E1）、`done/exp/analyze_margins_topk.py`（E2）、`done/exp/analyze_logit_shift_geometry.py`（E3）

**五、時程建議**
- T0（當天）：實作 Gate（輕量 margin 版）+ 10 epoch 小跑 → 產出 E1/E2/E3 指標與報告 commit
- T1：多層注入 10 epoch 小跑 → 同樣分析與 commit
- T2：挑選效果較佳者疊加 LR 調度/WD/SAM，做 50–100 epoch 長跑

