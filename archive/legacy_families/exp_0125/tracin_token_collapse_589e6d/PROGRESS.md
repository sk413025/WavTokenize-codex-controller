# Progress: TracIn token collapse (commit 589e6d)

## Checklist
- [x] Step A：準備 run/checkpoints + param scope 驗證（LoRA-only?）
- [x] Step B：量化 collapse/acc/margin（train/val）+ 建立 failure/success set
- [x] Step C：TracIn-CP 計算（至少 2k train candidates；≥50 val failures）
- [x] Step D：Influence 聚合（proponents/opponents profile）+ 圖表
- [x] Step E：最小反事實驗證（downweight/filter 短跑；可選但建議）
- [x] Step F：CONCLUSION（逐條判定 + Top-3 + Proposed Fix + 下一步）
- [x] Acceptance self-check（逐條對照 MUST/SHOULD/COULD）

---

## Step A：準備 run/checkpoints + param scope 驗證（完成）

結果摘要：
- **更新 run_dir**：改用 `exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613`（含完整 checkpoints）。
- 可用 checkpoint：`checkpoints/checkpoint_epoch010.pt` … `checkpoint_epoch300.pt` + `best_model.pt`，可支援 TracIn-CP 多 checkpoint。
- 參數集合驗證：`param_scope.json` 顯示 trainable 僅含 `lora_` 參數（trainable_param_count=3,704,576；non_lora_trainable_count=0）。

下一步：
- Step B：量化 collapse/acc/margin，建立 val failure/success set。

Blockers：
- 無（但缺多 checkpoint，需在結論標注限制）。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && python - <<'PY'\nfrom exp_0112_intermediate.models import TeacherStudentIntermediate\nfrom exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT\nfrom pathlib import Path\nimport json\nmodel = TeacherStudentIntermediate(wavtok_config=WAVTOK_CONFIG, wavtok_ckpt=WAVTOK_CKPT, lora_rank=256, lora_alpha=512, lora_dropout=0.2, intermediate_indices=[3,4,6], device='cpu')\ntrainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]\ntrainable_names = [n for n,_ in trainable]\nsummary = {\n  'trainable_param_count': int(sum(p.numel() for _,p in trainable)),\n  'total_param_count': int(sum(p.numel() for _,p in model.named_parameters())),\n  'trainable_percent': float(sum(p.numel() for _,p in trainable) / sum(p.numel() for _,p in model.named_parameters()) * 100),\n  'trainable_name_contains_lora_only': all('lora_' in n for n in trainable_names),\n  'non_lora_trainable_count': len([n for n in trainable_names if 'lora_' not in n]),\n  'trainable_name_sample': trainable_names[:20],\n  'trainable_name_patterns': ['lora_']\n}\nPath('exp_0125/tracin_token_collapse_589e6d/param_scope.json').write_text(json.dumps(summary, indent=2))\nPY`
- `find exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613 -maxdepth 3 -type f | sort`

---

## Step B：量化 collapse/acc/margin（train/val）+ 建立 failure/success set（完成）

結果摘要：
- 產出 `metrics_overview.json`、`failure_set.json`（run_dir: exp_k_v6）。
- strict acc fw：train **0.04315**、val **0.00913**（gap 仍存在）。
- val collapse 指標：entropy **6.066**、top‑k mass **0.197**、KL **1.247**。
- VQ margin（subset）：p50 train **0.01694** vs val **0.01085**（margin 下降）。
- failure set 大小 **199**（union top‑100 collapse + bottom‑100 acc），success set **100**。

下一步：
- Step C：TracIn-CP 計算（≥2k train candidates、≥50 val failures）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0125/tracin_token_collapse_589e6d/stepB_metrics_failure_set.py --batch_size 2 --num_workers 2 |& tee exp_0125/tracin_token_collapse_589e6d/stepB_metrics_failure_set.log`

---

## Step C：TracIn-CP 計算（完成，近似版）

結果摘要：
- 產出 `tracin_scores.csv`（rows=4000）與 `tracin_indices.json`（train candidates=2000、val failures=50）。  
- **近似設定**：`val_aggregate=True`（以 val failures 平均梯度代表），`approx_train_loss=True`（feature+intermediate，triplet=0），僅 L_train（未跑 L_anchor），checkpoints={epoch010, epoch300}。  
- train 梯度採 batch 近似（batch_size=4），同 batch 共享梯度值；影響力用 aggregated val loss 近似。  
- 註記：此為可行性近似；若需完整 per‑val/雙 loss TracIn，需更長時間或更大算力。

下一步：
- Step D：聚合 influence 分佈，建立 proponents/opponents profile 與圖表。

Blockers：
- 無（但 S1「兩種 loss」未滿足，需在結論標注限制）。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python exp_0125/tracin_token_collapse_589e6d/stepC_tracin_cp.py --approx_train_loss --train_batch_size 4 --val_grad_dtype float16 --val_grad_device cuda --loss_types train --val_aggregate --checkpoints checkpoints/checkpoint_epoch010.pt,checkpoints/checkpoint_epoch300.pt |& tee exp_0125/tracin_token_collapse_589e6d/stepC_tracin_cp.log`

補充（L_anchor 版 TracIn）：
- 產出 `tracin_scores_anchor.csv`（rows=2000）與 `tracin_indices_anchor.json`（train candidates=1000、val failures=50）。  
- 近似設定同樣為 `val_aggregate=True`，loss_type=anchor；為避免 OOM，採 `train_batch_size=1` 並將 val grad 存 CPU。  
- 用於滿足 SHOULD:S1（雙 loss 版本），但樣本數較小（後續可再擴充）。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python exp_0125/tracin_token_collapse_589e6d/stepC_tracin_cp.py --approx_train_loss --train_candidates 1000 --train_batch_size 1 --val_grad_dtype float32 --val_grad_device cpu --loss_types anchor --val_aggregate --checkpoints checkpoints/checkpoint_epoch010.pt,checkpoints/checkpoint_epoch300.pt --output_csv exp_0125/tracin_token_collapse_589e6d/tracin_scores_anchor.csv --meta_out exp_0125/tracin_token_collapse_589e6d/tracin_indices_anchor.json |& tee exp_0125/tracin_token_collapse_589e6d/stepC_tracin_cp_anchor.log`

---

## Step D：Influence 聚合（完成）

結果摘要：
- 產出 `proponents_profile.json`、`opponents_profile.json` 與 `plots/influence_vs_{snr,energy}.png`。
- proponents（正向影響）SNR 更低：mean **-2.61 dB**，Cohen’s d vs all = **-0.265**；noise type 以 **papercup** 為主（61/100）。
- opponents（負向影響）SNR 更高：mean **-1.30 dB**，Cohen’s d vs all = **+0.260**；noise type 以 **plastic/box** 為主（52/45）。
- 顯示「高 influence train 子集偏向噪音材質強」且材質分佈不同，支持 H3/H1 方向。

下一步：
- Step F：整合結論（Top‑3 root causes + Proposed Fix + 下一步實驗）。

Blockers：
- 無。

Commands / Entrypoints：
- `python exp_0125/tracin_token_collapse_589e6d/stepD_influence_profiles.py`
- `python exp_0125/tracin_token_collapse_589e6d/stepD_influence_profiles.py --scores_csv exp_0125/tracin_token_collapse_589e6d/tracin_scores_anchor.csv --meta_json exp_0125/tracin_token_collapse_589e6d/tracin_indices_anchor.json --out_dir exp_0125/tracin_token_collapse_589e6d/anchor_profiles`

---

## Step E：最小反事實驗證（完成）

結果摘要：
- 產出 `counterfactual/summary.json`、`counterfactual/summary.md`。  
- 採用 downweight/filter：從 train candidates 中移除 top‑200 proponents，保留 1800 samples，steps=800。  
- 結果 **未改善**：val strict acc **0.008242**（baseline 0.009135 ↓）、entropy **5.855**（↓）、top‑k mass **0.281**（↑）、KL **1.468**（↑）。  
- 結論：單純移除高 influence 噪音子集會 **惡化 collapse**，顯示需要更細緻的重權策略或保留代表性噪音多樣性。

下一步：
- 若要改進，可測「soft reweighting」或針對特定噪音材質做增強而非刪除。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0125/tracin_token_collapse_589e6d/stepE_counterfactual_short_run.py --steps 800 --batch_size 2 --grad_accum 2 |& tee exp_0125/tracin_token_collapse_589e6d/stepE_counterfactual_short_run.log`

---

## S3：音質交叉檢查（完成）

結果摘要：
- **failure set 重新推理 + 音質指標**：產出 `audio_quality/failure_set/failure_set_metrics.json`、`audio_quality/failure_set/summary.md`，對 val failure 前 50 筆做推理與音質評估（PESQ/STOI/SI‑SDR）。  
- student 音質統計（N=50）：PESQ mean/median **1.082/1.075**、STOI **0.533/0.535**、SI‑SDR **-31.02/-27.90**。  
- 生成 **bottom‑PESQ 子集**（N=30）：`audio_quality/failure_set/failure_set_bottom_pesq.json`。  
- **TracIn 交叉（bottom‑PESQ 子集）**：`audio_quality/tracin_scores_bottom_pesq.csv` + `audio_quality/bottom_pesq_profiles/{proponents,opponents}_profile.json`。  
  - proponents SNR mean **-2.10 dB**（papercup 59/100）、opponents SNR mean **-0.89 dB**（box 77/100）→ 音質最差樣本偏向低 SNR/紙杯材質，與 TracIn proponents 材質一致。  
- 原先 `audio_quality/pesq_stoi_summary.{json,md}`（2 samples）保留作最小 sanity check。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && python exp_0125/tracin_token_collapse_589e6d/stepS3_audio_pesq_stoi.py --epoch epoch_300 |& tee exp_0125/tracin_token_collapse_589e6d/stepS3_audio_pesq_stoi.log`
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 python exp_0125/tracin_token_collapse_589e6d/stepS3_failure_audio_quality.py --max_samples 50 --save_audio --compute_teacher --compute_noisy_vq`
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 python exp_0125/tracin_token_collapse_589e6d/stepC_tracin_cp.py --run_dir exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613 --failure_set exp_0125/tracin_token_collapse_589e6d/audio_quality/failure_set/failure_set_bottom_pesq.json --val_failures 30 --train_candidates 1000 --checkpoints checkpoints/checkpoint_epoch300.pt --loss_types train --val_aggregate --approx_train_loss --train_batch_size 2 --val_grad_device cpu --val_grad_dtype float32 --output_csv exp_0125/tracin_token_collapse_589e6d/audio_quality/tracin_scores_bottom_pesq.csv --meta_out exp_0125/tracin_token_collapse_589e6d/audio_quality/tracin_indices_bottom_pesq.json`
- `python exp_0125/tracin_token_collapse_589e6d/stepD_influence_profiles.py --scores_csv exp_0125/tracin_token_collapse_589e6d/audio_quality/tracin_scores_bottom_pesq.csv --meta_json exp_0125/tracin_token_collapse_589e6d/audio_quality/tracin_indices_bottom_pesq.json --out_dir exp_0125/tracin_token_collapse_589e6d/audio_quality/bottom_pesq_profiles --top_k 100`

---

## Step F：CONCLUSION（完成）

結果摘要：
- 完成 `CONCLUSION.md`：Top‑3 root causes（H3/H2/H1），回答 Q1–Q4，並提出 Primary Fix（noise‑aware reweighting + teacher anchor）。
- 明確標示 TracIn 近似限制（aggregate val loss、L_train 近似、未跑 L_anchor）。
- 完成 Acceptance self‑check（MUST 全滿；SHOULD 未滿部分已標註）。

下一步：
- 若要補 SHOULD：跑 L_anchor TracIn + counterfactual short‑run。

Blockers：
- 無。

---

## 5-checkpoint TracIn 更新（完成，2026-01-27）

動機：
- 原 TracIn 僅用 2 個 checkpoints（epoch010, epoch300），依 TracIn 論文建議，應使用多個 checkpoints（訓練初期、loss 下降最快期、訓練末期）以提高穩健性。

執行：
- 使用 5 個 checkpoints：epoch010、epoch050、epoch150、epoch250、epoch300。
- 產出 `tracin_scores_5ckpt.csv`（10001 rows = 1 header + 2000 train × 5 ckpt）。
- 產出 `profiles_5ckpt/proponents_profile.json` 與 `profiles_5ckpt/opponents_profile.json`。

結果比較：
| 指標 | 2-ckpt | 5-ckpt | 變化 |
|------|--------|--------|------|
| Proponents SNR | -2.61 dB | -2.24 dB | ↑ 0.37 dB |
| Proponents Cohen's d | -0.265 | -0.107 | 效應量下降 |
| Proponents papercup | 61% | 57% | ↓ 4% |
| Opponents SNR | -1.30 dB | -0.68 dB | ↑ 0.62 dB |
| Opponents Cohen's d | +0.260 | +0.499 | 效應量上升 |
| Opponents box | 45% | 58% | ↑ 13% |

結論：
- **方向一致**：proponents 仍偏向低 SNR + papercup，opponents 仍偏向 box。
- **效應量變化**：proponents 的 Cohen's d 從 -0.265 降至 -0.107（小效應），opponents 從 +0.260 升至 +0.499（中等效應）。
- 5-checkpoint 結果更穩健，支持原結論但效應量較保守。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python exp_0125/tracin_token_collapse_589e6d/stepC_tracin_cp.py --approx_train_loss --train_batch_size 4 --val_grad_dtype float16 --val_grad_device cuda --loss_types train --val_aggregate --checkpoints "checkpoints/checkpoint_epoch010.pt,checkpoints/checkpoint_epoch050.pt,checkpoints/checkpoint_epoch150.pt,checkpoints/checkpoint_epoch250.pt,checkpoints/checkpoint_epoch300.pt" --output_csv exp_0125/tracin_token_collapse_589e6d/tracin_scores_5ckpt.csv --meta_out exp_0125/tracin_token_collapse_589e6d/tracin_indices_5ckpt.json |& tee exp_0125/tracin_token_collapse_589e6d/stepC_tracin_5ckpt.log`
- `python exp_0125/tracin_token_collapse_589e6d/stepD_influence_profiles.py --scores_csv exp_0125/tracin_token_collapse_589e6d/tracin_scores_5ckpt.csv --meta_json exp_0125/tracin_token_collapse_589e6d/tracin_indices_5ckpt.json --out_dir exp_0125/tracin_token_collapse_589e6d/profiles_5ckpt --top_k 100`
