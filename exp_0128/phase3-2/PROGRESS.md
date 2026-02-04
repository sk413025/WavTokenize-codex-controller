# Phase 3-2 RVQ Fix (Exp 6a/6b/6c) — Progress Tracker

> Owner: Algorithm Experiment Engineer Agent  
> Scope: Phase 3-2 per `exp_0128/phase3-2/{PLAN,SPEC,ACCEPTANCE}.md`

## Run Context
- run_started_at: 2026-02-03T23:59:40-05:00
- run_finished_at: 2026-02-04T02:30:30-05:00
- git_commit: 34c1a4325b5a770d6fcb8e79ee6f9c15dd6df871
- machine: localhost.localdomain
- conda_env: base
- python: /home/sbplab/miniconda3/bin/python
- torch: 2.9.1+cu128 (torch.version.cuda=12.8)
- cuda_available:
  - python: True (torch.cuda.device_count=3)
  - nvidia-smi: works with escalated execution; fails in sandbox (NVML init error)
- gpus (`nvidia-smi -L`):
  - GPU 0: NVIDIA GeForce GTX 1080 Ti
  - GPU 1: NVIDIA GeForce RTX 2080 Ti
  - GPU 2: NVIDIA GeForce RTX 2080 Ti

## Experiment Tracker
| Exp | Variant | Status | GPU | cmd | output_dir | log | start | end | step200 (top10/used/feat_mse) | final (entropy/top10/used/joint/feat_mse) | P0 | P1 | P2 | P3 | notes |
|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| 6a | quant_align + commit(codebook-grad) | DONE (EARLY_STOP) | 1 | `bash exp_0128/phase3-2/run_exp6a.sh` | `exp_0128/phase3-2/run_exp6a_20260204_001415` | `exp_0128/phase3-2/run_exp6a_20260204_001415.log` | 2026-02-04T00:14:15-05:00 | 2026-02-04T00:18:17-05:00 | top10=1.0000 / used=9 / mse=0.0636 | ent=2.1227 / top10=1.0000 / used=9 / joint=0.0020 / mse=0.0636 | PASS | FAIL | FAIL | FAIL | collapse_flag@200 → early stop |
| 6b-0.25 | β=0.25 | SKIPPED (covered by Exp6a) | 1 | (reuse Exp6a) | `exp_0128/phase3-2/run_exp6a_20260204_001415` | `exp_0128/phase3-2/run_exp6a_20260204_001415.log` | 2026-02-04T00:14:15-05:00 | 2026-02-04T00:18:17-05:00 | top10=1.0000 / used=9 / mse=0.0636 | ent=2.1227 / top10=1.0000 / used=9 / joint=0.0020 / mse=0.0636 | PASS | FAIL | FAIL | FAIL | β=0.25 already evaluated in Exp6a (collapse_flag@200) |
| 6b-0.5 | β=0.5 | DONE (EARLY_STOP) | 1 | `bash exp_0128/phase3-2/run_exp6b.sh 0.5` | `exp_0128/phase3-2/run_exp6b_beta0p5_20260204_002017` | `exp_0128/phase3-2/run_exp6b_beta0p5_20260204_002017.log` | 2026-02-04T00:20:17-05:00 | 2026-02-04T00:24:40-05:00 | top10=1.0000 / used=10 / mse=0.0639 | ent=2.3988 / top10=1.0000 / used=10 / joint=0.0027 / mse=0.0639 | PASS | FAIL | FAIL | FAIL | collapse_flag@200 → early stop |
| 6b-1.0 | β=1.0 | DONE | 1 | `bash exp_0128/phase3-2/run_exp6b.sh 1.0` | `exp_0128/phase3-2/run_exp6b_beta1p0_20260204_002519` | `exp_0128/phase3-2/run_exp6b_beta1p0_20260204_002519.log` | 2026-02-04T00:25:19-05:00 | 2026-02-04T00:29:56-05:00 | top10=0.9996 / used=11 / mse=0.0639 | ent=3.0130 / top10=0.9996 / used=11 / joint=0.0039 / mse=0.0639 | PASS | FAIL | FAIL | FAIL | still collapsed (top10~1.0); no early-stop (used_codes=11 >= 0.01*K) |
| 6b-2.0 | β=2.0 | DONE (EARLY_STOP) | 1 | `bash exp_0128/phase3-2/run_exp6b.sh 2.0` | `exp_0128/phase3-2/run_exp6b_beta2p0_20260204_003034` | `exp_0128/phase3-2/run_exp6b_beta2p0_20260204_003034.log` | 2026-02-04T00:30:34-05:00 | 2026-02-04T00:34:54-05:00 | top10=1.0000 / used=9 / mse=0.0641 | ent=2.3988 / top10=1.0000 / used=9 / joint=0.0047 / mse=0.0641 | PASS | FAIL | FAIL | FAIL | collapse_flag@200 → early stop |
| 6c-th2 | EMA + reset (th=2) | DONE (CRASH) | 1 | `bash exp_0128/phase3-2/run_exp6c.sh 2 1.0` | `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_003605` | `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_003605.log` | 2026-02-04T00:36:05-05:00 | 2026-02-04T00:39:29-05:00 | crash@step0 (index_add fp16/fp32 mismatch) |  | FAIL | FAIL | FAIL | FAIL | fix required: cast EMA residuals to fp32 |
| 6c-th2-v2 | EMA + reset (th=2) | DONE (CRASH) | 1 | `bash exp_0128/phase3-2/run_exp6c.sh 2 1.0` | `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_004423` | `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_004423.log` | 2026-02-04T00:44:23-05:00 | 2026-02-04T00:47:49-05:00 | crash@step0 (loss_codebook float .item) |  | FAIL | FAIL | FAIL | FAIL | fix required: keep loss tensors across modes |
| 6c-th2-v3 | EMA + reset (th=2) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c.sh 2 1.0` | `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_004848` | `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_004848.log` | 2026-02-04T00:48:48-05:00 | 2026-02-04T00:53:09-05:00 | top10=0.1863 / used=671 / mse=0.0379 | ent=8.4276 / top10=0.1863 / used=671 / joint=0.9856 / mse=0.0379 | PASS | PASS | TBD | TBD | strong anti-collapse + low mse (EMA working) |
| 6c-th5 | EMA + reset (th=5) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c.sh 5 1.0` | `exp_0128/phase3-2/run_exp6c_ema_th5_beta1p0_20260204_005354` | `exp_0128/phase3-2/run_exp6c_ema_th5_beta1p0_20260204_005354.log` | 2026-02-04T00:53:54-05:00 | 2026-02-04T00:58:16-05:00 | top10=0.1934 / used=673 / mse=0.0376 | ent=8.4432 / top10=0.1934 / used=673 / joint=0.9865 / mse=0.0376 | PASS | PASS | TBD | TBD | similar to th=2 (both pass P1 strongly) |
| 6c-long-th2 | EMA + reset LONG (th=2) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_long.sh 2 1.0` | `exp_0128/phase3-2/run_exp6c_long_ema_th2_beta1p0_20260204_005935` | `exp_0128/phase3-2/run_exp6c_long_ema_th2_beta1p0_20260204_005935.log` | 2026-02-04T00:59:35-05:00 | 2026-02-04T01:07:48-05:00 | top10=0.1753 / used=671 / mse=0.0381 | ent=8.3017 / top10=0.2344 / used=671 / joint=0.9748 / mse=0.0339 | PASS | PASS | PASS | FAIL | P2 PASS; P3 FAIL (top10_mass not < 0.15) |
| 6c-tune-inter0.25 | EMA + reset (th=2), λ_inter=0.25 (screen) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 200 2 0.25 0 1.0 4 1024` | `exp_0128/phase3-2/run_exp6c_custom_steps200_ema_th2_beta1p0_inter0p25_warm0_20260204_011727` | `exp_0128/phase3-2/run_exp6c_custom_steps200_ema_th2_beta1p0_inter0p25_warm0_20260204_011727.log` | 2026-02-04T01:17:27-05:00 | 2026-02-04T01:21:50-05:00 | top10=0.1855 / used=671 / mse=0.0374 | ent=8.3927 / top10=0.1855 / used=671 / joint=0.9843 / mse=0.0374 | PASS | PASS | TBD | TBD | similar to 6c-long-th2 @ step200; no top10 improvement |
| 6c-tune-beta2 | EMA + reset (th=2), β=2.0 (screen) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 200 2 0.5 0 2.0 4 1024` | `exp_0128/phase3-2/run_exp6c_custom_steps200_ema_th2_beta2p0_inter0p5_warm0_20260204_012230` | `exp_0128/phase3-2/run_exp6c_custom_steps200_ema_th2_beta2p0_inter0p5_warm0_20260204_012230.log` | 2026-02-04T01:22:30-05:00 | 2026-02-04T01:26:59-05:00 | top10=0.1857 / used=671 / mse=0.0383 | ent=8.4747 / top10=0.1857 / used=671 / joint=0.9870 / mse=0.0383 | PASS | PASS | TBD | TBD | no top10 improvement vs β=1.0 |
| 6c-tune-K2048 | EMA + reset (th=2), K=2048 (screen) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 200 2 0.5 0 1.0 4 2048` | `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_20260204_012751` | `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_20260204_012751.log` | 2026-02-04T01:27:51-05:00 | 2026-02-04T01:32:14-05:00 | top10=0.1385 / used=1081 / mse=0.0380 | ent=9.1231 / top10=0.1385 / used=1081 / joint=0.9924 / mse=0.0380 | PASS | PASS | TBD | TBD | step200 meets P3 top10<0.15, but needs long-run confirmation |
| 6c-long-K2048 | EMA + reset (th=2), K=2048 (long) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 1000 2 0.5 0 1.0 4 2048` | `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_20260204_013230` | `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_20260204_013230.log` | 2026-02-04T01:32:30-05:00 | 2026-02-04T01:40:51-05:00 | top10=0.1290 / used=1080 / mse=0.0381 | ent=8.7480 / top10=0.2314 / used=1088 / joint=0.9834 / mse=0.0338 | PASS | PASS | PASS | FAIL | top10 drifts upward after step200 (P3 FAIL at step1000) |
| 6c-tune-up0.02-K2048 | EMA + reset (th=2), K=2048, usage_penalty=0.02 (screen) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 200 2 0.5 0 1.0 4 2048 0.02` | `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p02_20260204_015027` | `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p02_20260204_015027.log` | 2026-02-04T01:50:27-05:00 | 2026-02-04T01:54:51-05:00 | top10=0.1383 / used=1080 / mse=0.0378 | ent=9.1347 / top10=0.1383 / used=1080 / joint=0.9944 / mse=0.0378 | PASS | PASS | TBD | TBD | step200 still good; need long-run to see if penalty prevents top10 drift |
| 6c-long-up0.02-K2048 | EMA + reset (th=2), K=2048, usage_penalty=0.02 (long) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 1000 2 0.5 0 1.0 4 2048 0.02` | `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p02_20260204_015556` | `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p02_20260204_015556.log` | 2026-02-04T01:55:56-05:00 | 2026-02-04T02:04:09-05:00 | top10=0.1521 / used=1079 / mse=0.0381 | ent=8.8723 / top10=0.2042 / used=1084 / joint=0.9868 / mse=0.0342 | PASS | PASS | PASS | FAIL | top10 improved vs no-penalty long (0.2314 → 0.2042) but still not <0.15 |
| 6c-tune-up0.1-K2048 | EMA + reset (th=2), K=2048, usage_penalty=0.1 (screen) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 200 2 0.5 0 1.0 4 2048 0.1` | `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p1_20260204_020654` | `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p1_20260204_020654.log` | 2026-02-04T02:06:54-05:00 | 2026-02-04T02:11:18-05:00 | top10=0.1545 / used=1079 / mse=0.0379 | ent=9.0704 / top10=0.1545 / used=1079 / joint=0.9939 / mse=0.0379 | PASS | PASS | TBD | TBD | higher penalty slightly increases top10 at step200; need long-run for final impact |
| 6c-long-up0.1-K2048 | EMA + reset (th=2), K=2048, usage_penalty=0.1 (long) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 1000 2 0.5 0 1.0 4 2048 0.1` | `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p1_20260204_021214` | `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p1_20260204_021214.log` | 2026-02-04T02:12:14-05:00 | 2026-02-04T02:20:27-05:00 | top10=0.1354 / used=1075 / mse=0.0382 | ent=9.0286 / top10=0.1583 / used=1089 / joint=0.9917 / mse=0.0336 | PASS | PASS | PASS | FAIL | very close to P3 (top10 0.1583 vs <0.15) |
| 6c-long-up0.12-K2048 | EMA + reset (th=2), K=2048, usage_penalty=0.12 (long) | DONE | 1 | `bash exp_0128/phase3-2/run_exp6c_custom.sh 1000 2 0.5 0 1.0 4 2048 0.12` | `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p12_20260204_022217` | `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p12_20260204_022217.log` | 2026-02-04T02:22:17-05:00 | 2026-02-04T02:30:30-05:00 | top10=0.1378 / used=1079 / mse=0.0377 | ent=8.9630 / top10=0.1774 / used=1089 / joint=0.9890 / mse=0.0337 | PASS | PASS | PASS | FAIL | worse than up=0.1 at step1000; non-monotonic vs penalty |

## Timeline (append-only)
- 2026-02-03T23:59:40-05:00 initialized tracker
- 2026-02-04T00:00:45-05:00 preflight: `git rev-parse HEAD`=34c1a4325b5a770d6fcb8e79ee6f9c15dd6df871
- 2026-02-04T00:01:05-05:00 preflight: `python -c ... torch/cuda` => torch=2.9.1+cu128, cuda_available=True, device_count=3
- 2026-02-04T00:01:25-05:00 preflight: `python -m py_compile exp_0128/phase3/residual_vq/train_rvq_short_run.py` passed
- 2026-02-04T00:02:05-05:00 preflight: `nvidia-smi -L` recorded GPU list
- 2026-02-04T00:04:10-05:00 preflight: `python exp_0128/phase3/residual_vq/test_rvq.py` failed in sandbox (CUDA init error); rerun via `bash -lc` succeeded
- 2026-02-04T00:13:16-05:00 impl: Phase 3-2 loss/commitment/EMA flags wired into `train_rvq_short_run.py` + RVQ now returns `loss_commit`/`loss_codebook` and supports `rvq_update={grad,ema}`
- 2026-02-04T00:14:15-05:00 Exp6a started: `exp_0128/phase3-2/run_exp6a_20260204_001415` (GPU=1 via CUDA_VISIBLE_DEVICES=1)
- 2026-02-04T00:18:17-05:00 Exp6a early-stopped @ step200 (collapse_flag=true); P1 FAIL (top10=1.0, used=9/1024)
- 2026-02-04T00:20:17-05:00 Exp6b started: β=0.5 → `exp_0128/phase3-2/run_exp6b_beta0p5_20260204_002017`
- 2026-02-04T00:24:40-05:00 Exp6b finished: β=0.5, P1 FAIL (top10=1.0, used=10/1024), collapse_flag=true
- 2026-02-04T00:25:19-05:00 Exp6b started: β=1.0 → `exp_0128/phase3-2/run_exp6b_beta1p0_20260204_002519`
- 2026-02-04T00:29:56-05:00 Exp6b finished: β=1.0, P1 FAIL (top10=0.9996, used=11/1024)
- 2026-02-04T00:30:34-05:00 Exp6b started: β=2.0 → `exp_0128/phase3-2/run_exp6b_beta2p0_20260204_003034`
- 2026-02-04T00:34:54-05:00 Exp6b finished: β=2.0, P1 FAIL (top10=1.0, used=9/1024), collapse_flag=true
- 2026-02-04T00:36:05-05:00 Exp6c started: EMA+reset th=2 → `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_003605`
- 2026-02-04T00:39:29-05:00 Exp6c crashed @ step0: `RuntimeError: index_add_(): self (Float) and source (Half)` (EMA update under AMP)
- 2026-02-04T00:43:35-05:00 fix: cast EMA residuals to fp32 + gate EMA update to `model.train()` only (`exp_0128/phase3/residual_vq/models_rvq.py`)
- 2026-02-04T00:44:23-05:00 Exp6c retry started: EMA+reset th=2 → `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_004423`
- 2026-02-04T00:47:49-05:00 Exp6c retry crashed @ step0: `AttributeError: 'float' object has no attribute 'item'` (loss_codebook logging in EMA mode)
- 2026-02-04T00:48:24-05:00 fix: initialize RVQ losses as tensors so `.item()` works in EMA mode (`exp_0128/phase3/residual_vq/models_rvq.py`)
- 2026-02-04T00:48:48-05:00 Exp6c started: EMA+reset th=2 → `exp_0128/phase3-2/run_exp6c_ema_th2_beta1p0_20260204_004848`
- 2026-02-04T00:53:09-05:00 Exp6c finished: th=2, P1 PASS (top10=0.1863, used=671/1024, joint=0.9856, mse=0.0379)
- 2026-02-04T00:53:54-05:00 Exp6c started: EMA+reset th=5 → `exp_0128/phase3-2/run_exp6c_ema_th5_beta1p0_20260204_005354`
- 2026-02-04T00:58:16-05:00 Exp6c finished: th=5, P1 PASS (top10=0.1934, used=673/1024, joint=0.9865, mse=0.0376)
- 2026-02-04T00:59:35-05:00 Exp6c LONG started: EMA+reset th=2 → `exp_0128/phase3-2/run_exp6c_long_ema_th2_beta1p0_20260204_005935`
- 2026-02-04T01:07:48-05:00 Exp6c LONG finished: th=2, P2 PASS; P3 FAIL (final top10=0.2344, ent=8.3017, used=671/1024, joint=0.9748, mse=0.0339)
- 2026-02-04T01:17:27-05:00 Exp6c CUSTOM started: steps=200, K=1024, th=2, β=1.0, λ_inter=0.25 → `exp_0128/phase3-2/run_exp6c_custom_steps200_ema_th2_beta1p0_inter0p25_warm0_20260204_011727`
- 2026-02-04T01:21:50-05:00 Exp6c CUSTOM finished: steps=200, top10=0.1855, used=671/1024, mse=0.0374
- 2026-02-04T01:22:30-05:00 Exp6c CUSTOM started: steps=200, K=1024, th=2, β=2.0, λ_inter=0.5 → `exp_0128/phase3-2/run_exp6c_custom_steps200_ema_th2_beta2p0_inter0p5_warm0_20260204_012230`
- 2026-02-04T01:26:59-05:00 Exp6c CUSTOM finished: steps=200, top10=0.1857, used=671/1024, mse=0.0383
- 2026-02-04T01:27:51-05:00 Exp6c CUSTOM started: steps=200, K=2048, th=2, β=1.0, λ_inter=0.5 → `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_20260204_012751`
- 2026-02-04T01:32:14-05:00 Exp6c CUSTOM finished: steps=200, top10=0.1385, used=1081/2048, mse=0.0380
- 2026-02-04T01:32:30-05:00 Exp6c CUSTOM started: steps=1000, K=2048, th=2, β=1.0, λ_inter=0.5 → `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_20260204_013230`
- 2026-02-04T01:40:51-05:00 Exp6c CUSTOM finished: steps=1000, final top10=0.2314, ent=8.7480, used=1088/2048, joint=0.9834, mse=0.0338 (P3 FAIL at step1000)
- 2026-02-04T01:50:27-05:00 Exp6c CUSTOM started: steps=200, K=2048, th=2, β=1.0, λ_inter=0.5, usage_penalty=0.02 → `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p02_20260204_015027`
- 2026-02-04T01:54:51-05:00 Exp6c CUSTOM finished: steps=200, top10=0.1383, ent=9.1347, used=1080/2048, joint=0.9944, mse=0.0378
- 2026-02-04T01:55:56-05:00 Exp6c CUSTOM started: steps=1000, K=2048, th=2, β=1.0, λ_inter=0.5, usage_penalty=0.02 → `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p02_20260204_015556`
- 2026-02-04T02:04:09-05:00 Exp6c CUSTOM finished: steps=1000, final top10=0.2042, ent=8.8723, used=1084/2048, joint=0.9868, mse=0.0342 (P3 FAIL at step1000)
- 2026-02-04T02:06:54-05:00 Exp6c CUSTOM started: steps=200, K=2048, th=2, β=1.0, λ_inter=0.5, usage_penalty=0.1 → `exp_0128/phase3-2/run_exp6c_custom_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p1_20260204_020654`
- 2026-02-04T02:11:18-05:00 Exp6c CUSTOM finished: steps=200, top10=0.1545, ent=9.0704, used=1079/2048, joint=0.9939, mse=0.0379
- 2026-02-04T02:12:14-05:00 Exp6c CUSTOM started: steps=1000, K=2048, th=2, β=1.0, λ_inter=0.5, usage_penalty=0.1 → `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p1_20260204_021214`
- 2026-02-04T02:20:27-05:00 Exp6c CUSTOM finished: steps=1000, final top10=0.1583, ent=9.0286, used=1089/2048, joint=0.9917, mse=0.0336 (P3 still FAIL; close)
- 2026-02-04T02:22:17-05:00 Exp6c CUSTOM started: steps=1000, K=2048, th=2, β=1.0, λ_inter=0.5, usage_penalty=0.12 → `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p12_20260204_022217`
- 2026-02-04T02:30:30-05:00 Exp6c CUSTOM finished: steps=1000, final top10=0.1774, ent=8.9630, used=1089/2048, joint=0.9890, mse=0.0337 (P3 FAIL)

## Exp 6c-long-th2 — Eval checkpoints (from `metrics_history.json`)
| step | layer0_entropy | layer0_top10_mass | layer0_used_codes | layer0_usage_pct | joint_diversity | feature_mse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.0711 | 0.8100 | 56 | 5.4688 | 0.0125 | 0.0754 |
| 200 | 8.4842 | 0.1753 | 671 | 65.5273 | 0.9845 | 0.0381 |
| 400 | 7.9858 | 0.3111 | 652 | 63.6719 | 0.9739 | 0.0373 |
| 600 | 8.3402 | 0.2188 | 677 | 66.1133 | 0.9883 | 0.0365 |
| 800 | 8.2396 | 0.2344 | 658 | 64.2578 | 0.9843 | 0.0358 |
| 1000 | 8.3017 | 0.2344 | 671 | 65.5273 | 0.9748 | 0.0339 |

## Exp 6c-long-th2 — Loss & per-layer usage
- step 200: losses quant=0.0522, pre=0.0518, inter=0.6648, commit=0.0031, codebook=0.0000; layers L0 used=511 ent=5.7871 pct=49.90%, L1 used=647 ent=6.2603 pct=63.18%, L2 used=561 ent=6.0239 pct=54.79%, L3 used=472 ent=5.5496 pct=46.09%
- step 400: losses quant=0.0460, pre=0.0460, inter=0.6235, commit=0.0010, codebook=0.0000; layers L0 used=422 ent=5.3451 pct=41.21%, L1 used=596 ent=6.1534 pct=58.20%, L2 used=617 ent=6.1840 pct=60.25%, L3 used=520 ent=5.8943 pct=50.78%
- step 600: losses quant=0.0435, pre=0.0434, inter=0.6214, commit=0.0011, codebook=0.0000; layers L0 used=439 ent=5.5447 pct=42.87%, L1 used=626 ent=6.1982 pct=61.13%, L2 used=597 ent=6.1284 pct=58.30%, L3 used=540 ent=5.9383 pct=52.73%
- step 800: losses quant=0.0432, pre=0.0432, inter=0.6216, commit=0.0012, codebook=0.0000; layers L0 used=441 ent=5.5381 pct=43.07%, L1 used=594 ent=6.1319 pct=58.01%, L2 used=632 ent=6.2037 pct=61.72%, L3 used=553 ent=6.0054 pct=54.00%
- step 1000: losses quant=0.0412, pre=0.0412, inter=0.6149, commit=0.0012, codebook=0.0000; layers L0 used=494 ent=5.6977 pct=48.24%, L1 used=646 ent=6.2726 pct=63.09%, L2 used=625 ent=6.2205 pct=61.04%, L3 used=530 ent=5.9100 pct=51.76%

## Exp 6c-long-up0.02-K2048 — Eval checkpoints (from `metrics_history.json`)
| step | layer0_entropy | layer0_top10_mass | layer0_used_codes | layer0_usage_pct | joint_diversity | feature_mse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.5176 | 0.7001 | 63 | 3.0762 | 0.0153 | 0.0754 |
| 200 | 9.0496 | 0.1521 | 1079 | 52.6855 | 0.9937 | 0.0381 |
| 400 | 8.5135 | 0.2869 | 1025 | 50.0488 | 0.9859 | 0.0380 |
| 600 | 8.9617 | 0.1655 | 1086 | 53.0273 | 0.9939 | 0.0363 |
| 800 | 8.8420 | 0.1983 | 1038 | 50.6836 | 0.9907 | 0.0361 |
| 1000 | 8.8723 | 0.2042 | 1084 | 52.9297 | 0.9868 | 0.0342 |

## Exp 6c-long-up0.02-K2048 — Loss & per-layer usage
- step 200: losses quant=0.0521, pre=0.0518, inter=0.6641, commit=0.0029, codebook=0.0000; layers L0 used=654 ent=6.1016 pct=31.93%, L1 used=864 ent=6.5930 pct=42.19%, L2 used=749 ent=6.3597 pct=36.57%, L3 used=635 ent=6.1006 pct=31.01%
- step 400: losses quant=0.0460, pre=0.0460, inter=0.6232, commit=0.0007, codebook=0.0000; layers L0 used=537 ent=5.6353 pct=26.22%, L1 used=806 ent=6.4951 pct=39.36%, L2 used=799 ent=6.4594 pct=39.01%, L3 used=726 ent=6.3177 pct=35.45%
- step 600: losses quant=0.0434, pre=0.0434, inter=0.6214, commit=0.0008, codebook=0.0000; layers L0 used=575 ent=5.8740 pct=28.08%, L1 used=838 ent=6.5306 pct=40.92%, L2 used=821 ent=6.5181 pct=40.09%, L3 used=700 ent=6.2858 pct=34.18%
- step 800: losses quant=0.0426, pre=0.0426, inter=0.6213, commit=0.0009, codebook=0.0000; layers L0 used=610 ent=5.9702 pct=29.79%, L1 used=847 ent=6.5347 pct=41.36%, L2 used=840 ent=6.5190 pct=41.02%, L3 used=760 ent=6.3668 pct=37.11%
- step 1000: losses quant=0.0412, pre=0.0412, inter=0.6150, commit=0.0009, codebook=0.0000; layers L0 used=646 ent=5.9891 pct=31.54%, L1 used=890 ent=6.6287 pct=43.46%, L2 used=806 ent=6.4872 pct=39.36%, L3 used=757 ent=6.3460 pct=36.96%

## Exp 6c-long-up0.1-K2048 — Eval checkpoints (from `metrics_history.json`)
| step | layer0_entropy | layer0_top10_mass | layer0_used_codes | layer0_usage_pct | joint_diversity | feature_mse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.5176 | 0.7001 | 63 | 3.0762 | 0.0153 | 0.0754 |
| 200 | 9.0920 | 0.1354 | 1075 | 52.4902 | 0.9940 | 0.0382 |
| 400 | 8.6566 | 0.2449 | 1023 | 49.9512 | 0.9858 | 0.0365 |
| 600 | 8.8888 | 0.1800 | 1087 | 53.0762 | 0.9931 | 0.0355 |
| 800 | 8.7477 | 0.2168 | 1034 | 50.4883 | 0.9886 | 0.0358 |
| 1000 | 9.0286 | 0.1583 | 1089 | 53.1738 | 0.9917 | 0.0336 |

## Exp 6c-long-up0.1-K2048 — Loss & per-layer usage
- step 200: losses quant=0.0524, pre=0.0520, inter=0.6649, commit=0.0029, codebook=0.0000; layers L0 used=667 ent=6.1291 pct=32.57%, L1 used=865 ent=6.5905 pct=42.24%, L2 used=771 ent=6.4133 pct=37.65%, L3 used=646 ent=6.1135 pct=31.54%
- step 400: losses quant=0.0460, pre=0.0460, inter=0.6235, commit=0.0007, codebook=0.0000; layers L0 used=529 ent=5.6811 pct=25.83%, L1 used=849 ent=6.5450 pct=41.46%, L2 used=796 ent=6.4663 pct=38.87%, L3 used=711 ent=6.2765 pct=34.72%
- step 600: losses quant=0.0435, pre=0.0434, inter=0.6215, commit=0.0008, codebook=0.0000; layers L0 used=545 ent=5.7686 pct=26.61%, L1 used=832 ent=6.5246 pct=40.62%, L2 used=792 ent=6.4740 pct=38.67%, L3 used=712 ent=6.2768 pct=34.77%
- step 800: losses quant=0.0422, pre=0.0422, inter=0.6209, commit=0.0009, codebook=0.0000; layers L0 used=571 ent=5.8603 pct=27.88%, L1 used=816 ent=6.5002 pct=39.84%, L2 used=819 ent=6.5028 pct=39.99%, L3 used=737 ent=6.3133 pct=35.99%
- step 1000: losses quant=0.0411, pre=0.0411, inter=0.6150, commit=0.0009, codebook=0.0000; layers L0 used=639 ent=6.0657 pct=31.20%, L1 used=891 ent=6.6211 pct=43.51%, L2 used=841 ent=6.5069 pct=41.06%, L3 used=757 ent=6.3630 pct=36.96%

## Exp 6c-long-up0.12-K2048 — Eval checkpoints (from `metrics_history.json`)
| step | layer0_entropy | layer0_top10_mass | layer0_used_codes | layer0_usage_pct | joint_diversity | feature_mse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.5176 | 0.7001 | 63 | 3.0762 | 0.0153 | 0.0754 |
| 200 | 9.0966 | 0.1378 | 1079 | 52.6855 | 0.9931 | 0.0377 |
| 400 | 8.5597 | 0.2742 | 1025 | 50.0488 | 0.9902 | 0.0372 |
| 600 | 9.0055 | 0.1496 | 1094 | 53.4180 | 0.9939 | 0.0353 |
| 800 | 8.8137 | 0.1911 | 1036 | 50.5859 | 0.9911 | 0.0360 |
| 1000 | 8.9630 | 0.1774 | 1089 | 53.1738 | 0.9890 | 0.0337 |

## Exp 6c-long-up0.12-K2048 — Loss & per-layer usage
- step 200: losses quant=0.0522, pre=0.0518, inter=0.6641, commit=0.0029, codebook=0.0000; layers L0 used=653 ent=6.1148 pct=31.88%, L1 used=847 ent=6.5504 pct=41.36%, L2 used=749 ent=6.3789 pct=36.57%, L3 used=628 ent=6.0685 pct=30.66%
- step 400: losses quant=0.0459, pre=0.0459, inter=0.6234, commit=0.0007, codebook=0.0000; layers L0 used=540 ent=5.6705 pct=26.37%, L1 used=853 ent=6.5652 pct=41.65%, L2 used=795 ent=6.4655 pct=38.82%, L3 used=691 ent=6.2359 pct=33.74%
- step 600: losses quant=0.0433, pre=0.0433, inter=0.6215, commit=0.0008, codebook=0.0000; layers L0 used=566 ent=5.8591 pct=27.64%, L1 used=822 ent=6.5162 pct=40.14%, L2 used=775 ent=6.4485 pct=37.84%, L3 used=700 ent=6.2779 pct=34.18%
- step 800: losses quant=0.0424, pre=0.0424, inter=0.6211, commit=0.0009, codebook=0.0000; layers L0 used=577 ent=5.8920 pct=28.17%, L1 used=858 ent=6.5672 pct=41.89%, L2 used=854 ent=6.5524 pct=41.70%, L3 used=729 ent=6.3257 pct=35.60%
- step 1000: losses quant=0.0411, pre=0.0411, inter=0.6152, commit=0.0009, codebook=0.0000; layers L0 used=639 ent=6.0170 pct=31.20%, L1 used=889 ent=6.6207 pct=43.41%, L2 used=813 ent=6.4896 pct=39.70%, L3 used=729 ent=6.3318 pct=35.60%
