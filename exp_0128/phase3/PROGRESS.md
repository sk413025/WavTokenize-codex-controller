# Phase 3 RVQ (Exp 5a/5b/5c) — Progress Tracker

> Owner: Algorithm Experiment Engineer Agent
> Scope: `exp_0128/phase3/residual_vq` short-run (1000 steps) experiments

## Run Context
- run_started_at: 2026-02-03T03:37:45-05:00
- run_finished_at: 2026-02-03T05:21:56-05:00
- git_commit: 1b2ed5f1f1014c91331bb81c7685213e13ad3ac0
- machine: localhost.localdomain
- conda_env: base
- python: /home/sbplab/miniconda3/bin/python
- torch: 2.9.1+cu128 (torch.version.cuda=12.8)
- cuda_available:
  - sandbox: False (`/dev/nvidia*` open -> Permission denied)
  - escalated: True (torch.cuda.device_count=3)
- gpus:
  - sandbox: `nvidia-smi` fails (NVML init error)
  - escalated (`nvidia-smi -L`):
    - GPU 0: NVIDIA GeForce GTX 1080 Ti
    - GPU 1: NVIDIA GeForce RTX 2080 Ti
    - GPU 2: NVIDIA GeForce RTX 2080 Ti
- notes:
  - CUDA jobs must be run with escalated execution (sandbox blocks GPU device access).
  - `test_decoder.py` required a fallback loader/saver due to missing FFmpeg/torchcodec runtime libs.

## Experiment Tracker
| Exp | Status | GPU | cmd | output_dir | log | start | end | layer0_entropy | layer0_top10_mass | joint_diversity | feature_mse | success | notes |
|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 5a#1 | FAILED | 0 | `bash exp_0128/phase3/residual_vq/run_exp5a.sh` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_040926` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_040926.log` | 2026-02-03T04:09:26-05:00 | 2026-02-03T04:11:29-05:00 |  |  |  |  |  | crash @ initial eval: input audio dim mismatch ([B,T] vs expected [B,1,T]) |
| 5a#2 | FAILED | 0 | `bash exp_0128/phase3/residual_vq/run_exp5a.sh` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_041403` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_041403.log` | 2026-02-03T04:14:03-05:00 | 2026-02-03T04:17:05-05:00 |  |  |  |  |  | OOM @ initial eval: `torch.cdist` in RVQ (memory O(N*K*D)) |
| 5a#3 | FAILED | 0 | `bash exp_0128/phase3/residual_vq/run_exp5a.sh` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_043816` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_043816.log` | 2026-02-03T04:38:16-05:00 | 2026-02-03T04:41:25-05:00 |  |  |  |  |  | initial eval crashed: variable-length time dim across batches (concat error) |
| 5a#4 | FAILED | 0 | `bash exp_0128/phase3/residual_vq/run_exp5a.sh` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_044558` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_044558.log` | 2026-02-03T04:45:58-05:00 | 2026-02-03T04:49:26-05:00 | 4.0197 | 0.8165 | 0.0029 | 0.0754 |  | crashed at first train step: `IntermediateSupervisionLossV6` signature mismatch (fixed) |
| 5a | DONE | 0 | `bash exp_0128/phase3/residual_vq/run_exp5a.sh` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_045100` | `exp_0128/phase3/residual_vq/run_exp5a_20260203_045100.log` | 2026-02-03T04:51:00-05:00 | 2026-02-03T04:59:36-05:00 | 1.9476 | 1.0000 | 0.0005 | 0.0517 | false | severe collapse after training (layer0_used_codes=8/2048) |
| 5b#1 | FAILED | 0 | `bash exp_0128/phase3/residual_vq/run_exp5b.sh` | `exp_0128/phase3/residual_vq/run_exp5b_20260203_050058` | `exp_0128/phase3/residual_vq/run_exp5b_20260203_050058.log` | 2026-02-03T05:01:03-05:00 | 2026-02-03T05:01:03-05:00 |  |  |  |  |  | aborted early (log only header; no traceback captured) |
| 5b | DONE | 0 | `bash exp_0128/phase3/residual_vq/run_exp5b.sh` | `exp_0128/phase3/residual_vq/run_exp5b_20260203_050426` | `exp_0128/phase3/residual_vq/run_exp5b_20260203_050426.log` | 2026-02-03T05:04:30-05:00 | 2026-02-03T05:11:47-05:00 | 2.1229 | 1.0000 | 0.0112 | 0.0437 | false | collapse persists (layer0_used_codes=7/1024) |
| 5c | DONE | 1 | `bash exp_0128/phase3/residual_vq/run_exp5c.sh` | `exp_0128/phase3/residual_vq/run_exp5c_20260203_051334` | `exp_0128/phase3/residual_vq/run_exp5c_20260203_051334.log` | 2026-02-03T05:13:39-05:00 | 2026-02-03T05:21:56-05:00 | 1.7276 | 1.0000 | 0.3310 | 0.0383 | false | joint_diversity improves with more layers, but layer0 still collapses (layer0_used_codes=10/512); note: `CUDA_VISIBLE_DEVICES=1` so runtime device is `cuda:0` |

## Timeline (append-only)
- 2026-02-03T03:37:45-05:00 initialized tracker
- 2026-02-03T03:39:19-05:00 preflight: CUDA unavailable in sandbox; request escalated run for GPU access
- 2026-02-03T03:40:00-05:00 preflight: `test_rvq.py` passed (GPU via escalated execution)
- 2026-02-03T03:47:56-05:00 fix: `test_decoder.py` fallback to `soundfile` (torchaudio torchcodec/FFmpeg missing)
- 2026-02-03T03:48:00-05:00 preflight: `test_decoder.py` passed (GPU via escalated execution)
- 2026-02-03T04:11:29-05:00 Exp5a#1 failed at step 0 (audio dim mismatch → `wavtok_lora_patch` expected [B,C,T])
- 2026-02-03T04:12:35-05:00 fix: `train_rvq_short_run.py` auto-add channel dim when dataset returns [B,T]
- 2026-02-03T04:13:47-05:00 fix: add `set -o pipefail` to `run_exp5{a,b,c}.sh` (fail fast if python crashes)
- 2026-02-03T04:17:05-05:00 Exp5a#2 failed at step 0 (CUDA OOM in RVQ `torch.cdist`)
- 2026-02-03T04:17:57-05:00 fix: RVQ distance calc switched to matmul formula (avoid `torch.cdist` expansion)
- 2026-02-03T04:41:25-05:00 Exp5a#3 failed at step 0 (concat error due to variable-length batches)
- 2026-02-03T04:44:57-05:00 fix: eval metrics now mask padding + handle variable lengths; joint diversity via `torch.unique`
- 2026-02-03T04:49:26-05:00 Exp5a#4 failed at step 0→1 (IntermediateSupervisionLossV6 API mismatch)
- 2026-02-03T04:50:00-05:00 fix: call `IntermediateSupervisionLossV6(student_features, teacher_features)` (V6 returns `(loss, layer_losses)`)
- 2026-02-03T04:51:00-05:00 Exp5a attempt5 started (`run_exp5a_20260203_045100`)
- 2026-02-03T04:59:36-05:00 Exp5a DONE (success=false); final: layer0_entropy=1.9476, layer0_top10_mass=1.0000, joint_diversity=0.0005, feature_mse=0.0517
- 2026-02-03T05:01:03-05:00 Exp5b#1 aborted early (only header in log)
- 2026-02-03T05:04:30-05:00 Exp5b attempt2 started (`run_exp5b_20260203_050426`)
- 2026-02-03T05:11:47-05:00 Exp5b DONE (success=false); final: layer0_entropy=2.1229, layer0_top10_mass=1.0000, joint_diversity=0.0112, feature_mse=0.0437
- 2026-02-03T05:13:14-05:00 fix: `run_exp5c.sh` uses `CUDA_VISIBLE_DEVICES=1`, so set `--device cuda:0` (avoid invalid device ordinal)
- 2026-02-03T05:13:39-05:00 Exp5c started (`run_exp5c_20260203_051334`)
- 2026-02-03T05:21:56-05:00 Exp5c finished (success=false); final: layer0_entropy=1.7276, layer0_top10_mass=1.0000, joint_diversity=0.3310, feature_mse=0.0383
- 2026-02-03T05:25:37-05:00 recorded Exp5c step 600/800/1000 metrics + artifacts into this tracker

## Eval Logs (key metrics @ step 0/200/400/600/800/1000)

### Exp 5a
- step 0: layer0_entropy=4.0197, layer0_top10_mass=0.8165, layer0_used_codes=57/2048 (2.78%), joint_diversity=0.0029, feature_mse=0.0754 (Exp5a attempt5 RUNNING: `run_exp5a_20260203_045100`)
- step 200: layer0_entropy=2.1067, layer0_top10_mass=1.0000, layer0_used_codes=10/2048 (0.49%), joint_diversity=0.0004, feature_mse=0.0695; layers: L0 used=7 ent=1.5416 | L1 used=8 ent=1.5069
- step 400: layer0_entropy=2.1656, layer0_top10_mass=1.0000, layer0_used_codes=7/2048 (0.34%), joint_diversity=0.0003, feature_mse=0.0643; layers: L0 used=6 ent=1.4623 | L1 used=6 ent=1.3238
- step 600: layer0_entropy=2.1740, layer0_top10_mass=1.0000, layer0_used_codes=7/2048 (0.34%), joint_diversity=0.0004, feature_mse=0.0595; layers: L0 used=7 ent=1.4628 | L1 used=8 ent=1.5575
- step 800: layer0_entropy=2.1023, layer0_top10_mass=1.0000, layer0_used_codes=9/2048 (0.44%), joint_diversity=0.0005, feature_mse=0.0554; layers: L0 used=6 ent=1.3683 | L1 used=10 ent=1.7401
- step 1000: layer0_entropy=1.9476, layer0_top10_mass=1.0000, layer0_used_codes=8/2048 (0.39%), joint_diversity=0.0005, feature_mse=0.0517; layers: L0 used=7 ent=1.3641 | L1 used=13 ent=1.8124
- outputs:
  - `exp_0128/phase3/residual_vq/run_exp5a_20260203_045100/summary.json`
  - `exp_0128/phase3/residual_vq/run_exp5a_20260203_045100/metrics_history.json`
  - `exp_0128/phase3/residual_vq/run_exp5a_20260203_045100/loss_history.json`
  - `exp_0128/phase3/residual_vq/run_exp5a_20260203_045100/training_curves.png`
  - `exp_0128/phase3/residual_vq/run_exp5a_20260203_045100/checkpoints/`
  - `exp_0128/phase3/residual_vq/run_exp5a_20260203_045100/final_model.pt`

### Exp 5b
- step 0: layer0_entropy=4.0711, layer0_top10_mass=0.8100, layer0_used_codes=56/1024 (5.47%), joint_diversity=0.0125, feature_mse=0.0754 (Exp5b RUNNING: `run_exp5b_20260203_050426`)
- step 200: layer0_entropy=2.3602, layer0_top10_mass=1.0000, layer0_used_codes=10/1024 (0.98%), joint_diversity=0.0025, feature_mse=0.0640; layers: L0 9/1.4768 | L1 7/1.1907 | L2 6/1.5861 | L3 9/1.7066
- step 400: layer0_entropy=2.3072, layer0_top10_mass=1.0000, layer0_used_codes=8/1024 (0.78%), joint_diversity=0.0037, feature_mse=0.0570; layers: L0 7/1.4345 | L1 6/1.3870 | L2 8/1.5764 | L3 10/1.7090
- step 600: layer0_entropy=2.2073, layer0_top10_mass=1.0000, layer0_used_codes=8/1024 (0.78%), joint_diversity=0.0058, feature_mse=0.0509; layers: L0 7/1.3302 | L1 9/1.4629 | L2 9/1.7617 | L3 13/1.8516
- step 800: layer0_entropy=2.2395, layer0_top10_mass=1.0000, layer0_used_codes=8/1024 (0.78%), joint_diversity=0.0095, feature_mse=0.0469; layers: L0 8/1.2739 | L1 10/1.6988 | L2 11/1.9303 | L3 23/2.4233
- step 1000: layer0_entropy=2.1229, layer0_top10_mass=1.0000, layer0_used_codes=7/1024 (0.68%), joint_diversity=0.0112, feature_mse=0.0437; layers: L0 6/1.3136 | L1 9/1.6898 | L2 17/2.0572 | L3 18/2.5180
- outputs:
  - `exp_0128/phase3/residual_vq/run_exp5b_20260203_050426/summary.json`
  - `exp_0128/phase3/residual_vq/run_exp5b_20260203_050426/metrics_history.json`
  - `exp_0128/phase3/residual_vq/run_exp5b_20260203_050426/loss_history.json`
  - `exp_0128/phase3/residual_vq/run_exp5b_20260203_050426/training_curves.png`
  - `exp_0128/phase3/residual_vq/run_exp5b_20260203_050426/checkpoints/`
  - `exp_0128/phase3/residual_vq/run_exp5b_20260203_050426/final_model.pt`

### Exp 5c
- step 0: layer0_entropy=4.1626, layer0_top10_mass=0.7922, layer0_used_codes=46/512 (8.98%), joint_diversity=0.0509, feature_mse=0.0752 (Exp5c RUNNING: `run_exp5c_20260203_051334`)
- step 200: layer0_entropy=2.0587, layer0_top10_mass=1.0000, layer0_used_codes=9/512 (1.76%), joint_diversity=0.0269, feature_mse=0.0556; layers: L0 7/1.5224 | L1 8/1.4268 | L2 6/1.5677 | L3 9/1.6950 | L4 8/1.3224 | L5 8/1.6681 | L6 10/1.9100 | L7 13/1.3964
- step 400: layer0_entropy=2.0357, layer0_top10_mass=1.0000, layer0_used_codes=7/512 (1.37%), joint_diversity=0.1015, feature_mse=0.0477; layers: L0 7/1.4380 | L1 6/1.6183 | L2 8/1.5491 | L3 11/1.5934 | L4 10/1.5873 | L5 16/1.9430 | L6 18/2.2438 | L7 21/2.0685
- step 600: layer0_entropy=1.9923, layer0_top10_mass=1.0000, layer0_used_codes=7/512 (1.37%), joint_diversity=0.1600, feature_mse=0.0427; layers: L0 6/1.4093 | L1 10/1.5461 | L2 8/1.6092 | L3 12/1.8836 | L4 14/2.0219 | L5 21/2.3956 | L6 27/2.6504 | L7 28/2.8024
- step 800: layer0_entropy=2.0506, layer0_top10_mass=1.0000, layer0_used_codes=7/512 (1.37%), joint_diversity=0.2802, feature_mse=0.0407; layers: L0 6/1.3211 | L1 7/1.6918 | L2 10/1.6984 | L3 16/2.2354 | L4 19/2.3925 | L5 24/2.6642 | L6 35/3.0800 | L7 40/3.1442
- step 1000: layer0_entropy=1.7276, layer0_top10_mass=1.0000, layer0_used_codes=10/512 (1.95%), joint_diversity=0.3310, feature_mse=0.0383; layers: L0 7/1.1492 | L1 9/1.7083 | L2 12/1.9648 | L3 17/2.4484 | L4 20/2.5874 | L5 31/3.0145 | L6 39/3.3282 | L7 47/3.4879
- outputs:
  - `exp_0128/phase3/residual_vq/run_exp5c_20260203_051334/summary.json`
  - `exp_0128/phase3/residual_vq/run_exp5c_20260203_051334/metrics_history.json`
  - `exp_0128/phase3/residual_vq/run_exp5c_20260203_051334/loss_history.json`
  - `exp_0128/phase3/residual_vq/run_exp5c_20260203_051334/training_curves.png`
  - `exp_0128/phase3/residual_vq/run_exp5c_20260203_051334/checkpoints/`
  - `exp_0128/phase3/residual_vq/run_exp5c_20260203_051334/final_model.pt`
