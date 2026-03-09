# Phase 3-3 — Hot-Code Branching / Codebook Split（進度追蹤）

> Scope: `exp_0128/phase3-3`（Exp 7a/7b/7c/7d）  
> Owner: Codex Agent（Algorithm Experiment Engineer）

## Run Context（latest preflight）

- Timestamp: **2026-02-05T04:33:25-05:00**
- `git rev-parse HEAD`: `c3faea93530cd779af6e58dd110fcb8b3e768dc0`
- Conda env: `test`
- Python: `Python 3.10.13` (`/home/sbplab/miniconda3/envs/test/bin/python`)
- CUDA selection: **do not set** `CUDA_VISIBLE_DEVICES` (setting it currently triggers `cudaGetDeviceCount` Error 304 in this environment); select GPU via `--device cuda:1` / `cuda:2` (RTX 2080 Ti)
- PyTorch / CUDA: `torch 2.5.1` (torch.version.cuda=11.8); `torch.cuda.is_available()=True`; `torch.cuda.device_count()=3`
  - `torch.cuda.get_arch_list()` includes `sm_61`/`sm_75` → GTX 1080 Ti / RTX 2080 Ti are supported by this PyTorch build.
- `nvidia-smi -L`:
  - GPU 0: NVIDIA GeForce GTX 1080 Ti (UUID: GPU-222f3d63-5699-313a-bb97-b59ff09b662a)
  - GPU 1: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-a5b54173-a1ea-80b6-adcd-b5415a33d660)
  - GPU 2: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-2f7942cb-2e01-9194-6128-8e6840960365)

## Phase 3-2 Gate（建議先決條件）

- Gate condition: Phase 3-2 至少通過 P1（layer0_top10_mass 不再≈1.0；layer0_used_codes 不再個位數）
- Status: **PASS (P1 satisfied)**
- Evidence (reference run): `exp_0128/phase3-2/run_exp6c_custom_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_warm0_up0p2_ups600_upr200_seed43_20260204_100722`
  - step200: layer0_top10_mass=0.2723, layer0_used_codes=912/2048, layer0_entropy=8.3698, feature_mse=0.0374

## Experiment Tracker（Exp7a → 7b → 7c → 7d）

| Exp | Status | cmd | output_dir | log | start | end | final metrics (layer0_entropy / top10_mass / used_codes / feature_mse) | P1 | P2 | P3 | Notes |
|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| 7a | **COMPLETED** (full @ step1000) | `conda activate test; CUDA_VISIBLE_DEVICES=GPU-a5b5... python train_rvq_short_run.py --steps 1000 --enable_hot_split --split_k 3 --split_one_shot_step 0` | `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_034115` | `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_034115/train.log` | 2026-02-05T03:41:21-05:00 | 2026-02-05T03:49:35-05:00 | 8.9786 / 0.1698 / 1082 / 0.0339 (step1000) | PASS | PASS | N/A | P2 strong; but split group shows late child collapse (child_active_count=1 @ step800/1000) → Exp7c should address |
| 7b | **COMPLETED** | `conda activate test; CUDA_VISIBLE_DEVICES=GPU-a5b5... python train_rvq_short_run.py --steps 1000 --enable_hot_split --split_interval 200 --split_hot_k 1 --split_k 3` | `exp_0128/phase3-3/run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256` | `exp_0128/phase3-3/run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256/train.log` | 2026-02-05T03:53:02-05:00 | 2026-02-05T04:01:15-05:00 | 8.9774 / 0.1602 / 1104 / 0.0338 (step1000) | PASS | PASS | N/A | P1 PASS for all split events (first post-split eval child_active_count>=2); some transient collapse at step800 |
| 7c | **COMPLETED** | `conda activate test; CUDA_VISIBLE_DEVICES=GPU-a5b5... python train_rvq_short_run.py --steps 1000 --enable_hot_split --split_interval 200 --split_repulse_weight 1e-3` | `exp_0128/phase3-3/run_exp7c_periodic_int200_hot1_k3_repulse_w1e-3_sigma1_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_040845` | `exp_0128/phase3-3/run_exp7c_periodic_int200_hot1_k3_repulse_w1e-3_sigma1_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_040845/train.log` | 2026-02-05T04:08:50-05:00 | 2026-02-05T04:17:21-05:00 | 8.9116 / 0.1860 / 1104 / 0.0339 (step1000) | PASS | PASS | N/A | Repulsion prevents final child collapse (step1000 child_active_min=2) but one group still highly imbalanced (entropy_norm=0.262) |
| 7d | **COMPLETED** | `conda activate test; python analyze_proxy_separation.py --base_run_dir run_exp7b... --max_batches 50 --min_child_tokens 20` | `exp_0128/phase3-3/run_exp7d_proxy_snr_from_run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256_20260205_050353` | `exp_0128/phase3-3/run_exp7d_proxy_snr_from_run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256_20260205_050353/analysis.log` | 2026-02-05T05:03:53-05:00 | 2026-02-05T05:05:34-05:00 | N/A (analysis); abs_d_median=0.128, pass_groups=1/4 | N/A | N/A | FAIL | SNR separation is weak: only 1/4 split groups show abs(d)>=0.5 (and only one is moderately separated) |

## Gate Decision（Go / No-Go）

- P1 (split 被用到): **PASS**（Exp7a/7b/7c split 後第一個 eval，`child_active_count>=2`）
- P2 (collapse 指標改善): **PASS**（step1000 `layer0_top10_mass≈0.16~0.19`, `used_codes≈1080~1100`, `feature_mse≈0.034`）
- P3 (speech/noise proxy 分離): **FAIL / weak evidence**（sample-level SNR：只有 1/4 groups `abs(d)>=0.5`，其餘接近 0）
- Decision:
  - **GO**（作為 anti-collapse / capacity 增補機制，值得把 Exp7b periodic split 納入後續主線 ablation）
  - **NO-GO**（目前不足以宣稱 “speech/noise 被同一 code 混在一起且可被 split 穩定分開”；若此為核心目標需改 proxy/方法）

## Per-run Notes（固定回報格式）

### Exp7a

#### [Phase3-3][Exp7a][k=3][COMPLETED] (env=test, steps=1000, eval@0/200/400/600/800/1000)
- cmd: `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=GPU-a5b54173-a1ea-80b6-adcd-b5415a33d660 PYTHONUNBUFFERED=1 python -u exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 1000 --batch_size 8 --grad_accum 2 --lr 1e-4 --n_rvq_layers 4 --rvq_codebook_size 2048 --rvq_update ema --ema_decay 0.99 --ema_eps 1e-5 --ema_dead_code_threshold 2 --ema_usage_penalty 0.0 --lambda_quant 1.0 --lambda_pre 0.0 --lambda_inter 0.5 --beta_commit 1.0 --lambda_codebook 1.0 --inter_warmup_steps 0 --eval_interval 200 --eval_max_batches 50 --output_dir exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_034115 --seed 42 --device cuda:0 --cuda_preinit_retries 30 --cuda_preinit_sleep_s 1 --enable_hot_split --split_layer 0 --split_k 3 --split_hot_k 1 --split_interval 0 --split_one_shot_step 0 --split_init_method noise --split_init_std 1e-3`
- output_dir: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_034115`
- log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_034115/train.log`
- step: 1000 (completed)
- split_event (step/parent/children/init_method/std): step0 parent=617 -> children=[0,1], init=noise, std=1e-3
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
  - step0: 4.5176 / 0.7001 / 63
  - step200: 9.1194 / 0.1407 / 1091
  - step400: 8.5976 / 0.2581 / 1022
  - step600: 8.8177 / 0.2110 / 1077
  - step800: 8.7492 / 0.2236 / 1034
  - step1000: 8.9786 / 0.1698 / 1082
- feature_mse:
  - step0: 0.0754
  - step200: 0.0387
  - step400: 0.0370
  - step600: 0.0351
  - step800: 0.0360
  - step1000: 0.0339
- group metrics (child_active_count / group_entropy / top-child usage):
  - step200 group0(parent=617): child_active_count=2; children_counts=[93,19]; parent_count=107; group_entropy_children=0.4553 (norm=0.6569); top_child=0 (0.8304); parent_frac_in_group=0.4886
  - step400 group0(parent=617): child_active_count=2; children_counts=[45,142]; parent_count=0; group_entropy_children=0.5518 (norm=0.7961); top_child=1 (0.7594); parent_frac_in_group=0.0000
  - step600 group0(parent=617): child_active_count=2; children_counts=[23,42]; parent_count=107; group_entropy_children=0.6498 (norm=0.9375); top_child=1 (0.6462); parent_frac_in_group=0.6221
  - step800 group0(parent=617): child_active_count=1 (⚠️ late collapse); children_counts=[53,0]; parent_count=25; group_entropy_children=0.0000; top_child=0 (1.0000); parent_frac_in_group=0.3205
  - step1000 group0(parent=617): child_active_count=1 (⚠️ late collapse); children_counts=[25,0]; parent_count=24; group_entropy_children=0.0000; top_child=0 (1.0000); parent_frac_in_group=0.4898
- proxy separation (if enabled): N/A
- P1/P2/P3 status: PASS / PASS / N/A
- notes: P2 PASS (final collapse metrics strong). Split group shows **late-stage child collapse** (child_active_count=1 at step800/1000) → Exp7c should target this.
- next: proceed to Exp7b (periodic split interval=200) to test repeated splits + stability; then Exp7c (add balance reg) if collapse persists

#### [Phase3-3][Exp7a][k=3][COMPLETED] (env=test, steps=200, eval@0/200)
- cmd: `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=GPU-a5b54173-a1ea-80b6-adcd-b5415a33d660 PYTHONUNBUFFERED=1 python -u exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 200 --batch_size 8 --grad_accum 2 --lr 1e-4 --n_rvq_layers 4 --rvq_codebook_size 2048 --rvq_update ema --ema_decay 0.99 --ema_eps 1e-5 --ema_dead_code_threshold 2 --ema_usage_penalty 0.0 --lambda_quant 1.0 --lambda_pre 0.0 --lambda_inter 0.5 --beta_commit 1.0 --lambda_codebook 1.0 --inter_warmup_steps 0 --eval_interval 200 --eval_max_batches 50 --output_dir exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_033126 --seed 42 --device cuda:0 --cuda_preinit_retries 30 --cuda_preinit_sleep_s 1 --enable_hot_split --split_layer 0 --split_k 3 --split_hot_k 1 --split_interval 0 --split_one_shot_step 0 --split_init_method noise --split_init_std 1e-3`
- output_dir: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_033126`
- log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_033126/train.log`
- step: 200 (completed)
- split_event (step/parent/children/init_method/std): step0 parent=617 -> children=[0,1], init=noise, std=1e-3 (see `split_history.json`)
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
  - step0: 4.5176 / 0.7001 / 63
  - step200: 9.1634 / 0.1244 / 1092
- feature_mse:
  - step0: 0.0754
  - step200: 0.0388
- group metrics (child_active_count / group_entropy / top-child usage):
  - step200 group0(parent=617): child_active_count=2; children_counts=[104,27]; parent_count=160; group_entropy_children=0.5088 (norm=0.7340); top_child=0 (0.7939); parent_frac_in_group=0.5498
- proxy separation (if enabled): N/A
- P1/P2/P3 status: PASS / TBD (needs step1000) / N/A
- notes: P0 ok (no crash/NaN). P1 pass at step200: child_active_count=2 (split really used) and collapse metrics strong.
- next: run a step1000 run (Exp7a full or Exp7b step1000) to evaluate P2 at final.

#### [Phase3-3][Exp7a][k=3][FAILED] (attempt1)
- cmd: `CUDA_VISIBLE_DEVICES=0 python exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 200 --batch_size 8 --grad_accum 2 --lr 1e-4 --n_rvq_layers 4 --rvq_codebook_size 2048 --rvq_update ema --ema_decay 0.99 --ema_eps 1e-5 --ema_dead_code_threshold 2 --ema_usage_penalty 0.0 --lambda_quant 1.0 --lambda_pre 0.0 --lambda_inter 0.5 --beta_commit 1.0 --lambda_codebook 1.0 --inter_warmup_steps 0 --eval_interval 200 --output_dir exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_005538 --seed 42 --device cuda:0 --enable_hot_split --split_layer 0 --split_k 3 --split_hot_k 1 --split_interval 0 --split_one_shot_step 0 --split_init_method noise --split_init_std 1e-3`
- output_dir: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_005538`
- log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_005538.log`
- step: N/A (crash during CUDA init)
- split_event (step/parent/children/init_method/std): N/A (never reached)
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
- feature_mse:
- group metrics (child_active_count / group_entropy / top-child usage):
- proxy separation (if enabled):
- P1/P2/P3 status: FAIL (P0 crash)
- notes: `RuntimeError: Unexpected error from cudaGetDeviceCount (Error 304)` during `TeacherStudentRVQ(...).to(cuda:0)`. Repro: setting `CUDA_VISIBLE_DEVICES` triggers this in current env.
- next: rerun Exp7a **without** `CUDA_VISIBLE_DEVICES` (keep `--device cuda:0`), and set `PYTHONUNBUFFERED=1` for streaming logs.

#### [Phase3-3][Exp7a][k=3][FAILED] (attempt2)
- cmd: `PYTHONUNBUFFERED=1 python -u exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 200 --batch_size 8 --grad_accum 2 --lr 1e-4 --n_rvq_layers 4 --rvq_codebook_size 2048 --rvq_update ema --ema_decay 0.99 --ema_eps 1e-5 --ema_dead_code_threshold 2 --ema_usage_penalty 0.0 --lambda_quant 1.0 --lambda_pre 0.0 --lambda_inter 0.5 --beta_commit 1.0 --lambda_codebook 1.0 --inter_warmup_steps 0 --eval_interval 200 --output_dir exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_010324_attempt2 --seed 42 --device cuda:0 --enable_hot_split --split_layer 0 --split_k 3 --split_hot_k 1 --split_interval 0 --split_one_shot_step 0 --split_init_method noise --split_init_std 1e-3`
- output_dir: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_010324_attempt2`
- log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_010324_attempt2.log`
- step: N/A (crash during CUDA init)
- split_event (step/parent/children/init_method/std): N/A (never reached)
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
- feature_mse:
- group metrics (child_active_count / group_entropy / top-child usage):
- proxy separation (if enabled):
- P1/P2/P3 status: FAIL (P0 crash)
- notes: same `cudaGetDeviceCount (Error 304)` during `TeacherStudentRVQ(...).to(cuda:0)`.
- next: add in-script CUDA pre-init + retry, then rerun Exp7a (attempt3).

#### [Phase3-3][Exp7a][k=3][FAILED] (attempt3)
- cmd: `PYTHONUNBUFFERED=1 python -u exp_0128/phase3/residual_vq/train_rvq_short_run.py ... | tee <log>`
- output_dir: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_011355_attempt3`
- log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_011355_attempt3.log`
- step: N/A
- split_event (step/parent/children/init_method/std): N/A
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
- feature_mse:
- group metrics (child_active_count / group_entropy / top-child usage):
- proxy separation (if enabled):
- P1/P2/P3 status: FAIL (P0 crash)
- notes: `_cuda_preinit` retried 3 times but still failed (`cudaGetDeviceCount` error 304). Suspected shell redirection might be involved; validated by attempt4/5 that **redirect is not the root cause**.
- next: rerun Exp7a (attempt4) without shell redirection, log inside Python to `output_dir/train.log`, then reassess CUDA stability.

#### [Phase3-3][Exp7a][k=3][FAILED] (attempt4)
- cmd: `PYTHONUNBUFFERED=1 python -u exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 200 ... --device cuda:0 --enable_hot_split --split_k 3 --split_init_method noise --split_init_std 1e-3`
- output_dir: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_012127_attempt4`
- log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_012127_attempt4/train.log`
- step: N/A (crash during CUDA init)
- split_event (step/parent/children/init_method/std): N/A
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
- feature_mse:
- group metrics (child_active_count / group_entropy / top-child usage):
- proxy separation (if enabled):
- P1/P2/P3 status: FAIL (P0 crash)
- notes: Still fails at `_cuda_preinit` with `cudaGetDeviceCount` Error 304 even without any shell redirection.
- next: patch to **not hard-fail** on `_cuda_preinit` (log + continue), retry once; if still fails, pivot to CPU run or request system-level CUDA fix.

#### [Phase3-3][Exp7a][k=3][FAILED] (attempt5)
- cmd: `PYTHONUNBUFFERED=1 python -u exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 200 ... --device cuda:0 --enable_hot_split --split_k 3 --split_init_method noise --split_init_std 1e-3`
- output_dir: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_013538_attempt5`
- log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_013538_attempt5/train.log`
- step: N/A (crash during teacher `.to(cuda:0)`)
- split_event (step/parent/children/init_method/std): N/A
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
- feature_mse:
- group metrics (child_active_count / group_entropy / top-child usage):
- proxy separation (if enabled):
- P1/P2/P3 status: FAIL (P0 crash)
- notes: Even when allowing `_cuda_preinit` failure and continuing, it still crashes at teacher `.to(cuda:0)` with `cudaGetDeviceCount` Error 304. Additional evidence: `cuInit(0)=304`; `dmesg` shows repeated `NVRM: Going over RM unhandled interrupt threshold` → likely system-level CUDA driver/hardware instability.
- next: Decide: (A) switch Exp7a to `--device cpu` for functional validation (slow) or (B) request admin to fix GPU driver state / reset GPUs, then rerun Exp7a.

### Exp7b

#### [Phase3-3][Exp7b][interval=200][COMPLETED] (env=test, steps=1000)
- cmd: `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=GPU-a5b54173-a1ea-80b6-adcd-b5415a33d660 PYTHONUNBUFFERED=1 python -u exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 1000 --batch_size 8 --grad_accum 2 --lr 1e-4 --n_rvq_layers 4 --rvq_codebook_size 2048 --rvq_update ema --ema_decay 0.99 --ema_eps 1e-5 --ema_dead_code_threshold 2 --ema_usage_penalty 0.0 --lambda_quant 1.0 --lambda_pre 0.0 --lambda_inter 0.5 --beta_commit 1.0 --lambda_codebook 1.0 --inter_warmup_steps 0 --eval_interval 200 --eval_max_batches 50 --output_dir exp_0128/phase3-3/run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256 --seed 42 --device cuda:0 --cuda_preinit_retries 30 --cuda_preinit_sleep_s 1 --enable_hot_split --split_layer 0 --split_k 3 --split_hot_k 1 --split_interval 200 --split_one_shot_step 0 --split_init_method noise --split_init_std 1e-3`
- output_dir: `exp_0128/phase3-3/run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256`
- log: `exp_0128/phase3-3/run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256/train.log`
- step: 1000 (completed)
- split_event (step/parent/children/init_method/std):
  - step200 parent=220 -> children=[8,12], init=noise, std=1e-3
  - step400 parent=523 -> children=[7,13], init=noise, std=1e-3
  - step600 parent=875 -> children=[4,11], init=noise, std=1e-3
  - step800 parent=498 -> children=[1,10], init=noise, std=1e-3
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
  - step0: 4.5176 / 0.7001 / 63
  - step200: 9.0577 / 0.1550 / 1080
  - step400: 8.6044 / 0.2546 / 1023
  - step600: 8.8742 / 0.1877 / 1061
  - step800: 8.7216 / 0.2336 / 1036
  - step1000: 8.9774 / 0.1602 / 1104
- feature_mse:
  - step0: 0.0754
  - step200: 0.0376
  - step400: 0.0373
  - step600: 0.0362
  - step800: 0.0359
  - step1000: 0.0338
- group metrics (child_active_count / group_entropy / top-child usage):
  - step400 group0(parent=220): child_active_count=2; children_counts=[85,96]; parent_count=46; group_entropy_children=0.6913 (norm=0.9973); top_child=12 (0.5304); parent_frac_in_group=0.2026
  - step600 group0(parent=220): child_active_count=2; children_counts=[54,75]; parent_count=15; group_entropy_children=0.6798 (norm=0.9808); top_child=12 (0.5814); parent_frac_in_group=0.1042
  - step600 group1(parent=523): child_active_count=2; children_counts=[118,380]; parent_count=64; group_entropy_children=0.5475 (norm=0.7899); top_child=13 (0.7631); parent_frac_in_group=0.1139
  - step800 group0(parent=220): child_active_count=1 (collapse); children_counts=[0,99]; parent_count=38; group_entropy_children=0.0000; top_child=12 (1.0000); parent_frac_in_group=0.2774
  - step800 group1(parent=523): child_active_count=1 (collapse); children_counts=[0,64]; parent_count=0; group_entropy_children=0.0000; top_child=13 (1.0000); parent_frac_in_group=0.0000
  - step800 group2(parent=875): child_active_count=2; children_counts=[84,83]; parent_count=0; group_entropy_children=0.6931 (norm=1.0000); top_child=4 (0.5030); parent_frac_in_group=0.0000
  - step1000 group0(parent=220): child_active_count=2; children_counts=[155,39]; parent_count=79; group_entropy_children=0.5018 (norm=0.7240); top_child=8 (0.7990); parent_frac_in_group=0.2894
  - step1000 group1(parent=523): child_active_count=2; children_counts=[1157,357]; parent_count=11; group_entropy_children=0.5462 (norm=0.7880); top_child=7 (0.7642); parent_frac_in_group=0.0072
  - step1000 group2(parent=875): child_active_count=2; children_counts=[27,123]; parent_count=25; group_entropy_children=0.4714 (norm=0.6801); top_child=11 (0.8200); parent_frac_in_group=0.1429
  - step1000 group3(parent=498): child_active_count=2; children_counts=[12,42]; parent_count=0; group_entropy_children=0.5297 (norm=0.7642); top_child=10 (0.7778); parent_frac_in_group=0.0000
- proxy separation (if enabled): N/A
- P1/P2/P3 status: PASS / PASS / N/A
- notes: P1 PASS for all 4 split events at their first post-split eval (200→400, 400→600, 600→800, 800→1000). Some groups show transient collapse at step800 but recover by step1000 → suggests balance reg (Exp7c) could still help stability.
- next: decide whether to run Exp7c (add one balance reg) and Exp7d proxy separation

### Exp7c

#### [Phase3-3][Exp7c][repulse][COMPLETED] (env=test, steps=1000)
- cmd: `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=GPU-a5b54173-a1ea-80b6-adcd-b5415a33d660 PYTHONUNBUFFERED=1 python -u exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 1000 --batch_size 8 --grad_accum 2 --lr 1e-4 --n_rvq_layers 4 --rvq_codebook_size 2048 --rvq_update ema --ema_decay 0.99 --ema_eps 1e-5 --ema_dead_code_threshold 2 --ema_usage_penalty 0.0 --lambda_quant 1.0 --lambda_pre 0.0 --lambda_inter 0.5 --beta_commit 1.0 --lambda_codebook 1.0 --inter_warmup_steps 0 --eval_interval 200 --eval_max_batches 50 --output_dir exp_0128/phase3-3/run_exp7c_periodic_int200_hot1_k3_repulse_w1e-3_sigma1_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_040845 --seed 42 --device cuda:0 --cuda_preinit_retries 30 --cuda_preinit_sleep_s 1 --enable_hot_split --split_layer 0 --split_k 3 --split_hot_k 1 --split_interval 200 --split_one_shot_step 0 --split_init_method noise --split_init_std 1e-3 --split_repulse_weight 1e-3 --split_repulse_sigma 1.0`
- output_dir: `exp_0128/phase3-3/run_exp7c_periodic_int200_hot1_k3_repulse_w1e-3_sigma1_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_040845`
- log: `exp_0128/phase3-3/run_exp7c_periodic_int200_hot1_k3_repulse_w1e-3_sigma1_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_040845/train.log`
- step: 1000 (completed)
- split_event (step/parent/children/init_method/std):
  - step200 parent=616 -> children=[8,12], init=noise, std=1e-3
  - step400 parent=2 -> children=[7,13], init=noise, std=1e-3
  - step600 parent=2011 -> children=[4,11], init=noise, std=1e-3
  - step800 parent=498 -> children=[1,10], init=noise, std=1e-3
- layer0_entropy / layer0_top10_mass / layer0_used_codes:
  - step0: 4.5176 / 0.7001 / 63
  - step200: 9.1287 / 0.1288 / 1078
  - step400: 8.5642 / 0.2666 / 1023
  - step600: 8.8882 / 0.1833 / 1067
  - step800: 8.8335 / 0.2022 / 1039
  - step1000: 8.9116 / 0.1860 / 1104
- feature_mse:
  - step0: 0.0754
  - step200: 0.0384
  - step400: 0.0372
  - step600: 0.0363
  - step800: 0.0364
  - step1000: 0.0339
- group metrics (child_active_count / group_entropy / top-child usage):
  - step400 group0(parent=616): child_active_count=2; children_counts=[68,99]; parent_count=219; group_entropy_children=0.6758 (norm=0.9750); top_child=12 (0.5928); parent_frac_in_group=0.5674
  - step600 group0(parent=616): child_active_count=2; children_counts=[57,78]; parent_count=0; group_entropy_children=0.6810 (norm=0.9825); top_child=12 (0.5778); parent_frac_in_group=0.0000
  - step600 group1(parent=2): child_active_count=2; children_counts=[143,41]; parent_count=10; group_entropy_children=0.5305 (norm=0.7653); top_child=7 (0.7772); parent_frac_in_group=0.0515
  - step800 group0(parent=616): child_active_count=1 (collapse); children_counts=[0,23]; parent_count=0; group_entropy_children=0.0000; top_child=12 (1.0000); parent_frac_in_group=0.0000
  - step800 group1(parent=2): child_active_count=1 (collapse); children_counts=[0,22]; parent_count=14; group_entropy_children=0.0000; top_child=13 (1.0000); parent_frac_in_group=0.3889
  - step800 group2(parent=2011): child_active_count=2; children_counts=[71,52]; parent_count=0; group_entropy_children=0.6812 (norm=0.9827); top_child=4 (0.5772); parent_frac_in_group=0.0000
  - step1000 group0(parent=616): child_active_count=2; children_counts=[46,74]; parent_count=75; group_entropy_children=0.6657 (norm=0.9604); top_child=12 (0.6167); parent_frac_in_group=0.3846
  - step1000 group1(parent=2): child_active_count=2; children_counts=[12,258]; parent_count=0; group_entropy_children=0.1818 (norm=0.2623); top_child=13 (0.9556); parent_frac_in_group=0.0000
  - step1000 group2(parent=2011): child_active_count=2; children_counts=[210,188]; parent_count=0; group_entropy_children=0.6916 (norm=0.9978); top_child=4 (0.5276); parent_frac_in_group=0.0000
  - step1000 group3(parent=498): child_active_count=2; children_counts=[113,203]; parent_count=0; group_entropy_children=0.6520 (norm=0.9407); top_child=10 (0.6424); parent_frac_in_group=0.0000
- proxy separation (if enabled): N/A
- P1/P2/P3 status: PASS / PASS / N/A
- notes: Repulsion did not eliminate transient collapse (step800) and did not enforce balance in all groups (step1000 has one highly-skewed group: entropy_norm=0.262), but final eval recovers `child_active_count=2` for all groups.
- next: run Exp7d (proxy separation) using the best stable split run (recommend Exp7b) and report effect size / KS

### Exp7d

#### [Phase3-3][Exp7d][proxy=SNR][COMPLETED] (base=Exp7b, eval_max_batches=50)
- cmd: `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && PYTHONUNBUFFERED=1 python -u exp_0128/phase3-3/analyze_proxy_separation.py --base_run_dir exp_0128/phase3-3/run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256 --device cuda:1 --num_workers 0 --max_batches 50 --min_child_tokens 20`
- output_dir: `exp_0128/phase3-3/run_exp7d_proxy_snr_from_run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256_20260205_050353`
- log: `exp_0128/phase3-3/run_exp7d_proxy_snr_from_run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256_20260205_050353/analysis.log`
- step: val max_batches=50 (proxy aggregation over layer0 tokens; note: CUDA init error 304 → auto fallback to CPU)
- split_event (step/parent/children/init_method/std):
  - step200 parent=220 -> children=[8,12], init=noise, std=1e-3
  - step400 parent=523 -> children=[7,13], init=noise, std=1e-3
  - step600 parent=875 -> children=[4,11], init=noise, std=1e-3
  - step800 parent=498 -> children=[1,10], init=noise, std=1e-3
- layer0_entropy / layer0_top10_mass / layer0_used_codes (from base Exp7b step1000): 8.9774 / 0.1602 / 1104
- feature_mse (from base Exp7b step1000): 0.0338
- group metrics (child_active_count / group_entropy / top-child usage) (from base Exp7b step1000):
  - group0(parent=220): child_active_count=2; children_counts=[155,39]; top_child=8 (0.7990)
  - group1(parent=523): child_active_count=2; children_counts=[1157,357]; top_child=7 (0.7642)
  - group2(parent=875): child_active_count=2; children_counts=[27,123]; top_child=11 (0.8200)
  - group3(parent=498): child_active_count=2; children_counts=[12,42]; top_child=10 (0.7778)
- proxy separation (SNR dB; per-child mean/std + effect size d):
  - group0(parent=220): child8 mean=-1.25 std=3.06 (n=43) vs child12 mean=-1.01 std=2.96 (n=229) → abs(d)=0.083
  - group1(parent=523): child7 mean=-1.87 std=3.45 (n=43) vs child13 mean=-2.45 std=3.19 (n=47) → abs(d)=0.174
  - group2(parent=875): child4 mean=-1.67 std=3.12 (n=124) vs child11 mean=-3.37 std=2.86 (n=40) → abs(d)=0.557 (PASS per-group)
  - group3(parent=498): child1 mean=-0.89 std=3.03 (n=83) vs child10 mean=-0.96 std=3.05 (n=131) → abs(d)=0.024
  - Summary: abs_d_median=0.128; pass_groups=1/4 (threshold abs(d)>=0.5)
- P1/P2/P3 status: N/A / N/A / FAIL (weak evidence; only 1 group shows moderate SNR separation)
- notes: Proxy separation is not consistent across split groups under sample-level SNR; may require per-frame proxy or better split init (local kmeans) / explicit balance or hierarchical branch.
- next: if P3 is required, implement per-frame proxy (||noisy-clean|| per hop) and/or init-B (local kmeans) and rerun Exp7d on a run with higher child usage in multiple groups (or increase split_k / hot_k).

## Timeline（append-only）

- 2026-02-05T00:46:19-05:00 Preflight completed (git/python/torch ok; `nvidia-smi` NVML init failed but torch sees 3 CUDA devices)
- 2026-02-05T00:46:19-05:00 Sanity: `python -m py_compile exp_0128/phase3/residual_vq/train_rvq_short_run.py` PASS
- 2026-02-05T00:46:19-05:00 Phase 3-2 gate checked: PASS (see reference run step200 metrics)
- 2026-02-05T00:54:55-05:00 Implemented Phase3-3 hot-split plumbing in `exp_0128/phase3/residual_vq/train_rvq_short_run.py` (new args + `split_history.json` + eval-time group metrics); py_compile PASS
- 2026-02-05T00:55:38-05:00 Exp7a started: one-shot split (k=3, init=noise, std=1e-3), steps=200 → `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_005538`
- 2026-02-05T00:58:47-05:00 Exp7a attempt1 FAILED: CUDA init crash (`cudaGetDeviceCount` error 304) when `CUDA_VISIBLE_DEVICES` set; output_dir contains only `config.json`
- 2026-02-05T01:03:24-05:00 Exp7a attempt2 started: removed `CUDA_VISIBLE_DEVICES`, added `PYTHONUNBUFFERED=1` → `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_010324_attempt2`
- 2026-02-05T01:06:35-05:00 Exp7a attempt2 FAILED: same CUDA init crash (`cudaGetDeviceCount` error 304); output_dir contains only `config.json`
- 2026-02-05T01:13:14-05:00 Patch: added `_cuda_preinit(..., retries=3)` before model init in `exp_0128/phase3/residual_vq/train_rvq_short_run.py` to mitigate flaky CUDA init
- 2026-02-05T01:13:55-05:00 Exp7a attempt3 started: added `_cuda_preinit` (still used `| tee`) → `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_011355_attempt3`
- 2026-02-05T01:17:08-05:00 Exp7a attempt3 FAILED: `_cuda_preinit` could not init CUDA (Error 304). Diagnosis: CUDA init fails when shell redirects stdout/stderr to files (`tee`/`>`).
- 2026-02-05T01:21:07-05:00 Patch: tee console output inside Python to `output_dir/train.log` (avoid `tee`/`>`), keep `_cuda_preinit`
- 2026-02-05T01:21:32-05:00 Exp7a attempt4 started: removed shell redirection; in-script `train.log` → `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_012127_attempt4`
- 2026-02-05T01:24:39-05:00 Exp7a attempt4 FAILED: still `cudaGetDeviceCount` Error 304 during CUDA preinit (so not caused by `tee`/`>`); output_dir contains `config.json` + `train.log`
- 2026-02-05T01:35:43-05:00 Exp7a attempt5 started: allow CUDA preinit failure then continue to model init → `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_013538_attempt5`
- 2026-02-05T01:38:54-05:00 Exp7a attempt5 FAILED: crashes at teacher `.to(cuda:0)` with `cudaGetDeviceCount` Error 304; cannot reach step0
- 2026-02-05T01:42:23-05:00 Diagnostic: `nvidia-smi` reports 3 GPUs, but CUDA driver init fails (`cuInit(0)=304`); `dmesg` shows repeated `NVRM: Going over RM unhandled interrupt threshold` (irq 100/101) → likely system-level issue
- 2026-02-05T03:10:39-05:00 Preflight re-check PASS: `torch.cuda.is_available()=True`, `device_count=3`; CUDA alloc+matmul sanity PASS on `cuda:0` (RTX 2080 Ti)
- 2026-02-05T03:10:39-05:00 CUDA selection note: this env includes unsupported GTX 1080 Ti (sm_61) → must avoid `cuda:2`; optionally mask with `CUDA_VISIBLE_DEVICES=GPU-a5b54173-a1ea-80b6-adcd-b5415a33d660` (RTX 2080 Ti)
- 2026-02-05T03:23:10-05:00 Switched to conda env `test` (PyTorch/torchaudio 2.5.1): `torchaudio.load()` sanity PASS (no torchcodec/ffmpeg missing-lib crash)
- 2026-02-05T03:25:25-05:00 Patch: added `--eval_max_batches` to `exp_0128/phase3/residual_vq/train_rvq_short_run.py` (default=50) to speed sanity runs without changing experiment defaults
- 2026-02-05T03:25:53-05:00 Sanity started (env=test, eval_max_batches=2): `exp_0128/phase3-3/run_sanity_steps5_L4_K2048_ema_seed0_20260205_032546` (log: `exp_0128/phase3-3/run_sanity_steps5_L4_K2048_ema_seed0_20260205_032546/train.log`)
- 2026-02-05T03:29:21-05:00 Sanity PASS: completed 5 steps, saved `final_model.pt` and metrics (see `exp_0128/phase3-3/run_sanity_steps5_L4_K2048_ema_seed0_20260205_032546/train.log`)
- 2026-02-05T03:31:32-05:00 Exp7a started (one-shot split k=3 @ step0, init=noise std=1e-3): `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_033126` (log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_033126/train.log`)
- 2026-02-05T03:35:50-05:00 Exp7a completed (step200 early-check): P1 PASS; split group child_active_count=2 @ step200; outputs saved under `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps200_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_033126`
- 2026-02-05T03:40:36-05:00 Gate decision: **GO** (P1 PASS @ step200) → continue Phase 3-3; next run: Exp7a full (steps=1000) for P2 final check
- 2026-02-05T03:41:21-05:00 Exp7a full run started (steps=1000): `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_034115` (log: `exp_0128/phase3-3/run_exp7a_oneshot_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_034115/train.log`)
- 2026-02-05T03:49:35-05:00 Exp7a full completed: P1 PASS (step200 child_active_count=2), P2 PASS (step1000 top10_mass=0.1698, used_codes=1082, feature_mse=0.0339); note: late child collapse at step800/1000
- 2026-02-05T03:52:28-05:00 Patch: periodic split now skips `step==args.steps` (avoid splitting at final eval with no follow-up) in `exp_0128/phase3/residual_vq/train_rvq_short_run.py`; py_compile PASS
- 2026-02-05T03:53:02-05:00 Exp7b started (periodic split interval=200, hot_k=1, k=3): `exp_0128/phase3-3/run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256` (log: `exp_0128/phase3-3/run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256/train.log`)
- 2026-02-05T04:01:15-05:00 Exp7b completed: P1 PASS (all split events child_active_count>=2 at first post-split eval), P2 PASS (step1000 top10_mass=0.1602, used_codes=1104, feature_mse=0.0338)
- 2026-02-05T04:08:19-05:00 Patch: implemented Exp7c embedding repulsion via `--split_repulse_weight/--split_repulse_sigma` (EMA-safe: updates codebook + ema_embed_avg) in `exp_0128/phase3/residual_vq/train_rvq_short_run.py`; py_compile PASS
- 2026-02-05T04:08:50-05:00 Exp7c started (periodic split interval=200 + embedding repulsion w=1e-3): `exp_0128/phase3-3/run_exp7c_periodic_int200_hot1_k3_repulse_w1e-3_sigma1_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_040845` (log: `exp_0128/phase3-3/run_exp7c_periodic_int200_hot1_k3_repulse_w1e-3_sigma1_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_040845/train.log`)
- 2026-02-05T04:17:21-05:00 Exp7c completed: P1 PASS, P2 PASS (step1000 top10_mass=0.1860, used_codes=1104, feature_mse=0.0339); note: one group remains highly imbalanced (entropy_norm=0.262)
- 2026-02-05T04:33:25-05:00 Preflight (env=test): torch cuda ok with no `CUDA_VISIBLE_DEVICES` (is_available=True, device_count=3); setting `CUDA_VISIBLE_DEVICES` still triggers `cudaGetDeviceCount` Error 304 → use `--device cuda:1` / `cuda:2`
- 2026-02-05T04:58:29-05:00 Exp7d attempt1 started (proxy=SNR, base=Exp7b): `exp_0128/phase3-3/run_exp7d_proxy_snr_from_run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256_20260205_045829`
- 2026-02-05T05:01:40-05:00 Exp7d attempt1 ABORTED: excessive host RAM due to loading train cache in analysis; patched analysis to val-only loader + removed numpy (avoid OMP SHM2 crash)
- 2026-02-05T05:02:27-05:00 Exp7d pilot completed (max_batches=20, min_child_tokens=50): P3 FAIL (pass_groups=0/4); output_dir `exp_0128/phase3-3/run_exp7d_proxy_snr_from_run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256_20260205_050227`
- 2026-02-05T05:03:53-05:00 Exp7d started full (max_batches=50, min_child_tokens=20): `exp_0128/phase3-3/run_exp7d_proxy_snr_from_run_exp7b_periodic_int200_hot1_k3_noise_std1e-3_steps1000_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_035256_20260205_050353`
- 2026-02-05T05:05:34-05:00 Exp7d completed: P3 FAIL (pass_groups=1/4, abs_d_median=0.128); note: CUDA init error 304 → analysis ran on CPU
