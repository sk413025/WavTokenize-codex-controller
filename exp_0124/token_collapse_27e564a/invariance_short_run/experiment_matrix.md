# Invariance short-run experiment matrix

- run_dir: exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848
- checkpoint: best_model.pt
- train subset: 2000 samples (fixed seed)
- val subset: 500 samples (fixed seed)
- steps: 800 (optimizer steps)
- batch_size: 2
- grad_accum: 2 (effective batch = 4)
- amp: true
- snr_list (for synthetic view): 0,5,10 dB
- noise_sensitivity pairs: 30

Lambdas (L_invar):
- 0.0 (anchor-only baseline)
- 0.05
- 0.10
- 0.20 (optional)

Eval per setting (A-D):
- strict acc: train/val acc_frame_weighted
- collapse: entropy, top_k_mass (k=10), unique, KL(student||teacher)
- noise sensitivity: token_change_rate (controlled pairs)
- VQ margin: mean/p50/p90 (train/val)
