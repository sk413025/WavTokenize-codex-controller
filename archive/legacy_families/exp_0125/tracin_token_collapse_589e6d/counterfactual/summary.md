# Counterfactual short-run summary
- train subset: 1800 (drop_top_k=200)
- steps: 800, batch_size: 2, grad_accum: 2
- val strict acc: 0.008242 (baseline 0.009135)
- val entropy: 5.855 (baseline 6.066)
- val top_k_mass: 0.281 (baseline 0.197)
- val KL: 1.468 (baseline 1.247)