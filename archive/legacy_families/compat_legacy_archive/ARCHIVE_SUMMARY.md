# compat_legacy Archive Summary

Archived on 2026-03-09 from `families/compat_legacy/`.

## Reason for archival

These historical training scripts, analysis tools, shell launchers, experiment
logs (JSON/CSV), markdown documentation, and visualization scripts are no longer
referenced by any active pipeline. They were occupying 13 MB of code/data space
(after 630 MB of log files were removed in Phase 1) and polluting the planning
context with ~70 legacy Python files.

## What was archived

| Family | Contents | Key experiments |
|--------|----------|-----------------|
| curriculum_data | train_exp62~67, losses variants, diagnose/verify scripts, shell launchers | exp62 capacity, exp63 VQ-aware, exp64 curriculum, exp65 anti-collapse, exp66 post-VQ, exp67 curriculum-VQ |
| intermediate_stack | train.py~train_v5_continue.py, models_v2/v3, analysis/, evaluate scripts, visualization | exp_k through exp_k_v6: intermediate supervision evolution |
| plan_ori_vq | train_long/v2, plan_ori/train_single_vq_ema, test scripts, architecture docs | Original VQ plan with EMA, long training experiments |
| residual_vq_stack | All phase1-3 experiments, baseline token analysis, noise/sampling, codebook refresh, entropy reg | Comprehensive RVQ exploration: soft reweighting, entropy regularization, codebook refresh, split strategies |

## What was NOT archived (still in families/compat_legacy/)

These 5 core files remain because they are actively imported by `families/deps/`,
`families/official/`, and `families/eval/` pipelines:

1. `curriculum_data/data_curriculum.py` — CurriculumDataset, CurriculumSampler, collate_fn_curriculum
2. `intermediate_stack/models.py` — TeacherStudentIntermediate
3. `intermediate_stack/train_v6.py` — IntermediateSupervisionLossV6, verify_model_state
4. `plan_ori_vq/plan_ori/models_single_vq_ema.py` — TeacherStudentSingleVQ (model inheritance root)
5. `residual_vq_stack/phase3/residual_vq/models_rvq.py` — TeacherStudentRVQ

## Recovery

All archived files remain in git history at tag `pre-compat-legacy-cleanup` and
in this archive directory. To restore any file:

```bash
git show pre-compat-legacy-cleanup:families/compat_legacy/<path>
```
