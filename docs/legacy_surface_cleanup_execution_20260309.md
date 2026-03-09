# Legacy Surface Cleanup Execution

Start in `AGENTS.md`.

Date: 2026-03-09
Plan: `docs/legacy_surface_cleanup_plan_20260309.md`
Owner: `Codex(default)`

## Phase 1 Decision

Phase 1 cleanup is approved only for cold generated residue with direct evidence:

- `families/**/__pycache__/*.pyc`
- `families/eval/bwe_latent_hf/nohup_autolaunch.log`
- `families/official/hubert_then_distalign/nohup_train_distalign_gpu0.log`

The anchor log was initially blocked in Phase 1 because the dead
`anchor-then-material` shell path still referenced it.

## Evidence

Dependency evidence:

- safe-candidate and blocked-candidate mapping came from native `explorer`
- `families/compat_legacy/*`, active `families/deps/*`, active `families/eval/*`,
  and checkpoint-default run directories remain out of scope for Phase 1

Live-process evidence:

- active GPU processes at execution time were:
  - `exp_0305b/train_0224a_anchor.py` on `cuda:0`
  - `exp_0305b/train_0224a_anchor.py` on `cuda:1`
- these do not match the two approved `nohup` logs above

Coldness evidence:

- the two approved `nohup` logs both had last-modified timestamps on `2026-03-08`
- no repo source file referenced those two log paths

History preservation:

- deleted items are generated residue rather than source-of-truth artifacts
- cleanup rationale is preserved in this execution note and in git history
- removed historical compat content is recoverable via git tag
  `pre-compat-legacy-cleanup`

## Phase 1 Actions

- remove `67` compiled cache files under `families/**/__pycache__/`
- remove empty `__pycache__` directories created by those cache files
- remove `families/eval/bwe_latent_hf/nohup_autolaunch.log`
- remove `families/official/hubert_then_distalign/nohup_train_distalign_gpu0.log`

## Phase 1 Result

Completed on 2026-03-09 with these verified outcomes:

- the first post-cleanup verification pass observed `0` remaining
  `families/**/__pycache__/` directories and `0`
  `families/**/__pycache__/*.pyc` files
- later review and inspection steps may regenerate Python cache files, so cache
  counts should be treated as point-in-time cleanup evidence rather than a
  permanent invariant
- `families/eval/bwe_latent_hf/nohup_autolaunch.log`: removed
- `families/official/hubert_then_distalign/nohup_train_distalign_gpu0.log`: removed
- `families/official/anchor_then_material/nohup_expc_20260306_090111.log`: later removed together with the dead anchor family surface

## Not Done In Phase 1

- no deletion under `families/compat_legacy/*`
- no deletion under `families/deps/*/runs`
- no deletion of eval fixtures or baseline summaries
- no phase-1 deletion of the anchor log; it was removed later with the dead anchor family surface

## Next Candidate For Phase 2

There are two viable next cuts from native-role analysis:

- smallest safe cut:
  - extract `IntermediateSupervisionLossV6` and `verify_model_state` out of
    `families/compat_legacy/intermediate_stack/train_v6.py`
- highest fanout cut:
  - extract `TeacherStudentSingleVQ` and `SingleVQWithEMA` out of
    `families/compat_legacy/plan_ori_vq/plan_ori/models_single_vq_ema.py`

## Phase 2 Decision

User selected the highest-fanout cut on 2026-03-09:

- extract `TeacherStudentSingleVQ`
- extract `SingleVQWithEMA`

## Phase 2 Actions

- add `families/deps/encoder_vq_core/__init__.py`
- add `families/deps/encoder_vq_core/models_single_vq.py`
- convert `families/compat_legacy/plan_ori_vq/plan_ori/models_single_vq_ema.py`
  into a thin compatibility wrapper
- retarget active Python importers under:
  - `families/deps/encoder_aug/`
  - `families/deps/feat_align/`
  - `families/deps/no_vq_core/`
  - `families/deps/t453_weighted_baseline/`
  - `families/eval/decoder_lora_eval/`

## Phase 2 Validation

Successful import-smoke targets:

- `families.deps.encoder_vq_core.models_single_vq`
- `families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema`
- `families.deps.no_vq_core.models_no_vq`
- `families.deps.no_vq_core.models_no_vq_decoder_lora`
- `families.deps.feat_align.models_no_vq_e2e`
- `families.deps.encoder_aug.train_augmented`
- `families.deps.t453_weighted_baseline.train_t453_weighted`
- `families.eval.decoder_lora_eval.models_decoder_lora`

Post-change grep result:

- no active `families/` Python importer still points at
  `families.compat_legacy.plan_ori_vq.plan_ori.models_single_vq_ema`

## Phase 2 Remaining Risk

- `families/eval/decoder_lora_eval/generate_test_samples.py` still depends on the
  RVQ compat path for `TeacherStudentRVQ`
- that RVQ path was already flagged as broken before this extraction, so Phase 2
  option 3 does not resolve it

## Later Cleanup Waves

Later waves remove:

- dead `anchor-then-material` official control surface
- dead `anchor_regularization` dependency surface
- in-repo historical archive directories no longer kept as a live surface

Concrete later-wave deletions completed on 2026-03-09:

- remove `experiments/adapters/exp0305c_preflight_anchor_then_material.json`
- remove `experiments/adapters/exp0305c_train_expc.json`
- remove `experiments/manifests/exp0305c_to_exp0304_{preflight,smoke,short,controller}.json`
- remove `families/official/anchor_then_material/`
- remove `families/deps/anchor_regularization/`
- remove `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/`
- remove `families/deps/t453_weighted_baseline/COMMIT_5e859b0_VAL_AUDIO_ANALYSIS_{SPEC,PLAN}.md`
- remove `families/official/material_generalization/wavtokenizer_featuremap_14wav_extended/`
- remove `families/eval/decoder_lora_eval/test/`
- remove `archive/controller_runs/`
- remove `archive/legacy_artifacts/`
- remove `archive/legacy_families/`
- remove `archive/root-notes/`
- remove `archive/root-scripts/`

Later-wave validation notes:

- `experiments/registry.json` and `experiments/adapters/index.json` were updated
  to stop advertising the dead anchor line
- in-tree compat archive recovery text was replaced with git-history /
  git-tag recovery text
- `families/eval/decoder_lora_eval/generate_test_samples.py` still has a
  pre-existing baseline-path import failure unrelated to these deletions, so it
  remains a known residual risk rather than a cleanup regression
