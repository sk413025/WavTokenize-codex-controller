# Sanity Check: Offline Eval vs Training Log
Run dir: exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848
Checkpoint: exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848/best_model.pt
Checkpoint epoch: 141
Eval mode: model.eval() + torch.no_grad()
Train loader: CurriculumDataset + shuffle=False + full dataset
Val loader: CurriculumDataset + shuffle=False + full dataset

## Logged metrics at checkpoint epoch
- train_masked_acc (log, batch-mean): 0.025697
- val_masked_acc (log, batch-mean): 0.008990

## Offline eval metrics (strict, batch-mean)
- train strict acc_batch_mean: 0.033958
- val strict acc_batch_mean: 0.008991

## Differences (offline - log)
- train diff: +0.008261
- val diff: +0.000002
