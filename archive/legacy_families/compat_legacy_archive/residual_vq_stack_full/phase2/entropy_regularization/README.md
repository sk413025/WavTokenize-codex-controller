# Experiment 3: Entropy Regularization

## Purpose

Test whether explicitly maximizing entropy in the loss function can prevent token collapse by encouraging the model to use a diverse set of codebook tokens.

## Method

Add an entropy regularization term to the training loss:

```python
# Compute token distribution in current batch
token_counts = torch.bincount(student_codes.flatten(), minlength=4096).float()
token_probs = token_counts / token_counts.sum()

# Compute entropy: H = -sum(p * log(p))
entropy = -(token_probs * torch.log(token_probs + 1e-8)).sum()

# Maximize entropy by minimizing negative entropy
loss_entropy = -lambda_entropy * entropy

# Total loss
total_loss = intermediate_weight * loss_inter + (1 - intermediate_weight) * loss_main + loss_entropy
```

## Rationale

Based on Phase 1 failure analysis:
- **Root cause**: Training dynamics issue, not data sampling issue
- **Key finding**: Untrained model (Step 0) has better entropy (6.26) than trained model (6.07)
- **Hypothesis**: Training process lacks explicit diversity constraint

This approach directly addresses the problem by:
1. Adding explicit penalty for low entropy distributions
2. Encouraging model to spread predictions across more tokens
3. No changes to data sampling (keeping it simple and interpretable)

## Configurations

Test 3 different regularization strengths:

| Exp | λ | Description | Expected Behavior |
|-----|---|-------------|-------------------|
| 3a | 0.01 | Weak regularization | Gentle encouragement toward diversity |
| 3b | 0.05 | Medium regularization | Moderate diversity enforcement |
| 3c | 0.10 | Strong regularization | Strong diversity enforcement |

## Parameters (Consistent with Baseline)

All experiments use identical baseline configuration:

- **LoRA**: rank=256, alpha=512, dropout=0.2
- **Intermediate layers**: [3, 4, 6]
- **Layer weights**: {3: 0.3, 4: 0.5, 6: 0.5}
- **Loss weights**: intermediate=0.5
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Batch**: size=8, grad_accum=2 (effective=16)
- **Dataset**: CurriculumDataset with filter_clean_to_clean=True
- **Sampler**: Random (no weighted/balanced sampling)
- **Steps**: 1000 (short-run validation)
- **Eval interval**: 200 steps
- **Seed**: 42

**Only difference**: Added entropy regularization loss term

## Success Criteria

Compare final metrics (step 1000) against baseline (exp_k v6 @ epoch 300):

| Metric | Baseline | Target | Condition |
|--------|----------|--------|-----------|
| Entropy | 6.07 | Higher | > 6.07 |
| Top-10 Mass | 19.7% | Lower | < 19.7% |
| Strict Acc | 0.91% | Not worse | >= 0.82% (90% threshold) |

**Success condition**:
```python
success = (
    entropy > 6.07 AND
    top_10_mass < 0.197 AND
    strict_acc >= 0.0082
)
```

## How to Run

### Sequential (one GPU)

```bash
# Experiment 3a (λ=0.01) - 2-3 hours
bash exp_0128/phase2/entropy_regularization/run_exp3a.sh

# Experiment 3b (λ=0.05) - 2-3 hours
bash exp_0128/phase2/entropy_regularization/run_exp3b.sh

# Experiment 3c (λ=0.10) - 2-3 hours
bash exp_0128/phase2/entropy_regularization/run_exp3c.sh
```

### Parallel (three GPUs)

Modify the scripts to use different GPUs:

```bash
# Terminal 1 (GPU 0)
bash exp_0128/phase2/entropy_regularization/run_exp3a.sh

# Terminal 2 (GPU 1) - edit script to set CUDA_VISIBLE_DEVICES=1
bash exp_0128/phase2/entropy_regularization/run_exp3b.sh

# Terminal 3 (GPU 2) - edit script to set CUDA_VISIBLE_DEVICES=2
bash exp_0128/phase2/entropy_regularization/run_exp3c.sh
```

## Output Structure

Each experiment creates:

```
exp_0128/phase2/entropy_regularization/run_exp3{a,b,c}_TIMESTAMP/
├── config.json              # Experiment configuration
├── metrics_history.json     # Collapse metrics (every 200 steps)
├── loss_history.json        # Loss values (every step) + entropy loss
├── summary.json             # Final results + baseline comparison + success flag
├── training_curves.png      # Training curves (3 subplots)
├── final_model.pt           # Final model checkpoint
├── checkpoints/             # Intermediate checkpoints (every 200 steps)
│   ├── checkpoint_step0200.pt
│   ├── checkpoint_step0400.pt
│   └── ...
└── audio_samples/           # Audio samples for validation
    ├── train/step_XXXX/
    └── val/step_XXXX/
```

## Expected Training Curves

The `training_curves.png` will show 3 subplots:

1. **Training Loss**: Total, main, intermediate, and entropy loss over time
   - Entropy loss should be negative (maximizing entropy)
   - Total loss should still decrease (main task learning)

2. **Collapse Metrics**: Entropy and Top-10 Mass over time
   - Entropy should increase or stay stable (not decrease like Phase 1)
   - Top-10 Mass should decrease or stay low

3. **Entropy Loss**: Entropy regularization loss over time
   - Should remain negative (we're maximizing entropy)
   - Magnitude shows how strongly we're encouraging diversity

## Result Analysis

After completion, check `summary.json`:

```bash
cat exp_0128/phase2/entropy_regularization/run_exp3*/summary.json | jq
```

Compare all three configurations:

```bash
# Extract key metrics
for dir in exp_0128/phase2/entropy_regularization/run_exp3*; do
    echo "=== $(basename $dir) ==="
    jq '{lambda: .lambda_entropy, final: .final, success: .success}' $dir/summary.json
done
```

## Next Steps

### If any configuration succeeds:
1. Choose the best λ value based on metrics
2. Run full training (300 epochs) with that λ
3. Compare against baseline full training

### If all configurations fail:
1. Try larger λ values (0.2, 0.5)
2. Combine with Experiment 4 (Codebook Refresh)
3. Consider other regularization approaches

### If results are mixed:
1. Analyze training curves to understand behavior
2. Check if entropy increases initially but degrades later
3. Consider adaptive λ schedule (start low, increase over time)

## Notes

- Entropy is computed on **batch-level** token distribution (not dataset-level)
- This encourages diversity within each batch
- May need to tune λ carefully - too strong can hurt main task performance
- Watch for potential trade-off between diversity and accuracy

## Implementation Details

### Entropy Computation

```python
def compute_entropy_regularization(student_codes, codebook_size=4096):
    """
    Compute entropy of token distribution in current batch

    Args:
        student_codes: [B, T] token indices from VQ
        codebook_size: vocabulary size (4096 for WavTokenizer)

    Returns:
        entropy: scalar H = -sum(p * log(p))
    """
    batch_size, seq_len = student_codes.shape
    total_tokens = batch_size * seq_len

    # Count tokens
    token_counts = torch.bincount(student_codes.flatten(), minlength=codebook_size).float()
    token_probs = token_counts / total_tokens

    # Add epsilon for numerical stability
    epsilon = 1e-8
    token_probs = token_probs + epsilon

    # Compute entropy
    entropy = -(token_probs * torch.log(token_probs)).sum()

    return entropy
```

### Loss Integration

```python
# In training loop
with autocast():
    output = model(noisy_audio, clean_audio)

    # Main loss (feature + triplet)
    loss_main, _ = loss_fn(...)

    # Intermediate supervision loss
    loss_inter, _ = inter_loss_fn(...)

    # Entropy regularization (NEW!)
    entropy = compute_entropy_regularization(output['student_codes'])
    loss_entropy = -lambda_entropy * entropy  # Negative sign to maximize

    # Total loss
    loss = 0.5 * loss_inter + 0.5 * loss_main + loss_entropy
```

## Reference

- Phase 1 failure analysis: [exp_0128/RESULTS.md](../../RESULTS.md)
- Baseline configuration: [exp_0112_intermediate/train_v6.py](../../../exp_0112_intermediate/train_v6.py)
- TracIn diagnosis: [exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md](../../../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)
