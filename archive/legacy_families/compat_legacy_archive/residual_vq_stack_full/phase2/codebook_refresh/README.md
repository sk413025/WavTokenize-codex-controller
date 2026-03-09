# Experiment 4: Codebook Refresh

## Purpose

Test whether periodically resetting unused codebook entries can prevent codebook collapse and maintain diversity throughout training.

## Method

Track codebook usage over a sliding window and periodically refresh (reset) codes that are rarely used:

```python
# Initialize usage tracker
usage_tracker = CodebookUsageTracker(
    codebook_size=4096,
    window_size=refresh_interval
)

# During training
for step in range(total_steps):
    # Forward pass
    output = model(noisy_audio, clean_audio)

    # Track which codes are being used
    usage_tracker.update(output['student_codes'])

    # Every refresh_interval steps
    if step % refresh_interval == 0:
        # Identify unused codes
        usage_counts = usage_tracker.get_usage_counts()
        unused_mask = usage_counts < usage_threshold

        # Reset unused codes to random embeddings
        with torch.no_grad():
            codebook[unused_mask] = torch.randn_like(codebook[unused_mask])
```

## Rationale

Based on Phase 1 failure analysis:
- **Root cause**: Training dynamics issue leading to progressive code underutilization
- **Observation**: Model converges to using fewer and fewer codes over time
- **Hypothesis**: Codebook entries become "dead" and never recover during training

This approach addresses the problem by:
1. Actively monitoring which codes are being used
2. Giving unused codes a "second chance" by resetting them
3. Forcing the model to explore the full codebook space
4. No changes to loss function (orthogonal to Exp 3)

## Configurations

Test 4 combinations of refresh frequency and threshold:

| Exp | Refresh Interval | Usage Threshold | Description |
|-----|------------------|-----------------|-------------|
| 4a | 100 steps | 10 uses | Conservative: Less frequent refresh, higher threshold |
| 4b | 100 steps | 5 uses | Moderate threshold with 100-step interval |
| 4c | 50 steps | 10 uses | More frequent refresh with higher threshold |
| 4d | 50 steps | 5 uses | Aggressive: Most frequent refresh, lowest threshold |

**Interpretation**:
- **Refresh Interval**: How often to check and reset codes (50 or 100 steps)
  - Shorter = more aggressive maintenance
  - Longer = more stable training
- **Usage Threshold**: Minimum times a code must be used to keep it (5 or 10)
  - Lower = more codes get reset (more aggressive)
  - Higher = fewer resets (more conservative)

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

**Only difference**: Codebook refresh mechanism (no loss modification)

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
# Experiment 4a (r=100, t=10) - 2-3 hours
bash exp_0128/phase2/codebook_refresh/run_exp4a.sh

# Experiment 4b (r=100, t=5) - 2-3 hours
bash exp_0128/phase2/codebook_refresh/run_exp4b.sh

# Experiment 4c (r=50, t=10) - 2-3 hours
bash exp_0128/phase2/codebook_refresh/run_exp4c.sh

# Experiment 4d (r=50, t=5) - 2-3 hours
bash exp_0128/phase2/codebook_refresh/run_exp4d.sh
```

### Parallel (four GPUs)

Modify the scripts to use different GPUs:

```bash
# Terminal 1 (GPU 0)
bash exp_0128/phase2/codebook_refresh/run_exp4a.sh

# Terminal 2 (GPU 1) - edit script to set CUDA_VISIBLE_DEVICES=1
bash exp_0128/phase2/codebook_refresh/run_exp4b.sh

# Terminal 3 (GPU 2) - edit script to set CUDA_VISIBLE_DEVICES=2
bash exp_0128/phase2/codebook_refresh/run_exp4c.sh

# Terminal 4 (GPU 3) - edit script to set CUDA_VISIBLE_DEVICES=3
bash exp_0128/phase2/codebook_refresh/run_exp4d.sh
```

## Output Structure

Each experiment creates:

```
exp_0128/phase2/codebook_refresh/run_exp4{a,b,c,d}_TIMESTAMP/
├── config.json              # Experiment configuration
├── metrics_history.json     # Collapse metrics (every 200 steps)
├── loss_history.json        # Loss values (every step)
├── refresh_history.json     # Codebook refresh events (NEW!)
├── summary.json             # Final results + usage stats + success flag
├── training_curves.png      # Training curves (4 subplots)
├── final_model.pt           # Final model checkpoint + usage stats
├── checkpoints/             # Intermediate checkpoints (every 200 steps)
│   ├── checkpoint_step0200.pt
│   ├── checkpoint_step0400.pt
│   └── ...
└── audio_samples/           # Audio samples for validation
    ├── train/step_XXXX/
    └── val/step_XXXX/
```

## Expected Training Curves

The `training_curves.png` will show 4 subplots:

1. **Training Loss**: Total, main, and intermediate loss over time
   - Should decrease normally (no loss modification)

2. **Collapse Metrics**: Entropy and Top-10 Mass over time
   - Entropy should stay stable or increase (unlike Phase 1)
   - Top-10 Mass should stay low
   - Watch for "jumps" after refresh events

3. **Codebook Usage**: Number of used vs unused codes over time
   - Used codes should stay high
   - Unused codes should stay low
   - Shows effectiveness of refresh mechanism

4. **Codes Refreshed**: Number of codes reset at each interval
   - Shows how many codes were unused at each refresh
   - Should decrease over time if training is healthy
   - High numbers indicate ongoing collapse issues

## Result Analysis

After completion, check `summary.json`:

```bash
cat exp_0128/phase2/codebook_refresh/run_exp4*/summary.json | jq
```

Compare all four configurations:

```bash
# Extract key metrics
for dir in exp_0128/phase2/codebook_refresh/run_exp4*; do
    echo "=== $(basename $dir) ==="
    jq '{config: {refresh_interval, usage_threshold}, final: .final, usage: .final_usage_stats, success: .success}' $dir/summary.json
done
```

Check refresh events:

```bash
# See how many codes were refreshed over time
jq '.[] | {step, num_refreshed, unused_codes}' exp_0128/phase2/codebook_refresh/run_exp4a*/refresh_history.json
```

## Next Steps

### If any configuration succeeds:
1. Choose the best (interval, threshold) combination
2. Run full training (300 epochs) with those parameters
3. Consider combining with Exp 3 (Entropy Regularization)

### If all configurations fail:
1. Try more aggressive refresh (every 25 steps)
2. Try different refresh strategies (copy from frequent codes instead of random)
3. Analyze refresh events to understand codebook dynamics

### If results are mixed:
1. Check if frequent refresh helps initially but hurts later
2. Consider adaptive refresh schedule (more frequent early, less later)
3. Analyze which codes get repeatedly reset vs permanently used

## Implementation Details

### Usage Tracker

```python
class CodebookUsageTracker:
    """Track codebook usage over a sliding window"""

    def __init__(self, codebook_size=4096, window_size=1000):
        self.codebook_size = codebook_size
        self.window_size = window_size
        self.usage_counts = torch.zeros(codebook_size, dtype=torch.long)
        self.code_history = deque(maxlen=window_size)

    def update(self, codes):
        """Update with new batch of codes"""
        codes_flat = codes.flatten().cpu()
        new_counts = torch.bincount(codes_flat, minlength=self.codebook_size)

        # Remove oldest counts if window is full
        if len(self.code_history) >= self.window_size:
            oldest_codes = self.code_history[0]
            old_counts = torch.bincount(oldest_codes, minlength=self.codebook_size)
            self.usage_counts -= old_counts

        # Add new counts
        self.usage_counts += new_counts
        self.code_history.append(codes_flat)

    def get_unused_mask(self, threshold):
        """Get mask of codes used less than threshold"""
        return self.usage_counts < threshold
```

### Refresh Function

```python
def refresh_codebook(model, unused_mask, refresh_strategy='random'):
    """
    Refresh unused codebook entries

    Args:
        model: TeacherStudentIntermediate
        unused_mask: [codebook_size] boolean tensor
        refresh_strategy: 'random' or 'copy_frequent'

    Returns:
        num_refreshed: number of codes reset
    """
    num_unused = unused_mask.sum().item()
    if num_unused == 0:
        return 0

    # Get codebook
    codebook = model.student.feature_extractor.quantizer.vq.layers[0]._codebook.embed

    with torch.no_grad():
        if refresh_strategy == 'random':
            # Reset to random embeddings
            codebook[unused_mask] = torch.randn_like(codebook[unused_mask])
        elif refresh_strategy == 'copy_frequent':
            # Copy from frequently used codes (with noise)
            used_mask = ~unused_mask
            used_indices = torch.where(used_mask)[0]
            for i in torch.where(unused_mask)[0]:
                src_idx = used_indices[torch.randint(0, len(used_indices), (1,))]
                codebook[i] = codebook[src_idx] + 0.01 * torch.randn_like(codebook[src_idx])

    return num_unused
```

## Notes

- Codebook is refreshed **during training** (in-place modification)
- Usage is tracked over a **sliding window** (not cumulative from start)
- Window size = refresh_interval (only recent history matters)
- Refresh happens **before evaluation** to see immediate effect
- No gradient flows through refresh operation (torch.no_grad())

## Potential Issues

1. **Training instability**: Resetting codes mid-training might cause loss spikes
   - Monitor loss curves for anomalies after refresh
   - If unstable, increase refresh_interval or threshold

2. **Code oscillation**: Same codes get reset repeatedly
   - Check refresh_history to see if same codes reset each time
   - May indicate those codes are genuinely not useful

3. **No effect**: Codes get reset but still don't get used
   - Suggests problem is in training dynamics, not just codebook state
   - May need to combine with entropy regularization

## Reference

- Phase 1 failure analysis: [exp_0128/RESULTS.md](../../RESULTS.md)
- Baseline configuration: [exp_0112_intermediate/train_v6.py](../../../exp_0112_intermediate/train_v6.py)
- TracIn diagnosis: [exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md](../../../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)
- VQ-VAE codebook refresh paper: "Neural Discrete Representation Learning" (van den Oord et al., 2017)
