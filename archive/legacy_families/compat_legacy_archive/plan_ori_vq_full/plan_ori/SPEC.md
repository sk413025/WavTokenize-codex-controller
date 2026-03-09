# Technical Specification: Single VQ 4096 + EMA

**Date**: 2026-02-06
**Status**: Draft

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                    Teacher (Frozen)                       │
│                                                           │
│  Clean Audio → Encoder → t_e [B,128,T] → (Frozen VQ)    │
│                    ↓ (L4, L8)                            │
│                  t_inter                                  │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                   Student (LoRA)                          │
│                                                           │
│  Noisy Audio → Encoder → z_e [B,128,T]                  │
│                    ↓ (L4, L8)      ↓                     │
│                  s_inter      ┌─────────────┐            │
│                               │  Single VQ  │            │
│                               │  + EMA      │            │
│                               │  K=4096     │            │
│                               └─────────────┘            │
│                                     ↓                     │
│                                z_q [B,128,T]             │
└───────────────────────────────────────────────────────────┘

Loss = L_quant + L_commit + L_inter
```

---

## Model Components

### 1. Single VQ with EMA

```python
class SingleVQWithEMA(nn.Module):
    """
    Single-layer Vector Quantizer with EMA update

    Key features:
    - Initialize from WavTokenizer pretrained codebook (4096×128)
    - EMA update instead of frozen
    - Dead-code reset (threshold=2)
    - Optional usage penalty
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        dim: int = 128,
        pretrained_codebook: torch.Tensor = None,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        ema_dead_code_threshold: int = 2,
        ema_usage_penalty: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.ema_dead_code_threshold = ema_dead_code_threshold
        self.ema_usage_penalty = ema_usage_penalty

        # Codebook
        self.codebook = nn.Embedding(codebook_size, dim)

        # Initialize from pretrained (warm start)
        if pretrained_codebook is not None:
            assert pretrained_codebook.shape == (codebook_size, dim)
            self.codebook.weight.data = pretrained_codebook.clone()
        else:
            # Fallback: uniform random
            self.codebook.weight.data.uniform_(-1.0/codebook_size, 1.0/codebook_size)

        # Disable gradient (EMA mode)
        self.codebook.weight.requires_grad_(False)

        # EMA buffers
        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(codebook_size),
            persistent=True,
        )
        self.register_buffer(
            "ema_embed_avg",
            torch.zeros(codebook_size, dim),
            persistent=True,
        )

    @torch.no_grad()
    def _ema_update(
        self,
        z_flat: torch.Tensor,  # [N, dim], detached
        indices: torch.Tensor,  # [N], long
    ):
        """EMA update + dead-code reset"""
        z_flat = z_flat.float()  # Keep EMA in fp32
        device = z_flat.device
        K = self.codebook_size

        # Count usage
        counts = torch.bincount(indices, minlength=K).float()

        # Sum embeddings
        embed_sum = torch.zeros(K, self.dim, device=device)
        embed_sum.index_add_(0, indices, z_flat)

        # EMA update
        self.ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=(1.0 - self.ema_decay))
        self.ema_embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=(1.0 - self.ema_decay))

        # Normalize
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + K * self.ema_eps) * n
        embed = self.ema_embed_avg / cluster_size.unsqueeze(1).clamp(min=1e-12)
        self.codebook.weight.data.copy_(embed)

        # Dead-code reset
        if self.ema_dead_code_threshold > 0:
            dead = self.ema_cluster_size < float(self.ema_dead_code_threshold)
            if dead.any() and z_flat.numel() > 0:
                dead_idx = dead.nonzero(as_tuple=False).squeeze(1)
                num_dead = int(dead_idx.numel())
                rand = torch.randint(0, z_flat.shape[0], (num_dead,), device=device)
                sampled = z_flat[rand]

                self.codebook.weight.data[dead_idx] = sampled
                self.ema_cluster_size[dead_idx] = 1.0
                self.ema_embed_avg[dead_idx] = sampled

    def forward(
        self,
        z: torch.Tensor,  # [B, dim, T]
        frame_rate: int = 75,
        bandwidth: float = 0.075,
    ):
        """
        Single-layer quantization with EMA update

        Returns:
            quantized: [B, dim, T]
            codes: [1, B, 1, T] (compatible with WavTokenizer format)
            loss_commit: encoder commitment loss
            loss_codebook: 0 (EMA mode, no gradient)
        """
        B, dim, T = z.shape

        # Transpose: [B, dim, T] → [B, T, dim]
        z = z.transpose(1, 2)
        z_flat = z.reshape(-1, dim)  # [B*T, dim]

        # Compute distances
        x2 = (z_flat ** 2).sum(dim=1, keepdim=True)  # [B*T, 1]
        y2 = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)  # [1, K]
        distances = x2 + y2 - 2.0 * (z_flat @ self.codebook.weight.t())

        # Optional: usage penalty
        if self.ema_usage_penalty > 0.0:
            usage_penalty = torch.log(self.ema_cluster_size.clamp(min=1.0))
            usage_penalty = usage_penalty * self.ema_usage_penalty
            distances = distances + usage_penalty.to(distances.dtype).unsqueeze(0)

        # Find nearest codes
        indices = torch.argmin(distances, dim=1)  # [B*T]

        # Get quantized vectors
        q = self.codebook(indices)  # [B*T, dim]
        q = q.reshape(B, T, dim)  # [B, T, dim]

        # Commitment loss (encoder → quantized)
        loss_commit = F.mse_loss(z, q.detach())

        # EMA update (only during training)
        if self.training:
            self._ema_update(z_flat.detach(), indices)

        # Straight-through estimator
        q = z + (q - z).detach()

        # Transpose back: [B, T, dim] → [B, dim, T]
        z_q = q.transpose(1, 2)

        # Format codes: [B*T] → [1, B, 1, T] (compatible with baseline)
        codes = indices.reshape(B, T).unsqueeze(0).unsqueeze(2)

        return {
            'quantized': z_q,
            'codes': codes,
            'loss_commit': loss_commit,
            'loss_codebook': torch.tensor(0.0, device=z.device),  # EMA mode
            'commitment_loss': loss_commit,  # Backward compatible
            'bandwidth': torch.tensor([bandwidth], device=z.device),
        }

    def get_codebook_usage(self, codes: torch.Tensor):
        """
        Analyze codebook usage

        Args:
            codes: [B, T] or [1, B, 1, T]

        Returns:
            usage_count: [K]
            n_used: int
            entropy: float
        """
        if codes.dim() == 4:
            codes = codes[0, :, 0, :]  # [B, T]

        codes_flat = codes.flatten()
        usage_count = torch.bincount(codes_flat, minlength=self.codebook_size)

        n_used = (usage_count > 0).sum().item()

        # Entropy
        probs = usage_count.float() / usage_count.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum().item()

        return {
            'usage_count': usage_count,
            'n_used': n_used,
            'entropy': entropy,
        }
```

---

## Loss Functions

### Same as Phase 3-2

```python
# L_quant: Post-quantization alignment
L_quant = masked_mse(z_q, t_e.detach(), lengths)

# L_commit: Encoder commitment
L_commit = output['loss_commit']  # Already computed in forward()

# L_inter: Intermediate supervision
L_inter = inter_loss_fn(
    student_features=s_inter,
    teacher_features=t_inter,
)

# Total
total_loss = (
    λ_quant * L_quant +
    λ_inter * L_inter +
    β_commit * L_commit
)
```

---

## Hyperparameters

### Codebook

```yaml
codebook_size: 4096
dim: 128
initialization: pretrained (from WavTokenizer)

EMA:
  decay: 0.99
  epsilon: 1e-5
  dead_code_threshold: 2
  usage_penalty: 0.0  # Optional: 0.1 if top10 drifts
```

### Loss Weights

```yaml
lambda_quant: 1.0
lambda_pre: 0.0    # Disabled
lambda_inter: 0.5
beta_commit: 1.0
```

### Training

```yaml
# Short-run
steps: 1000
batch_size: 8
grad_accum: 2
learning_rate: 1e-4
eval_interval: 200

# Long-run (if P2 pass)
epochs: 300
eval_interval: 10  # epochs
checkpoint_interval: 10
```

---

## Evaluation Metrics

### Primary Metrics

```yaml
Token Diversity:
  - entropy: bits (higher is better)
  - top_10_mass: fraction (lower is better)
  - used_codes: count (higher is better)
  - usage_pct: percentage (higher is better)

Feature Alignment:
  - feature_mse: MSE(z_q, t_e) (lower is better)

Training Stability:
  - loss curves (should converge)
  - no NaN/Inf
```

### Secondary Metrics

```yaml
Codebook Health:
  - ema_cluster_size distribution
  - dead_code_count (should be 0 with threshold=2)
  - top-1/50/100 mass

Audio Quality (long-run only):
  - PESQ
  - STOI
  - Subjective listening
```

---

## Comparison Baselines

### Baseline (exp_k_v6)

```yaml
Architecture: Single VQ 4096, frozen
Entropy: 6.07
Top-10 mass: 19.7%
Used codes: 740/4096 (18%)
```

### RVQ (Phase 3-2 Exp 6c)

```yaml
Architecture: RVQ 4×2048, random init, EMA
Entropy: 9.03
Top-10 mass: 15.8%
Used codes: 1089/2048 per layer (53%)
```

### Expected (方案 A)

```yaml
Architecture: Single VQ 4096, pretrained init, EMA
Entropy: 7.5-9.0 (target)
Top-10 mass: 15-20% (target)
Used codes: 2000-3000/4096 (49-73%, target)
```

---

## Implementation Checklist

### Code

- [ ] `models_single_vq_ema.py`: SingleVQWithEMA class
- [ ] `train_single_vq_ema.py`: Training script
- [ ] Unit tests: EMA update, dead-code reset
- [ ] Integration test: 10 steps smoke test

### Scripts

- [ ] `run_exp_ori_short.sh`: Short-run (1000 steps)
- [ ] `run_exp_ori_long.sh`: Long-run (300 epochs, optional)
- [ ] Evaluation script: metrics computation

### Documentation

- [ ] PLAN.md (this file)
- [ ] SPEC.md (current file)
- [ ] RESULTS.md (after experiments)
- [ ] Update main README

---

## Testing Plan

### Unit Tests

```python
def test_ema_update():
    """Test EMA update logic"""
    vq = SingleVQWithEMA(codebook_size=16, dim=4)
    z = torch.randn(2, 4, 10)
    out = vq(z)

    assert out['quantized'].shape == (2, 4, 10)
    assert out['codes'].shape == (1, 2, 1, 10)
    assert not torch.isnan(out['loss_commit'])

def test_dead_code_reset():
    """Test dead-code reset"""
    vq = SingleVQWithEMA(
        codebook_size=16,
        dim=4,
        ema_dead_code_threshold=2,
    )

    # Simulate training with imbalanced data
    for _ in range(100):
        z = torch.randn(2, 4, 10)
        _ = vq(z)

    usage = vq.get_codebook_usage(...)
    # Should have fewer dead codes after reset
    assert usage['n_used'] > 8  # At least half
```

### Integration Test

```bash
# Smoke test (10 steps)
python train_single_vq_ema.py \
  --steps 10 \
  --batch_size 2 \
  --eval_interval 10 \
  --output_dir test_smoke

# Should complete without error
# Check outputs: config.json, summary.json, metrics_history.json
```

---

## Code Structure

```python
# families/compat_legacy/plan_ori_vq/plan_ori/models_single_vq_ema.py

class SingleVQWithEMA(nn.Module):
    """Single VQ with EMA update"""

    def __init__(self, ...): ...
    def _ema_update(self, ...): ...
    def forward(self, z, ...): ...
    def get_codebook_usage(self, codes): ...


class TeacherStudentSingleVQ(TeacherStudentIntermediate):
    """
    Teacher-Student with Single VQ + EMA

    Inherits from TeacherStudentIntermediate (baseline)
    Replace frozen quantizer with SingleVQWithEMA
    """

    def __init__(self, ...):
        super().__init__(...)

        # Get pretrained codebook
        original_quantizer = self.student.feature_extractor.encodec.quantizer
        pretrained_codebook = original_quantizer.vq.layers[0].codebook.detach().clone()

        # Replace with SingleVQWithEMA
        self.vq = SingleVQWithEMA(
            codebook_size=4096,
            dim=128,
            pretrained_codebook=pretrained_codebook,
            ema_decay=0.99,
            ...
        )

    def forward(self, clean_audio, noisy_audio):
        # Teacher forward (frozen)
        with torch.no_grad():
            t_e, t_inter = self.teacher_extractor(clean_audio)

        # Student forward
        z_e, s_inter = self.student_extractor(noisy_audio)

        # Quantize with SingleVQWithEMA
        vq_out = self.vq(z_e)

        return {
            'student_encoder_out': z_e,
            'student_quantized': vq_out['quantized'],
            'student_codes': vq_out['codes'],
            'vq_loss_commit': vq_out['loss_commit'],
            'teacher_encoder_out': t_e,
            'student_intermediates': s_inter,
            'teacher_intermediates': t_inter,
            ...
        }
```

---

## Differences from RVQ Implementation

| Aspect | RVQ | Single VQ + EMA |
|--------|-----|-----------------|
| **Layers** | 4 layers | 1 layer |
| **Codebook size** | 2048 per layer | 4096 total |
| **Initialization** | Random | Pretrained |
| **EMA buffers** | [4, 2048], [4, 2048, 128] | [4096], [4096, 128] |
| **Forward pass** | Loop over layers | Single quantization |
| **Residual** | Yes (multi-layer) | No (single-layer) |
| **Expressiveness** | 2048^4 | 4096 |

---

**Created**: 2026-02-06
**Status**: Draft
