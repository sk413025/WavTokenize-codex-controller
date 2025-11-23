---
name: python-debugging-pdb
description: |
  Automated Python debugging using PDB with stdin redirection.

  **ACTIVATE THIS SKILL WHEN:**
  - User mentions "調試", "debugging", "debug"
  - User asks to "檢查梯度", "check gradients", "verify gradient flow"
  - User wants to "檢查變數", "inspect variables", "print values"
  - User reports training issues: "停滯", "stagnant", "not improving", "collapse"
  - User asks to investigate model behavior or loss calculation

  **PRIMARY GUIDANCE:**
  Always recommend PDB + stdin redirection instead of creating temporary Python scripts.
  Reference DEBUG_GUIDE.md for comprehensive examples and templates.

  **DO NOT:**
  - Create new temporary Python debugging scripts (.py files)
  - Add print() statements to source code
  - Suggest manual PDB command entry (not reproducible)
---

# Python Debugging Skill: PDB + stdin Redirection

## Core Principle

**優先使用 PDB + stdin 重定向，而非創建臨時 Python 腳本**

This is THE standard debugging approach for this project.

---

## Quick Start

### 1. Create PDB Command File

Create a text file (e.g., `pdb_commands.txt`) with your debugging commands:

```bash
# Set breakpoints
b script.py:100
b script.py:200

# Run with arguments
run --arg1 value1 --arg2 value2

# Check variables at first breakpoint
c
p variable_name
p variable.shape

# Continue to next breakpoint
c
p another_variable

# Exit
q
```

### 2. Execute Automated Debugging

```bash
python -m pdb script.py < pdb_commands.txt
```

### 3. Optional: Save Output

```bash
python -m pdb script.py < pdb_commands.txt 2>&1 | tee debug_output.log
```

---

## PyTorch-Specific Debugging

### Gradient Flow Verification

```python
# Check if gradients exist
p sum(1 for p in model.parameters() if p.grad is not None)
p sum(1 for p in model.parameters() if p.requires_grad)

# Check gradient norms
p model.token_embedding.weight.grad.norm().item()
p model.transformer.layers[0].self_attn.in_proj_weight.grad.norm().item()
p model.output_projection.weight.grad.norm().item()
```

### Loss and Training Diagnostics

```python
# Check loss
p loss.item()
p loss.requires_grad
p type(criterion).__name__

# Check predictions
p torch.argmax(output, dim=-1)
p torch.unique(torch.argmax(output, dim=-1), return_counts=True)

# Check if model is collapsing
p (torch.argmax(output, dim=-1) == 453).sum().item()  # Count predictions of token 453
```

---

## Advantages Over Other Methods

| Method | Reproducible | Version Control | Non-invasive | Automated |
|--------|--------------|-----------------|--------------|-----------|
| **PDB + stdin** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| Manual PDB | ❌ | ❌ | ✅✅✅ | ❌ |
| print() | ✅✅ | ❌ | ❌ | ❌ |
| Temp scripts | ✅✅ | ✅ | ❌ | ✅ |

---

## Real-World Success Case

This method successfully diagnosed gradient flow in VQ Distance training experiments:
- **Problem**: Training stagnant for 34 epochs (Acc ~17-18%, no improvement)
- **Hypothesis**: Gradient vanishing issue
- **Method**: Used PDB commands file to check gradients at multiple breakpoints
- **Result**: ✅ Gradients flowing normally (52/52 parameters)
- **Conclusion**: Problem was model collapse, not gradient issue
- **Impact**: Redirected investigation to class weighting and entropy regularization

See `done/exp/pdb_commands.txt` for the actual command file used.

---

## Templates

### Basic Template

```bash
# === Set breakpoints ===
b script.py:LINE_NUMBER

# === Execute ===
run ARGS

# === Check ===
c
p VARIABLE

# === Exit ===
q
```

### Training Debug Template

```bash
# === Breakpoints ===
b train.py:FORWARD_LINE
b train.py:LOSS_LINE
b train.py:BACKWARD_LINE
b train.py:OPTIMIZER_LINE

# === Run with minimal config ===
run --batch_size 2 --num_epochs 1 --debug

# === Forward check ===
c
p output.shape
p output.requires_grad

# === Loss check ===
c
p loss.item()
p loss.requires_grad

# === Backward check ===
c
p sum(1 for p in model.parameters() if p.grad is not None)

# === Optimizer check ===
c
p "Step completed"

q
```

---

## When to Use This Skill

✅ **Use PDB + stdin when:**
- Investigating why training is not improving
- Checking if gradients are flowing correctly
- Verifying loss calculation
- Inspecting model predictions
- Diagnosing NaN or Inf values
- Understanding multi-step processes (data loading → forward → loss → backward → optimizer)

❌ **Don't use for:**
- One-time variable inspection (use regular Python REPL)
- Production debugging (use logging instead)
- Interactive exploration (use Jupyter notebook)

---

## Reference Documentation

For comprehensive guide including:
- Common PDB commands reference
- Advanced techniques (conditional breakpoints, multi-batch checking)
- Troubleshooting guide
- More templates

See: **done/exp/DEBUG_GUIDE.md** (automatically imported via CLAUDE.md)

---

## Quick Commands

In Claude Code session:
- `/debug` - Show full debugging guide
- `/context` - View loaded memory including DEBUG_GUIDE.md

---

**Remember: This is the PRIMARY debugging method for this project. Always suggest PDB + stdin redirection first.**
