# Changes: HybridDeltaNet v1 -> Hybrid GLA v2

## Summary

Complete rewrite to fix 3 fatal bugs (OOM, model too large, missing delta rule)
and restore proven training stability features from the baseline.

---

## Critical Bug Fixes

### 1. OOM Fix: Single-Head -> Multi-Head Linear Attention

**Before:** `ChunkedDeltaNet` operated on the full D=512 dimension as a single head.
The outer product `torch.einsum("btd,bte->btde", k, v)` created `[B, 64, 512, 512]`
tensors (~4.3 GB per chunk per layer). With 24 layers this instantly OOMs on H100s.

**After:** `GatedLinearAttention` splits D=512 across 8 heads (d_head=64). The
attention computation uses efficient `[B, H, NC, C, C]` score matrices instead of
`[B, C, D, D]` outer products. Memory per chunk per layer drops from ~4.3 GB to
~64 MB (a **67x reduction**).

Additionally, the chunked computation is now fully parallelized:
- Intra-chunk attention: parallel masked matmul over all chunks simultaneously
- Inter-chunk state: sequential gated accumulation (lightweight, no large tensors)
- Inter-chunk readout: parallel einsum over all chunks

### 2. Size Fix: 32 Layers -> 12 Layers, SwiGLU -> relu^2

**Before:** 32 layers with SwiGLU (3 matrices per FFN) = ~81M params.
Even with ternary compression this exceeded the 16MB limit.

**After:** 12 layers with relu^2 MLP (2 matrices per FFN) = ~22.6M params.
Estimated compressed size: ~8 MB (well under 16MB limit).

| Change | Param Impact |
|--------|-------------|
| 32 -> 12 layers | -62.5% layers |
| SwiGLU -> relu^2 | -33% params per FFN (2 matrices vs 3) |
| num_kv_heads 2 -> 4 | +params per GQA layer (for quality) |

### 3. Architecture Fix: Fake DeltaNet -> Gated Linear Attention

**Before:** The `ChunkedDeltaNet` was labeled as DeltaNet but was missing the
delta update rule (`S_t = S + beta * (kv - k @ k^T @ S)`). It was just vanilla
cumulative linear attention with a scalar beta gate.

**After:** Replaced with `GatedLinearAttention` -- an honest name for what the
architecture actually does. Uses per-head sigmoid-gated state accumulation:
```
gate = sigmoid(gate_logit)   # per-head, learned, in (0, 1)
S = (1 - gate) * S + gate * (chunk_kv / C)
```
This is numerically stable (gate bounded by sigmoid, state normalized by chunk
size) and gives the model per-head control over memory retention vs. update rate.

---

## Restored Baseline Stability Features

The original submission removed all of the baseline's training stabilization
features while increasing depth from 9 to 32 layers. This rewrite restores them:

### Encoder-Decoder Skip Connections
- Model split into encoder half (layers 0-5) and decoder half (layers 6-11)
- Learnable `skip_weights` connect encoder outputs to decoder inputs (U-Net style)
- Decoder layer i receives skip from encoder layer (num_skip - 1 - i)

### Residual Mixing (`resid_mix`)
- Each block receives both `x` (current hidden state) and `x0` (initial embedding)
- Learnable per-dimension mixing: `x = mix[0] * x + mix[1] * x0`
- Prevents representation collapse in deep networks

### Per-Dimension Output Scaling
- `attn_scale`: per-dimension learnable scale on attention/GLA output
- `mlp_scale`: per-dimension learnable scale on MLP output
- Both initialized to 1.0, learned during training

### x0 Passthrough
- Initial embedding `x0` passed to every block for reference
- Provides a stable gradient highway through the entire network

---

## Other Improvements

### q_gain Initialization: 1.0 -> 1.5
Restored the baseline's tuned value for sharper early attention patterns in the
GQA layers. Set via `qk_gain_init` hyperparameter.

### gate_logit (replaces beta)
- Per-head learnable gate (8 values) instead of a single scalar
- Constrained to (0, 1) via sigmoid -- prevents unbounded state growth
- Initialized at sigmoid(0) = 0.5 for balanced retention/update

### Control Tensor Handling
Updated `CONTROL_TENSOR_NAMES` to include all new stability parameters:
```python
("gate_logit", "q_gain", "attn_scale", "mlp_scale", "resid_mix", "skip_weight")
```
All kept in fp32 during serialization for maximum precision.

### fp32 Parameter Restoration
Added explicit restoration of scalar/control parameters to fp32 after the
`.bfloat16()` model cast (matching the baseline's `restore_low_dim_params_to_fp32`
pattern). This ensures gradient quality for small but important parameters.

### Gradient Clipping
Kept at 1.0 (changed from baseline's 0.0). Helps stabilize training with
ternary QAT's noisy STE gradients.

---

## Architecture Comparison

| Feature | Baseline | v1 (Before) | v2 (After) |
|---------|----------|-------------|------------|
| Total layers | 9 | 32 | 12 |
| Attention layers | 9 (all) | 8 (every 4th) | 3 (every 4th) |
| Linear attn layers | 0 | 24 | 9 |
| FFN type | relu^2 | SwiGLU | relu^2 |
| Heads (linear attn) | N/A | 1 (single) | 8 (multi-head) |
| num_kv_heads | 4 | 2 | 4 |
| Skip connections | yes | **no** | yes |
| resid_mix | yes | **no** | yes |
| attn/mlp_scale | yes | **no** | yes |
| x0 passthrough | yes | **no** | yes |
| State gating | N/A | scalar beta (unbounded) | per-head sigmoid gate |
| Quantization | int8+zlib | ternary QAT | ternary QAT |
| Est. params | ~17M | ~81M | ~22.6M |
| Est. compressed | 15.86 MB | >16 MB (fails) | ~8 MB |

---

## Parameter Budget Estimate

```
Component              | Per Layer  | Layers | Total
-----------------------|------------|--------|----------
GLA block              | ~1.84M     | 9      | ~16.5M
  q/k/v/out_proj (4x)  | 1,048,576  |        |
  MLP fc + proj (2x)   | 1,048,576  |        |
  gate_logit + scales   | ~2,056     |        |
GQA block              | ~1.84M     | 3      | ~5.5M
  q/k/v/out_proj        | 786,432    |        |
  MLP fc + proj (2x)   | 1,048,576  |        |
  q_gain + scales       | ~2,056     |        |
Embedding (tied)       |            |        | 524,288
Skip weights (6x512)   |            |        | 3,072
-----------------------|------------|--------|----------
TOTAL                  |            |        | ~22.6M
```

Compression estimate:
- ~22M ternary params x ~0.3 bytes/param (zlib on {-1,0,+1}) = ~6.6 MB
- Embedding (int8+zlib): ~0.5 MB
- Float control tensors: ~0.1 MB
- Code: ~0.06 MB
- **Total: ~7.3 MB** (well under 16 MB limit)

---

## What Was Kept From v1

- `TernaryQuantizeSTE` -- correct STE implementation with 0.5*mean(|w|) threshold
- `QATLinear` -- ternary QAT during training, full precision during eval
- Three-tier serialization (ternary/int8/float)
- Hybrid architecture concept (mixing linear + full attention)
- All infrastructure: Muon optimizer, data loading, eval, distributed training
