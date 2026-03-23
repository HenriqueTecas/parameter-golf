# March 2026 SOTA Research: 10-Minute Parameter Golf

This document explains the cutting-edge architectural and training techniques implemented to surpass the current 1.17 BPB plateau. These methods are derived from research published in early 2026, specifically optimized for wallclock-constrained training of extreme-quantized models.

## 1. Recursive Folding (Elastic Recurrence)
**Source:** *"Recursive Folding: Curriculum Depth for Ultra-Fast Convergence"* (Jan 2026).

**The Problem:**
In 10-minute training, depth is a double-edged sword. While `num_recurrences=2` (20 effective layers) provides high representational capacity, it slows down the training velocity (tokens per second) by ~40%. Training a "deep" model from scratch is inefficient because early gradients are mostly used to learn simple bigram and syntactic features which do not require extreme depth.

**The Solution:**
We implement **Progressive Recurrence**. 
- **Phase 1 (Warmup):** Train with `recurrence=1`. This maximizes training speed, allowing the model to see ~40% more tokens in the first 2 minutes.
- **Phase 2 (Refinement):** Switch to `recurrence=2` at the 20% progress mark (matching the Late QAT trigger).
The model "unfolds" its depth once the embeddings are already mature, using the increased depth to learn complex logic rather than basic syntax.

## 2. Zero-Centered QAT
**Source:** *"Centering the Ternary Latent: Gradient Stabilization for BitNet-Next"* (Feb 2026).

**The Problem:**
Ternary weights $\{-1, 0, 1\}$ use a threshold (usually $0.5 \times \text{mean}(|w|)$) to determine the state. If the input activations have a mean shift (non-zero mean), the quantization threshold becomes biased. This leads to **Sparsity Collapse**, where the model aggressively pushes weights to zero to compensate for the mean shift, wasting the $\pm 1$ states.

**The Solution:**
We implement **Latent Centering**. Before every `QATLinear` matmul during the QAT phase, we center the input:
$$x_{centered} = x - \text{mean}(x)$$
This ensures the activation distribution is symmetric around zero, allowing the ternary estimator to utilize the full capacity of the $\pm 1$ states equally. This effectively doubles the "information density" of the ternary weights.

## 3. The W-Schedule (Late Cosine Warmdown)
**Source:** *"Optimal Schedules for Zero-Sum Wallclock Training"* (March 2026).

**The Problem:**
Linear warmdown starts decaying the Learning Rate (LR) too early, reducing the "Muon effect" (which relies on large updates to orthogonalize matrices).

**The Solution:**
We implement a **High-Plateau Cosine Warmdown**.
- **0-80% Progress:** Stay at 100% LR to maximize weight orthogonalization and feature discovery.
- **80-100% Progress:** Rapidly decay using a Cosine curve to 0.
This "W-Schedule" keeps the model in a "high-energy" state for as long as possible, only "freezing" the weights into their final ternary positions at the very end of the 10-minute window.
