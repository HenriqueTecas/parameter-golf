"""
Hybrid GLA (Gated Linear Attention) + Full Attention with Ternary QAT + Depth Recurrence
Architecture: 12 unique layers (9 GLA + 3 full attention) with inner-loop recurrence
  - 2 entry layers + 8 recurrent layers (×N) + 2 exit layers = 20+ effective layers
Compression: Ternary QAT for weight matrices, int8 for embedding, fp32 for control tensors
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_stride = int(os.environ.get("VAL_STRIDE", 64))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    chunk_size = int(os.environ.get("CHUNK_SIZE", 64))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))
    num_recurrences = int(os.environ.get("NUM_RECURRENCES", 2))
    num_entry_layers = int(os.environ.get("NUM_ENTRY_LAYERS", 2))
    num_exit_layers = int(os.environ.get("NUM_EXIT_LAYERS", 2))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 0))
    diag_mask_last_n = int(os.environ.get("DIAG_MASK_LAST_N", 0))
    late_qat_frac = float(os.environ.get("LATE_QAT_FRAC", 0.0))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov,
                weight_decay=weight_decay,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Sliding window evaluation: each window is train_seq_len tokens of context,
    # but only the last val_stride tokens contribute to the loss / BPB metric.
    # This gives every evaluated position at least (seq_len - stride) tokens of
    # left context, reducing the "cold start" penalty of non-overlapping chunks.
    seq_len = args.train_seq_len
    stride = min(args.val_stride, seq_len)
    if stride <= 0:
        stride = seq_len

    total_tokens = val_tokens.numel() - 1
    num_windows = max(1, (total_tokens - seq_len) // stride + 1)

    # Distribute windows evenly across ranks
    win_start = (num_windows * rank) // world_size
    win_end = (num_windows * (rank + 1)) // world_size

    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    batch_size = max(1, local_batch_tokens // seq_len)

    # Create an O(1) strided view of overlapping windows to avoid heavy CPU indexing
    windows = val_tokens.unfold(0, seq_len + 1, stride)

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for b_start in range(win_start, win_end, batch_size):
            b_end = min(b_start + batch_size, win_end)
            actual_batch = b_end - b_start

            # Fast basic slicing from the strided view
            tokens = windows[b_start:b_end]                  # CPU, uint16, fast view
            x = tokens[:, :-1].to(device=device, dtype=torch.int64, non_blocking=True)
            y = tokens[:, 1:].to(device=device, dtype=torch.int64, non_blocking=True)

            # Forward pass on full context to get logits
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x, None).detach()  # [actual_batch, seq_len, vocab]

            # Only score the last `stride` positions
            logits_s = logits[:, -stride:, :].contiguous()
            y_s = y[:, -stride:].contiguous()
            per_token_loss = F.cross_entropy(
                logits_s.reshape(-1, logits_s.size(-1)),
                y_s.reshape(-1),
                reduction="none",
            )

            val_loss_sum += per_token_loss.to(torch.float64).sum()
            val_token_count += float(actual_batch * stride)

            # Byte counting for BPB (stride portion only)
            prev_ids = x[:, -stride:].reshape(-1)
            tgt_ids = y_s.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# TERNARY QUANTIZATION AWARE TRAINING
# -----------------------------


class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor) -> Tensor:
        scale = weight.abs().mean().clamp(min=1e-5)
        # fp16 scale simulation: match serialization precision to minimize roundtrip gap
        scale = scale.half().float()
        threshold = 0.5 * scale
        w_ternary = torch.where(
            weight > threshold,
            torch.ones_like(weight),
            torch.where(
                weight < -threshold, -torch.ones_like(weight), torch.zeros_like(weight)
            ),
        )
        ctx.save_for_backward(weight, scale)
        return w_ternary * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


class QATLinear(nn.Module):
    # Global toggle for late QAT: when False, all QATLinear layers use fp32 weights
    qat_globally_enabled: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        apply_qat: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.apply_qat = apply_qat
        nn.init.kaiming_normal_(self.weight)
        if zero_init:
            nn.init.zeros_(self.weight)
        self._zero_init = zero_init

    def forward(self, x: Tensor) -> Tensor:
        if self.apply_qat and self.training and QATLinear.qat_globally_enabled:
            w_q = TernaryQuantizeSTE.apply(self.weight)
            return F.linear(x, w_q.to(x.dtype), None)
        return F.linear(x, self.weight.to(x.dtype), None)


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


# -----------------------------
# GATED LINEAR ATTENTION (multi-head, chunked)
# -----------------------------


class GatedLinearAttention(nn.Module):
    """Multi-head chunked linear attention with per-head gated recurrent state."""

    def __init__(self, dim: int, num_heads: int, chunk_size: int = 64):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        self.q_proj = QATLinear(dim, dim, apply_qat=True)
        self.k_proj = QATLinear(dim, dim, apply_qat=True)
        self.v_proj = QATLinear(dim, dim, apply_qat=True)
        self.out_proj = QATLinear(dim, dim, apply_qat=True, zero_init=True)

        self.gate_logit = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        causal = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim
        C = self.chunk_size
        NC = T // C

        q = self.q_proj(x).reshape(B, T, H, d).permute(0, 2, 1, 3)
        k = F.normalize(
            self.k_proj(x).reshape(B, T, H, d).permute(0, 2, 1, 3), dim=-1
        )
        v = self.v_proj(x).reshape(B, T, H, d).permute(0, 2, 1, 3)

        q = q.reshape(B, H, NC, C, d)
        k = k.reshape(B, H, NC, C, d)
        v = v.reshape(B, H, NC, C, d)

        # Intra-chunk: causal linear attention via masked matmul (parallel)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores.masked_fill(~self.causal_mask, 0.0)
        intra = torch.matmul(scores, v)

        # Pre-compute chunk-level KV summaries for state updates
        chunk_kv = torch.einsum("bhncd,bhnce->bhnde", k, v)

        # Sequential gated state accumulation
        gate = torch.sigmoid(self.gate_logit).to(x.dtype)
        g = gate[None, :, None, None]
        S = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        states = []
        for ci in range(NC):
            states.append(S)
            S = (1.0 - g) * S + g * (chunk_kv[:, :, ci] / C)
        S_all = torch.stack(states, dim=2)

        # Inter-chunk: query accumulated states (parallel)
        inter = torch.einsum("bhncd,bhnde->bhnce", q, S_all)

        out = (intra + inter).reshape(B, H, T, d)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(out)


# -----------------------------
# GQA SELF-ATTENTION
# -----------------------------


class GQASelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = QATLinear(dim, dim, apply_qat=True)
        self.k_proj = QATLinear(dim, kv_dim, apply_qat=True)
        self.v_proj = QATLinear(dim, kv_dim, apply_qat=True)
        self.out_proj = QATLinear(dim, dim, apply_qat=True, zero_init=True)

        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False  # Enabled selectively on deeper layers
        self.use_diag_mask = False  # Diagonal masking (ternary-safe XSA alternative)
        # Will be set via set_diag_mask() before torch.compile
        self.register_buffer("diag_mask", None, persistent=False)

    def set_diag_mask(self, seq_len: int, device: torch.device) -> None:
        """Pre-build the causal+diagonal mask for torch.compile compatibility."""
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        diag = torch.eye(seq_len, dtype=torch.bool, device=device)
        diag[0, 0] = False  # Keep position 0's self-attention (no other context with causal)
        self.diag_mask = causal & ~diag

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        """Exclusive Self Attention: subtract self-value projection (GQA-aware, zero-alloc).
        y: [B, H, T, D], v: [B, Hkv, T, D]."""
        B, H, T, D = y.shape
        Hkv = v.size(1)
        group = H // Hkv
        # Reshape y into KV head groups — free view, no memory alloc
        y_g = y.reshape(B, Hkv, group, T, D)
        vn = F.normalize(v, dim=-1).unsqueeze(2)  # [B, Hkv, 1, T, D]
        # Project out self-value component per KV head group
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, H, T, D)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = (
            self.q_proj(x)
            .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        if self.use_diag_mask:
            # Expand KV heads for GQA compat (Flash/mem_efficient don't support mask+GQA)
            if self.num_kv_heads != self.num_heads:
                group = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(group, dim=1)
                v = v.repeat_interleave(group, dim=1)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=self.diag_mask,
                is_causal=False,
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        if self.use_xsa:
            y = self._xsa(y, v)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.out_proj(y)


# -----------------------------
# relu^2 MLP
# -----------------------------


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = QATLinear(dim, hidden, apply_qat=True)
        self.proj = QATLinear(hidden, dim, apply_qat=True, zero_init=True)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


# -----------------------------
# HYBRID BLOCK (with baseline stability features)
# -----------------------------


class HybridBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        is_attention: bool,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        chunk_size: int,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()

        if is_attention:
            self.mixer = GQASelfAttention(
                dim, num_heads, num_kv_heads, rope_base, qk_gain_init
            )
        else:
            self.mixer = GatedLinearAttention(dim, num_heads, chunk_size)

        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.mixer(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x)
        )
        return x


# -----------------------------
# HYBRID GPT MODEL (encoder-decoder with skip connections)
# -----------------------------


class HybridGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        chunk_size: int,
        logit_softcap: float,
        tied_embed_init_std: float,
        qk_gain_init: float,
        num_recurrences: int = 2,
        num_entry_layers: int = 2,
        num_exit_layers: int = 2,
    ):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tied_embed_init_std = tied_embed_init_std
        self.num_recurrences = num_recurrences
        self.num_entry_layers = num_entry_layers
        self.num_exit_layers = num_exit_layers

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

        # Recurrent block: layers between entry and exit
        num_recurrent_layers = num_layers - num_entry_layers - num_exit_layers
        self.num_recurrent_layers = num_recurrent_layers
        # U-Net skip connections within the recurrent block
        rec_encoder = num_recurrent_layers // 2
        rec_decoder = num_recurrent_layers - rec_encoder
        self.rec_encoder_count = rec_encoder
        self.rec_decoder_count = rec_decoder
        self.num_skip_weights = min(rec_encoder, rec_decoder)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )

        # RMSNorm applied between recurrence passes for gradient stability
        self.recurrence_norm = RMSNorm()

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            is_attention = i % 4 == 3
            self.blocks.append(
                HybridBlock(
                    model_dim,
                    is_attention=is_attention,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_mult=mlp_mult,
                    rope_base=rope_base,
                    chunk_size=chunk_size,
                    qk_gain_init=qk_gain_init,
                )
            )

        self.final_norm = RMSNorm()

    def forward(
        self, input_ids: Tensor, target_ids: Tensor | None = None
    ) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        entry_end = self.num_entry_layers
        rec_start = entry_end
        rec_end = rec_start + self.num_recurrent_layers
        exit_start = rec_end

        # Entry layers (unique, run once)
        for i in range(entry_end):
            x = self.blocks[i](x, x0)

        # Recurrent layers (run num_recurrences times)
        for r in range(self.num_recurrences):
            if r > 0:
                x = self.recurrence_norm(x)

            # Encoder half of recurrent block: save skip connections
            skip_connections: list[Tensor] = []
            for i in range(self.rec_encoder_count):
                x = self.blocks[rec_start + i](x, x0)
                skip_connections.append(x)

            # Decoder half of recurrent block: consume skip connections
            for j in range(self.rec_decoder_count):
                block_idx = rec_start + self.rec_encoder_count + j
                if j < self.num_skip_weights:
                    skip = skip_connections[self.num_skip_weights - 1 - j]
                    sw = self.skip_weights[j].to(dtype=x.dtype)[None, None, :]
                    x = x + sw * skip
                x = self.blocks[block_idx](x, x0)

        # Exit layers (unique, run once)
        for i in range(exit_start, len(self.blocks)):
            x = self.blocks[i](x, x0)

        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if target_ids is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# -----------------------------
# SERIALIZATION
# -----------------------------

CONTROL_TENSOR_NAMES = (
    "gate_logit",
    "q_gain",
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "skip_weights",
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_CLIP_Q = 0.9999984


def quantize_to_ternary(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    scale = t32.abs().mean().clamp(min=1e-5)
    threshold = 0.5 * scale
    w_ternary = torch.where(
        t32 > threshold,
        torch.ones_like(t32),
        torch.where(t32 < -threshold, -torch.ones_like(t32), torch.zeros_like(t32)),
    )
    return w_ternary.to(torch.int8), scale.to(torch.float16)


def quantize_float_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        clipped = torch.maximum(
            torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None]
        )
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(torch.float16).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def serialize_model(model: nn.Module, code: str) -> dict:
    state_dict = model.state_dict()

    ternary_weights: dict = {}
    int8_weights: dict = {}
    float_weights: dict = {}

    ternary_patterns = ("q_proj", "k_proj", "v_proj", "out_proj", "fc", "proj")

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        is_control = any(p in name for p in CONTROL_TENSOR_NAMES)
        is_small = t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL
        is_ternary_target = (
            any(p in name for p in ternary_patterns) and "weight" in name
        )

        if is_control:
            float_weights[name] = t.float()
        elif is_small:
            float_weights[name] = t.to(torch.float16)
        elif is_ternary_target:
            w_q, scale = quantize_to_ternary(t)
            ternary_weights[name] = {"w": w_q, "s": scale}
        else:
            q, s = quantize_float_tensor_int8(t)
            int8_weights[name] = {"q": q, "s": s}

    return {
        "format": "hybrid_ternary_v2",
        "ternary": ternary_weights,
        "int8": int8_weights,
        "float": float_weights,
        "code": code,
    }


def deserialize_model(obj: dict, model: nn.Module) -> None:
    state_dict = {}

    for name, data in obj["ternary"].items():
        w = data["w"].float() * data["s"].float()
        state_dict[name] = w

    for name, data in obj["int8"].items():
        q, s = data["q"], data["s"]
        if s.ndim > 0:
            w = q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1))).float()
        else:
            w = q.float() * s.item()
        state_dict[name] = w

    for name, t in obj["float"].items():
        state_dict[name] = t

    model.load_state_dict(state_dict, strict=True)


# -----------------------------
# MAIN TRAINING
# -----------------------------


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 8 // world_size))
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(True)  # Needed for diagonal mask (flash doesn't support attn_mask)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"Script only setup for SentencePiece .model file: {args.tokenizer_path}"
        )
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError("VOCAB_SIZE mismatch")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    log0("val_bpb:enabled tokenizer_kind=sentencepiece")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards tokens:{val_tokens.numel() - 1}")

    base_model = (
        HybridGPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            rope_base=args.rope_base,
            chunk_size=args.chunk_size,
            logit_softcap=args.logit_softcap,
            tied_embed_init_std=args.tied_embed_init_std,
            qk_gain_init=args.qk_gain_init,
            num_recurrences=args.num_recurrences,
            num_entry_layers=args.num_entry_layers,
            num_exit_layers=args.num_exit_layers,
        )
        .to(device)
        .bfloat16()
    )

    # Enable XSA on the last N attention layers
    attn_blocks = [b for b in base_model.blocks if isinstance(b.mixer, GQASelfAttention)]
    if args.xsa_last_n > 0:
        for b in attn_blocks[-args.xsa_last_n:]:
            b.mixer.use_xsa = True
        log0(f"xsa:enabled on last {min(args.xsa_last_n, len(attn_blocks))} of {len(attn_blocks)} attention layers")

    # Enable diagonal masking on the last N attention layers (ternary-safe XSA alternative)
    if args.diag_mask_last_n > 0:
        for b in attn_blocks[-args.diag_mask_last_n:]:
            b.mixer.use_diag_mask = True
            b.mixer.set_diag_mask(args.train_seq_len, device)
        log0(f"diag_mask:enabled on last {min(args.diag_mask_last_n, len(attn_blocks))} of {len(attn_blocks)} attention layers")

    # Keep QATLinear weights in fp32 for gradient quality
    for module in base_model.modules():
        if isinstance(module, QATLinear):
            module.weight.data = module.weight.data.float()

    # Keep control/scalar parameters in fp32
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if (
                param.ndim < 2
                or any(p in name for p in CONTROL_TENSOR_NAMES)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # Optimizer parameter grouping
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pn in name for pn in CONTROL_TENSOR_NAMES)
    ]
    block_scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pn in name for pn in CONTROL_TENSOR_NAMES)
    ]
    scalar_params = block_scalar_params + [base_model.skip_weights]

    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    gla_layers = args.num_layers - args.num_layers // 4
    attn_layers = args.num_layers // 4
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    effective_depth = args.num_entry_layers + base_model.num_recurrent_layers * args.num_recurrences + args.num_exit_layers
    log0(
        f"architecture:hybrid_gla num_layers:{args.num_layers} "
        f"gla_layers:{gla_layers} attention_layers:{attn_layers} "
        f"entry_layers:{args.num_entry_layers} recurrent_layers:{base_model.num_recurrent_layers} "
        f"exit_layers:{args.num_exit_layers} num_recurrences:{args.num_recurrences} "
        f"effective_depth:{effective_depth}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len}"
    )
    log0(
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds}"
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return (
            remaining_ms / max(warmdown_ms, 1e-9)
            if remaining_ms <= warmdown_ms
            else 1.0
        )

    if args.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1
                    )
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (
                args.warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == args.warmup_steps
            ):
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )

    # Late QAT: start with QAT disabled if late_qat_frac > 0
    if args.late_qat_frac > 0:
        QATLinear.qat_globally_enabled = False

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )

        should_validate = last_step or (
            args.val_loss_every > 0 and step % args.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Late QAT: train full precision early, enable ternary STE when LR starts decaying
        if args.late_qat_frac > 0:
            progress = elapsed_ms / max_wallclock_ms if max_wallclock_ms else step / max(args.iterations, 1)
            was_enabled = QATLinear.qat_globally_enabled
            QATLinear.qat_globally_enabled = progress >= (1.0 - args.late_qat_frac)
            if QATLinear.qat_globally_enabled and not was_enabled:
                log0(f"late_qat: enabling ternary QAT at step {step} (progress={progress:.3f})")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            1 - frac
        ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if master_process:
        obj = serialize_model(base_model, code)
        buf = io.BytesIO()
        torch.save(obj, buf)
        raw_bytes = len(buf.getvalue())
        compressed = zlib.compress(buf.getvalue(), level=9)

        with open("final_model.ternary.ptz", "wb") as f:
            f.write(compressed)

        quant_file_bytes = os.path.getsize("final_model.ternary.ptz")
        code_bytes = len(code.encode("utf-8"))
        total_bytes = quant_file_bytes + code_bytes

        log0(f"Serialized model: {quant_file_bytes} bytes (raw_torch:{raw_bytes})")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {total_bytes} bytes")

        if total_bytes > 16_000_000:
            log0(f"WARNING: Exceeds 16MB limit by {total_bytes - 16_000_000} bytes")

    if distributed:
        dist.barrier()

    with open("final_model.ternary.ptz", "rb") as f:
        loaded_obj = torch.load(
            io.BytesIO(zlib.decompress(f.read())),
            map_location="cpu",
            weights_only=False,
        )
    deserialize_model(loaded_obj, base_model)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
