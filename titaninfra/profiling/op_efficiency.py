"""Roofline analysis for operator efficiency and optimization headroom.

GEMM utilization compares achieved FLOP/s with the BF16 hardware peak,
communication compares bytes/s with NVLink bandwidth, and memory-bound kernels
compare bytes/s with HBM bandwidth. headroom_ms = time_ms * (1 - utilization)
prioritizes expensive operators that remain far from their roof.

FLOPs are derived analytically from model dimensions with no torch dependency.
"""

from dataclasses import dataclass

# Dimensions for the DeepSeek V3 refmoe-20B configuration.
REFMOE_20B_DIMS: dict[str, int] = {
    "dim": 2048,
    "n_layers": 32,
    "n_dense_layers": 1,        # First layer is dense FFN; remaining 31 are MoE
    "n_heads": 16,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dense_hidden_dim": 7168,
    "moe_hidden_dim": 768,
    "num_experts": 128,
    "num_shared_experts": 1,
    "router_top_k": 4,
}


@dataclass(frozen=True)
class HardwareRoof:
    """Hardware roof: BF16 FLOP/s plus NVLink/HBM bytes per second."""

    name: str
    peak_flops: float       # BF16 dense FLOP/s
    nvlink_bw: float        # Unidirectional aggregate NVLink bytes/s
    hbm_bw: float           # HBM bytes/s for memory-bound kernels


# Default to H200 SXM with NVLink-4; peak_flops matches TorchTitan.
ROOFS: dict[str, HardwareRoof] = {
    "H200": HardwareRoof("H200", peak_flops=989e12, nvlink_bw=900e9, hbm_bw=4.8e12),
    "H100": HardwareRoof("H100", peak_flops=989e12, nvlink_bw=900e9, hbm_bw=3.35e12),
    "H800": HardwareRoof("H800", peak_flops=989e12, nvlink_bw=400e9, hbm_bw=3.35e12),
    "H20": HardwareRoof("H20", peak_flops=148e12, nvlink_bw=900e9, hbm_bw=4.0e12),
    "A100": HardwareRoof("A100", peak_flops=312e12, nvlink_bw=600e9, hbm_bw=2.0e12),
}
DEFAULT_ROOF = "H200"


# Analytical FLOPs per forward call. GEMM [M,K]x[K,N] = 2*M*K*N;
# tokens = local_batch_size * seq_len.

def attn_flops_per_call(dims: dict, tokens: int, batch: int, seq_len: int) -> float:
    """Estimate FLOPs for one MLA layer forward call.

    Includes five projection GEMMs and a causal upper bound for attention;
    block-causal packing may perform less work than this bound.
    """
    d = dims
    qk_head_dim = d["qk_nope_head_dim"] + d["qk_rope_head_dim"]
    nh = d["n_heads"]
    # Projection GEMMs.
    wq_a = 2 * tokens * d["dim"] * d["q_lora_rank"]
    wq_b = 2 * tokens * d["q_lora_rank"] * (nh * qk_head_dim)
    wkv_a = 2 * tokens * d["dim"] * (d["kv_lora_rank"] + d["qk_rope_head_dim"])
    wkv_b = 2 * tokens * d["kv_lora_rank"] * (nh * (d["qk_nope_head_dim"] + d["v_head_dim"]))
    wo = 2 * tokens * (nh * d["v_head_dim"]) * d["dim"]
    proj = wq_a + wq_b + wkv_a + wkv_b + wo
    # Attention core (QK^T and AV), halved for causal masking.
    core = 2 * batch * nh * (seq_len ** 2) * (qk_head_dim + d["v_head_dim"]) * 0.5
    return float(proj + core)


def experts_flops_per_call(dims: dict, tokens: int) -> float:
    """Estimate routed-expert FLOPs for top_k SwiGLU experts per token."""
    d = dims
    routed = tokens * d["router_top_k"]
    return float(routed * 6 * d["dim"] * d["moe_hidden_dim"])


def shared_flops_per_call(dims: dict, tokens: int) -> float:
    """Estimate shared-expert FLOPs for one layer forward call."""
    d = dims
    return float(tokens * d["num_shared_experts"] * 6 * d["dim"] * d["moe_hidden_dim"])


def router_flops_per_call(dims: dict, tokens: int) -> float:
    """Estimate FLOPs for the router-gate projection."""
    d = dims
    return float(2 * tokens * d["dim"] * d["num_experts"])


def dense_mlp_flops_per_call(dims: dict, tokens: int) -> float:
    """Estimate FLOPs for one dense SwiGLU FFN layer."""
    d = dims
    return float(tokens * 6 * d["dim"] * d["dense_hidden_dim"])


# Map each region to its per-forward FLOP estimator.
def per_call_flops(region: str, dims: dict, *, tokens: int, batch: int, seq_len: int) -> float | None:
    if region == "attn":
        return attn_flops_per_call(dims, tokens, batch, seq_len)
    if region == "moe.experts":
        return experts_flops_per_call(dims, tokens)
    if region == "moe.shared":
        return shared_flops_per_call(dims, tokens)
    if region == "moe.router":
        return router_flops_per_call(dims, tokens)
    return None  # Memory-bound regions such as norms have no GEMM FLOPs.


# Roofline metrics.

def achieved_flops(per_call: float, calls: int, time_ms: float) -> float:
    """Return achieved FLOP/s = per_call * calls / elapsed_time.

    calls includes activation recomputation and microbatches so work and
    elapsed time use the same measurement scope.
    """
    if time_ms <= 0:
        return 0.0
    return (per_call * calls) / (time_ms / 1000.0)


def mfu(per_call: float, calls: int, time_ms: float, roof: HardwareRoof) -> float:
    """Return achieved FLOP/s divided by the hardware peak."""
    if roof.peak_flops <= 0:
        return 0.0
    return achieved_flops(per_call, calls, time_ms) / roof.peak_flops


def bandwidth_util(bytes_moved: float, time_ms: float, roof: HardwareRoof) -> float:
    """Return achieved bytes/s divided by the NVLink peak."""
    if time_ms <= 0 or roof.nvlink_bw <= 0:
        return 0.0
    return (bytes_moved / (time_ms / 1000.0)) / roof.nvlink_bw


def headroom_ms(time_ms: float, util: float) -> float:
    """Return optimization headroom: time_ms * (1 - utilization)."""
    return time_ms * (1.0 - max(0.0, min(1.0, util)))
