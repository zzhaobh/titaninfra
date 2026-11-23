"""Kernel implementations; each module registers its operators on import."""

# Import side-effect registers each op's KernelSpec. deepseek_v4_hc loads the
# pure HC math by file path (no torchtitan), so it registers in any env.
from titaninfra.kernels.ops import deepseek_v4_hc as _deepseek_v4_hc  # noqa: F401
from titaninfra.kernels.ops import deepseek_v4_sparse_gather as _deepseek_v4_sparse_gather  # noqa: F401
from titaninfra.kernels.ops import kimi_moonvit_merge as _kimi_moonvit_merge  # noqa: F401
from titaninfra.kernels.ops import kimi_moonvit_rope as _kimi_moonvit_rope  # noqa: F401
