"""DeepSeek V4 family registration + MoE balance spec.

The registration assertions need torchtitan (triton, absent on this Mac) since
importing the family triggers model-module imports; they SKIP locally and run on
the GPU node. The MoE-balance assertion is pure-Python (declared in
``models/__init__.py`` independent of torchtitan) and runs anywhere.
"""

import pytest


def test_moe_balance_declared():
    # MoE balance is declared in models/__init__.py without importing torchtitan.
    from titaninfra.models.moe_balance import get_moe_balance

    spec = get_moe_balance("deepseek_v4")
    assert spec.bias_gamma == 1e-3
    assert spec.aux_coeff == 1e-4
    assert spec.aux_scope == "sequence"


def test_family_registered():
    pytest.importorskip("torchtitan.models.common.attention")
    pytest.importorskip("torchtitan.models.common.moe")
    from titaninfra.models import is_supported

    assert is_supported("deepseek_v4")


def test_spec_loads():
    pytest.importorskip("torchtitan.models.common.attention")
    pytest.importorskip("torchtitan.models.common.moe")
    from titaninfra.models import load_model_spec
    from titaninfra.models.deepseek_v4.model import DeepSeekV4Model

    # load_model_spec returns the bare Config directly (no ModelSpec wrapper
    # -- parallelize/pipeline are inherited BaseModel methods,
    # state_dict_adapter_cls/_register_optimizer_hooks are class attributes/
    # classmethods on DeepSeekV4Model, matching every other migrated family).
    spec = load_model_spec("deepseek_v4", "debugmodel")
    assert isinstance(spec, DeepSeekV4Model.Config)
