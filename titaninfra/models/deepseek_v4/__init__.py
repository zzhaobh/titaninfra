"""DeepSeek V4 model family (titaninfra-maintained).

Native V4 architecture (Hyper-Connections, low-rank compressed attention with
CSA/HCA KV compression + Lightning Indexer + grouped output + attention
sinks, sqrtsoftplus + hash MoE router, MTP head structure), trainable from
scratch in bf16 with dense/sliding-window + compressed attention. HF weight
loading via the state_dict adapter is supported (see state_dict_adapter.py);
real MTP training loss is a later phase (see model.py's module docstring for
the exact scope boundary).
"""

from titaninfra.models.deepseek_v4.config_registry import (
    deepseek_v4_configs,
    model_registry,
)
from titaninfra.models.deepseek_v4.model import (
    DeepSeekV4Attention,
    DeepSeekV4Model,
    DeepSeekV4TransformerBlock,
)

__all__ = [
    "DeepSeekV4Attention",
    "DeepSeekV4Model",
    "DeepSeekV4TransformerBlock",
    "deepseek_v4_configs",
    "model_registry",
]
