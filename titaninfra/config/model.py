"""Model architecture, attention, precision, and parallel-strategy configuration.

ModelConfig selects a family/flavor and maps it to TorchTitan model and
parallelism configs. Flex is the only attention backend. Algorithm-defining
fields have no hidden defaults so recipes are reproducible."""

import warnings
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_KNOWN_MOE_PATTERNS = [
    "A3B", "A22B",       # Qwen3-30B-A3B、Qwen3-235B-A22B
    "A4B",               # Gemma4-26B-A4B
    "MoE", "moe",
    "ixtral",            # Mixtral
    "DeepSeek-V3",
    "deepseek_v3",
    "deepseek_v4",       # titaninfra V4 family
    "gpt_oss",
    "gemma4",            # titaninfra Gemma4 MoE
]


class ModelConfig(BaseModel):
    """Model selection and distributed/attention/precision settings.

Required algorithm fields are dtype, param_dtype, attn_backend,
attn_mask_type, ac_mode, and compile. Parallel degrees default to non-enabled
except dp_shard=-1 for automatic FSDP sharding."""

    name: str = Field(..., description="模型名称，格式 'family/flavor'，如 'qwen3/8B'")

    dp_shard: int = Field(default=-1, description="FSDP sharding 度，-1 表示自动（全量 sharding）")
    dp_replicate: int = Field(default=1, description="HSDP/DDP 复制度，通常为 1")

    tp: int = Field(default=1, ge=1, description="Tensor parallelism 度")
    pp: int = Field(default=1, ge=1, description="Pipeline parallelism 度")
    cp: int = Field(default=1, ge=1, description="Context parallelism 度（长序列）")
    ep: int = Field(default=1, ge=1, description="Expert parallelism 度（MoE 模型）")
    ep_backend: Literal["standard", "deepep", "hybridep"] = Field(
        default="standard",
        description="Expert parallelism 通信后端。deepep/hybridep 需要 H100+ GPU 和安装 DeepEP 库",
    )

    # ============================================================
    # ============================================================
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = Field(
        default="default",
        description=(
            "FSDP reshard 策略，控制 forward 后参数行为。\n"
            "- 'default': 智能默认（推荐）\n"
            "- 'always': 总是 reshard，最省显存但通信多\n"
            "- 'never': 从不 reshard，显存占用高但通信少"
        ),
    )
    fsdp_explicit_prefetch: bool = Field(
        default=False,
        description=(
            "启用 FSDP2 显式 fwd/bwd prefetch 链（set_modules_to_forward/backward_prefetch）。\n"
            "默认 False（用 FSDP2 隐式预取）。开启后在 deepseek_v3 的 apply_fsdp 末尾按层链式 prefetch，\n"
            "治 host-bound（CPU launch 跟不上致隐式预取来不及）+ 提升 compute-comm overlap。\n"
            "代价：多持有 1 层完整参数，显存峰值升。仅 ep_degree>1 + pp=1 生效。\n"
            "设计：docs/superpowers/specs/2026-06-30-fsdp2-fwd-bwd-overlap-design.md §3.1"
        ),
    )

    # ============================================================
    # ============================================================
    pipeline_parallel_schedule: str = Field(
        default="1F1B",
        description=(
            "Pipeline 并行调度策略。\n"
            "- '1F1B': 标准 1F1B 调度\n"
            "- 'dualpipev': DualPipeV（支持 EP + PP 重叠，需要 pp>1, ep>1）\n"
            "- 'Interleaved1F1B': 交错 1F1B（需要虚拟流水线）"
        ),
    )
    # ============================================================
    # ============================================================
    context_parallel_load_balancer: Literal["headtail", "ptrr"] | None = Field(
        default="ptrr",
        description=(
            "CP 负载均衡器类型。\n"
            "- 'headtail': 适用于 SDPA\n"
            "- 'ptrr': 适用于 FlexInnerAttention（推荐）\n"
            "- None: 禁用负载均衡"
        ),
    )

    # ============================================================
    # ============================================================
    enable_async_tensor_parallel: bool = Field(
        default=False,
        description=(
            "是否启用异步 TP。\n"
            "注意：必须同时启用 compile=True"
        ),
    )

    # ============================================================
    # ============================================================

    dtype: Literal["bf16", "fp16", "fp32", "mxfp8"] = Field(
        ...,
        description=(
            "计算精度 / 量化模式轴（不是参数存储精度！）。决定 fwd/bwd 计算 dtype 以及"
            "是否启用 MXFP8 量化（use_mxfp8）。mxfp8 的计算基底仍是 bf16。"
            "参数 master 存储精度由 param_dtype 控制；二者解耦——标准混合精度是 "
            "dtype=bf16 + param_dtype=fp32。mxfp8 需要 Hopper+ GPU。"
        ),
    )
    param_dtype: Literal["fp32", "bf16"] = Field(
        ...,
        description=(
            "参数 / master weight 存储精度（optimizer 更新落点）。\n"
            "- fp32（默认，标准混合精度）：master 存 fp32，FSDP2 前向反向 cast 到 dtype"
            " 计算。对齐 Megatron --bf16 / DeepSeek-V3。sub-ULP 的小更新能在 fp32 上累加。\n"
            "- bf16（纯低精度训练）：无 fp32 master copy。warmup 期 lr~1e-6 ≪ bf16 "
            "ULP(~1e-4) → weight+=update 被 round 掉 → 权重学不动、loss plateau。"
            "仅 SFT / 短训可接受，长训预训练务必用 fp32。"
        ),
    )
    ac_mode: Literal["none", "selective", "full", "memory_budget"] = Field(
        ..., description="激活重计算模式"
    )
    ac_memory_budget: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Activation checkpointing 内存预算系数（仅当 ac_mode='memory_budget' 时生效）。\n"
            "- 0.0: 最省显存（接近 full 模式，重计算最多）\n"
            "- 1.0: 最快（接近 selective 模式，显存占用最高）\n"
            "- 0.5: 默认平衡点\n"
            "注意：需要同时启用 compile=True，因为该模式依赖 torch.compile 的分区器"
        ),
    )
    compile: bool = Field(..., description="是否启用 torch.compile")
    cpu_offload: bool = Field(default=False, description="是否将 optimizer states offload 到 CPU")

    kernel_impls: list[str] = Field(
        default_factory=list,
        description=(
            "启用自定义 kernel 的算子名 list（registry 注册后填，如 ['hc_pre','hc_post']）。\n"
            "空 list（默认）= 全部走 pytorch 参考实现，保持灵活性。list 里的算子名才被\n"
            "模块化替换为 Triton / Helion / torch.compile kernel；titaninfra 负责替换控制\n"
            "（见 titaninfra/kernels/）。算子名必须已在 registry 注册（拼错会 fail-loud）；\n"
            "硬件算力不足时自动回退 pytorch。tilelang 已弃用（2026-07）。\n"
            "设计文档：docs/tilelang_kernel_library_research.md"
        ),
    )

    attn_backend: Literal["flex", "varlen"] = Field(
        ...,
        description=(
            "Attention 计算后端。\n"
            "- flex: Flex Attention（支持 packing + block_causal + Context Parallel）\n"
            "- varlen: 变长注意力（Kimi K3 的 KDA 层等场景必需；gpt_oss 可选，"
            "但训练时 sinks 梯度需要 FA4 硬件或 torch nightly 20261125+）"
        ),
    )
    attn_mask_type: Literal["causal", "block_causal"] = Field(
        ...,
        description=(
            "Attention 掩码类型。\n"
            "- causal: 标准 causal mask（每个 token 只能 attend to 之前 tokens）\n"
            "- block_causal: 块级 causal mask（支持 packing，每个文档内部 causal，文档间隔离）"
        ),
    )

    moe_load_balance_coeff: float | None = Field(
        default=None,
        ge=0,
        description=(
            "覆盖家族声明的 expert_bias 更新步长 γ（仅当家族已声明 bias 均衡）。\n"
            "均衡算法由 model spec 的 MoEBalanceSpec 决定，此处仅调数值。\n"
            "家族未声明 bias 时此值被忽略并 warn-loud。\n"
            "- None: 使用家族声明的默认值\n"
            "- 1e-3 ~ 1e-2: 推荐范围\n"
            "注意：仅对已声明 bias 均衡的 MoE 家族有效"
        ),
    )
    moe_top_k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "MoE 每个 token 选择的专家数量。\n"
            "- None: 使用模型默认值（通常为 2 或 8）\n"
            "- 1: 每个 token 只选择 1 个专家（分布最不均匀）\n"
            "- 2~8: 推荐，分布更均衡\n"
            "注意：仅对 MoE 模型有效，dense 模型会忽略此配置"
        ),
    )

    moe_aux_loss_coeff: float | None = Field(
        default=None,
        ge=0,
        description=(
            "覆盖家族声明的 aux-loss 权重 α（仅当家族已声明 aux 均衡）。\n"
            "sequence/global 作用域由 MoEBalanceSpec.aux_scope 决定。\n"
            "家族未声明 aux 时此值被忽略并 warn-loud。\n"
            "- None: 使用家族声明的默认值\n"
            "- 1e-4 ~ 1e-2: 推荐范围\n"
            "可与 expert_bias（moe_load_balance_coeff）耦合（DeepSeek V3/V4 方案）。"
        ),
    )

    lora_rank: int | None = Field(
        default=None,
        gt=0,
        description=(
            "LoRA 秩（启用 PEFT 时必填）。\n"
            "- None: 不启用 LoRA（默认，全量训练）\n"
            "- >0: 启用 LoRA，冻结 base 权重，只训练 adapter\n"
            "GroupedLinear 要求 rank % 8 == 0。\n"
            "启用后必须提供 initial_load_path 或 initial_load_in_hf 作为 base 权重来源。"
        ),
    )
    lora_alpha: float = Field(
        default=16.0,
        description="LoRA 缩放因子，输出乘 alpha/rank。常用值 16~64。",
    )
    lora_target_modules: list[str] | None = Field(
        default=None,
        description=(
            "LoRA 目标模块（FQN 后缀匹配，非 glob）。\n"
            "- None: 所有支持的 projection（默认）\n"
            "- 示例: ['wqkv', 'wo', 'w13', 'w2']（attention + MoE experts）\n"
            "注意: 空 list [] 表示不匹配任何层（通常无意义）。"
        ),
    )
    lora_trainable_modules: list[str] = Field(
        default_factory=list,
        description=(
            "保持全量可训练的模块路径（config 路径匹配）。\n"
            "这些模块下的所有 config 保持全量可训练，不做 Frozen。\n"
            "示例: ['vision_encoder']（VLM 场景视觉塔全量训练 + 语言塔 LoRA）。"
        ),
    )
    lora_expert_adapter_layout: Literal["per_expert", "shared_outer"] = Field(
        default="per_expert",
        description=(
            "MoE routed expert 的 LoRA adapter 布局。\n"
            "- per_expert: 每个 expert 独立的 A/B 因子（默认）\n"
            "- shared_outer: 共享模型维侧的因子（w13 的 A、w2 的 B），EP 上 replicate"
        ),
    )

    @model_validator(mode="after")
    def validate_ep_config(self) -> "ModelConfig":
        """Validate the supplied configuration."""
        if self.ep > 1 and not self.is_moe_model():
            warnings.warn(
                f"⚠️  配置了 ep={self.ep}，但模型 '{self.name}' 不是 MoE 结构。\n"
                f"   Expert Parallelism 仅对 MoE 模型有效，dense 模型会忽略此配置。",
                UserWarning,
                stacklevel=2,
            )

        if self.ep_backend in ("deepep", "hybridep") and self.ep <= 1:
            original_backend = self.ep_backend
            object.__setattr__(self, "ep_backend", "standard")
            warnings.warn(
                f"⚠️  ep_backend='{original_backend}' 需要 ep>1，但当前 ep={self.ep}。\n"
                f"   已自动降级为 ep_backend='standard'。\n"
                f"   如需使用 {original_backend.upper()}，请设置 ep>1。",
                UserWarning,
                stacklevel=2,
            )

        if self.ep > 1 and self.ep_backend == "standard" and self.is_moe_model():
            warnings.warn(
                f"ℹ️  ep={self.ep} + ep_backend='standard'：使用 NCCL AllToAll 进行专家通信。\n"
                f"   DeepEP 后端可提供更优的通信性能，如硬件支持可设置 ep_backend='deepep'。",
                UserWarning,
                stacklevel=2,
            )

        if self.dp_replicate > 1 and self.ep_backend in ("deepep", "hybridep") and self.ep > 1:
            mode = "HSDP" if self.dp_shard > 1 else "DDP"
            warnings.warn(
                f"ℹ️  多副本 + DeepEP 模式已启用：\n"
                f"   - dp_replicate={self.dp_replicate}（副本数）\n"
                f"   - dp_shard={self.dp_shard}（副本内 FSDP 分片）\n"
                f"   - ep={self.ep}（副本内 EP 度）\n"
                f"   - ep_backend='{self.ep_backend}'（EP 通信后端）\n"
                f"   → 使用 {mode} 进行跨副本梯度同步\n"
                f"   → 每个副本内部使用 DeepEP 进行专家通信",
                UserWarning,
                stacklevel=2,
            )

        return self

    @model_validator(mode="after")
    def validate_memory_budget_config(self) -> "ModelConfig":
        """Validate the supplied configuration."""
        if self.ac_mode == "memory_budget" and not self.compile:
            warnings.warn(
                "⚠️  ac_mode='memory_budget' 需要启用 torch.compile！\n"
                "   memory_budget 模式依赖编译器的分区器来决定最优的激活值保存/重计算策略。\n"
                "   已自动将 compile 从 False 修正为 True。\n"
                "   如果不想启用 compile，请使用 ac_mode='selective' 或 'full'。",
                UserWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "compile", True)
        return self

    @model_validator(mode="after")
    def validate_kernel_impls(self) -> "ModelConfig":
        """Validate the supplied configuration."""
        if self.kernel_impls:
            from titaninfra.kernels import validate_enabled

            validate_enabled(self.kernel_impls)
        return self

    @model_validator(mode="after")
    def validate_param_dtype_master_weights(self) -> "ModelConfig":
        """Validate the supplied configuration."""
        if self.param_dtype == "bf16":
            warnings.warn(
                "⚠️  param_dtype='bf16'：纯低精度训练，无 fp32 master weights。\n"
                "   warmup 期 lr~1e-6 ≪ bf16 ULP(~1e-4) → weight+=update 被 round 掉 →\n"
                "   权重学不动、loss plateau（曾导致 refmoe 20B loss 完全不降）。\n"
                "   长训预训练请用 param_dtype='fp32'（标准混合精度，对齐 Megatron/DeepSeek）。\n"
                "   仅当确认是 SFT/短训且接受精度损失时才用 bf16。",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_dual_pipe_v_config(self) -> "ModelConfig":
        """Validate the supplied configuration."""
        is_dual_pipe_v = (
            self.pipeline_parallel_schedule.lower() == "dualpipev"
            and self.pp > 1
            and self.ep > 1
        )
        if is_dual_pipe_v and self.ac_mode != "none":
            raise ValueError(
                f"❌ DualPipeV + Activation Checkpointing 不兼容！\n"
                f"   当前配置：pipeline_parallel_schedule='dualpipev', ac_mode='{self.ac_mode}'\n"
                f"   解决方案：设置 ac_mode='none' 或改用其他 pipeline_parallel_schedule"
            )

        if self.enable_async_tensor_parallel and not self.compile:
            warnings.warn(
                "⚠️  enable_async_tensor_parallel=True 需要 compile=True！\n"
                "   已自动将 compile 从 False 修正为 True。",
                UserWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "compile", True)

        return self

    @model_validator(mode="after")
    def validate_name(self) -> "ModelConfig":
        """Validate the supplied configuration."""
        if "/" not in self.name:
            raise ValueError(
                f"model name 必须包含 '/'，格式为 'family/flavor'，例如 'qwen3/8B'，当前值：{self.name!r}"
            )
        family, flavor = self.name.split("/", 1)
        if not family or not flavor:
            raise ValueError(
                f"model name 格式错误，family 和 flavor 都不能为空，当前值：{self.name!r}"
            )
        return self


    def is_moe_model(self) -> bool:
        """Is moe model."""
        return any(pattern in self.name for pattern in _KNOWN_MOE_PATTERNS)


    @property
    def needs_packing(self) -> bool:
        """Needs packing."""
        return self.attn_mask_type == "block_causal"


    def parse_name(self) -> tuple[str, str]:
        """Parse and normalize the input."""
        family, flavor = self.name.split("/", 1)
        return family, flavor

    @property
    def family(self) -> str:
        """Family."""
        return self.name.split("/", 1)[0]

    @property
    def flavor(self) -> str:
        """Flavor."""
        return self.name.split("/", 1)[1]


    def torchtitan_dtype(self) -> str:
        """Torchtitan dtype."""
        mapping = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
            "mxfp8": "bfloat16",
        }
        return mapping[self.dtype]

    def torchtitan_param_dtype(self) -> str:
        """Torchtitan param dtype."""
        return {"fp32": "float32", "bf16": "bfloat16"}[self.param_dtype]

    def to_torchtitan_activation_checkpoint_config(self):
        """To torchtitan activation checkpoint config."""
        from torchtitan.distributed.activation_remat import (
            DefaultActivationRematConfig,
            FullActivationRematConfig,
        )

        if self.ac_mode == "none":
            return None
        if self.ac_mode == "selective":
            return DefaultActivationRematConfig()
        if self.ac_mode == "full":
            return FullActivationRematConfig()
        if self.ac_mode == "memory_budget":
            raise ValueError(
                "ac_mode='memory_budget' 在当前 torchtitan 架构下没有等价物："
                "旧的 torch.compile 分区器驱动的自动内存预算搜索（MemoryBudgetAC）"
                "已被删除，新架构改成模型声明具名区域 + glob save/recompute 匹配"
                "（ActivationRematConfig），两者不是同一机制，不能机械改名。"
                "请改用 ac_mode='selective'（映射到 upstream 自己的默认折中策略）"
                "或 ac_mode='full'（重算除模型代码硬写区域外的一切）。"
            )
        raise ValueError(f"未知 ac_mode: {self.ac_mode!r}")

    @property
    def use_mxfp8(self) -> bool:
        """Use mxfp8."""
        return self.dtype == "mxfp8"

    def to_torchtitan_parallelism_config(self):
        """To torchtitan parallelism config."""
        from torchtitan.config.parallelism import ParallelismConfig
        from torchtitan.distributed.context_parallel import (
            HeadTailCPLoadBalancer,
            PTRRFlexAttentionCPLoadBalancer,
        )

        _load_balancer_map = {
            "headtail": HeadTailCPLoadBalancer.Config,
            "ptrr": PTRRFlexAttentionCPLoadBalancer.Config,
        }
        if self.context_parallel_load_balancer is None:
            load_balancer = None
        else:
            load_balancer = _load_balancer_map[self.context_parallel_load_balancer]()

        return ParallelismConfig(
            data_parallel_replicate_degree=self.dp_replicate,
            data_parallel_shard_degree=self.dp_shard,
            tensor_parallel_degree=self.tp,
            pipeline_parallel_degree=self.pp,
            context_parallel_degree=self.cp,
            expert_parallel_degree=self.ep,
            fsdp_reshard_after_forward=self.fsdp_reshard_after_forward,
            fsdp_enable_cpu_offloading=self.cpu_offload,
            pipeline_parallel_schedule=self.pipeline_parallel_schedule,
            context_parallel_load_balancer=load_balancer,
        )
