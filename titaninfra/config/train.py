"""Optimizer, scheduler, runtime, and checkpoint configuration.

TrainConfig is shared by Pretrain, SFT, and CPT. Optimizer and scheduler fields
must be explicit, and warmup_steps or warmup_ratio must be chosen explicitly."""

import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MUON_REQUIRED_FIELDS = (
    "muon_lr",
    "muon_momentum",
    "muon_nesterov",
    "muon_ns_steps",
    "muon_weight_decay",
    "muon_min_ndim_size",
    "muon_scale_factor",
    "muon_ns_dtype",
    "muon_gather_mode",
    "muon_include_router",
)


class MoEBiasSchedule(BaseModel):
    """Mo ebias schedule configuration and behavior."""

    calibration_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "冷启动校准步数。0 = 不启用。\n"
            "校准期间所有参数 lr=0（不更新），expert_bias 以加速 γ 收敛。\n"
            "推荐：bias 冷启动用 100-200 步（γ×50 时约 20-50 步即可收敛）。"
        ),
    )
    calibration_multiplier: float = Field(
        default=50.0,
        gt=0,
        description=(
            "校准期间 γ (load_balance_coeff) 的加速倍率。\n"
            "γ_calibration = moe_load_balance_coeff × calibration_multiplier。\n"
            "默认 50× → γ=0.05，约 20-50 步收敛。领域迁移推荐 5-10×。"
        ),
    )
    freeze_after: int | None = Field(
        default=None,
        ge=1,
        description=(
            "≥ 此训练步停止更新 expert_bias（γ→0，硬截断）。1-based。\n"
            "recipe 自行 token→step 换算（如 DeepSeek-V3 末 500B token）。\n"
            "None = 全程更新（DCP resume 续训无需设置）。"
        ),
    )

from titaninfra.config.data import StreamConfig  # noqa: E402 — placed below MoEBiasSchedule to avoid circular import


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    """Model architecture and parallel-strategy configuration."""

    local_batch_size: int = Field(..., ge=1, description="每个 GPU 的 micro batch size")
    global_batch_size: int = Field(
        ..., ge=-1,
        description="全局 batch size（-1 表示自动计算，等于 local_batch_size × dp_world_size）",
    )
    seq_len: int = Field(..., ge=1, description="训练序列长度")
    steps: int = Field(default=100, ge=1, description="总训练步数（proportion 模式必填，choose/repeat 自动推导）")

    optimizer_name: Literal["Adam", "AdamW", "MuonAdamW", "DistMuon"] = Field(
        ...,
        description=(
            "优化器类型。\n"
            "- 'Adam' / 'AdamW': 标准 Adam/AdamW\n"
            "- 'MuonAdamW': 混合优化器（Muon for 2D linear params + AdamW for rest）\n"
            "- 'DistMuon': 分布式 Muon（FlexShard 重分布 + Newton-Schulz，"
            "配合 Dist-MoE 使用；按 FQN 声明 compute layout，支持 EP/EDP 分片）"
        ),
    )
    lr: float = Field(..., gt=0, description="学习率（AdamW 部分，MuonAdamW 模式下用于 non-2D params）")
    beta1: float = Field(..., ge=0, le=1, description="Adam beta1（一阶矩衰减率）")
    beta2: float = Field(..., ge=0, le=1, description="Adam beta2（二阶矩衰减率）")
    eps: float = Field(..., gt=0, description="Adam epsilon（数值稳定性）")
    weight_decay: float = Field(..., ge=0, description="权重衰减")
    optimizer_implementation: Literal[
        "fused", "foreach", "for-loop", "fused_opt_states_bf16"
    ] = Field(
        default="fused",
        description=(
            "优化器实现方式（仅 Adam/AdamW 部分）。\n"
            "- 'fused': CUDA 融合实现（最快，推荐）；优化器状态精度跟随参数 dtype\n"
            "- 'foreach': 水平融合实现（较快）\n"
            "- 'for-loop': 默认实现（最慢）\n"
            "- 'fused_opt_states_bf16': fp32 参数 + bf16 优化器状态（省 ~1/3 优化器显存，"
            "DeepSeek-V3 671B 同款）。**前提：参数必须是 fp32**（即 mixed_precision，"
            "ModelConfig.dtype 决定计算精度，参数 master copy 始终 fp32）。"
            "若参数本身是 bf16，此选项退化为纯 bf16、无效果。"
        ),
    )
    optimizer_state_dtype: Literal["fp32", "bf16"] = Field(
        ...,
        description=(
            "Adam/AdamW 优化器状态(exp_avg/exp_avg_sq)存储精度。\n"
            "- fp32（默认，标准）：动量/方差全精度。\n"
            "- bf16：省 ~1/3 优化器显存（DeepSeek-V3 671B 同款）。**要求 "
            "ModelConfig.param_dtype=fp32**（否则 params+states 全 bf16 = 退化纯 bf16、"
            "无效，会被校验拒绝）。内部翻译为 optimizer_implementation='fused_opt_states_bf16'，"
            "仅 Adam/AdamW 支持。"
        ),
    )

    muon_lr: float = Field(
        default=0.02, gt=0,
        description="Muon 学习率（2D linear params），通常比 AdamW lr 高 10-100x",
    )
    muon_momentum: float = Field(
        default=0.95, ge=0, lt=1,
        description="Muon 动量衰减因子（EMA beta）",
    )
    muon_nesterov: bool = Field(
        default=True,
        description="Muon 是否使用 Nesterov 动量",
    )
    muon_ns_steps: int = Field(
        default=5, ge=1, le=20,
        description="Newton-Schulz 迭代次数（5 通常足够收敛）",
    )
    muon_weight_decay: float = Field(
        default=0.0, ge=0,
        description="Muon 解耦权重衰减系数",
    )
    muon_min_ndim_size: int = Field(
        default=4096, ge=1,
        description="2D 参数最小 numel 阈值，低于此值的小矩阵使用 AdamW",
    )
    muon_include_router: bool = Field(
        default=False,
        description=(
            "是否让 MoE router/gate 走 Muon（默认 False = 走 AdamW）。\n"
            "- False（默认）：router gate 走 AdamW（KellerJordan 小模型经验：\n"
            "  正交化会扰乱 softmax routing）。\n"
            "- True：router gate 走 Muon。Moonlight（arXiv 2502.16982）实测大模型\n"
            "  上 router 走 Muon 的 SVD-entropy 更高（更新更多样）→ 对 MoE 有益。\n"
            "  仅在 16B+ MoE 上验证过；小模型慎开。"
        ),
    )
    muon_ns_dtype: Literal["bf16", "fp32"] = Field(
        default="bf16",
        description=(
            "Newton-Schulz 迭代的计算精度。\n"
            "- 'bf16'（默认）：BFloat16，与 PyTorch 官方和 NVIDIA 实现一致\n"
            "- 'fp32'：Float32，更高精度但更慢"
        ),
    )
    muon_scale_factor: float = Field(
        default=1.0, gt=0,
        description=(
            "Muon 更新缩放因子，乘以 sqrt(max(m,n))。\n"
            "- 1.0（默认）：KellerJordan 原始设定，搭配 muon_lr=0.02\n"
            "- 0.2：Moonlight/Kimi K2 设定，可共享 AdamW 学习率"
        ),
    )
    muon_gather_mode: bool = Field(
        default=False,
        description=(
            "是否对 2D 权重（router/attention/dense）做精确全矩阵 Newton-Schulz。\n"
            "- False（默认）：per-shard NS（BlockMuon 近似），零行为变更。\n"
            "- True：FSDP 维 all-gather 还原全矩阵做精确 NS，再切回。\n"
            "  3D expert 由 skip-guard 跳过（EP 下整专家分布已精确）。\n"
            "  额外显存峰值 <0.1%（瞬时、逐参数、不累积）。\n"
            "  见 docs/superpowers/specs/2026-06-16-muon-gather-mode-design.md。"
        ),
    )
    muon_qk_clip_tau: float | None = Field(
        default=None, gt=0,
        description=(
            "QK-Clip 阈值 τ（仅 MuonAdamW 模式）。None = 禁用。\n"
            "每步 optimizer.step 后检查 Q/K 权重的最大注意力 logit，\n"
            "超过 τ 时按 head 缩放 Q/K 权重。参考 Kimi K2 QK-Clip。\n"
            "推荐值：30-100。"
        ),
    )

    enable_sv_entropy_monitor: bool = Field(
        default=False,
        description=(
            "启用奇异值熵监控。\n"
            "低频采样 Muon 管理的 2D 权重矩阵，在 CPU 上计算 SVD 熵，\n"
            "上报 sv_entropy/ namespace 指标到 WandB。\n"
            "熵趋近 1.0 表示权重矩阵接近正交（Muon 目标），趋近 0 表示低秩塌缩。"
        ),
    )
    sv_entropy_interval: int = Field(
        default=100, ge=1,
        description=(
            "奇异值熵采样间隔（每 N 个训练步计算一次）。推荐 50-200。\n"
            "必须能被 log_interval 整除（否则永远不会触发）。"
        ),
    )
    sv_entropy_max_layers: int = Field(
        default=4, ge=1,
        description="每次采样的最大层数（从 Muon 参数中随机选取），控制 CPU 开销",
    )

    layer_lr_overrides: dict[str, float] | None = Field(
        default=None,
        description=(
            "为不同层的参数设置独立学习率倍数（相对于 base_lr 的比例）。\n"
            "\n"
            "格式：{\"参数名模式\": 倍数}\n"
            "- 参数名模式：支持部分匹配（如 \"router\" 匹配所有包含 router 的参数）\n"
            "- 倍数：该层实际 lr = base_lr × 倍数，跟随 LR scheduler 正确缩放\n"
            "\n"
            "示例（将 router 降为 1/10）：\n"
            "  {\"router\": 0.1, \"embed\": 0.01}\n"
        ),
    )
    freeze_params: list[str] | None = Field(
        default=None,
        description=(
            "冻结指定参数（lr_multiplier=0，参数不更新）。\n"
            "\n"
            "格式：参数名模式列表，支持部分匹配。\n"
            "示例：\n"
            "  [\"router\"]          — 冻结所有 router 参数（SFT 推荐，防止路由分布漂移）\n"
            "  [\"router\", \"embed\"] — 同时冻结 router 和 embedding\n"
        ),
    )

    warmup_steps: int = Field(default=200, ge=0, description="学习率 warmup 步数（与 warmup_ratio 二选一）")
    warmup_ratio: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "Warmup 占总训练步数的比例（与 warmup_steps 二选一）。\n"
            "设值后自动计算 warmup_steps = int(warmup_ratio × steps)，覆盖 warmup_steps。\n"
            "推荐 0.05~0.1"
        ),
    )
    min_lr_ratio: float = Field(..., ge=0, le=1, description="最小学习率比例（相对于初始 lr）")
    lr_decay_type: Literal["cosine", "linear", "sqrt", "constant"] = Field(
        ...,
        description=(
            "LR 衰减类型。\n"
            "- 'cosine': warmup 后余弦衰减到 min_lr_ratio × lr（默认）\n"
            "- 'linear': warmup 后线性衰减\n"
            "- 'sqrt': warmup 后平方根衰减\n"
            "- 'constant': warmup 后保持 lr 不变（LIMO paper 使用此模式）"
        ),
    )
    decay_ratio: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "LR 衰减阶段占总训练步数的比例（Warmup-Stable-Decay 调度）。\n"
            "None（默认）：warmup 后立即进入 cosine decay（标准 cosine 调度）。\n"
            "设值后（如 0.2）：warmup → 稳定期（lr 不变）→ 最后 20% 步数衰减。\n"
            "推荐 CPT 场景使用 0.1~0.3，参考 https://arxiv.org/abs/2404.06395"
        ),
    )
    max_grad_norm: float = Field(..., ge=0, description="梯度裁剪阈值（0 表示不裁剪）")

    chunked_loss: bool = Field(
        default=False,
        description=(
            "是否启用分块交叉熵 loss（显存优化，upstream 原生 ChunkedLossWrapper）。\n"
            "启用后模型最后一段以 _skip_lm_head=True 运行，forward 返回隐藏态 "
            "[B,L,D]，loss 逐段应用 lm_head 并计算 CE，避免一次性物化 [B,L,V] "
            "logits（V≈150K）。峰值 logits 显存降到 1/loss_num_chunks。"
        ),
    )
    loss_num_chunks: int = Field(
        default=1,
        ge=1,
        description=(
            "chunked_loss=True 时沿序列维（L）切分的段数。\n"
            "1（默认）退化为不分块。典型 4~8：段数越多峰值显存越低但 kernel "
            "启动开销略增。仅 chunked_loss=True 时生效。"
        ),
    )
    # === NEFTune（Noisy Embedding Fine-Tuning）===
    neftune_noise_alpha: float | None = Field(
        default=None,
        gt=0,
        description=(
            "NEFTune 噪声强度。None（默认）表示不启用。\n"
            "训练时在 embedding 输出上注入均匀噪声 α/√(L×d)，缓解 SFT 过拟合。\n"
            "推荐值 5~15，参考 https://arxiv.org/abs/2310.05914"
        ),
    )

    # === Checkpoint ===
    ckpt_subdir: str = Field(
        default="checkpoints",
        description="Checkpoint 子目录名（相对于 dump_folder，实际路径为 dump_folder/ckpt_subdir/step-N）",
    )
    ckpt_interval: int = Field(default=500, ge=1, description="每隔多少步保存一次 checkpoint")
    resume_from: str | None = Field(
        default=None,
        description=(
            "从指定 checkpoint 路径断点续训。\n"
            "支持两种用法：\n"
            "1. 断点续训（推荐）：不设置，自动从 dump_folder/ckpt_subdir 加载最新\n"
            "2. Checkpoint 迁移：设置 resume_from 指向其他路径的 checkpoint"
        ),
    )
    keep_latest_k: int = Field(
        default=3,
        ge=0,
        description=(
            "保留最近的 k 个 checkpoint，自动清理旧的。\n"
            "0 表示保留所有。不能设置为 1。"
        ),
    )
    exclude_from_loading: list[str] = Field(
        default_factory=list,
        description=(
            "从 checkpoint 加载时排除的状态键列表。\n"
            "可选值：'optimizer', 'lr_scheduler', 'dataloader', 'train_state'。"
        ),
    )

    @field_validator("exclude_from_loading", mode="before")
    @classmethod
    def _parse_exclude_from_loading(cls, v):
        """Parse and normalize the input."""
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    async_checkpoint: str = Field(
        default="disabled",
        description="异步 checkpoint 写入模式：'disabled'（同步）, 'async'（异步）, 'async_with_pinned_mem'",
    )
    last_save_model_only: bool = Field(
        default=False,
        description="训练结束时是否只保存 model weights（不含 optimizer 状态）",
    )
    last_save_in_hf: bool = Field(
        default=False,
        description=(
            "训练结束时是否额外把最终权重导出为 HF safetensors（可直接推理）。\n"
            "torchtitan 约束：要求 last_save_model_only=True 且 dump_folder 为本地路径。\n"
            "开启后无需再单独跑 convert.py。"
        ),
    )
    export_dtype: str = Field(
        default="float32",
        description="last_save_in_hf 导出的权重精度：'float16' / 'bfloat16' / 'float32'。",
    )
    enable_first_step_checkpoint: bool = Field(
        default=False,
        description=(
            "是否在 step 1 就保存一次 checkpoint。\n"
            "用于长跑前验证 checkpoint 路径/存储可写，避免跑几小时才发现存不了盘。"
        ),
    )
    seed: int | None = Field(
        default=None,
        description=(
            "全局训练 RNG 基础种子（torchtitan DebugConfig.seed）。\n"
            "None 表示不固定种子。设定后模型初始化/数据打乱等可复现。\n"
            "注意：与 shuffle_seed（数据流打乱种子）是不同层级的种子。"
        ),
    )

    log_interval: int = Field(default=10, ge=1, description="每隔多少步打印一次日志")
    dump_folder: str = Field(default="./outputs", description="训练产物保存目录")

    task_name: str | None = Field(
        default=None,
        description="任务名称，用作 checkpoint 和输出目录的后缀。None 时使用默认名称",
    )

    comm_init_timeout_seconds: int = Field(
        default=3600,
        gt=0,
        description="进程组初始化超时（秒）。透传到 torchtitan CommConfig.init_timeout_seconds。",
    )
    comm_train_timeout_seconds: int = Field(
        default=3600,
        gt=0,
        description=(
            "训练期集合通信 PG 超时（秒）。透传到 torchtitan CommConfig.train_timeout_seconds，"
            "覆盖 launcher 的 NCCL_TIMEOUT。256-rank 首步含 compile 需 ≥1800；默认 3600。"
        ),
    )

    shuffle: bool = Field(default=False, description="是否打乱数据顺序（MosaicML StreamingDataset shuffle）")
    shuffle_seed: int = Field(default=9176, description="打乱种子")
    num_workers: int = Field(default=1, ge=0, description="DataLoader worker 进程数（0=主进程）")
    persistent_workers: bool = Field(
        default=False,
        description="是否跨 epoch 保持 worker 进程存活（仅在 num_workers > 0 时有效）",
    )
    pin_memory: bool = Field(
        default=True,
        description="将数据复制到 CUDA pinned memory，加速 CPU→GPU 传输。GPU 训练默认启用。",
    )
    prefetch_factor: int | None = Field(
        default=None,
        ge=1,
        description="每个 worker 预取的批次数（仅在 num_workers > 0 时有效）",
    )
    image_decode_workers: int | None = Field(
        default=None,
        ge=1,
        description=(
            "多模态 pretrain 专用：StreamingDataset 内部图 decode 线程池大小。\n"
            "None → min(8, cpu-2)。torchvision decode/resize 释 GIL，8 线程实测 7× 加速。\n"
            "不影响 DataLoader worker（num_workers 铁律仍为 1，避免多流 epoch barrier 死锁）。\n"
            "纯文本训练忽略此字段。"
        ),
    )
    image_decode_prefetch: int | None = Field(
        default=None,
        ge=1,
        description=(
            "多模态 pretrain 专用：图 decode future 队列深度（背压上限）。\n"
            "None → 2×image_decode_workers。纯文本训练忽略。"
        ),
    )
    predownload: int | None = Field(
        default=None,
        ge=1,
        description=(
            "StreamingDataset 提前下载的 sample 数。None 时默认 8×batch_size。\n"
            "本地数据集建议设为 total_samples（全量预加载）。"
        ),
    )
    shuffle_block_size: int | None = Field(
        default=None,
        ge=1,
        description="StreamingDataset shuffle block 大小。小数据集建议设小值（如 256）。",
    )
    shuffle_algo: str | None = Field(
        default="py1br",
        description="MosaicML shuffle 算法。默认 'py1br'：resume 时只重建当前 block，"
        "epoch 重建快（num_canonical_nodes 大时 MosaicML 默认 py1s/py2s 会物化整个 epoch "
        "样本数组 → resume fast-forward 卡几十分钟）。设 None 回退 MosaicML 默认。",
    )
    fsdp_gas_no_sync: bool = Field(
        default=False,
        description=(
            "GAS 梯度累积期间跳过 HSDP 跨机 all-reduce（仅最后一个 microbatch 同步）。\n"
            "用 FSDP2 set_requires_all_reduce：保留机内 reduce-scatter（梯度 sharded，显存零增长），\n"
            "只省跨机 all-reduce（IB，GAS 里最贵的通信）。省 (GAS-1)/GAS 跨机梯度通信。\n"
            "默认 False（每 microbatch 都 all-reduce，现状）。仅非 PP + GAS>1 生效。\n"
            "设计：docs/superpowers/specs/2026-06-30-fsdp2-fwd-bwd-overlap-design.md §3.2"
        ),
    )
    skip_corrupted_samples: bool = Field(
        default=True,
        description="是否跳过损坏样本（True = 跳过并 warn，False = raise）",
    )
    max_consecutive_errors: int = Field(
        default=10,
        ge=1,
        description="连续错误上限，超过后终止训练",
    )

    val_streams: list[StreamConfig] | None = Field(
        default=None,
        description="验证集 MDS 数据流配置。格式同 streams，使用 StreamConfig 对象。",
    )
    eval_freq: int = Field(default=500, ge=1, description="每隔多少步做一次 eval")
    eval_steps: int = Field(
        default=-1,
        description="每次 eval 跑多少步。val_streams 模式必须设正数。",
    )

    # === Tokenizer ===
    tokenizer_path: str | None = Field(
        default=None,
        description="Tokenizer 路径。None 时需配合 hf_ckpt_path 传入 Trainer",
    )

    enable_wandb: bool = Field(default=False, description="是否启用 Weights & Biases 日志记录")
    wandb_project: str | None = Field(default=None, description="WandB 项目名")
    wandb_entity: str | None = Field(default=None, description="WandB 团队名")
    wandb_run_name: str | None = Field(default=None, description="WandB 运行名称")
    wandb_run_id: str | None = Field(default=None, description="WandB 运行 ID（用于恢复运行）")
    wandb_tags: str | None = Field(default=None, description="WandB 标签（逗号分隔）")
    wandb_notes: str | None = Field(default=None, description="WandB 运行备注")
    wandb_mode: str | None = Field(
        default="offline",
        description="WandB 模式：offline, dryrun, online",
    )

    enable_moe_health_monitor: bool = Field(
        default=False,
        description=(
            "启用 MoE 专家负载健康度监控。\n"
            "开启后通过 forward hook 采集每层 tokens_per_expert 分布，\n"
            "上报 moe/ namespace 指标到 WandB（负载 CV、死 expert 比例等）。\n"
            "开销极低（每层每步 ~0.5μs），但默认关闭以避免非 MoE 模型的无效 hook。"
        ),
    )
    moe_health_ep_reduce_interval: int = Field(
        default=0,
        description=(
            "EP 全局统计频率（每 N 个训练步 reduce 一次到 rank-0 计算全局指标）。\n"
            "必须能被 log_interval 整除（否则永远不会触发）。\n"
            "0 = 禁用全局统计（方案C local-only）。推荐 50。\n"
            "需要 enable_moe_health_monitor=True 且 EP > 1 才生效。"
        ),
    )
    moe_bias_schedule: MoEBiasSchedule | None = Field(
        default=None,
        description=(
            "MoE bias 生命周期调度（校准 + 末期冻结）。\n"
            "Continue pretrain / 领域迁移时校准让 expert_bias 冷启动收敛；\n"
            "训练末期 freeze_after 冻结 bias 贴近推理固定路由。\n"
            "None = 不启用。"
        ),
    )

    enable_profiling: bool = Field(
        default=False,
        description="启用 torch.profiler 采集。生产训练关闭；开发调试打开。",
    )
    profile_freq: int = Field(
        default=100,
        ge=1,
        description="每 N 步采集一次 trace cycle（cycle = wait + warmup + active）。",
    )
    profile_active: int = Field(
        default=2,
        ge=1,
        description="连续采集步数。2 步可看 GAS 内 microbatch pipeline。",
    )
    profile_warmup: int = Field(
        default=3,
        ge=0,
        description="预热步数。排除 torch.compile JIT 编译噪声。",
    )
    profile_skip_first: int = Field(
        default=5,
        ge=0,
        description="跳过前 N 步。排除 NCCL init + DeepEP AllToAll startup。",
    )
    profile_export_dir: str = Field(
        default="profile_traces",
        description="trace 导出子目录（相对于 dump_folder）。",
    )
    profile_memory: bool = Field(
        default=False,
        description="启用 memory snapshot。重度 I/O，仅诊断 OOM 时使用。",
    )
    profile_with_stack: bool = Field(
        default=False,
        description="记录 Python callstack。+20% overhead，仅定位 Python 瓶颈时开。",
    )
    profile_with_flops: bool = Field(
        default=True,
        description="估算 FLOPS（GEMM + Conv）。零 overhead，可直接算 kernel 级 MFU。",
    )

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_fields(cls, values):
        """Handle legacy fields."""
        if isinstance(values, dict):
            # ckpt_dir → ckpt_subdir
            if "ckpt_dir" in values and "ckpt_subdir" not in values:
                values["ckpt_subdir"] = values.pop("ckpt_dir")
        return values

    @model_validator(mode="after")
    def _validate_train_config(self) -> "TrainConfig":
        """Validate the supplied configuration."""
        explicit = self.model_fields_set

        warmup_steps_set = "warmup_steps" in explicit
        warmup_ratio_set = "warmup_ratio" in explicit and self.warmup_ratio is not None
        if not warmup_steps_set and not warmup_ratio_set:
            raise ValueError(
                "必须显式指定 warmup_steps 或 warmup_ratio 之一（算法复现规格无隐藏 default）。\n"
                "  - warmup_steps=N    绝对步数\n"
                "  - warmup_ratio=0.05 占总步数比例（优先于 warmup_steps）"
            )
        if warmup_steps_set and warmup_ratio_set:
            warnings.warn(
                f"同时显式指定了 warmup_ratio={self.warmup_ratio} 和 "
                f"warmup_steps={self.warmup_steps}。\n"
                "warmup_ratio 优先，warmup_steps 将被覆盖。",
                UserWarning,
                stacklevel=2,
            )

        if self.optimizer_name == "MuonAdamW":
            missing = [f for f in _MUON_REQUIRED_FIELDS if f not in explicit]
            if missing:
                raise ValueError(
                    "optimizer_name='MuonAdamW' 要求显式写出全部 Muon 算法字段"
                    "（算法复现规格无隐藏 default）。缺少：\n"
                    f"  {', '.join(missing)}\n"
                    "请在 TrainConfig 中显式赋值，即使采用与默认相同的值。"
                )

        if self.global_batch_size > 0 and self.local_batch_size > 0:
            warnings.warn(
                f"同时指定了 global_batch_size={self.global_batch_size} 和 "
                f"local_batch_size={self.local_batch_size}。\n"
                "TorchTitan 会以 global_batch_size 为准计算 gradient_accumulation_steps。",
                UserWarning,
                stacklevel=2,
            )

        if self.keep_latest_k == 1:
            raise ValueError(
                "keep_latest_k=1 无效：TorchTitan 至少需要保留 2 个 checkpoint。\n"
                "请设置 keep_latest_k >= 2，或设置 0 保留全部。"
            )

        if self.last_save_in_hf:
            if not self.last_save_model_only:
                raise ValueError(
                    "last_save_in_hf=True 要求 last_save_model_only=True"
                    "（HF safetensors 导出只含 model 权重，不含 optimizer 状态）。"
                )
        if self.export_dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(
                f"export_dtype={self.export_dtype!r} 无效，"
                "只支持 'float16' / 'bfloat16' / 'float32'。"
            )

        if self.val_streams is not None and self.eval_steps == -1:
            raise ValueError(
                "val_streams 模式必须设置 eval_steps > 0。\n"
                "建议：eval_steps=10~50"
            )

        if (
            self.enable_sv_entropy_monitor
            and self.sv_entropy_interval % self.log_interval != 0
        ):
            raise ValueError(
                f"sv_entropy_interval={self.sv_entropy_interval} 不能被 "
                f"log_interval={self.log_interval} 整除。\n"
                "collect() 只在 log_interval 步被调用，不整除会导致 SV entropy 永远不触发。"
            )

        if (
            self.moe_health_ep_reduce_interval > 0
            and self.moe_health_ep_reduce_interval % self.log_interval != 0
        ):
            raise ValueError(
                f"moe_health_ep_reduce_interval={self.moe_health_ep_reduce_interval} "
                f"不能被 log_interval={self.log_interval} 整除。\n"
                "collect() 只在 log_interval 步被调用，不整除会导致 EP reduce 永远不触发。"
            )

        return self



def resolve_warmup_steps(
    train_config: "TrainConfig",
    total_steps: int,
) -> int:
    """Resolve warmup steps."""
    if train_config.warmup_ratio is not None:
        resolved = int(train_config.warmup_ratio * total_steps)
        import logging
        logging.getLogger(__name__).info(
            f"warmup_ratio={train_config.warmup_ratio} × total_steps={total_steps} "
            f"→ warmup_steps={resolved}"
        )
        return resolved
    return train_config.warmup_steps
