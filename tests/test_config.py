"""Unit tests for config."""

import warnings

import pytest
from pydantic import ValidationError

from tests.conftest import make_test_model_config, make_test_train_config


class TestModelConfig:
    def test_basic_creation(self):
        """Verify basic creation."""
        m = make_test_model_config(name="qwen3/8B")
        assert m.name == "qwen3/8B"
        assert m.dp_shard == -1
        assert m.tp == 1
        assert m.dtype == "bf16"

    def test_parse_name(self):
        """Verify parse name."""
        m = make_test_model_config(name="llama3/70B")
        family, flavor = m.parse_name()
        assert family == "llama3"
        assert flavor == "70B"

    def test_family_and_flavor_properties(self):
        """Verify family and flavor properties."""
        m = make_test_model_config(name="qwen3/30B-A3B", attn_backend="flex")
        assert m.family == "qwen3"
        assert m.flavor == "30B-A3B"

    def test_invalid_name_no_slash(self):
        """Verify invalid name no slash."""
        with pytest.raises(ValidationError, match="格式"):
            make_test_model_config(name="qwen3-8B")

    def test_invalid_name_empty_family(self):
        """Verify invalid name empty family."""
        with pytest.raises(ValidationError):
            make_test_model_config(name="/8B")

    def test_parallel_config(self):
        """Verify parallel config."""
        m = make_test_model_config(name="qwen3/8B", tp=4, pp=2, dp_shard=8)
        assert m.tp == 4
        assert m.pp == 2
        assert m.dp_shard == 8

    def test_torchtitan_dtype_conversion(self):
        """Verify torchtitan dtype conversion."""
        assert make_test_model_config(name="qwen3/8B", dtype="bf16").torchtitan_dtype() == "bfloat16"
        assert make_test_model_config(name="qwen3/8B", dtype="fp16").torchtitan_dtype() == "float16"
        assert make_test_model_config(name="qwen3/8B", dtype="fp32").torchtitan_dtype() == "float32"

    def test_mxfp8_dtype(self):
        """Verify mxfp8 dtype."""
        m = make_test_model_config(name="qwen3/8B", dtype="mxfp8")
        assert m.use_mxfp8 is True
        assert m.torchtitan_dtype() == "bfloat16"

    def test_bf16_not_mxfp8(self):
        """Verify bf16 not mxfp8."""
        m = make_test_model_config(name="qwen3/8B", dtype="bf16")
        assert m.use_mxfp8 is False

    def test_attn_backend_explicit(self):
        """Verify attn backend explicit."""
        m = make_test_model_config(name="qwen3/8B", attn_backend="flex", attn_mask_type="causal")
        assert m.attn_backend == "flex"
        assert m.attn_mask_type == "causal"

    def test_attn_backend_rejects_sdpa(self):
        """Verify attn backend rejects sdpa."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_test_model_config(name="qwen3/8B", attn_backend="sdpa")

    def test_attn_backend_flex_keeps_causal(self):
        """Verify attn backend flex keeps causal."""
        m = make_test_model_config(name="qwen3/8B", attn_backend="flex", attn_mask_type="causal")
        assert m.attn_mask_type == "causal"

    def test_moe_detection(self):
        """Verify moe detection."""
        assert make_test_model_config(name="qwen3/30B-A3B", attn_backend="flex").is_moe_model() is True
        assert make_test_model_config(name="DeepSeek-V3/8B", attn_backend="flex").is_moe_model() is True
        assert make_test_model_config(name="mixtral/8x7B", attn_backend="flex").is_moe_model() is True

        assert make_test_model_config(name="qwen3/8B").is_moe_model() is False
        assert make_test_model_config(name="llama3/70B").is_moe_model() is False

    def test_needs_packing_property(self):
        """Verify needs packing property."""
        m1 = make_test_model_config(name="qwen3/8B", attn_backend="flex", attn_mask_type="block_causal")
        assert m1.needs_packing is True

        m2 = make_test_model_config(name="qwen3/8B", attn_backend="flex", attn_mask_type="causal")
        assert m2.needs_packing is False



class TestNoDefaultRequiredFields:
    """Tests for test no default required fields."""

    @pytest.mark.parametrize(
        "missing",
        ["dtype", "param_dtype", "attn_backend", "attn_mask_type", "ac_mode", "compile"],
    )
    def test_model_missing_a_field_raises(self, missing):
        """Verify model missing a field raises."""
        from titaninfra.config.model import ModelConfig

        kwargs = {
            "name": "qwen3/8B",
            "dtype": "bf16",
            "param_dtype": "fp32",
            "attn_backend": "flex",
            "attn_mask_type": "causal",
            "ac_mode": "none",
            "compile": False,
        }
        kwargs.pop(missing)
        with pytest.raises(ValidationError, match=missing):
            ModelConfig(**kwargs)

    def test_model_bare_name_raises(self):
        """Verify model bare name raises."""
        from titaninfra.config.model import ModelConfig

        with pytest.raises(ValidationError):
            ModelConfig(name="qwen3/8B")

    @pytest.mark.parametrize(
        "missing",
        [
            "local_batch_size", "global_batch_size", "seq_len", "optimizer_name",
            "lr", "beta1", "beta2", "eps", "weight_decay", "optimizer_state_dtype",
            "min_lr_ratio", "lr_decay_type", "max_grad_norm",
        ],
    )
    def test_train_missing_a_field_raises(self, missing):
        """Verify train missing a field raises."""
        from titaninfra.config.train import TrainConfig

        kwargs = {
            "local_batch_size": 1,
            "global_batch_size": -1,
            "seq_len": 4096,
            "optimizer_name": "AdamW",
            "lr": 1e-4,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
            "weight_decay": 0.1,
            "optimizer_state_dtype": "fp32",
            "min_lr_ratio": 0.1,
            "lr_decay_type": "cosine",
            "max_grad_norm": 1.0,
            "warmup_steps": 200,
        }
        kwargs.pop(missing)
        with pytest.raises(ValidationError, match=missing):
            TrainConfig(**kwargs)

    def test_train_bare_raises(self):
        """Verify train bare raises."""
        from titaninfra.config.train import TrainConfig

        with pytest.raises(ValidationError):
            TrainConfig()

    def test_warmup_must_be_explicit(self):
        """Verify warmup must be explicit."""
        from titaninfra.config.train import TrainConfig

        base = {
            "local_batch_size": 1, "global_batch_size": -1, "seq_len": 4096,
            "optimizer_name": "AdamW", "lr": 1e-4, "beta1": 0.9, "beta2": 0.95,
            "eps": 1e-8, "weight_decay": 0.1, "optimizer_state_dtype": "fp32",
            "min_lr_ratio": 0.1, "lr_decay_type": "cosine", "max_grad_norm": 1.0,
        }
        with pytest.raises(ValidationError, match="warmup"):
            TrainConfig(**base)
        assert TrainConfig(**base, warmup_steps=100).warmup_steps == 100
        assert TrainConfig(**base, warmup_ratio=0.05).warmup_ratio == 0.05

    def test_muon_requires_all_muon_fields(self):
        """Verify muon requires all muon fields."""
        from titaninfra.config.train import TrainConfig

        base = {
            "local_batch_size": 1, "global_batch_size": -1, "seq_len": 4096,
            "optimizer_name": "MuonAdamW", "lr": 1e-4, "beta1": 0.9, "beta2": 0.95,
            "eps": 1e-8, "weight_decay": 0.1, "optimizer_state_dtype": "fp32",
            "min_lr_ratio": 0.1, "lr_decay_type": "cosine", "max_grad_norm": 1.0,
            "warmup_steps": 200,
        }
        with pytest.raises(ValidationError, match="Muon"):
            TrainConfig(**base)

    def test_muon_not_required_for_adamw(self):
        """Verify muon not required for adamw."""
        c = make_test_train_config(optimizer_name="AdamW", warmup_steps=100)
        assert c.optimizer_name == "AdamW"

    def test_factory_muon_fills_fields(self):
        """Verify factory muon fills fields."""
        c = make_test_train_config(optimizer_name="MuonAdamW", warmup_steps=100)
        assert c.optimizer_name == "MuonAdamW"
        assert c.muon_lr == 0.02




class TestTrainConfig:
    def test_factory_defaults(self):
        """Verify factory defaults."""
        c = make_test_train_config()
        assert c.local_batch_size == 1
        assert c.seq_len == 4096
        assert c.lr == 1e-4
        assert c.steps == 100
        assert c.warmup_steps == 200

    def test_custom_values(self):
        """Verify custom values."""
        c = make_test_train_config(local_batch_size=8, lr=1e-5, seq_len=8192, steps=5000)
        assert c.local_batch_size == 8
        assert c.lr == 1e-5
        assert c.seq_len == 8192

    def test_invalid_lr(self):
        """Verify invalid lr."""
        with pytest.raises(ValidationError):
            make_test_train_config(lr=-1e-4)

    def test_comm_timeout_defaults(self):
        """Verify comm timeout defaults."""
        c = make_test_train_config()
        assert c.comm_init_timeout_seconds == 3600
        assert c.comm_train_timeout_seconds == 3600

    def test_comm_timeout_custom(self):
        """Verify comm timeout custom."""
        c = make_test_train_config(comm_init_timeout_seconds=7200, comm_train_timeout_seconds=1800)
        assert c.comm_init_timeout_seconds == 7200
        assert c.comm_train_timeout_seconds == 1800

    def test_comm_timeout_must_be_positive(self):
        """Verify comm timeout must be positive."""
        with pytest.raises(ValidationError):
            make_test_train_config(comm_train_timeout_seconds=0)

    def test_checkpoint_fields(self):
        """Verify checkpoint fields."""
        c = make_test_train_config(ckpt_subdir="my_ckpts", ckpt_interval=200, resume_from="/path/to/ckpt")
        assert c.ckpt_subdir == "my_ckpts"
        assert c.ckpt_interval == 200
        assert c.resume_from == "/path/to/ckpt"

    def test_train_on_inputs_field_removed(self):
        """Verify train on inputs field removed."""
        from titaninfra.config import TrainConfig
        assert "train_on_inputs" not in TrainConfig.model_fields
        c = make_test_train_config(train_on_inputs=True)  # extra="ignore"
        assert not hasattr(c, "train_on_inputs")

    def test_chat_template_field_removed(self):
        """Verify chat template field removed."""
        from titaninfra.config import TrainConfig
        assert "chat_template" not in TrainConfig.model_fields
        c = make_test_train_config(chat_template="qwen3")
        assert not hasattr(c, "chat_template")

    def test_new_dataloader_fields_defaults(self):
        """Verify new dataloader fields defaults."""
        c = make_test_train_config()
        assert c.skip_corrupted_samples is True
        assert c.max_consecutive_errors == 10

    def test_new_dataloader_fields_custom(self):
        """Verify new dataloader fields custom."""
        c = make_test_train_config(skip_corrupted_samples=False, max_consecutive_errors=5)
        assert c.skip_corrupted_samples is False
        assert c.max_consecutive_errors == 5

    def test_stream_cache_limit_field(self):
        """Verify stream cache limit field."""
        from titaninfra.config import StreamConfig, TrainConfig
        assert "cache_limit" not in TrainConfig.model_fields
        assert StreamConfig(local="/x", tokenizer_name="qwen3").cache_limit is None
        assert StreamConfig(local="/x", tokenizer_name="qwen3", cache_limit="10gb").cache_limit == "10gb"




class TestStreamConfig:
    def test_text_stream(self):
        """Verify text stream."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(tokenizer_name="qwen3", local="/tmp/data", data_type="text", proportion=0.5)
        assert s.data_type == "text"
        assert s.proportion == 0.5

    def test_messages_stream(self):
        """Verify messages stream."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/sft",
            data_type="messages",
            family="qwen3",
            add_bos_eos=False,
            proportion=0.3,
        )
        assert s.data_type == "messages"
        assert s.family == "qwen3"

    def test_valid_proportion_config(self):
        """Verify valid proportion config."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/data",
            proportion=0.1,
        )
        assert s.proportion == 0.1
        assert s.repeat is None
        assert s.choose is None

    def test_valid_repeat_config(self):
        """Verify valid repeat config."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/data",
            repeat=2.0,
        )
        assert s.repeat == 2.0

    def test_valid_choose_config(self):
        """Verify valid choose config."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/data",
            choose=1000,
        )
        assert s.choose == 1000

    def test_multiple_mixing_specs_raises(self):
        """Verify multiple mixing specs raises."""
        from titaninfra.config import StreamConfig

        with pytest.raises(ValidationError, match="互斥"):
            StreamConfig(tokenizer_name="qwen3", local="/tmp/data", proportion=0.5, repeat=2.0)

    def test_no_mixing_spec_ok(self):
        """Verify no mixing spec ok."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(tokenizer_name="qwen3", local="/tmp/data")
        assert s.proportion is None
        assert s.repeat is None
        assert s.choose is None

    def test_resolve_stream_id(self):
        """Verify resolve stream id."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(tokenizer_name="qwen3", local="/data/math/mds/train")
        assert s.resolve_stream_id() == "train"

    def test_custom_stream_id(self):
        """Verify custom stream id."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(tokenizer_name="qwen3", local="/tmp/data", stream_id="my-stream")
        assert s.resolve_stream_id() == "my-stream"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def test_add_bos_eos_defaults_true(self):
        """Verify add bos eos defaults true."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(tokenizer_name="qwen3", local="/tmp/data")
        assert s.add_bos_eos is True

    def test_add_bos_eos_true_on_text_stream_ok(self):
        """Verify add bos eos true on text stream ok."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/data",
            data_type="text",
            add_bos_eos=True,
        )
        assert s.add_bos_eos is True
        assert s.data_type == "text"

    def test_add_bos_eos_false_on_messages_stream_ok(self):
        """Verify add bos eos false on messages stream ok."""
        from titaninfra.config import StreamConfig

        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/sft",
            data_type="messages",
            family="qwen3",
            add_bos_eos=False,
        )
        assert s.add_bos_eos is False

    def test_add_bos_eos_true_on_messages_stream_raises(self):
        """Verify add bos eos true on messages stream raises."""
        from titaninfra.config import StreamConfig

        with pytest.raises(ValidationError, match="add_bos_eos=True 仅对 data_type='text' 有效"):
            StreamConfig(
                tokenizer_name="qwen3",
                local="/tmp/sft",
                data_type="messages",
                family="qwen3",
                add_bos_eos=True,
            )

    def test_add_bos_eos_compatible_with_all_pack_strategies(self):
        """Verify add bos eos compatible with all pack strategies."""
        from titaninfra.config import PackStrategy, StreamConfig

        for strategy in (
            PackStrategy.HARD_TRUNCATE,
            PackStrategy.FULL_SAMPLE,
            PackStrategy.NO_PACK,
        ):
            s = StreamConfig(
                tokenizer_name="qwen3",
                local="/tmp/data",
                data_type="text",
                pack_strategy=strategy,
                add_bos_eos=True,
            )
            assert s.add_bos_eos is True
            assert s.pack_strategy == strategy

    def test_add_bos_eos_with_sentence_boundary_ok(self):
        """Verify add bos eos with sentence boundary ok."""
        from titaninfra.config import PackStrategy, StreamConfig

        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/data",
            data_type="text",
            pack_strategy=PackStrategy.SENTENCE_BOUNDARY,
            sentence_end_tokens=["。", "\n"],
            add_bos_eos=True,
        )
        assert s.add_bos_eos is True
        assert s.pack_strategy == PackStrategy.SENTENCE_BOUNDARY

    def test_add_bos_eos_non_bool_raises(self):
        """Verify add bos eos non bool raises."""
        from titaninfra.config import StreamConfig

        with pytest.raises(ValidationError):
            StreamConfig(
                tokenizer_name="qwen3",
                local="/tmp/data",
                data_type="text",
                add_bos_eos=[True],
            )




class TestTrainConfigWarnings:
    def test_global_batch_size_with_local_emits_user_warning(self):
        """Verify global batch size with local emits user warning."""

        with pytest.warns(UserWarning, match="同时指定了 global_batch_size"):
            make_test_train_config(global_batch_size=128, local_batch_size=4)

    def test_global_batch_size_negative_no_warning(self):
        """Verify global batch size negative no warning."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_test_train_config(global_batch_size=-1, local_batch_size=4)

            batch_warnings = [
                x for x in w
                if "global_batch_size" in str(x.message)
            ]
            assert len(batch_warnings) == 0




class TestTrainConfigExtraIgnore:
    def test_extra_fields_are_ignored(self):
        """Verify extra fields are ignored."""

        c = make_test_train_config(
            lr=1e-4,
            nonexistent_field="should_be_ignored",
            another_unknown=42,
        )
        assert c.lr == 1e-4
        assert not hasattr(c, "nonexistent_field")
        assert not hasattr(c, "another_unknown")




class TestModelConfigMoEDetection:
    def test_is_moe_a3b_pattern(self):
        """Verify is moe a3b pattern."""

        assert make_test_model_config(name="qwen3/30B-A3B").is_moe_model() is True

    def test_is_moe_deepseek_v3(self):
        """Verify is moe deepseek v3."""

        assert make_test_model_config(name="DeepSeek-V3/236B").is_moe_model() is True

    def test_is_moe_mixtral(self):
        """Verify is moe mixtral."""

        assert make_test_model_config(name="mixtral/8x7B").is_moe_model() is True

    def test_is_not_moe_dense_model(self):
        """Verify is not moe dense model."""

        assert make_test_model_config(name="llama3/70B").is_moe_model() is False
        assert make_test_model_config(name="qwen3/8B").is_moe_model() is False
        assert make_test_model_config(name="qwen3/0.6B").is_moe_model() is False

    def test_is_moe_ixtral_pattern(self):
        """Verify is moe ixtral pattern."""

        assert make_test_model_config(name="mistral/ixtral-8x7B").is_moe_model() is True

    def test_is_moe_titaninfra_deepseek_v3_lowercase(self):
        """Verify is moe titaninfra deepseek v3 lowercase."""

        assert make_test_model_config(name="deepseek_v3/16B_titaninfra").is_moe_model() is True
        assert make_test_model_config(name="deepseek_v3/20B_titaninfra").is_moe_model() is True
        assert make_test_model_config(name="deepseek_v3/49B_titaninfra").is_moe_model() is True
        assert make_test_model_config(name="deepseek_v3/236B_titaninfra").is_moe_model() is True
        assert make_test_model_config(name="deepseek_v3/671B_titaninfra").is_moe_model() is True

    def test_is_moe_titaninfra_deepseek_v4(self):
        """Verify is moe titaninfra deepseek v4."""

        assert make_test_model_config(name="deepseek_v4/16B").is_moe_model() is True

    def test_is_moe_titaninfra_gpt_oss(self):
        """Verify is moe titaninfra gpt oss."""

        assert make_test_model_config(name="gpt_oss/20B").is_moe_model() is True

    def test_is_moe_qwen3_a22b(self):
        """Verify is moe qwen3 a22b."""

        assert make_test_model_config(name="qwen3/235B-A22B").is_moe_model() is True


class TestModelConfigNeedsPacking:
    def test_needs_packing_flex_block_causal(self):
        """Verify needs packing flex block causal."""

        m = make_test_model_config(name="qwen3/8B", attn_backend="flex", attn_mask_type="block_causal")
        assert m.needs_packing is True

    def test_no_packing_sdpa_causal(self):
        """Verify no packing sdpa causal."""

        m = make_test_model_config(name="qwen3/8B")
        assert m.needs_packing is False

    def test_no_packing_flex_causal(self):
        """Verify no packing flex causal."""

        m = make_test_model_config(name="qwen3/8B", attn_backend="flex", attn_mask_type="causal")
        assert m.needs_packing is False




class TestModelConfigEPWarning:
    def test_ep_on_dense_model_warns(self):
        """Verify ep on dense model warns."""

        with pytest.warns(UserWarning, match="不是 MoE 结构"):
            make_test_model_config(name="llama3/70B", ep=8)

    def test_ep_on_moe_model_no_ep_warning(self):
        """Verify ep on moe model no ep warning."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_test_model_config(name="qwen3/30B-A3B", ep=8, attn_backend="flex")

            ep_warnings = [
                x for x in w
                if "不是 MoE 结构" in str(x.message)
            ]
            assert len(ep_warnings) == 0




class TestTrainConfigNewFields:
    def test_keep_latest_k_default(self):
        """Verify keep latest k default."""

        c = make_test_train_config()
        assert c.keep_latest_k == 3

    def test_keep_latest_k_zero_keeps_all(self):
        """Verify keep latest k zero keeps all."""

        c = make_test_train_config(keep_latest_k=0)
        assert c.keep_latest_k == 0

    def test_wandb_fields_defaults(self):
        """Verify wandb fields defaults."""

        c = make_test_train_config()
        assert c.enable_wandb is False
        assert c.wandb_project is None
        assert c.wandb_entity is None
        assert c.wandb_mode == "offline"


# ============================================================
# ============================================================


class TestTrainConfigGuards:
    """Tests for test train config guards."""

    def test_keep_latest_k_one_raises(self):
        """Verify keep latest k one raises."""

        with pytest.raises(ValidationError, match="keep_latest_k=1 无效"):
            make_test_train_config(keep_latest_k=1)

    def test_keep_latest_k_two_ok(self):
        """Verify keep latest k two ok."""

        c = make_test_train_config(keep_latest_k=2)
        assert c.keep_latest_k == 2

    def test_keep_latest_k_zero_ok(self):
        """Verify keep latest k zero ok."""

        c = make_test_train_config(keep_latest_k=0)
        assert c.keep_latest_k == 0

    def test_batch_size_warning_explains_grad_accum(self):
        """Verify batch size warning explains grad accum."""

        with pytest.warns(UserWarning, match="gradient_accumulation_steps"):
            make_test_train_config(global_batch_size=64, local_batch_size=2)


class TestWarmupRatio:
    """Tests for test warmup ratio."""

    def test_warmup_ratio_default_none(self):
        """Verify warmup ratio default none."""
        config = make_test_train_config()
        assert config.warmup_ratio is None

    def test_warmup_ratio_accepted(self):
        """Verify warmup ratio accepted."""
        config = make_test_train_config(warmup_ratio=0.1)
        assert config.warmup_ratio == 0.1

    def test_resolve_warmup_steps_with_ratio(self):
        """Verify resolve warmup steps with ratio."""
        from titaninfra.config.train import resolve_warmup_steps
        config = make_test_train_config(warmup_ratio=0.1, steps=10_000)
        result = resolve_warmup_steps(config, total_steps=10_000)
        assert result == 1000

    def test_resolve_warmup_steps_without_ratio(self):
        """Verify resolve warmup steps without ratio."""
        from titaninfra.config.train import resolve_warmup_steps
        config = make_test_train_config(warmup_steps=500, steps=10_000)
        result = resolve_warmup_steps(config, total_steps=10_000)
        assert result == 500

    def test_resolve_warmup_steps_ratio_overrides_steps(self):
        """Verify resolve warmup steps ratio overrides steps."""
        from titaninfra.config.train import resolve_warmup_steps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = make_test_train_config(warmup_ratio=0.05, warmup_steps=999, steps=10_000)
        result = resolve_warmup_steps(config, total_steps=10_000)
        assert result == 500  # 0.05 x 10000 = 500, not 999

    def test_resolve_warmup_steps_epoch_mode(self):
        """Verify resolve warmup steps epoch mode."""
        from titaninfra.config.train import resolve_warmup_steps
        config = make_test_train_config(warmup_ratio=0.1, steps=100_000)
        result = resolve_warmup_steps(config, total_steps=6000)
        assert result == 600  # 0.1 x 6000

    def test_warmup_ratio_conflict_warns(self):
        """Verify warmup ratio conflict warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_test_train_config(warmup_ratio=0.1, warmup_steps=500)
            warmup_warnings = [
                x for x in w if "warmup_ratio" in str(x.message) and "warmup_steps" in str(x.message)
            ]
            assert len(warmup_warnings) >= 1


# ============================================================================
# Scheduling combination validation
# ============================================================================


class TestSchedulingCombinationValidation:
    """Tests for test scheduling combination validation."""

    @staticmethod
    def _stream(**kw):
        from titaninfra.config.data import StreamConfig
        defaults = {"local": "/tmp/fake_mds"}
        defaults.update(kw)
        return StreamConfig(tokenizer_name="qwen3", **defaults)

    @staticmethod
    def _validate(streams, **train_kw):
        from titaninfra.config.data import validate_scheduling_combination
        validate_scheduling_combination(streams, make_test_train_config(**train_kw))

    # --- Legal combos ---

    def test_choose_auto_valid(self):
        streams = [self._stream(choose=1000), self._stream(choose=2000)]
        self._validate(streams)

    def test_choose_auto_with_steps_sentinel(self):
        streams = [self._stream(choose=1000)]
        self._validate(streams, steps=10000)

    def test_repeat_auto_valid(self):
        streams = [self._stream(repeat=2.0), self._stream(repeat=1.0)]
        self._validate(streams)

    def test_repeat_auto_with_steps_sentinel(self):
        streams = [self._stream(repeat=3.0)]
        self._validate(streams, steps=10000)

    def test_proportion_steps_valid(self):
        streams = [self._stream(proportion=0.6), self._stream(proportion=0.4)]
        self._validate(streams, steps=50000)

    def test_single_stream_steps_valid(self):
        self._validate([self._stream()], steps=1000)

    def test_no_mode_multi_stream_steps_valid(self):
        streams = [self._stream(), self._stream()]
        self._validate(streams, steps=5000)

    # --- Illegal combos (ValueError) ---

    def test_proportion_auto_no_steps_errors(self):
        streams = [self._stream(proportion=0.5), self._stream(proportion=0.5)]
        with pytest.raises(ValueError, match="proportion.*steps"):
            self._validate(streams)

    def test_default_auto_no_steps_errors(self):
        with pytest.raises(ValueError, match="proportion.*steps"):
            self._validate([self._stream()])


class TestPrecisionConfig:
    """Tests for test precision config."""

    def _model(self, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return make_test_model_config(name="deepseek_v3/debugmodel_titaninfra", **kw)

    def test_defaults_are_fp32_master(self):
        m = self._model()
        t = make_test_train_config(steps=10)
        assert m.param_dtype == "fp32"
        assert t.optimizer_state_dtype == "fp32"

    def test_dtype_and_param_dtype_decoupled(self):
        m = self._model(dtype="bf16", param_dtype="fp32")
        assert m.torchtitan_dtype() == "bfloat16"
        assert m.torchtitan_param_dtype() == "float32"

    def test_param_dtype_bf16_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_test_model_config(name="deepseek_v3/debugmodel_titaninfra", param_dtype="bf16")
        assert any("param_dtype" in str(x.message) and "master" in str(x.message) for x in w)

    def test_mxfp8_does_not_regress(self):
        m = self._model(dtype="mxfp8")
        assert m.use_mxfp8 is True
        assert m.torchtitan_dtype() == "bfloat16"
        assert m.param_dtype == "fp32"

    def test_bf16_state_requires_fp32_master_raises(self):
        from titaninfra import Trainer
        from titaninfra.config import StreamConfig
        m = self._model(param_dtype="bf16")
        t = make_test_train_config(steps=10, optimizer_state_dtype="bf16")
        s = [StreamConfig(local="/tmp/x", tokenizer_name="refmoe",
                          data_type="text", proportion=1.0)]
        with pytest.raises(ValueError, match="param_dtype='fp32'"):
            Trainer(model=m, train=t, streams=s)

    def test_optimizer_state_dtype_bf16_maps_to_bf16_moments(self):
        # optimizer_state_dtype=bf16 -> _build_optimizer_config -> catch-all
        # AdamW.Config(fused=True, moment_dtype="bfloat16"). upstream removed the
        # container-level `implementation` string ("fused_opt_states_bf16"); the
        # per-entry moment_dtype field replaced it (and requires fused=True,
        # upstream _validate_moment_dtype). Needs torchtitan; skip without it.
        pytest.importorskip("torchtitan")
        from titaninfra.engine._pretrain_helpers import _build_optimizer_config

        def catch_all(cfg):
            # param groups are first-match-wins; the ".*" catch-all is always last.
            entry = cfg.optimizers[-1]
            assert entry.pattern == r".*"
            return entry

        # AdamW branch
        cfg = _build_optimizer_config(
            make_test_train_config(steps=10, optimizer_name="AdamW", optimizer_state_dtype="bf16"), []
        )
        assert catch_all(cfg).moment_dtype == "bfloat16"
        assert catch_all(cfg).fused is True
        # fp32 state -> moments follow the parameter dtype; default impl is fused
        cfg_fp32 = _build_optimizer_config(
            make_test_train_config(steps=10, optimizer_name="AdamW", optimizer_state_dtype="fp32"), []
        )
        assert catch_all(cfg_fp32).moment_dtype == "parameter"
        assert catch_all(cfg_fp32).fused is True
        # MuonAdamW branch translates the same way (shared _adamw_shared_kwargs)
        cfg_muon = _build_optimizer_config(
            make_test_train_config(steps=10, optimizer_name="MuonAdamW", optimizer_state_dtype="bf16"), []
        )
        assert catch_all(cfg_muon).moment_dtype == "bfloat16"
        assert catch_all(cfg_muon).fused is True


def test_fsdp_explicit_prefetch_default_false():
    from titaninfra import ModelConfig
    cfg = ModelConfig(name="deepseek_v3/20B_titaninfra", dtype="bf16", param_dtype="fp32",
                      attn_backend="flex", attn_mask_type="block_causal", ac_mode="full", compile=True)
    assert cfg.fsdp_explicit_prefetch is False


def test_fsdp_explicit_prefetch_settable():
    from titaninfra import ModelConfig
    cfg = ModelConfig(name="deepseek_v3/20B_titaninfra", dtype="bf16", param_dtype="fp32",
                      attn_backend="flex", attn_mask_type="block_causal", ac_mode="full", compile=True,
                      fsdp_explicit_prefetch=True)
    assert cfg.fsdp_explicit_prefetch is True


def test_fsdp_gas_no_sync_default_false():
    cfg = make_test_train_config()
    assert cfg.fsdp_gas_no_sync is False


def test_fsdp_gas_no_sync_settable():
    cfg = make_test_train_config(fsdp_gas_no_sync=True)
    assert cfg.fsdp_gas_no_sync is True


# ============================================================
# ============================================================

def test_seed_default_none():
    cfg = make_test_train_config()
    assert cfg.seed is None


def test_seed_settable():
    cfg = make_test_train_config(seed=1234)
    assert cfg.seed == 1234


def test_last_save_in_hf_default_false():
    cfg = make_test_train_config()
    assert cfg.last_save_in_hf is False
    assert cfg.export_dtype == "float32"


def test_last_save_in_hf_requires_model_only():
    with pytest.raises(ValidationError, match="last_save_model_only=True"):
        make_test_train_config(last_save_in_hf=True, last_save_model_only=False)


def test_last_save_in_hf_ok_with_model_only():
    cfg = make_test_train_config(last_save_in_hf=True, last_save_model_only=True,
                                 export_dtype="bfloat16")
    assert cfg.last_save_in_hf is True
    assert cfg.export_dtype == "bfloat16"


def test_export_dtype_invalid_raises():
    with pytest.raises(ValidationError, match="export_dtype"):
        make_test_train_config(export_dtype="int8")


def test_enable_first_step_checkpoint_default_false():
    cfg = make_test_train_config()
    assert cfg.enable_first_step_checkpoint is False


def test_enable_first_step_checkpoint_settable():
    cfg = make_test_train_config(enable_first_step_checkpoint=True)
    assert cfg.enable_first_step_checkpoint is True
