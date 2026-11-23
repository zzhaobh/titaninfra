"""Validate the complete public Pretrain and SFT recipes without training."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODEL_FIELDS = {
    "dtype", "param_dtype", "attn_backend", "attn_mask_type", "ac_mode", "compile",
}
REQUIRED_TRAIN_FIELDS = {
    "local_batch_size", "global_batch_size", "seq_len", "optimizer_name", "lr",
    "beta1", "beta2", "eps", "weight_decay", "optimizer_state_dtype",
    "min_lr_ratio", "lr_decay_type", "max_grad_norm",
}


def _load_example(name: str):
    path = REPO_ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"public_{name}_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["pretrain", "sft"])
def test_example_configs_validate(name: str) -> None:
    """Every required field is explicit and scheduling accepts the configuration."""
    from titaninfra.config.data import validate_scheduling_combination, validate_streams_mixing_mode

    module = _load_example(name)
    model, train, streams = module.build_config()
    assert REQUIRED_MODEL_FIELDS <= model.model_fields_set
    assert REQUIRED_TRAIN_FIELDS <= train.model_fields_set
    assert {"warmup_steps", "warmup_ratio"} & train.model_fields_set
    assert streams
    validate_streams_mixing_mode(streams)
    validate_scheduling_combination(streams, train)


def test_pretrain_recipe_uses_token_proportions() -> None:
    """Pretrain combines two text streams with explicit steps and packing."""
    from titaninfra.config.data import PackStrategy

    _, train, streams = _load_example("pretrain").build_config()
    assert train.steps == 100_000
    assert train.seq_len == 16_384
    assert [stream.data_type for stream in streams] == ["text", "text"]
    assert [stream.proportion for stream in streams] == [0.7, 0.3]
    assert all(stream.pack_strategy == PackStrategy.HARD_TRUNCATE for stream in streams)


def test_sft_recipe_uses_native_messages_and_repeat() -> None:
    """SFT uses the native Qwen3 encoder, full samples, and three epochs."""
    from titaninfra.config.data import PackStrategy

    _, train, streams = _load_example("sft").build_config()
    assert train.seq_len == 8_192
    assert len(streams) == 1
    stream = streams[0]
    assert stream.data_type == "messages"
    assert stream.family == "qwen3"
    assert stream.repeat == 3.0
    assert stream.pack_strategy == PackStrategy.FULL_SAMPLE
    assert "add_bos_eos" not in stream.model_fields_set


def test_api_tour_mixed_recipe_and_model_families() -> None:
    """The opening API tour builds valid mixed streams and model families."""
    from titaninfra.config.data import validate_scheduling_combination, validate_streams_mixing_mode

    module = _load_example("api_tour")
    models = module.build_model_configs()
    assert tuple(models) == module.MODEL_NAMES
    model, train, streams = module.build_config()
    assert model.name == "qwen3/debugmodel"
    assert {stream.data_type for stream in streams} == {"text", "messages"}
    validate_streams_mixing_mode(streams)
    validate_scheduling_combination(streams, train)
    module.main()


def test_message_conversion_apis_roundtrip_tool_and_thinking() -> None:
    """Documented rich-block/HF conversion APIs preserve structure and weights."""
    from titaninfra.data.sft.message_converter import normalize_to_hf_messages, to_ir_messages

    rich = [
        {"role": "user", "blocks": [{"type": "text", "text": "Find TitanInfra."}], "weight": 0},
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "text": "Search first."},
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "TitanInfra"}},
            ],
            "weight": 1,
        },
        {
            "role": "tool",
            "blocks": [{"type": "tool_result", "tool_use_id": "call_1", "text": "Documentation"}],
            "weight": 0,
        },
    ]
    hf_messages, weights = normalize_to_hf_messages(rich, system="Be helpful.")
    assert weights == [0.0, 0, 1, 0]
    assert hf_messages[0] == {"role": "system", "content": "Be helpful."}
    assert hf_messages[2]["tool_calls"][0]["function"]["name"] == "search"
    assert hf_messages[3]["tool_call_id"] == "call_1"

    restored = to_ir_messages(hf_messages[1:])
    block_types = [block["type"] for message in restored for block in message["blocks"]]
    assert block_types == ["text", "thinking", "tool_use", "tool_result"]


def test_plain_thinking_roundtrip_preserves_blocks() -> None:
    """Non-tool assistant thinking+text must not collapse into one text block."""
    from titaninfra.data.sft.message_converter import normalize_to_hf_messages, to_ir_messages

    rich = [{
        "role": "assistant",
        "blocks": [
            {"type": "thinking", "text": "Add the numbers."},
            {"type": "text", "text": "4"},
        ],
        "weight": 1,
    }]
    hf_messages, _ = normalize_to_hf_messages(rich)
    restored = to_ir_messages(hf_messages)
    assert [block["type"] for block in restored[0]["blocks"]] == ["thinking", "text"]
    assert restored[0]["blocks"][0]["text"] == "Add the numbers."
    assert restored[0]["blocks"][1]["text"] == "4"
