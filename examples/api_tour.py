"""One-file tour of TitanInfra's public data, training, and utility APIs.

Importing this module has no training or network side effects. Run the nested
operations only after preparing the paths and checkpoints used by your recipe.
"""

import json

from titaninfra import (
    ModelConfig,
    MoEBiasSchedule,
    MoEHealthPlugin,
    PackStrategy,
    StreamConfig,
    TrainConfig,
    Trainer,
    TrainResult,
    compare_losses,
    diagnose_hf_checkpoint,
    download_hf_assets,
    from_hf,
    get_template,
    list_templates,
    register_template,
    to_hf,
)
from titaninfra.data.converter import DataConvertConfig, DataConverter, convert_messages_to_mds
from titaninfra.data.schema import decode_sample
from titaninfra.data.sft.dispatch import tokenize_sample
from titaninfra.data.sft.message_converter import normalize_to_hf_messages, to_ir_messages

MODEL_DEFAULTS = {
    "dtype": "bf16",
    "param_dtype": "fp32",
    "attn_backend": "flex",
    "attn_mask_type": "block_causal",
    "ac_mode": "none",
    "compile": False,
}
MODEL_NAMES = (
    "qwen3/debugmodel",
    "qwen2_5/debugmodel_titaninfra",
    "llama3/debugmodel",
    "deepseek_v3/debugmodel",
    "deepseek_v4/debugmodel",
    "gpt_oss/debugmodel",
    "gemma4/debugmodel",
    "qwen3_5/debugmodel",
)


def build_model_configs() -> dict[str, ModelConfig]:
    """Build several model families with the same parallel/precision policy."""
    return {name: ModelConfig(name=name, **MODEL_DEFAULTS) for name in MODEL_NAMES}


def build_config() -> tuple[ModelConfig, TrainConfig, list[StreamConfig]]:
    """Build a mixed Pretrain + SFT (CPT) recipe with explicit hyperparameters."""
    model = ModelConfig(name="qwen3/debugmodel", **MODEL_DEFAULTS)
    train = TrainConfig(
        local_batch_size=1,
        global_batch_size=8,
        seq_len=4096,
        steps=1000,
        optimizer_name="AdamW",
        lr=3e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        optimizer_state_dtype="fp32",
        min_lr_ratio=0.1,
        lr_decay_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
    )
    streams = [
        StreamConfig(
            local="/data/mds/general/train",
            tokenizer_name="qwen3",
            data_type="text",
            proportion=0.7,
            pack_strategy=PackStrategy.HARD_TRUNCATE,
        ),
        StreamConfig(
            local="/data/mds/sft/train",
            tokenizer_name="qwen3",
            data_type="messages",
            family="qwen3",
            proportion=0.3,
            pack_strategy=PackStrategy.FULL_SAMPLE,
        ),
    ]
    return model, train, streams


def convert_data() -> None:
    """Convert raw text and message records to MDS when the inputs exist."""
    config = DataConvertConfig(
        source="/data/raw/pretrain.jsonl",
        output="/data/mds/general/train",
        tokenizer_path="/models/Qwen3-8B",
        tokenizer_name="qwen3",
        text_key="text",
        num_shards=16,
    )
    DataConverter(config).convert()
    convert_messages_to_mds(
        source="/data/raw/sft.jsonl",
        output="/data/mds/sft/train",
        tokenizer_path="/models/Qwen3-8B",
        tokenizer_name="qwen3",
        family="qwen3",
        num_shards=16,
        overwrite=True,
    )


def encode_messages(tokenizer) -> None:
    """Validate, convert, and natively tokenize a rich message sample."""
    rich = [
        {"role": "user", "blocks": [{"type": "text", "text": "Solve 2+2."}]},
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "text": "Add the numbers."},
                {"type": "text", "text": "4"},
            ],
        },
    ]
    raw = json.dumps({"system": "Be helpful.", "messages": rich})
    decode_sample(raw)
    hf_messages, weights = normalize_to_hf_messages(rich, system="Be helpful.")
    restored = to_ir_messages(hf_messages[1:])
    assert [block["type"] for block in restored[0]["blocks"]] == ["thinking", "text"]
    print(f"HF messages={len(hf_messages)}, weights={weights}, restored={len(restored)}")
    tokenize_sample(json.loads(raw), tokenizer=tokenizer, family="qwen3")


def train() -> TrainResult:
    """Run mixed Pretrain + SFT after conversion and model assets are ready."""
    model, train_config, streams = build_config()
    return Trainer(
        model=model,
        train=train_config,
        streams=streams,
        hf_ckpt_path="/models/Qwen3-8B",
        task_name="api-tour-cpt",
    )


def convert_checkpoints() -> None:
    """Convert DCP to HF, or import HF weights into DCP."""
    to_hf(
        dcp_path="/outputs/api-tour-cpt/checkpoints/step-1000",
        output_path="/exports/qwen3-8b",
        model_name="qwen3/8B",
        hf_assets_path="/models/Qwen3-8B",
    )
    from_hf(
        hf_path="/models/Qwen3-8B",
        output_path="/outputs/qwen3-8b-dcp",
        model_name="qwen3/8B",
    )


def assets_and_diagnostics() -> None:
    """Download assets, inspect checkpoints, and compare training losses."""
    model_dir = download_hf_assets("Qwen/Qwen3-8B", asset_types="tokenizer")
    diagnose_hf_checkpoint(model_dir, model_name="Qwen3-8B", verbose=True)
    compare_losses(["/outputs/run-a", "/outputs/run-b"], show=False)


def main() -> None:
    """Demonstrate side-effect-free configuration and utility construction."""
    models = build_model_configs()
    model, train_config, streams = build_config()
    template = get_template("qwen3")
    register_template(template)
    health = MoEHealthPlugin(model_parts=[])
    bias_schedule = MoEBiasSchedule(calibration_steps=100)
    print(f"models={list(models)}, stage=cpt, steps={train_config.steps}")
    print(f"streams={len(streams)}, templates={list_templates()[:3]}")
    print(f"MoE available={health.has_moe}, calibration={bias_schedule.calibration_steps}")


if __name__ == "__main__":
    main()
