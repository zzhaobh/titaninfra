"""Complete SFT recipe: three epochs of native Qwen3 message streams."""

from titaninfra import ModelConfig, PackStrategy, StreamConfig, TrainConfig, Trainer


def build_config() -> tuple[ModelConfig, TrainConfig, list[StreamConfig]]:
    """Return a fully explicit configuration without reading data or starting training."""
    model = ModelConfig(
        name="qwen3/8B",
        dp_shard=-1,
        dp_replicate=1,
        tp=1,
        pp=1,
        cp=1,
        ep=1,
        dtype="bf16",
        param_dtype="fp32",
        attn_backend="flex",
        attn_mask_type="block_causal",
        ac_mode="selective",
        compile=False,
    )
    train = TrainConfig(
        local_batch_size=1,
        global_batch_size=-1,
        seq_len=8_192,
        optimizer_name="AdamW",
        lr=1e-5,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        optimizer_state_dtype="fp32",
        warmup_steps=200,
        min_lr_ratio=0.1,
        lr_decay_type="cosine",
        max_grad_norm=1.0,
        ckpt_interval=500,
        keep_latest_k=3,
        dump_folder="/outputs/sft-qwen3-8b",
        task_name="sft-qwen3-8b",
        log_interval=10,
        seed=42,
    )
    streams = [
        StreamConfig(
            stream_id="instruction-messages",
            local="/data/mds/sft/train",
            tokenizer_name="qwen3",
            data_type="messages",
            family="qwen3",
            repeat=3.0,
            pack_strategy=PackStrategy.FULL_SAMPLE,
            epoch_size_margin=0.1,
        )
    ]
    return model, train, streams


def main() -> None:
    """Run the recipe after replacing the neutral data/checkpoint paths."""
    model, train, streams = build_config()
    result = Trainer(
        model=model,
        train=train,
        streams=streams,
        hf_ckpt_path="/models/Qwen3-8B",
    )
    print(result)


if __name__ == "__main__":
    main()
