"""Data loading, conversion, inspection, and chat-template utilities.

StreamingTextDataLoader unifies Pretrain, SFT, and CPT over MosaicML Streaming.
Conversion and inspection APIs are imported lazily to avoid heavy dependencies.
"""

from titaninfra.data.templates import (
    ChatTemplate,
    Role,
    apply_template,
    encode_with_template,
    get_template,
    infer_template_from_model_name,
    list_templates,
    register_template,
)


def StreamingTextDataLoader(*args, **kwargs):
    """Instantiate StreamingTextDataLoader through a lazy import."""
    from titaninfra.data.streaming_dataset import StreamingTextDataLoader as _Loader

    return _Loader(*args, **kwargs)


def __getattr__(name: str):
    """Resolve converter and MDS-inspection APIs on first access."""
    _converter_names = {
        "DataConvertConfig", "DataConverter", "OutputPath",
        "ShardValidationError", "WriterFormat", "convert_to_mds",
        "convert_messages_to_mds", "validate_mds_shards",
    }
    _inspect_names = {"get_mds_info", "inspect_mds", "print_mds_samples"}

    if name in _converter_names:
        import titaninfra.data.converter as _mod
        return getattr(_mod, name)
    if name in _inspect_names:
        import titaninfra.data.inspect_mds as _mod  # noqa: F811
        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatTemplate",
    "Role",
    "get_template",
    "list_templates",
    "register_template",
    "apply_template",
    "encode_with_template",
    "infer_template_from_model_name",
    "StreamingTextDataLoader",
    "DataConverter",
    "ShardValidationError",
    "DataConvertConfig",
    "convert_to_mds",
    "convert_messages_to_mds",
    "validate_mds_shards",
    "WriterFormat",
    "OutputPath",
    "inspect_mds",
    "get_mds_info",
    "print_mds_samples",
]
