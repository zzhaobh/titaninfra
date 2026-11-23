"""Compatibility helpers for TorchTitan's config-tree quantization converters.

Upstream removed the free functions find_pad_multiple and
find_float8_linear_config after converters became self-applying config trees.
Fork-local parallelization still reads these values from converter lists.

Converter mapping: Float8GroupedMMConverter -> Float8GroupedExpertsConverter
(padding 16), MXFP8Converter -> MXFP8GroupedExpertsConverter (padding 32), and
Float8LinearConverter remains unchanged.
"""

PAD_MULTIPLE_MAP: dict[str, int] = {
    "float8": 16,
    "mxfp8": 32,
}


def find_pad_multiple(converters: list) -> int | None:
    """Return the token-group padding required by grouped quantized GEMMs."""
    from torchtitan.config.transform.quantization import (
        Float8GroupedExpertsConverter,
        MXFP8GroupedExpertsConverter,
    )

    for c in converters:
        if isinstance(c, Float8GroupedExpertsConverter.Config):
            return PAD_MULTIPLE_MAP["float8"]
        if isinstance(c, MXFP8GroupedExpertsConverter.Config):
            return PAD_MULTIPLE_MAP["mxfp8"]
    return None


def find_float8_linear_config(converters: list):
    """Return the first Float8LinearConverter.Config, or None if absent."""
    from torchtitan.config.transform.quantization import Float8LinearConverter

    return next(
        (c for c in converters if isinstance(c, Float8LinearConverter.Config)),
        None,
    )
