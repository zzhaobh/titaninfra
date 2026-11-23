"""Router — picks the encoder name for a ``(NormalizedSample, family)`` pair.

Current contract: route() always returns the family-native encoder.
The earlier 2-rule structural table (`_has_image` / `_is_multi_turn_tool`)
was dead code: `_has_image` always returned False (no ImageBlock in
NormalizedSample), and `_is_multi_turn_tool` resolved to the same family
encoder it would have used anyway. Keeping the function as a thin
seam so callers (dispatch.py, rl_prompt.py) don't have to change when
a real structural trigger is introduced.
"""

from __future__ import annotations

from titaninfra.data.sft.types import NormalizedSample


def route(sample: NormalizedSample, family: str) -> str:
    del sample
    return family


__all__ = ["route"]
