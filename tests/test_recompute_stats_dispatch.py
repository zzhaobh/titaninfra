"""recompute_stats messages-mode token counter must use dispatch."""

import inspect

from titaninfra.data import recompute_stats


def test_token_counter_uses_dispatch():
    src = inspect.getsource(recompute_stats)
    # The whole module file must no longer import the Jinja helper.
    assert "tokenize_with_hf_template" not in src
    assert "dispatch.tokenize_sample" in src or "tokenize_sample" in src
