"""Unit tests for asset-pattern resolution in the tools package."""

import pytest

from titaninfra.tools import ASSET_PATTERNS, _resolve_download_patterns


def test_resolve_single_type():
    """A single asset type resolves to its registered patterns."""
    patterns = _resolve_download_patterns(["tokenizer"], None)
    assert patterns == ASSET_PATTERNS["tokenizer"]


def test_resolve_multiple_types():
    """Multiple asset types concatenate their registered patterns."""
    patterns = _resolve_download_patterns(["tokenizer", "config"], None)
    expected = ASSET_PATTERNS["tokenizer"] + ASSET_PATTERNS["config"]
    assert patterns == expected


def test_resolve_with_additional_patterns():
    """Additional glob patterns are appended to the result."""
    patterns = _resolve_download_patterns(["config"], ["README.md"])
    assert "README.md" in patterns
    assert patterns[-1] == "README.md"


def test_resolve_unknown_type_raises():
    """Unknown asset types raise ValueError."""
    with pytest.raises(ValueError, match="未知资产类型"):
        _resolve_download_patterns(["unknown_type"], None)


def test_resolve_empty_types():
    """An empty asset-type list produces no patterns."""
    patterns = _resolve_download_patterns([], None)
    assert patterns == []
