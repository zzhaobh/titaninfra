"""Unit tests for history policy v2."""
import msgspec
import pytest

from titaninfra.data.schema import (
    DEFAULT_HISTORY_POLICY,
    HistoryAction,
    HistoryPolicy,
    SFTSample,
)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def test_keep_last_n_defaults_none():
    p = HistoryPolicy()
    assert p.keep_last_n is None


def test_keep_last_n_roundtrip():
    p = msgspec.json.decode(
        msgspec.json.encode({"keep_last_n": 2, "per_type": {}}),
        type=HistoryPolicy,
    )
    assert p.keep_last_n == 2


def test_default_history_policy_drop_families():
    for fam in ("qwen3", "gpt_oss", "gemma4"):
        pol = DEFAULT_HISTORY_POLICY[fam]
        assert pol.per_type["thinking"].action == "drop", fam


def test_default_history_policy_keep_families():
    for fam in ("qwen2_5", "deepseek_v3", "llama3", "deepseek_v4"):
        pol = DEFAULT_HISTORY_POLICY[fam]
        assert pol.per_type == {}, fam


def test_sft_sample_no_longer_has_history_policy():
    assert "history_policy" not in SFTSample.__struct_fields__


def test_sft_sample_ignores_legacy_history_policy_field():
    from titaninfra.data.schema import decode_sample
    raw = b"""{
        "messages": [{"role": "user", "blocks": [{"type": "text", "text": "hi"}]}],
        "history_policy": {"per_type": {"thinking": {"action": "drop"}}}
    }"""
    sample = decode_sample(raw)
    assert not hasattr(sample, "history_policy")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
from titaninfra.config.data import StreamConfig  # noqa: E402


def _sc(history_policy):
    return StreamConfig(
        local="/tmp/x", data_type="messages", family="qwen3",
        tokenizer_name="qwen3", history_policy=history_policy,
        add_bos_eos=False,
    )


def test_streamconfig_history_policy_none_ok():
    sc = _sc(None)
    assert sc.history_policy is None


def test_streamconfig_history_policy_complete_ok():
    sc = _sc(HistoryPolicy(
        keep_last_n=2,
        per_type={"thinking": HistoryAction(action="keep"),
                  "tool_use": HistoryAction(action="drop"),
                  "tool_result": HistoryAction(action="drop")}))
    assert sc.history_policy.keep_last_n == 2


def test_streamconfig_history_policy_dict_coerced():
    sc = _sc({"keep_last_n": 1, "per_type": {"thinking": {"action": "keep"}}})
    assert isinstance(sc.history_policy, HistoryPolicy)
    assert sc.history_policy.keep_last_n == 1


def test_streamconfig_missing_per_type_raises():
    with pytest.raises(ValueError, match="per_type"):
        _sc(HistoryPolicy(keep_last_n=2))


def test_streamconfig_missing_keep_last_n_raises():
    with pytest.raises(ValueError, match="keep_last_n"):
        _sc(HistoryPolicy(per_type={"thinking": HistoryAction(action="keep")}))


def test_streamconfig_keep_last_n_zero_raises():
    with pytest.raises(ValueError, match="keep_last_n"):
        _sc(HistoryPolicy(keep_last_n=0,
                          per_type={"thinking": HistoryAction(action="keep")}))


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
from titaninfra.data.sft.dispatch import _resolve_history_policy  # noqa: E402


def test_resolve_no_user_returns_none():
    assert _resolve_history_policy("gpt_oss", None) is None
    assert _resolve_history_policy("qwen3", None) is None


def test_resolve_user_policy_passthrough_no_merge():
    user = HistoryPolicy(keep_last_n=2,
                         per_type={"thinking": HistoryAction(action="keep")})
    eff = _resolve_history_policy("qwen3", user)
    assert eff.keep_last_n == 2
    assert eff.per_type["thinking"].action == "keep"


def test_resolve_unknown_family_no_user_returns_none():
    assert _resolve_history_policy("nonexistent_family", None) is None


# ----------------------------------------------------------------------
# Task 4: normalize — _compute_history_boundary + history loss=0
# ----------------------------------------------------------------------
import json  # noqa: E402

from titaninfra.data.sft import normalize  # noqa: E402

_MATH_TOOLS = "tests/fixtures/datasets/math_tools_multiturn.jsonl"


def _load_multiturn_sample(min_asst=3):
    """Load multiturn sample."""
    with open(_MATH_TOOLS) as f:
        for line in f:
            o = json.loads(line)
            msgs = o.get("messages") or []
            n_asst = sum(1 for m in msgs if m.get("role") == "assistant")
            if n_asst >= min_asst:
                return o
    raise AssertionError("no multiturn sample found")


def test_keep_last_n_marks_earlier_history_weight_zero():
    sample = _load_multiturn_sample(min_asst=3)
    sample = {**sample, "history_policy": HistoryPolicy(keep_last_n=1)}
    ns = normalize.from_sft_sample(sample)
    asst_msgs = [m for m in ns.messages if m.role == "assistant"]
    assert len(asst_msgs) >= 3
    for m in asst_msgs[:-1]:
        assert m.weight == 0.0, "history assistant 必须 weight=0"
    assert asst_msgs[-1].weight > 0.0, "current assistant 必须保留 loss"


def test_history_messages_not_deleted():
    sample = _load_multiturn_sample(min_asst=3)
    n_before = len(sample["messages"])
    sample2 = {**sample, "history_policy": HistoryPolicy(
        keep_last_n=1, per_type={"thinking": HistoryAction(action="drop")})}
    ns = normalize.from_sft_sample(sample2)
    assert len(ns.messages) == n_before, "history 消息条目不可删除（索引保护）"





# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def test_all_families_declare_history_drop():
    import importlib

    from titaninfra.data.sft.capabilities import FEATURE_HISTORY_POLICY_DROP
    for fam in ("qwen3", "qwen2_5", "llama3", "deepseek_v3",
                "deepseek_v4", "gpt_oss", "gemma4"):
        mod = importlib.import_module(f"titaninfra.data.sft.encoders.{fam}")
        assert FEATURE_HISTORY_POLICY_DROP in mod.SUPPORTED_FEATURES, fam


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
import pathlib  # noqa: E402

from titaninfra.data.sft import dispatch  # noqa: E402
from titaninfra.data.sft.types import IGNORE_INDEX  # noqa: E402

_HAS_QWEN3 = pathlib.Path("tests/fixtures/tokenizers/qwen3/tokenizer.json").exists()


@pytest.mark.skipif(not _HAS_QWEN3, reason="qwen3 tokenizer fixture 未下载")
def test_history_thinking_dropped_shortens_e2e():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("tests/fixtures/tokenizers/qwen3")
    sample = _load_multiturn_sample(min_asst=3)
    enc_drop = dispatch.tokenize_sample(
        sample, tok, "qwen3",
        history_policy=HistoryPolicy(
            keep_last_n=1, per_type={"thinking": HistoryAction(action="drop")}))
    enc_keep = dispatch.tokenize_sample(
        sample, tok, "qwen3",
        history_policy=HistoryPolicy(
            keep_last_n=1, per_type={"thinking": HistoryAction(action="keep")}))
    assert len(enc_drop.input_ids) < len(enc_keep.input_ids), \
        "drop 历史 thinking 后序列应更短"
    assert any(lbl != IGNORE_INDEX for lbl in enc_drop.labels), "current 必须有 loss"


@pytest.mark.parametrize("fam", ["qwen3", "qwen2_5", "llama3",
                                 "deepseek_v3", "gpt_oss", "gemma4"])
def test_contract_history_masked_current_unmasked(fam):
    """Verify contract history masked current unmasked."""
    if not pathlib.Path(f"tests/fixtures/tokenizers/{fam}/tokenizer.json").exists():
        pytest.skip(f"{fam} tokenizer fixture 未下载")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"tests/fixtures/tokenizers/{fam}")
    sample = _load_multiturn_sample(min_asst=3)
    enc = dispatch.tokenize_sample(
        sample, tok, fam,
        history_policy=HistoryPolicy(
            keep_last_n=1,
            per_type={"thinking": HistoryAction(action="keep")}))
    assert any(lbl == IGNORE_INDEX for lbl in enc.labels), f"{fam}: history 应被 mask"
    assert any(lbl != IGNORE_INDEX for lbl in enc.labels), f"{fam}: current 应有 loss"


def test_streamconfig_keep_last_n_with_insert_summary_rejected():
    # Audit #11 (2026-06-06): keep_last_n and insert_summary use DIFFERENT
    # boundaries (current_start vs last_user_idx). Co-setting them lets a
    # message in the overlap zone be labeled 'current' (kept) yet also have
    # its HistorySummaryBlock collected into the summary head -> duplicate
    # content. insert_summary encoder rendering is also not implemented this
    # period, so reject the ambiguous co-set loud at config time.
    with pytest.raises(ValueError, match="insert_summary"):
        _sc(HistoryPolicy(
            keep_last_n=2, insert_summary=True,
            per_type={"thinking": HistoryAction(action="keep")}))


def test_streamconfig_insert_summary_without_keep_last_n_allowed():
    """insert_summary owns the whole history region and needs no per_type."""
    sc = _sc(HistoryPolicy(insert_summary=True))
    assert sc.history_policy.insert_summary is True
    assert sc.history_policy.keep_last_n is None
