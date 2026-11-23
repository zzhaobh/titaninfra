"""
Tests for DeepSeek-V4 SFT Encoder.

P0: bit-exact alignment with reference encode_messages() output.
P1: behavioral unit tests (thinking, skip marker, tool handling, labels).
"""

from __future__ import annotations

import pathlib

import pytest

from titaninfra.data.schema import decode_sample

_FIXTURES = pathlib.Path("tests/fixtures/tokenizers/deepseek_v4")
_ENCODING_TESTS = _FIXTURES / "encoding" / "tests"
_TOKENIZER_AVAILABLE = (_FIXTURES / "tokenizer.json").exists()

pytestmark = pytest.mark.skipif(
    not _TOKENIZER_AVAILABLE,
    reason="deepseek_v4 tokenizer fixture not downloaded; run tests/fixtures/tokenizers/download_all.py",
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(_FIXTURES), trust_remote_code=True)


@pytest.fixture(scope="module")
def encoder(tok):
    from titaninfra.data.sft.encoders.deepseek_v4 import DeepSeekV4Encoder
    return DeepSeekV4Encoder(tok)


def _load_titaninfra_fixture(name: str):
    path = _ENCODING_TESTS / name
    with open(path, "rb") as f:
        return decode_sample(f.read())


def _gold(name: str) -> str:
    return (_ENCODING_TESTS / name).read_text(encoding="utf-8").rstrip("\n")


# ── P0: alignment tests ────────────────────────────────────────────────────

def test_align_case_1(tok, encoder):
    """Thinking + DSML tool calls: token sequence matches reference gold output."""
    sample = _load_titaninfra_fixture("titaninfra_case_1.json")
    input_ids, labels = encoder.encode(sample.messages, tools=sample.tools)
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    assert decoded == _gold("test_output_1.txt")


def test_align_case_2(tok, encoder):
    """Multi-turn, no tools: first assistant uses </think> skip marker, second has thinking."""
    sample = _load_titaninfra_fixture("titaninfra_case_2.json")
    input_ids, labels = encoder.encode(sample.messages, tools=None)
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    assert decoded == _gold("test_output_2.txt")


def test_align_case_3(tok, encoder):
    """latest_reminder + developer turn + 3 tools + thinking: matches reference gold output."""
    sample = _load_titaninfra_fixture("titaninfra_case_3.json")
    input_ids, labels = encoder.encode(sample.messages, tools=sample.tools)
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    assert decoded == _gold("test_output_3.txt")


def test_align_case_4(tok, encoder):
    """latest_reminder + weight=0 history masking + action task: matches reference gold output."""
    sample = _load_titaninfra_fixture("titaninfra_case_4.json")
    input_ids, labels = encoder.encode(sample.messages, tools=None)
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    assert decoded == _gold("test_output_4.txt")


def test_align_case_4_labels(tok, encoder):
    """weight=0 assistant tokens are IGNORE; last assistant (after task) carries loss."""
    from titaninfra.data.sft.encoders import IGNORE_INDEX
    sample = _load_titaninfra_fixture("titaninfra_case_4.json")
    input_ids, labels = encoder.encode(sample.messages, tools=None)

    assert len(input_ids) == len(labels)

    # weight=0 assistant: find first <｜Assistant｜> token, verify up through first EOS is IGNORE
    asst_id = tok.encode("<｜Assistant｜>", add_special_tokens=False)[0]
    eos_id = tok.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)[0]
    first_asst_pos = input_ids.index(asst_id)
    first_eos_pos = input_ids.index(eos_id)
    assert all(lbl == IGNORE_INDEX for lbl in labels[first_asst_pos: first_eos_pos + 1])

    # Last assistant (after action task): some tokens must carry loss
    assert any(lbl != IGNORE_INDEX for lbl in labels[first_eos_pos + 1:])


def test_align_case_1_labels(tok, encoder):
    """P0 labels: system/user tokens are IGNORE; assistant content tokens carry loss."""
    from titaninfra.data.sft.encoders import IGNORE_INDEX
    sample = _load_titaninfra_fixture("titaninfra_case_1.json")
    input_ids, labels = encoder.encode(sample.messages, tools=sample.tools)

    assert len(input_ids) == len(labels), "input_ids and labels must be same length"

    # Find first assistant position (after <｜Assistant｜> token)
    asst_id = tok.encode("<｜Assistant｜>", add_special_tokens=False)[0]
    first_asst_pos = input_ids.index(asst_id)

    # Everything up to and including <｜Assistant｜> must be IGNORE
    assert all(lbl == IGNORE_INDEX for lbl in labels[: first_asst_pos + 1])
    # At least some content after the role marker must carry loss
    assert any(lbl != IGNORE_INDEX for lbl in labels[first_asst_pos + 1 :])


@pytest.mark.parametrize(
    "case_name",
    ["titaninfra_case_1.json", "titaninfra_case_2.json", "titaninfra_case_3.json", "titaninfra_case_4.json"],
)
def test_align_cases_train_on_inputs_full_loss(tok, encoder, case_name):
    """train_on_inputs=True on real fixtures: token stream unchanged, all non-BOS labels = input_ids.

    Guards against accidental masking introduced by future per-role branches in the encoder.
    weight=0 messages must also become loss-carrying when train_on_inputs overrides masking.
    """
    from titaninfra.data.sft.encoders import IGNORE_INDEX
    sample = _load_titaninfra_fixture(case_name)
    ids_default, _ = encoder.encode(sample.messages, tools=sample.tools)
    ids_train_in, labels_train_in = encoder.encode(
        sample.messages, tools=sample.tools, train_on_inputs=True
    )
    # Token stream identical regardless of label policy
    assert ids_default == ids_train_in
    # BOS stays IGNORE; everything else carries loss
    assert labels_train_in[0] == IGNORE_INDEX
    assert labels_train_in[1:] == ids_train_in[1:]


def test_loss_config_placeholder_no_op(tok, encoder):
    """loss_config is reserved/unused in the binary path; None / {} / arbitrary keys → identical output.

    Locks current contract so a future weight-scaling implementation must update this test
    rather than silently change behavior.
    """
    sample = _load_titaninfra_fixture("titaninfra_case_1.json")
    base_ids, base_lbl = encoder.encode(sample.messages, tools=sample.tools)
    empty_ids, empty_lbl = encoder.encode(sample.messages, tools=sample.tools, loss_config={})
    junk_ids, junk_lbl = encoder.encode(
        sample.messages, tools=sample.tools, loss_config={"future_key": 0.5}
    )
    assert base_ids == empty_ids == junk_ids
    assert base_lbl == empty_lbl == junk_lbl


# ── P1: behavioral unit tests ──────────────────────────────────────────────

def _make_msg(role: str, blocks: list) -> dict:
    """Build a minimal Message-like object from raw block dicts for quick construction."""
    from titaninfra.data.schema import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
    block_map = {
        "text": lambda d: TextBlock(text=d["text"]),
        "thinking": lambda d: ThinkingBlock(text=d["text"]),
        "tool_use": lambda d: ToolUseBlock(id=d.get("id", "x"), name=d["name"], input=d["input"]),
        "tool_result": lambda d: ToolResultBlock(tool_use_id=d.get("tool_use_id", "x"), text=d["text"]),
    }
    return Message(role=role, blocks=[block_map[b["type"]](b) for b in blocks])


def test_thinking_block_produces_think_tokens(tok, encoder):
    from titaninfra.data.schema import Message, TextBlock, ThinkingBlock
    msgs = [
        Message(role="user", blocks=[TextBlock(text="hi")]),
        Message(role="assistant", blocks=[ThinkingBlock(text="think"), TextBlock(text="ok")]),
    ]
    ids, _ = encoder.encode(msgs)
    think_s = tok.encode("<think>", add_special_tokens=False)[0]
    think_e = tok.encode("</think>", add_special_tokens=False)[0]
    assert think_s in ids
    assert think_e in ids
    # <think> must appear before </think>
    assert ids.index(think_s) < ids.index(think_e)


def test_no_thinking_produces_skip_marker(tok, encoder):
    from titaninfra.data.schema import Message, TextBlock
    msgs = [
        Message(role="user", blocks=[TextBlock(text="hi")]),
        Message(role="assistant", blocks=[TextBlock(text="hello")]),
    ]
    ids, _ = encoder.encode(msgs)
    think_s = tok.encode("<think>", add_special_tokens=False)[0]
    think_e = tok.encode("</think>", add_special_tokens=False)[0]
    assert think_s not in ids, "<think> should not appear when no ThinkingBlock"
    assert think_e in ids, "</think> skip marker must always appear"


def test_tool_result_merged_into_user_turn(tok, encoder):
    """role='tool' → produces a <｜User｜> token (not a standalone tool marker)."""
    from titaninfra.data.schema import Message, TextBlock, ToolResultBlock
    msgs = [
        Message(role="user", blocks=[TextBlock(text="call it")]),
        Message(role="assistant", blocks=[TextBlock(text="ok")]),
        Message(role="tool", blocks=[ToolResultBlock(tool_use_id="x", text="result")]),
        Message(role="assistant", blocks=[TextBlock(text="done")]),
    ]
    ids, _ = encoder.encode(msgs)
    user_id = tok.encode("<｜User｜>", add_special_tokens=False)[0]
    # Should have exactly 2 user tokens: one for user turn, one for tool result turn
    assert ids.count(user_id) == 2
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert "<tool_result>result</tool_result>" in decoded


def test_consecutive_tool_results_in_one_user_turn(tok, encoder):
    """Multiple consecutive role='tool' messages → single <｜User｜> turn."""
    from titaninfra.data.schema import Message, TextBlock, ToolResultBlock
    msgs = [
        Message(role="user", blocks=[TextBlock(text="go")]),
        Message(role="assistant", blocks=[TextBlock(text="calling")]),
        Message(role="tool", blocks=[ToolResultBlock(tool_use_id="a", text="r1")]),
        Message(role="tool", blocks=[ToolResultBlock(tool_use_id="b", text="r2")]),
        Message(role="assistant", blocks=[TextBlock(text="done")]),
    ]
    ids, _ = encoder.encode(msgs)
    user_id = tok.encode("<｜User｜>", add_special_tokens=False)[0]
    # Should have only 2 user tokens: original user + merged tool turn
    assert ids.count(user_id) == 2
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert "<tool_result>r1</tool_result>" in decoded
    assert "<tool_result>r2</tool_result>" in decoded


def test_multi_turn_labels_correctness(tok, encoder):
    """system + user labels = IGNORE; assistant content labels = token ids."""
    from titaninfra.data.schema import Message, TextBlock
    from titaninfra.data.sft.encoders import IGNORE_INDEX
    msgs = [
        Message(role="system", blocks=[TextBlock(text="sys")]),
        Message(role="user", blocks=[TextBlock(text="u")]),
        Message(role="assistant", blocks=[TextBlock(text="a")]),
    ]
    ids, labels = encoder.encode(msgs)
    asst_id = tok.encode("<｜Assistant｜>", add_special_tokens=False)[0]
    asst_pos = ids.index(asst_id)
    # Before and including <｜Assistant｜>: all IGNORE
    assert all(lbl == IGNORE_INDEX for lbl in labels[: asst_pos + 1])
    # After <｜Assistant｜>: must include real token ids (not IGNORE)
    assert any(lbl != IGNORE_INDEX for lbl in labels[asst_pos + 1:])


def test_train_on_inputs_all_loss(tok, encoder):
    """train_on_inputs=True → labels == input_ids (except BOS stays IGNORE)."""
    from titaninfra.data.schema import Message, TextBlock
    from titaninfra.data.sft.encoders import IGNORE_INDEX
    msgs = [
        Message(role="user", blocks=[TextBlock(text="hello")]),
        Message(role="assistant", blocks=[TextBlock(text="hi")]),
    ]
    ids, labels = encoder.encode(msgs, train_on_inputs=True)
    # BOS (index 0) is always IGNORE
    assert labels[0] == IGNORE_INDEX
    # All other tokens should equal their corresponding input_id
    assert ids[1:] == labels[1:]


def test_tools_injected_into_system(tok, encoder):
    """Tools defined at top-level appear in the system prompt section."""
    from titaninfra.data.schema import Message, TextBlock, Tool
    msgs = [
        Message(role="system", blocks=[TextBlock(text="You are helpful.")]),
        Message(role="user", blocks=[TextBlock(text="use tool")]),
        Message(role="assistant", blocks=[TextBlock(text="ok")]),
    ]
    tools = [Tool(name="my_tool", description="does stuff", input_schema={"type": "object"})]
    ids, _ = encoder.encode(msgs, tools=tools)
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert "## Tools" in decoded
    assert "my_tool" in decoded
    assert "does stuff" in decoded


def test_dsml_string_attribute_inference(tok, encoder):
    """str args → string='true'; int/list/dict → string='false'."""
    from titaninfra.data.sft.encoders.deepseek_v4 import _DSML, _encode_dict_to_dsml_params
    params = _encode_dict_to_dsml_params({"city": "Beijing", "count": 3, "flags": [1, 2]}, _DSML)
    assert 'name="city" string="true"' in params
    assert 'name="count" string="false"' in params
    assert 'name="flags" string="false"' in params


def test_max_length_truncation(tok, encoder):
    """max_length clips both input_ids and labels to that length."""
    from titaninfra.data.schema import Message, TextBlock
    msgs = [
        Message(role="user", blocks=[TextBlock(text="hello world " * 50)]),
        Message(role="assistant", blocks=[TextBlock(text="response " * 50)]),
    ]
    ids, labels = encoder.encode(msgs, max_length=32)
    assert len(ids) == 32
    assert len(labels) == 32


def test_bos_is_always_first(tok, encoder):
    """First token is always BOS regardless of content."""
    from titaninfra.data.schema import Message, TextBlock
    msgs = [Message(role="user", blocks=[TextBlock(text="x")])]
    ids, _ = encoder.encode(msgs)
    bos_id = tok.encode("<｜begin▁of▁sentence｜>", add_special_tokens=False)[0]
    assert ids[0] == bos_id


def test_no_double_bos(tok, encoder):
    """Encoder must not produce two consecutive BOS tokens."""
    from titaninfra.data.schema import Message, TextBlock
    msgs = [
        Message(role="system", blocks=[TextBlock(text="sys")]),
        Message(role="user", blocks=[TextBlock(text="hello")]),
        Message(role="assistant", blocks=[TextBlock(text="hi")]),
    ]
    ids, _ = encoder.encode(msgs)
    bos_id = tok.encode("<｜begin▁of▁sentence｜>", add_special_tokens=False)[0]
    assert ids.count(bos_id) == 1, f"Expected exactly 1 BOS, got {ids.count(bos_id)}"


def test_dsv4_parallel_tool_calls_rejected():
    """Audit #6 (2026-06-06): DSv4's DSML wire is single-call-only (encoder
    docstring). A parallel-tool-call assistant turn (2 ToolUseBlocks) must be
    rejected loud — not silently rendered as a single <｜DSML｜tool_calls> block
    with two <｜DSML｜invoke> elements the model cannot reproduce at inference.
    """
    from transformers import AutoTokenizer

    from titaninfra.data.sft.dispatch import tokenize_sample
    tok = AutoTokenizer.from_pretrained("tests/fixtures/tokenizers/deepseek_v4")
    sample = {"messages": [
        {"role": "user", "blocks": [{"type": "text", "text": "weather+time"}]},
        {"role": "assistant", "blocks": [
            {"type": "tool_use", "name": "get_weather", "input": {"city": "SF"}, "id": "w"},
            {"type": "tool_use", "name": "get_time", "input": {"tz": "PT"}, "id": "t"},
        ]},
    ]}
    from titaninfra.data.sft.capabilities import EmptyAssistantError
    with pytest.raises(EmptyAssistantError):
        tokenize_sample(sample, tok, family="deepseek_v4")
