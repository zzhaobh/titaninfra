"""Unit tests for schema."""

import json
from pathlib import Path

import pytest

from titaninfra.data.schema import (
    ImageBlock,
    LossConfig,
    Message,
    SFTSample,
    TextBlock,
    ThinkingBlock,
    Tool,
    ToolResultBlock,
    ToolUseBlock,
    decode_json_array,
    decode_jsonl,
    decode_sample,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLES_PATH = FIXTURES_DIR / "sft_samples.json"


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestBlockConstruction:

    def test_text_block(self):
        b = TextBlock(text="hello")
        assert isinstance(b, TextBlock)
        assert b.text == "hello"
        assert b.weight is None

    def test_thinking_block(self):
        b = ThinkingBlock(text="分析中...")
        assert isinstance(b, ThinkingBlock)
        assert b.text == "分析中..."

    def test_tool_use_block(self):
        b = ToolUseBlock(id="c1", name="calc", input={"expr": "1+1"})
        assert isinstance(b, ToolUseBlock)
        assert b.id == "c1"
        assert b.name == "calc"
        assert b.input == {"expr": "1+1"}

    def test_tool_result_block(self):
        b = ToolResultBlock(tool_use_id="c1", text="2")
        assert isinstance(b, ToolResultBlock)
        assert b.is_error is False

    def test_tool_result_error(self):
        b = ToolResultBlock(tool_use_id="c1", text="err", is_error=True)
        assert b.is_error is True

    def test_image_block(self):
        b = ImageBlock(source={"type": "url", "data": "https://example.com/img.jpg"})
        assert isinstance(b, ImageBlock)

    def test_block_weight_override(self):
        b = TextBlock(text="x", weight=0.7)
        assert b.weight == 0.7


# ═══════════════════════════════════════════
#  M1: Message / LossConfig / SFTSample
# ═══════════════════════════════════════════


class TestMessage:

    def test_basic_message(self):
        m = Message(role="user", blocks=[TextBlock(text="hi")])
        assert m.role == "user"
        assert len(m.blocks) == 1
        assert m.weight is None
        assert m.name is None

    def test_message_with_weight(self):
        m = Message(role="assistant", blocks=[], weight=0)
        assert m.weight == 0

    def test_message_with_name(self):
        m = Message(role="assistant", blocks=[], name="agent_1")
        assert m.name == "agent_1"


class TestLossConfig:

    def test_defaults(self):
        cfg = LossConfig()
        assert cfg.type_overrides == {}

    def test_custom_overrides(self):
        cfg = LossConfig(type_overrides={"thinking": 0.5})
        assert cfg.type_overrides["thinking"] == 0.5


class TestSFTSample:

    def test_minimal(self):
        s = SFTSample(messages=[
            Message(role="user", blocks=[TextBlock(text="hi")]),
            Message(role="assistant", blocks=[TextBlock(text="hello")]),
        ])
        assert len(s.messages) == 2
        assert s.system is None
        assert s.tools is None
        assert s.loss_config.type_overrides == {}

    def test_with_system_and_tools(self):
        s = SFTSample(
            messages=[Message(role="user", blocks=[TextBlock(text="hi")])],
            system="你是助手",
            tools=[Tool(name="calc", description="计算")],
        )
        assert s.system == "你是助手"
        assert len(s.tools) == 1
        assert s.tools[0].name == "calc"


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestBlockUnion:

    def test_mixed_blocks_in_message(self):
        m = Message(role="assistant", blocks=[
            ThinkingBlock(text="思考"),
            TextBlock(text="回答"),
            ToolUseBlock(id="c1", name="f", input={}),
        ])
        assert isinstance(m.blocks[0], ThinkingBlock)
        assert isinstance(m.blocks[1], TextBlock)
        assert isinstance(m.blocks[2], ToolUseBlock)


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestDecodeSample:

    def test_minimal_json(self):
        data = (
            b'{"messages": [{"role": "user", "blocks": [{"type": "text", "text": "hi"}]},'
            b' {"role": "assistant", "blocks": [{"type": "text", "text": "hello"}]}]}'
        )
        s = decode_sample(data)
        assert len(s.messages) == 2
        assert s.messages[0].blocks[0].text == "hi"

    def test_str_input(self):
        s = decode_sample('{"messages": [{"role": "user", "blocks": [{"type": "text", "text": "x"}]}]}')
        assert s.messages[0].blocks[0].text == "x"

    def test_defaults_filled(self):
        s = decode_sample(b'{"messages": []}')
        assert s.system is None
        assert s.tools is None
        assert s.loss_config.type_overrides == {}

    def test_unknown_fields_ignored(self):
        s = decode_sample(b'{"_comment": "test", "_scenario": "foo", "messages": []}')
        assert len(s.messages) == 0

    def test_thinking_block(self):
        data = b'{"messages": [{"role": "assistant", "blocks": [{"type": "thinking", "text": "hmm"}]}]}'
        s = decode_sample(data)
        assert isinstance(s.messages[0].blocks[0], ThinkingBlock)

    def test_tool_use_block(self):
        data = (
            b'{"messages": [{"role": "assistant", "blocks": '
            b'[{"type": "tool_use", "id": "c1", "name": "f", "input": {"x": 1}}]}]}'
        )
        s = decode_sample(data)
        b = s.messages[0].blocks[0]
        assert isinstance(b, ToolUseBlock)
        assert b.input == {"x": 1}

    def test_image_block(self):
        data = b'{"messages": [{"role": "user", "blocks": [{"type": "image", "source": {"type": "url", "data": "http://img"}}]}]}'
        s = decode_sample(data)
        assert isinstance(s.messages[0].blocks[0], ImageBlock)

    def test_loss_config_parsed(self):
        data = (
            b'{"loss_config": {"type_overrides": {"thinking": 0.5}}, "messages": []}'
        )
        s = decode_sample(data)
        assert s.loss_config.type_overrides == {"thinking": 0.5}

    def test_invalid_json_raises(self):
        with pytest.raises(msgspec.DecodeError):
            decode_sample(b"not json")

    def test_invalid_block_type_raises(self):
        data = b'{"messages": [{"role": "user", "blocks": [{"type": "unknown_type", "text": "x"}]}]}'
        with pytest.raises(msgspec.ValidationError):
            decode_sample(data)

    def test_message_weight(self):
        data = b'{"messages": [{"role": "assistant", "weight": 0, "blocks": [{"type": "text", "text": "x"}]}]}'
        s = decode_sample(data)
        assert s.messages[0].weight == 0

    def test_block_weight(self):
        data = b'{"messages": [{"role": "assistant", "blocks": [{"type": "text", "text": "x", "weight": 0.7}]}]}'
        s = decode_sample(data)
        assert s.messages[0].blocks[0].weight == 0.7


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestDecodeFixtures:

    def test_all_24_fixtures_parse(self):
        """Verify all 24 fixtures parse."""
        samples = decode_json_array(SAMPLES_PATH)
        assert len(samples) == 24

    @pytest.mark.parametrize("idx", range(24))
    def test_each_fixture(self, idx):
        """Verify each fixture."""
        with open(SAMPLES_PATH) as f:
            raw_list = json.load(f)
        sample = decode_sample(json.dumps(raw_list[idx]).encode())
        assert isinstance(sample, SFTSample)
        assert len(sample.messages) >= 0

    def test_fixture_scenarios_present(self):
        """Verify fixture scenarios present."""
        with open(SAMPLES_PATH) as f:
            raw_list = json.load(f)
        scenarios = {item.get("_scenario") for item in raw_list}
        expected = {
            "basic_single_turn", "with_system", "multi_turn_default_history",
            "thinking_cot", "single_tool_call", "image_caption",
            "train_on_inputs", "empty_blocks", "multi_agent",
        }
        assert expected.issubset(scenarios), f"Missing: {expected - scenarios}"


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestDecodeJsonl:

    def test_jsonl_roundtrip(self, tmp_path):
        """Verify jsonl roundtrip."""
        lines = [
            '{"messages": [{"role": "user", "blocks": [{"type": "text", "text": "a"}]}]}',
            '{"messages": [{"role": "user", "blocks": [{"type": "text", "text": "b"}]}]}',
        ]
        p = tmp_path / "test.jsonl"
        p.write_text("\n".join(lines))
        samples = decode_jsonl(p)
        assert len(samples) == 2
        assert samples[0].messages[0].blocks[0].text == "a"
        assert samples[1].messages[0].blocks[0].text == "b"

    def test_jsonl_empty_lines_skipped(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"messages": []}\n\n\n{"messages": []}\n')
        samples = decode_jsonl(p)
        assert len(samples) == 2


import msgspec  # noqa: E402
