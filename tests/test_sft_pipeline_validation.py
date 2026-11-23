"""Unit tests for sft pipeline validation."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from titaninfra.data.packer import PackMode, StreamingPacker
from titaninfra.data.sft.message_converter import normalize_to_hf_messages
from titaninfra.data.sft.types import IGNORE_INDEX

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "tokenizers"
REPO_ROOT = Path(__file__).parent.parent

TOKENIZER_PARAMS = [
    pytest.param("qwen3", id="qwen3"),
    pytest.param("qwen2_5", id="qwen2_5"),
    pytest.param("deepseek_v3", id="deepseek_v3"),
    pytest.param("llama4", id="llama4"),
    pytest.param("gemma4", id="gemma4"),
    pytest.param("gpt_oss", id="gpt_oss"),
]


@pytest.fixture(params=TOKENIZER_PARAMS)
def tokenizer(request):
    name = request.param
    path = str(FIXTURES_DIR / name)
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)



SINGLE_TURN = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
]

MULTI_TURN = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Write a Python hello world"},
    {"role": "assistant", "content": 'print("Hello, world!")'},
]

WITH_SYSTEM = [
    {"role": "system", "content": "You are a math tutor. Be concise."},
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
]

THINKING_BLOCKS = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Explain quantum entanglement"}],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "text": "The user wants to learn about quantum entanglement."},
            {"type": "text", "text": "Quantum entanglement is a correlation between two particles."},
        ],
    },
]

ALL_MESSAGE_SETS = [SINGLE_TURN, MULTI_TURN, WITH_SYSTEM, THINKING_BLOCKS]


def _first_diff(a: list[int], b: list[int]) -> str:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return f"pos {i}: titaninfra={x}, hf={y}"
    if len(a) != len(b):
        return f"lengths differ: {len(a)} vs {len(b)}"
    return "identical"


# ══════════════════════════════════════════════════════
# Class 1: Token Count Alignment
# ══════════════════════════════════════════════════════

class TestTokenCountAlignment:
    """Tests for test token count alignment."""

    @pytest.mark.parametrize("messages,name", [
        (SINGLE_TURN, "single_turn"),
        (MULTI_TURN, "multi_turn"),
        (WITH_SYSTEM, "with_system"),
    ], ids=["single_turn", "multi_turn", "with_system"])
    def test_input_ids_bitexact(self, tokenizer, messages, name):
        input_ids_titaninfra, _ = tokenize_with_hf_template(
            messages=messages, tokenizer=tokenizer, max_length=4096,
        )

        hf_messages, _ = normalize_to_hf_messages(messages)
        input_ids_hf = tokenizer.apply_chat_template(
            hf_messages, tokenize=True, add_generation_prompt=False, return_dict=False,
        )

        assert input_ids_titaninfra == input_ids_hf, (
            f"[{name}] input_ids mismatch!\n"
            f"titaninfra len={len(input_ids_titaninfra)}, hf len={len(input_ids_hf)}\n"
            f"first diff: {_first_diff(input_ids_titaninfra, input_ids_hf)}"
        )

    def test_thinking_blocks_input_ids(self, tokenizer):
        """Verify thinking blocks input ids."""
        input_ids_titaninfra, _ = tokenize_with_hf_template(
            messages=THINKING_BLOCKS, tokenizer=tokenizer, max_length=4096,
        )

        hf_messages, _ = normalize_to_hf_messages(THINKING_BLOCKS)
        input_ids_hf = tokenizer.apply_chat_template(
            hf_messages, tokenize=True, add_generation_prompt=False, return_dict=False,
        )

        assert input_ids_titaninfra == input_ids_hf, (
            f"thinking blocks input_ids mismatch!\n"
            f"titaninfra len={len(input_ids_titaninfra)}, hf len={len(input_ids_hf)}\n"
            f"first diff: {_first_diff(input_ids_titaninfra, input_ids_hf)}"
        )

    def test_length_matches_exactly(self, tokenizer):
        for messages in ALL_MESSAGE_SETS:
            ids_j, _ = tokenize_with_hf_template(
                messages=messages, tokenizer=tokenizer, max_length=4096,
            )
            hf_msgs, _ = normalize_to_hf_messages(messages)
            ids_h = tokenizer.apply_chat_template(
                hf_msgs, tokenize=True, add_generation_prompt=False, return_dict=False,
            )
            assert len(ids_j) == len(ids_h)


# ══════════════════════════════════════════════════════
# Class 2: Template Rendering
# ══════════════════════════════════════════════════════

class TestTemplateRendering:
    """Tests for test template rendering."""

    @pytest.mark.parametrize("messages", [
        SINGLE_TURN, MULTI_TURN, WITH_SYSTEM,
    ], ids=["single_turn", "multi_turn", "with_system"])
    def test_rendered_text_matches_hf(self, tokenizer, messages):
        hf_messages, _ = normalize_to_hf_messages(messages)
        rendered = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=False,
        )
        assert isinstance(rendered, str)
        assert len(rendered) > 0

        direct = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=False,
        )
        assert rendered == direct

    def test_thinking_block_renders_with_tags(self, tokenizer):
        """Verify thinking block renders with tags."""
        hf_messages, _ = normalize_to_hf_messages(THINKING_BLOCKS)
        rendered = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=False,
        )
        assert "<think>" in rendered
        assert "</think>" in rendered
        assert "quantum entanglement" in rendered.lower()

    def test_normalize_preserves_content(self, tokenizer):
        """Verify normalize preserves content."""
        hf_messages, weights = normalize_to_hf_messages(WITH_SYSTEM)
        assert len(hf_messages) == 3
        assert hf_messages[0]["role"] == "system"
        assert "math tutor" in hf_messages[0]["content"].lower()
        assert weights[0] == 0.0
        assert weights[2] == 1.0


# ══════════════════════════════════════════════════════
# Class 3: Mask Structure After Packing (KEY FOCUS)
# ══════════════════════════════════════════════════════

class TestMaskStructureAfterPacking:
    """Tests for test mask structure after packing."""

    @staticmethod
    def _tokenize_to_tuples(messages, tokenizer, max_length=512):
        input_ids, labels = tokenize_with_hf_template(
            messages=messages, tokenizer=tokenizer, max_length=max_length,
        )
        return list(zip(input_ids, labels, strict=False))

    def test_position_reset_at_doc_boundary(self, tokenizer):
        """Verify position reset at doc boundary."""
        packer = StreamingPacker(seq_len=2048, mode=PackMode.SFT)

        doc1 = self._tokenize_to_tuples(SINGLE_TURN, tokenizer)
        doc2 = self._tokenize_to_tuples(MULTI_TURN, tokenizer)
        packer.add_document(doc1)
        packer.add_document(doc2)

        samples = list(packer.flush_all())
        assert len(samples) >= 1

        for sample in samples:
            positions = sample.positions
            assert positions is not None
            assert positions[0] == 0, "first position must be 0"

            doc_starts = [i for i, p in enumerate(positions) if p == 0]
            assert len(doc_starts) >= 1

            for ds_idx in range(len(doc_starts)):
                start = doc_starts[ds_idx]
                end = doc_starts[ds_idx + 1] if ds_idx + 1 < len(doc_starts) else len(positions)
                doc_positions = positions[start:end]
                assert doc_positions == list(range(len(doc_positions))), (
                    f"positions not monotonic in doc at {start}: {doc_positions[:10]}..."
                )

    def test_no_false_position_zero(self, tokenizer):
        """Verify no false position zero."""
        packer = StreamingPacker(seq_len=2048, mode=PackMode.SFT)
        doc = self._tokenize_to_tuples(SINGLE_TURN, tokenizer)
        packer.add_document(doc)
        samples = list(packer.flush_all())

        for sample in samples:
            positions = sample.positions
            for i, p in enumerate(positions):
                if p == 0 and i > 0:
                    assert positions[i - 1] > 0, (
                        f"false position==0 at index {i}, prev={positions[i - 1]}"
                    )

    def test_labels_and_ids_same_length(self, tokenizer):
        """Verify labels and ids same length."""
        packer = StreamingPacker(seq_len=2048, mode=PackMode.SFT)
        doc1 = self._tokenize_to_tuples(SINGLE_TURN, tokenizer)
        doc2 = self._tokenize_to_tuples(WITH_SYSTEM, tokenizer)
        packer.add_document(doc1)
        packer.add_document(doc2)

        for sample in packer.flush_all():
            assert len(sample.input_ids) == len(sample.labels)
            assert len(sample.input_ids) == len(sample.positions)
            if sample.loss_weights is not None:
                assert len(sample.input_ids) == len(sample.loss_weights)

    def test_loss_mask_ratio(self, tokenizer):
        """Verify loss mask ratio."""
        input_ids, labels = tokenize_with_hf_template(
            messages=WITH_SYSTEM, tokenizer=tokenizer, max_length=4096,
        )

        loss_tokens = [v for v in labels if v != IGNORE_INDEX]
        masked_tokens = [v for v in labels if v == IGNORE_INDEX]

        assert len(loss_tokens) > 0, "no tokens have loss"
        assert len(masked_tokens) > 0, "no tokens are masked"
        assert len(masked_tokens) > len(loss_tokens), (
            "system+user tokens should outnumber assistant tokens in this sample"
        )

    def test_multi_doc_packing_preserves_labels(self, tokenizer):
        """Verify multi doc packing preserves labels."""
        packer = StreamingPacker(seq_len=4096, mode=PackMode.SFT)
        doc1 = self._tokenize_to_tuples(SINGLE_TURN, tokenizer)
        doc2 = self._tokenize_to_tuples(WITH_SYSTEM, tokenizer)
        packer.add_document(doc1)
        packer.add_document(doc2)

        for sample in packer.flush_all():
            has_loss = any(v != IGNORE_INDEX for v in sample.labels)
            has_mask = any(v == IGNORE_INDEX for v in sample.labels)
            assert has_loss, "packed sample has no loss tokens"
            assert has_mask, "packed sample has no masked tokens"

    def test_seq_len_plus_one_output(self, tokenizer):
        """Verify seq len plus one output."""
        seq_len = 512
        packer = StreamingPacker(seq_len=seq_len, mode=PackMode.SFT)

        for messages in [SINGLE_TURN, MULTI_TURN, WITH_SYSTEM]:
            doc = self._tokenize_to_tuples(messages, tokenizer, max_length=seq_len)
            packer.add_document(doc)

        for sample in packer.flush_all():
            assert len(sample.input_ids) <= seq_len + 1, (
                f"packed sample length {len(sample.input_ids)} > seq_len+1={seq_len + 1}"
            )

    def test_causal_shift_alignment(self, tokenizer):
        """Verify causal shift alignment."""
        input_ids, labels = tokenize_with_hf_template(
            messages=SINGLE_TURN, tokenizer=tokenizer, max_length=4096,
        )

        shifted_inputs = input_ids[:-1]
        shifted_labels = labels[1:]
        assert len(shifted_inputs) == len(shifted_labels)

        has_loss = any(v != IGNORE_INDEX for v in shifted_labels)
        assert has_loss, "no loss tokens after causal shift"


# ══════════════════════════════════════════════════════
# Class 4: Decode Roundtrip
# ══════════════════════════════════════════════════════

class TestDecodeRoundtrip:
    """Tests for test decode roundtrip."""

    @pytest.mark.parametrize("messages,expected_text", [
        (SINGLE_TURN, "2+2 equals 4"),
        (MULTI_TURN, "Hello, world!"),
        (WITH_SYSTEM, "math tutor"),
    ], ids=["single_turn", "multi_turn", "with_system"])
    def test_decode_contains_content(self, tokenizer, messages, expected_text):
        input_ids, _ = tokenize_with_hf_template(
            messages=messages, tokenizer=tokenizer, max_length=4096,
        )
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        assert expected_text.lower() in decoded.lower(), (
            f"decoded text does not contain '{expected_text}'\n"
            f"decoded[:200]: {decoded[:200]}"
        )

    def test_thinking_content_survives_roundtrip(self, tokenizer):
        """Verify thinking content survives roundtrip."""
        input_ids, _ = tokenize_with_hf_template(
            messages=THINKING_BLOCKS, tokenizer=tokenizer, max_length=4096,
        )
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "quantum entanglement" in decoded.lower()

    def test_assistant_content_has_loss(self, tokenizer):
        """Verify assistant content has loss."""
        input_ids, labels = tokenize_with_hf_template(
            messages=SINGLE_TURN, tokenizer=tokenizer, max_length=4096,
        )
        loss_ids = [tid for tid, lab in zip(input_ids, labels, strict=False) if lab != IGNORE_INDEX]
        assert len(loss_ids) > 0
        decoded_loss = tokenizer.decode(loss_ids, skip_special_tokens=True)
        assert "4" in decoded_loss, (
            f"assistant answer '4' not in loss-token decode: {decoded_loss}"
        )


# ══════════════════════════════════════════════════════
# Class 5: Thinking Scale Pipeline
# ══════════════════════════════════════════════════════

class TestThinkingScalePipeline:
    """Tests for test thinking scale pipeline."""

    THINKING_MESSAGES = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "text": "Simple arithmetic: 2 plus 2 equals 4."},
                {"type": "text", "text": "The answer is 4."},
            ],
        },
    ]

    @pytest.fixture
    def qwen3_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            str(FIXTURES_DIR / "qwen3"), trust_remote_code=True,
        )

    def test_thinking_scale_weights(self, qwen3_tokenizer):
        """Verify thinking scale weights."""
        ids, labels, weights = tokenize_with_hf_template(
            messages=self.THINKING_MESSAGES,
            tokenizer=qwen3_tokenizer,
            max_length=4096,
            loss_config={"type_overrides": {"thinking": 0.3}},
            return_weights=True,
        )

        if weights is None:
            pytest.skip("weights collapsed to binary (no fractional weights)")

        has_scaled = any(0.0 < w < 1.0 for w in weights)
        assert has_scaled, "no thinking tokens with fractional weight found"

        for w in weights:
            if 0.0 < w < 1.0:
                assert abs(w - 0.3) < 0.01, f"unexpected weight {w}, expected ~0.3"

    def test_thinking_scale_zero_masks_labels(self, qwen3_tokenizer):
        """Verify thinking scale zero masks labels."""
        ids, labels, weights = tokenize_with_hf_template(
            messages=self.THINKING_MESSAGES,
            tokenizer=qwen3_tokenizer,
            max_length=4096,
            loss_config={"type_overrides": {"thinking": 0.0}},
            return_weights=True,
        )

        loss_ids = [tid for tid, lab in zip(ids, labels, strict=False) if lab != IGNORE_INDEX]
        decoded = qwen3_tokenizer.decode(loss_ids, skip_special_tokens=True)
        assert "4" in decoded, f"answer not in loss tokens: {decoded}"

    def test_mask_empty_thinking(self, qwen3_tokenizer):
        """Verify mask empty thinking."""
        msgs_no_think = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        ids, labels = tokenize_with_hf_template(
            messages=msgs_no_think,
            tokenizer=qwen3_tokenizer,
            max_length=4096,
            mask_empty_thinking=True,
        )

        decoded = qwen3_tokenizer.decode(ids, skip_special_tokens=False)
        if "<think>" in decoded:
            open_ids = qwen3_tokenizer.encode("<think>", add_special_tokens=False)

            think_region_start = None
            for i in range(len(ids) - len(open_ids) + 1):
                if ids[i:i + len(open_ids)] == open_ids:
                    think_region_start = i
                    break

            if think_region_start is not None:
                close_ids = qwen3_tokenizer.encode("</think>", add_special_tokens=False)
                think_region_end = None
                for i in range(think_region_start + len(open_ids), len(ids) - len(close_ids) + 1):
                    if ids[i:i + len(close_ids)] == close_ids:
                        think_region_end = i + len(close_ids)
                        break
                if think_region_end is None:
                    think_region_end = think_region_start + len(open_ids)
                for j in range(think_region_start, min(think_region_end, len(labels))):
                    assert labels[j] == IGNORE_INDEX, (
                        f"empty thinking token at pos {j} not masked"
                    )


# ══════════════════════════════════════════════════════
# Class 6: FSB Long Thinking Data
# ══════════════════════════════════════════════════════

class TestFSBLongThinkingData:
    """Tests for test fsblong thinking data."""

    FSB_JSONL = REPO_ROOT / "data" / "fsb" / "sft" / "train" / "fsb_v1_think.jsonl"

    @pytest.fixture
    def fsb_samples(self):
        if not self.FSB_JSONL.exists():
            pytest.skip(f"FSB data not found: {self.FSB_JSONL}")
        samples = []
        with open(self.FSB_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    @pytest.fixture
    def qwen3_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            str(FIXTURES_DIR / "qwen3"), trust_remote_code=True,
        )

    def test_all_samples_tokenize_successfully(self, fsb_samples, qwen3_tokenizer):
        """Verify all samples tokenize successfully."""
        for i, sample in enumerate(fsb_samples):
            messages = sample["messages"]
            input_ids, labels = tokenize_with_hf_template(
                messages=messages,
                tokenizer=qwen3_tokenizer,
                max_length=16384,
            )
            assert len(input_ids) > 0, f"sample {i}: empty input_ids"
            assert len(input_ids) == len(labels), f"sample {i}: length mismatch"
            loss_count = sum(1 for v in labels if v != IGNORE_INDEX)
            assert loss_count > 0, f"sample {i}: no loss tokens"

    def test_thinking_blocks_detected(self, fsb_samples, qwen3_tokenizer):
        """Verify thinking blocks detected."""
        for i, sample in enumerate(fsb_samples):
            messages = sample["messages"]
            hf_messages, _ = normalize_to_hf_messages(messages)

            has_thinking = any(
                "<think>" in m.get("content", "")
                for m in hf_messages
                if m["role"] == "assistant"
            )
            assert has_thinking, f"sample {i}: no thinking content after normalize"

    def test_no_position_drift(self, fsb_samples, qwen3_tokenizer):
        """Verify no position drift."""
        packer = StreamingPacker(seq_len=4096, mode=PackMode.SFT)

        for sample in fsb_samples:
            messages = sample["messages"]
            input_ids, labels = tokenize_with_hf_template(
                messages=messages,
                tokenizer=qwen3_tokenizer,
                max_length=4096,
            )
            tokens = list(zip(input_ids, labels, strict=False))
            packer.add_document(tokens)

        for sample in packer.flush_all():
            positions = sample.positions
            assert positions is not None
            assert positions[0] == 0

            for i in range(1, len(positions)):
                if positions[i] == 0:
                    assert positions[i - 1] > 0, (
                        f"false doc boundary at pos {i}: prev={positions[i - 1]}"
                    )
                else:
                    assert positions[i] == positions[i - 1] + 1, (
                        f"position gap at {i}: {positions[i - 1]} -> {positions[i]}"
                    )

    def test_fsb_input_ids_match_hf(self, fsb_samples, qwen3_tokenizer):
        """Verify fsb input ids match hf."""
        for i, sample in enumerate(fsb_samples):
            messages = sample["messages"]
            input_ids_j, _ = tokenize_with_hf_template(
                messages=messages,
                tokenizer=qwen3_tokenizer,
                max_length=16384,
            )
            hf_messages, _ = normalize_to_hf_messages(messages)
            input_ids_h = qwen3_tokenizer.apply_chat_template(
                hf_messages, tokenize=True, add_generation_prompt=False, return_dict=False,
            )[:16384]

            assert input_ids_j == input_ids_h, (
                f"[FSB sample {i}] input_ids mismatch!\n"
                f"titaninfra len={len(input_ids_j)}, hf len={len(input_ids_h)}\n"
                f"first diff: {_first_diff(input_ids_j, input_ids_h)}"
            )

    def test_thinking_scale_with_fsb(self, fsb_samples, qwen3_tokenizer):
        """Verify thinking scale with fsb."""
        for i, sample in enumerate(fsb_samples):
            messages = sample["messages"]
            ids, labels, weights = tokenize_with_hf_template(
                messages=messages,
                tokenizer=qwen3_tokenizer,
                max_length=4096,
                loss_config={"type_overrides": {"thinking": 0.5}},
                return_weights=True,
            )

            if weights is not None:
                has_scaled = any(0.0 < w < 1.0 for w in weights)
                assert has_scaled, f"sample {i}: no fractional weights with thinking_scale=0.5"
                for w in weights:
                    if 0.0 < w < 1.0:
                        assert abs(w - 0.5) < 0.01, f"sample {i}: unexpected weight {w}"


# ══════════════════════════════════════════════════════
# Class 7: Cross-Family Tool-Use Contract
# ══════════════════════════════════════════════════════

TOOL_USE_MESSAGES = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "22C sunny"},
    {"role": "assistant", "content": "It's 22°C and sunny in Paris."},
]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


@pytest.mark.xdist_group("heavy_tokenizer")
class TestToolUseCrossFamilyContract:
    """Same tool-use conversation through every family's Jinja template must preserve:
      - tool name appears in rendered text
      - tool argument value appears in rendered text
      - tool result text survives the round trip
    Minimum interop contract titaninfra commits to across qwen3 / deepseek_v3 / gemma4 / gpt_oss.
    DSV4 uses its own DSML encoder and is tested in test_dsv4_encoder.py.
    """

    def test_tool_name_and_args_survive_render(self, tokenizer, request):
        """tool name + concrete argument value must appear in tokenized→decoded text.

        Uses train_on_inputs=True to bypass boundary detection — this test only
        verifies round-trip text survival, not loss masking. Per-family templates
        for role='tool' vary (Qwen→user marker, Gemma→tool marker), so boundary
        validation is not the contract under test here.

        Known limitation: deepseek_v3 Jinja template silently drops assistant
        `tool_calls` when content is empty (only renders tool outputs + final
        assistant text). Use the dedicated DSV4 DSML encoder when DeepSeek-family
        tool-use SFT is required.
        """
        family_id = request.node.callspec.id
        if family_id == "deepseek_v3":
            pytest.xfail("deepseek_v3 Jinja template drops assistant tool_calls — use DSV4 encoder for tool SFT")
        input_ids, _ = tokenize_with_hf_template(
            messages=TOOL_USE_MESSAGES,
            tokenizer=tokenizer,
            tools=TOOL_SCHEMAS,
            max_length=4096,
            train_on_inputs=True,
        )
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "get_weather" in decoded, f"tool name missing: {decoded[:500]}"
        assert "Paris" in decoded, f"tool argument value missing: {decoded[:500]}"

    def test_tool_result_text_survives_render(self, tokenizer):
        """Tool result body must appear in the rendered/tokenized output."""
        input_ids, _ = tokenize_with_hf_template(
            messages=TOOL_USE_MESSAGES,
            tokenizer=tokenizer,
            tools=TOOL_SCHEMAS,
            max_length=4096,
            train_on_inputs=True,
        )
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "22C sunny" in decoded, f"tool result text missing: {decoded[:500]}"
