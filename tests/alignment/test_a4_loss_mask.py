"""Unit tests for a4 loss mask."""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

TOKENIZERS_DIR = Path(__file__).parent.parent / "fixtures" / "tokenizers"
IGNORE_INDEX = -100


def _load_tokenizer(model_name: str = "qwen3"):
    return AutoTokenizer.from_pretrained(
        str(TOKENIZERS_DIR / model_name), trust_remote_code=True
    )



SINGLE_TURN = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]

MULTI_TURN = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "天气如何？"},
    {"role": "assistant", "content": "今天天气很好。"},
]

WITH_SYSTEM = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]


class TestA4LossMask:
    """Tests for test a4 loss mask."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _load_tokenizer()


    def test_labels_same_length_as_input_ids(self, tokenizer):
        """Verify labels same length as input ids."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

        input_ids, labels = tokenize_with_hf_template(
            messages=SINGLE_TURN, tokenizer=tokenizer, max_length=4096
        )
        assert len(input_ids) == len(labels)

    def test_user_tokens_masked(self, tokenizer):
        """Verify user tokens masked."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        input_ids, labels = tokenize_with_hf_template(
            messages=SINGLE_TURN, tokenizer=tokenizer, max_length=4096
        )
        _, boundaries, _ = _compute_message_boundaries(SINGLE_TURN, tokenizer)
        user_start, user_end = boundaries[0], boundaries[1]

        user_labels = labels[user_start:user_end]
        assert all(v == IGNORE_INDEX for v in user_labels), (
            f"user 区间存在非 -100 label: "
            f"{[v for v in user_labels if v != IGNORE_INDEX][:5]}"
        )

    def test_assistant_tokens_have_loss(self, tokenizer):
        """Verify assistant tokens have loss."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        input_ids, labels = tokenize_with_hf_template(
            messages=SINGLE_TURN, tokenizer=tokenizer, max_length=4096
        )
        _, boundaries, _ = _compute_message_boundaries(SINGLE_TURN, tokenizer)
        asst_start, asst_end = boundaries[1], boundaries[2]

        asst_labels = labels[asst_start:asst_end]
        asst_ids = input_ids[asst_start:asst_end]
        has_loss = [v for v in asst_labels if v != IGNORE_INDEX]
        assert len(has_loss) > 0, "assistant 区间没有任何参与 loss 的 token"

        for i, (tok_id, label) in enumerate(zip(asst_ids, asst_labels, strict=False)):
            if label != IGNORE_INDEX:
                assert label == tok_id, (
                    f"assistant token[{i}]: label={label} != token_id={tok_id}"
                )

    def test_system_tokens_masked(self, tokenizer):
        """Verify system tokens masked."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        input_ids, labels = tokenize_with_hf_template(
            messages=WITH_SYSTEM, tokenizer=tokenizer, max_length=4096
        )
        _, boundaries, _ = _compute_message_boundaries(WITH_SYSTEM, tokenizer)
        sys_start, sys_end = boundaries[0], boundaries[1]

        sys_labels = labels[sys_start:sys_end]
        assert all(v == IGNORE_INDEX for v in sys_labels), (
            "system 区间存在非 -100 label"
        )


    def test_auto_mode_all_assistants_have_loss(self, tokenizer):
        """Verify auto mode all assistants have loss."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

        _, labels = tokenize_with_hf_template(
            messages=MULTI_TURN, tokenizer=tokenizer, max_length=4096,
        )
        loss_count = sum(1 for v in labels if v != IGNORE_INDEX)
        assert loss_count > 0

    # ── train_on_inputs ───────────────────────────────────────

    def test_train_on_inputs_all_tokens_have_loss(self, tokenizer):
        """Verify train on inputs all tokens have loss."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

        input_ids, labels = tokenize_with_hf_template(
            messages=SINGLE_TURN, tokenizer=tokenizer, max_length=4096,
            train_on_inputs=True,
        )
        loss_count = sum(1 for v in labels if v != IGNORE_INDEX)
        assert loss_count == len(labels), (
            f"train_on_inputs=True 时应全部参与 loss: "
            f"{loss_count}/{len(labels)}"
        )


    def test_labels_not_shifted(self, tokenizer):
        """Verify labels not shifted."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

        input_ids, labels = tokenize_with_hf_template(
            messages=SINGLE_TURN, tokenizer=tokenizer, max_length=4096,
        )
        for i in range(len(labels)):
            if labels[i] != IGNORE_INDEX:
                assert labels[i] == input_ids[i], (
                    f"位置 {i}: labels[{i}]={labels[i]} != input_ids[{i}]={input_ids[i]}，"
                    f"labels 不应预先 shift"
                )


    def test_empty_messages(self, tokenizer):
        """Verify empty messages."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

        input_ids, labels = tokenize_with_hf_template(
            messages=[], tokenizer=tokenizer, max_length=4096,
        )
        assert input_ids == []
        assert labels == []

    def test_truncation_preserves_alignment(self, tokenizer):
        """Verify truncation preserves alignment."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

        input_ids, labels = tokenize_with_hf_template(
            messages=MULTI_TURN, tokenizer=tokenizer,
            max_length=30,
        )
        assert len(input_ids) <= 30
        assert len(input_ids) == len(labels)

    # ── sft_mask_mode="manual" ───────────────────────────────

    def test_manual_mode_respects_weights(self, tokenizer):
        """Verify manual mode respects weights."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        manual_msgs = [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "回答一", "weight": 0},
            {"role": "user", "content": "问题二"},
            {"role": "assistant", "content": "回答二"},  # default weight=1
        ]

        _, labels = tokenize_with_hf_template(
            messages=manual_msgs, tokenizer=tokenizer, max_length=4096,
            sft_mask_mode="manual",
        )

        _, boundaries, _ = _compute_message_boundaries(manual_msgs, tokenizer)

        asst1_labels = labels[boundaries[1]:boundaries[2]]
        asst1_loss = sum(1 for v in asst1_labels if v != IGNORE_INDEX)
        assert asst1_loss == 0, f"weight=0 的 assistant 不应有 loss, got {asst1_loss}"

        asst2_labels = labels[boundaries[3]:boundaries[4]]
        asst2_loss = sum(1 for v in asst2_labels if v != IGNORE_INDEX)
        assert asst2_loss > 0, "weight=1 的 assistant 应有 loss"


    def test_auto_mode_weight_zero_always_respected(self, tokenizer):
        """Verify auto mode weight zero always respected."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        msgs_with_zero = [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "回答一"},
            {"role": "user", "content": "问题二"},
            {"role": "assistant", "content": "回答二", "weight": 0},
            {"role": "user", "content": "问题三"},
            {"role": "assistant", "content": "回答三"},
        ]

        _, labels = tokenize_with_hf_template(
            messages=msgs_with_zero, tokenizer=tokenizer, max_length=4096,
        )

        _, boundaries, _ = _compute_message_boundaries(msgs_with_zero, tokenizer)

        asst1_labels = labels[boundaries[1]:boundaries[2]]
        assert sum(1 for v in asst1_labels if v != IGNORE_INDEX) > 0

        asst2_labels = labels[boundaries[3]:boundaries[4]]
        asst2_loss = sum(1 for v in asst2_labels if v != IGNORE_INDEX)
        assert asst2_loss == 0, (
            f"weight=0 应覆盖 history=all, 但 assistant[3] 有 {asst2_loss} 个 loss token"
        )

        asst3_labels = labels[boundaries[5]:boundaries[6]]
        assert sum(1 for v in asst3_labels if v != IGNORE_INDEX) > 0
