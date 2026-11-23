"""ToolUseAdapter base class and DefaultAdapter (no-op wrapper).

TEST INFRA ONLY — relocated under ``tests/oracle/_legacy/`` because the
adapter pattern is reachable only from the jinja oracle baseline
(``tests/oracle/_jinja_oracle``) and a handful of legacy alignment tests.
The production SFT pipeline routes every family through pure-Python
segment encoders in ``titaninfra/data/sft/encoders/`` (see CLAUDE.md
"Jinja status").

The default adapter delegates to the pattern-based boundary builder in
``tests.oracle._legacy._boundary._compute_message_boundaries``, kept
alongside as the load-bearing baseline for the oracle parity tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ToolUseAdapter(ABC):
    """Per-template hook for tool-use rendering and boundary detection."""

    def preprocess_messages(
        self,
        hf_messages: list[dict],
        tools: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict] | None]:
        """Transform ``hf_messages`` and ``tools`` before HF ``apply_chat_template``.

        Default: identity. Override when the template Jinja expects a specific
        argument format (e.g. qwen3_5 needs ``tool_call.function.arguments``
        as a dict, not a JSON string).

        MUST NOT mutate caller's input — return new objects when changes apply.
        """
        return hf_messages, tools

    @abstractmethod
    def compute_boundaries(
        self,
        hf_messages: list[dict],
        tokenizer,
        tools: list[dict] | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return ``(full_ids, boundaries, content_starts)``.

        Contract: ``boundaries`` has length ``len(hf_messages) + 1``;
        ``content_starts`` has length ``len(hf_messages)``.
        Loss range for message i = ``[content_starts[i], boundaries[i+1])``.
        """
        ...


class DefaultAdapter(ToolUseAdapter):
    """No-op adapter: dispatches to the existing pattern-based pipeline."""

    def compute_boundaries(
        self,
        hf_messages: list[dict],
        tokenizer,
        tools: list[dict] | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        return _compute_message_boundaries(hf_messages, tokenizer, tools=tools)
