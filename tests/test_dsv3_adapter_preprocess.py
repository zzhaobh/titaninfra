"""DeepSeekV3Adapter.preprocess_messages — DSV3-specific content=None rewrite.

Why this is at the adapter layer (not in normalize_to_hf_messages):

DSV3/R1 Jinja gates the tool_call branch on `message['content'] is none`:

    {%- if message['role'] == 'assistant' and message['content'] is none %}
        ... <｜tool▁call▁begin｜> ... <｜tool▁call▁end｜> ...
    {%- elif message['role'] == 'assistant' and message['content'] is not none %}
        ... <｜Assistant｜>{{content}}<｜end▁of▁sentence｜> ...   (tool_calls dropped)
    {%- endif %}

An empty string falls through to the text branch and silently drops tool_calls
(no loss on tool tokens during SFT). But this rewrite cannot live in
normalize_to_hf_messages because gpt_oss / llama4 Jinja iterate the content
string (e.g. `if "tool_call" in message['content']`) and crash on None with
`TypeError: argument of type 'NoneType' is not iterable`.

Per-family adapter is the correct fix point — see
titaninfra/data/sft/tool_use_adapters/deepseek_v3.py.
"""

from __future__ import annotations

from tests.oracle._legacy.tool_use_adapters.deepseek_v3 import DeepSeekV3Adapter


class TestDeepSeekV3PreprocessMessages:
    def test_assistant_tool_calls_empty_content_rewritten_to_none(self):
        adapter = DeepSeekV3Adapter()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "search", "arguments": '{"q":"x"}'}},
                ],
            }
        ]
        out_msgs, out_tools = adapter.preprocess_messages(msgs, tools=None)
        assert out_msgs[0]["content"] is None
        assert out_msgs[0]["tool_calls"][0]["function"]["name"] == "search"
        assert out_tools is None

    def test_assistant_tool_calls_with_text_content_preserved(self):
        """DSV3 XOR limit: text + tool_calls keeps the text (Jinja will drop tool_calls).

        The adapter does not try to rescue this case here; the alignment-layer
        xfail tracks it separately.
        """
        adapter = DeepSeekV3Adapter()
        msgs = [
            {
                "role": "assistant",
                "content": "Searching...",
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "search", "arguments": '{"q":"x"}'}},
                ],
            }
        ]
        out_msgs, _ = adapter.preprocess_messages(msgs)
        assert out_msgs[0]["content"] == "Searching..."

    def test_assistant_no_tool_calls_unchanged(self):
        adapter = DeepSeekV3Adapter()
        msgs = [{"role": "assistant", "content": "Hello"}]
        out_msgs, _ = adapter.preprocess_messages(msgs)
        assert out_msgs[0]["content"] == "Hello"
        assert "tool_calls" not in out_msgs[0]

    def test_assistant_empty_content_no_tool_calls_unchanged(self):
        """Empty content without tool_calls stays as '' (no spurious None rewrite)."""
        adapter = DeepSeekV3Adapter()
        msgs = [{"role": "assistant", "content": ""}]
        out_msgs, _ = adapter.preprocess_messages(msgs)
        assert out_msgs[0]["content"] == ""

    def test_user_and_tool_messages_untouched(self):
        adapter = DeepSeekV3Adapter()
        msgs = [
            {"role": "user", "content": ""},
            {"role": "tool", "content": "", "tool_call_id": "c1"},
        ]
        out_msgs, _ = adapter.preprocess_messages(msgs)
        assert out_msgs[0]["content"] == ""
        assert out_msgs[1]["content"] == ""

    def test_caller_input_not_mutated(self):
        """Adapter must not mutate the list/dicts passed in."""
        adapter = DeepSeekV3Adapter()
        original_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}},
            ],
        }
        msgs = [original_msg]
        out_msgs, _ = adapter.preprocess_messages(msgs)
        assert original_msg["content"] == ""
        assert out_msgs[0] is not original_msg
        assert out_msgs[0]["content"] is None

    def test_tools_passed_through(self):
        adapter = DeepSeekV3Adapter()
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        msgs = [{"role": "user", "content": "hi"}]
        out_msgs, out_tools = adapter.preprocess_messages(msgs, tools=tools)
        assert out_tools is tools

    def test_mixed_batch_only_rewrites_matching_messages(self):
        adapter = DeepSeekV3Adapter()
        msgs = [
            {"role": "user", "content": "find me x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "search", "arguments": '{"q":"x"}'}},
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "Found it."},
        ]
        out_msgs, _ = adapter.preprocess_messages(msgs)
        assert out_msgs[0]["content"] == "find me x"
        assert out_msgs[1]["content"] is None
        assert out_msgs[2]["content"] == "result"
        assert out_msgs[3]["content"] == "Found it."

    def test_sequential_multi_turn_only_last_tool_call_rewritten(self):
        """DSV3 Jinja breaks on sequential multi-turn tool_calls.

        Only the LAST assistant tool_call in the conversation gets `content=None`
        (so its tool_call branch renders correctly); earlier ones keep `""` and
        fall through the text branch (tool_calls silently dropped). This keeps
        the conversation structurally parseable for boundary computation.
        """
        adapter = DeepSeekV3Adapter()
        tc = lambda cid: [
            {"id": cid, "type": "function",
             "function": {"name": "search", "arguments": '{"q":"x"}'}},
        ]
        msgs = [
            {"role": "user", "content": "do two searches"},
            {"role": "assistant", "content": "", "tool_calls": tc("c1")},
            {"role": "tool", "content": "r1", "tool_call_id": "c1"},
            {"role": "assistant", "content": "", "tool_calls": tc("c2")},
            {"role": "tool", "content": "r2", "tool_call_id": "c2"},
            {"role": "assistant", "content": "Done."},
        ]
        out_msgs, _ = adapter.preprocess_messages(msgs)
        # First tool_call assistant: kept as "" (text branch will drop tool_calls)
        assert out_msgs[1]["content"] == ""
        # Last tool_call assistant: rewritten to None (tool branch renders calls)
        assert out_msgs[3]["content"] is None
        # Final text assistant: untouched
        assert out_msgs[5]["content"] == "Done."

    def test_single_tool_call_still_rewritten(self):
        """Single tool_call episode: that one assistant message gets None."""
        adapter = DeepSeekV3Adapter()
        msgs = [
            {"role": "user", "content": "search x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "search", "arguments": '{"q":"x"}'}},
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "Found."},
        ]
        out_msgs, _ = adapter.preprocess_messages(msgs)
        assert out_msgs[1]["content"] is None
