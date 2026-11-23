"""Immutable chat-template data structures and built-in definitions.

Role stores role prefix/suffix values. ChatTemplate contains only declarative
formatting data; formatting behavior lives in separate functions.
"""

from dataclasses import dataclass

# Matches the ignore label used by Transformers and TorchTitan.
IGNORE_INDEX = -100


# Data structures.

@dataclass(frozen=True)
class Role:
    """Prefix and suffix strings that wrap one message role."""
    prefix: str          # Role prefix
    suffix: str          # Role suffix


@dataclass(frozen=True)
class ChatTemplate:
    """Declarative configuration for formatting a message sequence.

    Fields define role wrappers, optional BOS/EOS, default system content,
    stop words, thinking tags, tool/image roles, and special-token prefixes.
    """
    name: str
    system_role: Role
    user_role: Role
    assistant_role: Role
    bos_token: str = ""
    eos_token: str = ""
    sep: str = ""
    default_system: str = ""
    stop_words: tuple[str, ...] = ()
    thought_words: tuple[str, str] | None = None  # (start_tag, end_tag)
    enable_thinking: bool = True  # Include reasoning content in loss
    mask_empty_thinking: bool = False  # Mask loss on empty Jinja thinking tags
    # Optional special-token roles and prefixes
    tool_role: Role | None = None
    image_role: Role | None = None
    tool_call_prefix: str = ""
    tool_call_words: tuple[str, str] | None = None
    image_token: str = ""


# Built-in templates, starting with ChatML.
_CHATML = ChatTemplate(
    name="chatml",
    system_role=Role(
        prefix="<|im_start|>system\n",
        suffix="<|im_end|>\n",
    ),
    user_role=Role(
        prefix="<|im_start|>user\n",
        suffix="<|im_end|>\n",
    ),
    assistant_role=Role(
        prefix="<|im_start|>assistant\n",
        suffix="<|im_end|>\n",
    ),
    eos_token="<|im_end|>",
    stop_words=("<|im_end|>", ""),
)

# Qwen3: ChatML plus reasoning support.
_QWEN3 = ChatTemplate(
    name="qwen3",
    system_role=_CHATML.system_role,
    user_role=_CHATML.user_role,
    assistant_role=_CHATML.assistant_role,
    eos_token=_CHATML.eos_token,
    stop_words=_CHATML.stop_words,
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    thought_words=("<think>\n", "\n</think>\n\n"),
    enable_thinking=True,
    mask_empty_thinking=True,
    tool_call_words=("<tool_call>\n", "\n</tool_call>"),
)

# Qwen2.5: ChatML with a default system prompt and no reasoning tags.
_QWEN2_5 = ChatTemplate(
    name="qwen2_5",
    system_role=_CHATML.system_role,
    user_role=_CHATML.user_role,
    assistant_role=_CHATML.assistant_role,
    eos_token=_CHATML.eos_token,
    stop_words=_CHATML.stop_words,
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    thought_words=None,
    enable_thinking=False,
)

# Llama 2 format.
_LLAMA2 = ChatTemplate(
    name="llama2",
    system_role=Role(
        prefix="<<SYS>>\n",
        suffix="\n<</SYS>>\n\n",
    ),
    user_role=Role(
        prefix="[INST] ",
        suffix=" ",
    ),
    assistant_role=Role(
        prefix="",
        suffix=" </s>",
    ),
    bos_token="<s>",
    eos_token="</s>",
    stop_words=("</s>",),
)

# Llama 3 format.
_LLAMA3 = ChatTemplate(
    name="llama3",
    system_role=Role(
        prefix="<|start_header_id|>system<|end_header_id|>\n\n",
        suffix="<|eot_id|>",
    ),
    user_role=Role(
        prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        suffix="<|eot_id|>",
    ),
    assistant_role=Role(
        prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        suffix="<|eot_id|>",
    ),
    bos_token="<|begin_of_text|>",
    eos_token="<|eot_id|>",
    stop_words=("<|eot_id|>", "<|end_of_text|>"),
    default_system="You are a helpful assistant.",
)

# Mistral v0.1 Instruct format.
_MISTRAL = ChatTemplate(
    name="mistral",
    system_role=Role(
        prefix="",  # Mistral v0.1 merges system content into user.
        suffix="",
    ),
    user_role=Role(
        prefix="[INST] ",
        suffix=" [/INST]",
    ),
    assistant_role=Role(
        prefix=" ",
        suffix="</s>",
    ),
    bos_token="<s>",
    eos_token="</s>",
    stop_words=("</s>",),
)

# DeepSeek R1 format.
_DEEPSEEK_R1 = ChatTemplate(
    name="deepseek_r1",
    system_role=Role(
        prefix="<｜System｜>",
        suffix="\n",
    ),
    user_role=Role(
        prefix="<｜User｜>",
        suffix="\n",
    ),
    assistant_role=Role(
        prefix="<｜Assistant｜>",
        suffix="<｜end▁of▁sentence｜>",
    ),
    bos_token="<｜begin▁of▁sentence｜>",
    eos_token="<｜end▁of▁sentence｜>",
    stop_words=("<｜end▁of▁sentence｜>",),
)

# DeepSeek R1 reasoning template.
_DEEPSEEK_R1_REASONING = ChatTemplate(
    name="deepseek_r1_reasoning",
    system_role=Role(
        prefix="<｜System｜>",
        suffix="\n",
    ),
    user_role=Role(
        prefix="<｜User｜>",
        suffix="\n",
    ),
    assistant_role=Role(
        prefix="<｜Assistant｜>",
        suffix="<｜end▁of▁sentence｜>",
    ),
    bos_token="<｜begin▁of▁sentence｜>",
    eos_token="<｜end▁of▁sentence｜>",
    stop_words=("<｜end▁of▁sentence｜>",),
    thought_words=("<｜begin of think｜>\n", "\n<｜end of think｜>\n\n"),
    enable_thinking=True,
)

# Plain format for unstructured text continuation.
_PLAIN = ChatTemplate(
    name="plain",
    system_role=Role(prefix="", suffix="\n"),
    user_role=Role(prefix="", suffix=""),
    assistant_role=Role(prefix="", suffix=""),
)

# Gemma4: asymmetric turns, channel thinking, and declaration tools.
_GEMMA4 = ChatTemplate(
    name="gemma4",
    system_role=Role(prefix="<|turn>system\n", suffix="<turn|>\n"),
    user_role=Role(prefix="<|turn>user\n", suffix="<turn|>\n"),
    assistant_role=Role(prefix="<|turn>model\n", suffix="<turn|>\n"),
    eos_token="<turn|>",
    stop_words=("<turn|>",),
    thought_words=("<|channel>thought\n", "\n<channel|>"),
    enable_thinking=True,
    tool_call_words=("<|tool_call>", "<tool_call|>"),
)

# GPT-OSS: channel-based assistant output and TypeScript tool declarations.
_GPT_OSS = ChatTemplate(
    name="gpt_oss",
    system_role=Role(prefix="<|start|>system<|message|>", suffix="<|end|>"),
    user_role=Role(prefix="<|start|>user<|message|>", suffix="<|end|>"),
    assistant_role=Role(
        prefix="<|start|>assistant<|channel|>final<|message|>",
        suffix="<|end|>",
    ),
    eos_token="<|end|>",
    stop_words=("<|end|>", "<|return|>", "<|call|>"),
    thought_words=("<|channel|>analysis<|message|>", "<|end|>"),
    enable_thinking=True,
)


# DeepSeek V4 encoder path; Jinja role/block fields are unused.
_DEEPSEEK_V4 = ChatTemplate(
    name="deepseek_v4",
    system_role=Role(prefix="", suffix=""),
    user_role=Role(prefix="<｜User｜>", suffix=""),
    assistant_role=Role(prefix="<｜Assistant｜>", suffix="<｜end▁of▁sentence｜>"),
    bos_token="<｜begin▁of▁sentence｜>",
    eos_token="<｜end▁of▁sentence｜>",
    stop_words=("<｜end▁of▁sentence｜>",),
    thought_words=("<think>", "</think>"),
    enable_thinking=True,
)


# Built-in template instances exported to the registry.

# All builtin template instances
BUILTIN_TEMPLATES = [
    _CHATML,
    _QWEN3,
    _QWEN2_5,
    _LLAMA2,
    _LLAMA3,
    _MISTRAL,
    _DEEPSEEK_R1,
    _DEEPSEEK_R1_REASONING,
    _PLAIN,
    _GEMMA4,
    _GPT_OSS,
    _DEEPSEEK_V4,
]
