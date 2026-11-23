"""CPU regression for the offline VLM SFT converter path.

Covers the merge-blocker fix: ``convert_messages_to_mds`` must thread
``resolution_policy`` + ``spatial_merge_size`` into the tokenizer worker so
that a ``family="qwen3_5"`` messages stream carrying an ImageBlock expands
vision tokens offline (byte-identical to the runtime ``tokenize_sample``
result) instead of raising ``ValueError("IMAGE segment present but
resolution_policy is None")``.

Everything Mac may lack (PIL, the qwen3_5 fixture, streaming/MDS deps) is
guarded — the test SKIPs rather than errors when a dependency is missing.
"""

from __future__ import annotations

import base64
import io
import json
import os

import pytest

QWEN3_5_TOKENIZER = os.path.join(
    os.path.dirname(__file__), "fixtures", "tokenizers", "qwen3_5"
)


def _b64_image(color=(220, 20, 60)) -> str:
    from PIL import Image

    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_convert_messages_to_mds_vlm_qwen3_5(tmp_path):
    # ── Skip cleanly when any Mac-optional dependency is missing ──
    try:
        from PIL import Image  # noqa: F401

        from titaninfra.config.data import ResolutionPolicy
        from titaninfra.data.converter import convert_messages_to_mds
        from titaninfra.data.schema import StreamStats
    except Exception as exc:  # ImportError or transitive dep failures
        pytest.skip(f"VLM SFT converter deps unavailable: {exc}")

    if not os.path.isdir(QWEN3_5_TOKENIZER):
        pytest.skip("qwen3_5 tokenizer fixture not present")

    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(QWEN3_5_TOKENIZER, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"qwen3_5 tokenizer failed to load: {exc}")

    # ── Tiny synth JSONL: one user turn with a base64 image + text ──
    try:
        img_b64 = _b64_image()
    except Exception as exc:
        pytest.skip(f"PIL image synth failed: {exc}")

    rows = []
    for i in range(8):
        rows.append({"messages": [
            {"role": "user", "blocks": [
                {"type": "image", "source": {"type": "base64", "data": img_b64}},
                {"type": "text", "text": "What color is this image?"},
            ]},
            {"role": "assistant", "blocks": [
                {"type": "text", "text": "The image is a solid color block."},
            ]},
        ]})

    src_jsonl = tmp_path / "vlm_sft_src.jsonl"
    with open(src_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_dir = tmp_path / "mds_out"

    # ── Must NOT raise (pre-fix this raised ValueError on the image seg) ──
    stats = convert_messages_to_mds(
        str(src_jsonl),
        str(out_dir),
        tokenizer_path=QWEN3_5_TOKENIZER,
        family="qwen3_5",
        stream_id="vlm_sft",
        num_shards=2,
        resolution_policy=ResolutionPolicy(min_pixels=256, max_pixels=1048576),
        spatial_merge_size=2,
    )

    assert stats["num_samples"] > 0, stats
    assert stats["total_tokens"] > 0, stats

    # ── stream_stats.json must record positive qwen3_5 tokens (image_pad
    #    tokens counted → proves the expansion ran offline) ──
    loaded = StreamStats.load(str(out_dir))
    assert loaded is not None, "stream_stats.json missing / unloadable"
    total, avg = loaded.resolve_tokens("qwen3_5")
    assert total > 0, f"expected positive qwen3_5 total_tokens, got {total}"
