"""Materialize small MDS datasets required by integration tests.

The public branch stores only raw fixtures. Generated MDS lives under the
ignored data/ directory and is created with the same conversion APIs users run.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "datasets"
TOKENIZER = REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "qwen3"


def ensure_test_mds() -> None:
    """Generate missing MDS datasets when runtime dependencies are available."""
    if importlib.util.find_spec("streaming") is None or importlib.util.find_spec("pydantic") is None:
        return

    from titaninfra.data.converter import DataConvertConfig, DataConverter, convert_messages_to_mds

    jobs = [
        ("aime/pretrain/train/aime.jsonl", "aime/mds/train", "text"),
        ("aime/pretrain/eval/aime.jsonl", "aime/mds/eval", "text"),
        ("aime/sft/train/aime.jsonl", "aime/mds_sft/train", "messages"),
        ("aime/sft/eval/aime.jsonl", "aime/mds_sft/eval", "messages"),
        ("sft_test/basic/train.jsonl", "sft_test/basic/mds/train", "messages"),
        ("sft_test/basic/eval.jsonl", "sft_test/basic/mds/eval", "messages"),
    ]
    for relative, output, data_type in jobs:
        source = FIXTURES / relative
        destination = REPO_ROOT / "data" / output
        if (destination / "index.json").exists():
            continue
        if data_type == "messages":
            convert_messages_to_mds(
                source=str(source),
                output=str(destination),
                tokenizer_path=str(TOKENIZER),
                tokenizer_name="qwen3",
                family="qwen3",
                num_shards=2,
                num_workers=1,
                overwrite=True,
            )
        else:
            config = DataConvertConfig(
                source=str(source),
                output=str(destination),
                tokenizer_path=str(TOKENIZER),
                tokenizer_name="qwen3",
                text_key="text",
                num_shards=2,
                num_workers=1,
                overwrite=True,
            )
            DataConverter(config).convert()

    # Build a tiny synthetic Lance fixture without network access when the
    # optional multimodal test dependency is installed. The generated directory
    # is ignored and never becomes a public release asset.
    lance_fixture = REPO_ROOT / "tests" / "fixtures" / "multimodal" / "datatrove_sample.lance"
    if not lance_fixture.exists() and importlib.util.find_spec("lance") is not None:
        from tests.fixtures.multimodal.build_fixture import build

        build(n_caption=0, n_interleaved=0)
