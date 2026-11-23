# Development and Testing

[简体中文](../zh-CN/development.md) · [README](../../README.md)

## Environment

Use Python 3.12 and the verified PyTorch 2.15 baseline. Do not replace the pinned TorchTitan engine through implicit dependency resolution.

```bash
git clone https://github.com/zzhaobh/titaninfra.git
cd titaninfra
git checkout open-source
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core and development dependencies.
python -m pip install -r requirements-dev.txt

# Compatible public TorchTitan runtime.
git clone https://github.com/pytorch/torchtitan.git ../torchtitan
git -C ../torchtitan checkout 692c9bb5b26f606e5fb739acd55effb6f8850bc9
python -m pip install --no-deps -e ../torchtitan

# Install MDS streaming from source to avoid its legacy Transformers<5 cap.
git clone --depth 1 https://github.com/mosaicml/streaming.git ../streaming
(cd ../streaming && git fetch --depth 1 origin 1381ca1213fc95b6573efb44e5ec9f85c765ce1c \
  && git checkout --detach FETCH_HEAD \
  && python -m pip install --no-build-isolation --no-deps .)

# TitanInfra itself.
python -m pip install --no-deps -e ".[data,dev]"
```

For CPU-only lint/release checks, a lightweight virtual environment with `pytest` is enough. Runtime tests require the dependencies above.

## Test fixtures

Download tokenizer assets before running the full suite:

```bash
python tests/fixtures/tokenizers/download_all.py
```

These files are generated test inputs and are intentionally ignored by Git. Some upstream assets require access approval or may be unavailable. Small FSB, MATH500, AIME/basic, and multi-turn fixtures are tracked under `tests/fixtures/datasets/`. During pytest startup, `tests/_fixture_mds.py` uses the public conversion APIs to materialize their MDS form under the ignored `data/` directory when runtime dependencies are available.

## CPU tests

```bash
pytest tests/ -v -n auto -k "not roundtrip" \
    --ignore=tests/alignment \
    --basetemp=$TMPDIR/titaninfra-pytest
```

Run focused tests while developing:

```bash
pytest tests/test_public_snapshot.py tests/test_public_examples.py -v
pytest tests/test_basic.py tests/test_trainer.py tests/test_config.py -v
python -m py_compile examples/pretrain.py examples/sft.py
```

The public-example tests validate configurations only; they do not start training. Roundtrip, multi-GPU, external-weight, and some tokenizer tests require additional hardware, assets, or network access and must not be reported as self-contained CPU coverage.

## Static and release checks

```bash
git diff --check
ruff check titaninfra/ tests/ scripts/ examples/
python -m build
bash scripts/verify_install.sh
```

`tests/test_public_snapshot.py` checks required documents, local Markdown links, internal-path cleanup, private residue, BSD metadata, and removal of the legacy RL public API. `tests/test_public_examples.py` validates the complete Pretrain/SFT configurations against the real models and scheduling validators.

## Documentation changes

Update both languages in the same change:

- `README.md` ↔ `README.zh-CN.md`
- `docs/en/api.md` ↔ `docs/zh-CN/api.md`
- `docs/en/data.md` ↔ `docs/zh-CN/data.md`
- `docs/en/training.md` ↔ `docs/zh-CN/training.md`
- `docs/en/development.md` ↔ `docs/zh-CN/development.md`

Keep examples executable and consistent with current source. Python comments and docstrings use English.

## Before submitting

1. Review the final diff and confirm no private path, credential, generated dataset, checkpoint, or agent-specific file is included.
2. Run relevant behavioral tests and the release checks.
3. Build the wheel/sdist and verify the public import.
4. Document Python/PyTorch/TorchTitan revisions and any unverified hardware/model combinations.
5. Preserve BSD 3-Clause and third-party notices.

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the pull-request workflow.
