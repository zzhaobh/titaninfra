#!/usr/bin/env python3
"""Download all tokenizer fixtures from HuggingFace.

Usage:
    python tests/fixtures/tokenizers/download_all.py

Environment:
    HF_ENDPOINT  — override hub URL (default: https://hf-mirror.com for China)

Each tokenizer is defined by an HF model ID and a list of files to fetch.
Only files that don't exist locally are downloaded; pass --force to re-download all.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

TOKENIZER_REGISTRY: dict[str, dict] = {
    "qwen3": {
        "hf_model_id": "Qwen/Qwen3-8B",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "qwen2_5": {
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "deepseek_v3": {
        "hf_model_id": "deepseek-ai/DeepSeek-V3",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "deepseek_v4": {
        "hf_model_id": "deepseek-ai/DeepSeek-V4",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "llama3": {
        "hf_model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    },
    "llama4": {
        "hf_model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "gpt_oss": {
        "hf_model_id": "openai/gpt-oss-20b",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "gemma4": {
        "hf_model_id": "google/gemma-4-12b-it",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "qwen3_5": {
        # (<|image_pad|>=248056 / <|vision_start|>=248053 / <|vision_end|>=248054)。
        "hf_model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "files": ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
    },
    "moonlight": {
        "hf_model_id": "moonshotai/Moonlight-16B-A3B",
        "files": ["tiktoken.model", "tokenization_moonshot.py", "tokenizer_config.json"],
    },
    "kimi_vl": {
        "hf_model_id": "moonshotai/Kimi-VL-A3B-Instruct",
        "files": ["tiktoken.model", "tokenization_moonshot.py", "tokenizer_config.json"],
    },
}


def download_file(url: str, dest: Path, *, timeout: int = 60) -> bool:
    """Download a single file. Returns True on success, False on 404/error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "titaninfra-test-fixtures/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        print(f"  HTTP {e.code} downloading {url}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  Error downloading {url}: {e}", file=sys.stderr)
        return False


def download_tokenizer(name: str, spec: dict, *, force: bool = False) -> bool:
    """Download one tokenizer fixture. Returns True if all required files exist after."""
    dest_dir = FIXTURES_DIR / name
    model_id = spec["hf_model_id"]
    required_files = spec["files"]

    all_present = all((dest_dir / f).exists() for f in required_files)
    if all_present and not force:
        print(f"  {name}: already present, skipping")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = True

    for filename in required_files:
        dest_path = dest_dir / filename
        if dest_path.exists() and not force:
            continue

        url = f"{HF_ENDPOINT}/{model_id}/resolve/main/{filename}"
        print(f"  {name}/{filename} <- {url}")
        if not download_file(url, dest_path):
            success = False
            print(f"  FAILED: {name}/{filename}")

    return success


def check_missing() -> list[str]:
    """Return names of tokenizers with missing files."""
    missing = []
    for name, spec in TOKENIZER_REGISTRY.items():
        dest_dir = FIXTURES_DIR / name
        for f in spec["files"]:
            if not (dest_dir / f).exists():
                missing.append(name)
                break
    return missing


def download_all(*, force: bool = False) -> tuple[int, int]:
    """Download all tokenizers. Returns (success_count, fail_count)."""
    ok = 0
    fail = 0
    for name, spec in TOKENIZER_REGISTRY.items():
        if download_tokenizer(name, spec, force=force):
            ok += 1
        else:
            fail += 1
    return ok, fail


def main():
    parser = argparse.ArgumentParser(description="Download tokenizer fixtures from HuggingFace")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("names", nargs="*", help="Specific tokenizers to download (default: all)")
    args = parser.parse_args()

    print(f"HF endpoint: {HF_ENDPOINT}")
    print(f"Fixtures dir: {FIXTURES_DIR}")
    print()

    if args.names:
        registry = {k: v for k, v in TOKENIZER_REGISTRY.items() if k in args.names}
        unknown = set(args.names) - set(TOKENIZER_REGISTRY.keys())
        if unknown:
            print(f"Unknown tokenizers: {unknown}", file=sys.stderr)
            print(f"Available: {sorted(TOKENIZER_REGISTRY.keys())}", file=sys.stderr)
            sys.exit(1)
    else:
        registry = TOKENIZER_REGISTRY

    ok, fail = 0, 0
    for name, spec in registry.items():
        if download_tokenizer(name, spec, force=args.force):
            ok += 1
        else:
            fail += 1

    print()
    print(f"Done: {ok} ok, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
