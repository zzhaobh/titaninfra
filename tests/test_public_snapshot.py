"""Executable checks for the public open-source snapshot contract."""

import ast
import io
import json
import re
import subprocess
import tokenize
import xml.etree.ElementTree as ET
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PUBLIC_DOCS = (
    "README.md",
    "README.zh-CN.md",
    "CONTRIBUTING.md",
    "docs/en/api.md",
    "docs/en/config.md",
    "docs/en/data.md",
    "docs/en/training.md",
    "docs/en/development.md",
    "docs/zh-CN/api.md",
    "docs/zh-CN/config.md",
    "docs/zh-CN/data.md",
    "docs/zh-CN/training.md",
    "docs/zh-CN/development.md",
)

FORBIDDEN_PUBLIC_PATHS = (
    "CLAUDE.md",
    ".claude",
    "docs/superpowers",
    "experiments",
    "amp",
    "data",
    "env_scripts",
)


def test_required_public_docs_exist() -> None:
    """A public reader receives every document in the bilingual release set."""
    missing = [path for path in REQUIRED_PUBLIC_DOCS if not (REPO_ROOT / path).is_file()]
    assert not missing, f"missing public documents: {missing}"


def test_forbidden_public_paths_absent() -> None:
    """Internal agent notes, experiments, and root datasets stay out of the release."""
    tracked = _tracked_files()
    present = [
        path
        for path in FORBIDDEN_PUBLIC_PATHS
        if path in tracked or any(relative.startswith(path + "/") for relative in tracked)
    ]
    assert not present, f"internal paths still tracked: {present}"


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [path for path in result.stdout.splitlines() if path]


def test_no_forbidden_release_content() -> None:
    """Private workstation/cluster details cannot leak into the public snapshot."""
    forbidden = re.compile(
        r"/Users/|/mnt/workspace/|root@|session-[0-9a-f]{8}|your-org|"
        r"\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b|"
        r"\b192\.168\.[0-9]{1,3}\.[0-9]{1,3}\b",
        re.IGNORECASE,
    )
    findings = []
    for relative in _tracked_files():
        if relative == "tests/test_public_snapshot.py":
            continue
        path = REPO_ROOT / relative
        if not path.is_file() or path.suffix.lower() not in {
            ".py", ".md", ".txt", ".toml", ".sh", ".json", ".jsonl", ".yml", ".yaml",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if forbidden.search(line):
                findings.append(f"{relative}:{line_number}: {line.strip()[:160]}")
    assert not findings, "private release residue:\n" + "\n".join(findings[:20])


def test_small_dataset_fixtures_exist() -> None:
    """Dataset cleanup preserves the exact small inputs used by behavioral tests."""
    fixtures = REPO_ROOT / "tests" / "fixtures" / "datasets"
    fsb = fixtures / "fsb.json"
    math500 = fixtures / "math500.jsonl"
    math_tools = fixtures / "math_tools_multiturn.jsonl"
    assert len(fsb.read_text(encoding="utf-8").splitlines()) == 10
    assert len(math500.read_text(encoding="utf-8").splitlines()) == 500
    record = json.loads(math_tools.read_text(encoding="utf-8").strip())
    assert sum(message.get("role") == "assistant" for message in record["messages"]) >= 3


def _literal_all_assignment(module: Path) -> list[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"__all__ not found in {module}")


def test_no_verl_or_legacy_rl_api() -> None:
    """The release exposes no VERL dependency or unsupported legacy RL API."""
    for module in (REPO_ROOT / "titaninfra" / "__init__.py", REPO_ROOT / "titaninfra" / "config" / "__init__.py"):
        exports = _literal_all_assignment(module)
        assert "RLConfig" not in exports
        assert "RLDataConfig" not in exports
    for relative in (
        "titaninfra/engine/rl.py",
        "titaninfra/data/sft/rl_assembler.py",
        "titaninfra/data/sft/rl_prompt.py",
    ):
        assert not (REPO_ROOT / relative).exists(), relative

    findings = []
    for relative in _tracked_files():
        if not relative.startswith(("titaninfra/", "docs/")) and relative not in {
            "pyproject.toml", "requirements.txt", "requirements-dev.txt", "THIRD_PARTY_NOTICES.md",
        }:
            continue
        path = REPO_ROOT / relative
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt", ".toml"}:
            text = path.read_text(encoding="utf-8")
            if re.search(r"\bverl\b", text, re.IGNORECASE):
                findings.append(relative)
    assert not findings, f"VERL references remain: {findings}"


def test_standard_docker_runtime() -> None:
    """The public Dockerfile reproduces the pinned CPU/GPU runtime."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04" in text
    assert "python3-venv" in text
    assert "692c9bb5b26f606e5fb739acd55effb6f8850bc9" in text
    assert 'pip install --no-deps -e ".[data]"' in text
    assert "requirements.txt" in text
    assert "mosaicml/streaming.git" in text
    assert "python -m pip install --no-build-isolation --no-deps ." in text
    assert "1381ca1213fc95b6573efb44e5ec9f85c765ce1c" in text
    dockerfiles = [relative for relative in _tracked_files() if Path(relative).name.lower() == "dockerfile"]
    assert dockerfiles == ["Dockerfile"]
    assert (REPO_ROOT / ".dockerignore").is_file()


def test_bsd_license_metadata() -> None:
    """The public package consistently uses BSD 3-Clause like TorchTitan."""
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("BSD 3-Clause License")
    assert "TitanInfra Contributors" in license_text
    for clause in (
        "Redistributions of source code must retain",
        "Redistributions in binary form must reproduce",
        "Neither the name of the copyright holder",
    ):
        assert clause in license_text

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "BSD-3-Clause"
    assert not any("MIT License" in classifier for classifier in project["classifiers"])

    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "TorchTitan (BSD 3-Clause)" in notices
    assert "BSD 3-Clause" in notices
    assert "verl" not in notices.lower()


def _public_markdown_files() -> list[Path]:
    return sorted(REPO_ROOT.glob("*.md")) + sorted((REPO_ROOT / "docs").rglob("*.md"))


def _markdown_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)


def test_bilingual_doc_pairs_exist() -> None:
    """Every English guide has a Chinese counterpart with the same filename."""
    english = {path.name for path in (REPO_ROOT / "docs" / "en").glob("*.md")}
    chinese = {path.name for path in (REPO_ROOT / "docs" / "zh-CN").glob("*.md")}
    assert english == {"api.md", "config.md", "data.md", "training.md", "development.md"}
    assert chinese == english
    assert (REPO_ROOT / "README.md").is_file()
    assert (REPO_ROOT / "README.zh-CN.md").is_file()


def _heading_slug(heading: str) -> str:
    """Approximate GitHub's heading anchors, including Chinese headings."""
    text = re.sub(r"[`*_~]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- \u3400-\u9fff]", "", text)
    return text.replace(" ", "-")


def test_markdown_links_resolve() -> None:
    """Local documentation links and their optional heading anchors resolve."""
    broken = []
    for document in _public_markdown_files():
        for link in _markdown_links(document):
            clean = link.strip()
            target, _, anchor = clean.partition("#")
            target = target.split(" ", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(REPO_ROOT)} -> {link}")
                continue
            if anchor and resolved.suffix == ".md":
                headings = {
                    _heading_slug(match.group(1))
                    for match in re.finditer(r"^#{1,6}\s+(.+?)\s*$", resolved.read_text(encoding="utf-8"), re.MULTILINE)
                }
                if anchor not in headings:
                    broken.append(f"{document.relative_to(REPO_ROOT)} -> missing anchor {link}")
    assert not broken, "broken local Markdown links:\n" + "\n".join(broken)


def test_readme_release_focus() -> None:
    """Both READMEs lead with the data pipeline and complete recipes."""
    for relative, data_guide in (("README.md", "docs/en/data.md"), ("README.zh-CN.md", "docs/zh-CN/data.md")):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for target in ("examples/api_tour.py", "examples/pretrain.py", "examples/sft.py", data_guide, "LICENSE"):
            assert target in text, f"{relative} does not link {target}"
        for topic in ("proportion", "repeat", "choose", "pack", "stream_stats.json"):
            assert topic in text, f"{relative} lacks {topic}"


def test_no_removed_doc_links() -> None:
    """Navigation never sends public readers to removed internal material."""
    forbidden = re.compile(r"docs/superpowers|CLAUDE\.md|\.claude/|docs/research", re.IGNORECASE)
    findings = []
    for document in _public_markdown_files():
        for link in _markdown_links(document):
            if forbidden.search(link):
                findings.append(f"{document.relative_to(REPO_ROOT)} -> {link}")
    assert not findings, findings


def test_brand_assets() -> None:
    """The local vector logo and image-generation prompt express full-stack LM development."""
    logo = REPO_ROOT / "docs" / "assets" / "titaninfra-logo.svg"
    ET.parse(logo)
    for relative in ("README.md", "README.zh-CN.md"):
        assert "docs/assets/titaninfra-logo.svg" in (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_python_comments_and_docstrings_are_english() -> None:
    """Public Python comments and docstrings contain no Chinese characters."""
    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    findings = []
    for relative in _tracked_files():
        if not relative.endswith(".py") or not relative.startswith(("titaninfra/", "tests/", "scripts/", "examples/")):
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
            tree = ast.parse(text)
        except (tokenize.TokenError, SyntaxError):
            continue
        for token in tokens:
            if token.type == tokenize.COMMENT and cjk.search(token.string):
                findings.append(f"{relative}:{token.start[0]}: {token.string.strip()[:100]}")
        for node in ast.walk(tree):
            # Covers module/class/function docstrings and standalone attribute
            # documentation strings after dataclass/Pydantic field declarations.
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and cjk.search(node.value.value)
            ):
                findings.append(f"{relative}:{node.lineno}: documentation string contains CJK")
    assert not findings, "non-English Python comments/docstrings:\n" + "\n".join(findings[:30])
