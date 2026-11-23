#!/usr/bin/env bash
set -euo pipefail

# Verify titaninfra installs correctly from wheel (non-editable).
# Run this BEFORE pushing any packaging changes.
#
# Usage:
#   bash scripts/verify_install.sh          # full check (build + install + import)
#   bash scripts/verify_install.sh --quick  # import-only (assumes already installed)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

if [[ "${1:-}" != "--quick" ]]; then
    echo "=== Building wheel ==="
    cd "$PROJECT_DIR"
    rm -rf build/ dist/
    "${PYTHON_BIN:-python3}" -m build --wheel --quiet 2>&1 || fail "python -m build failed"
    pass "Wheel built"

    WHEEL=$(ls dist/*.whl 2>/dev/null | head -1)
    [[ -n "$WHEEL" ]] || fail "No wheel found in dist/"

    echo ""
    echo "=== Wheel contents (subpackages) ==="
    "${PYTHON_BIN:-python3}" -c "
import zipfile, sys, re
whl = zipfile.ZipFile(sys.argv[1])
pkgs = sorted({re.match(r'([^/]+/[^/]+)/', n).group(1)
               for n in whl.namelist()
               if re.match(r'titaninfra/[^/]+/__init__\.py', n)})
print(f'  Found {len(pkgs)} subpackages:')
for p in pkgs:
    print(f'    {p}')
if len(pkgs) < 10:
    print(f'  WARNING: expected ≥10 subpackages, got {len(pkgs)}', file=sys.stderr)
    sys.exit(1)
" "$WHEEL" || fail "Wheel missing subpackages"
    pass "Wheel subpackage count OK"

    echo ""
    echo "=== Installing in clean venv ==="
    VENV_DIR=$(mktemp -d)
    trap "rm -rf $VENV_DIR" EXIT
    "${PYTHON_BIN:-python3}" -m venv --system-site-packages "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --no-deps "$WHEEL" 2>&1 || fail "pip install wheel failed"
    pass "Wheel installed"
    pass "Wheel installed"

    PYTHON="$VENV_DIR/bin/python"
else
    PYTHON="${PYTHON_BIN:-python3}"
fi

echo ""
echo "=== Import smoke tests ==="

$PYTHON -c "from titaninfra.config import ModelConfig, TrainConfig, StreamConfig" \
    || fail "titaninfra.config import failed"
pass "titaninfra.config"

$PYTHON -c "from titaninfra.config.data import PackStrategy, StreamConfig" \
    || fail "titaninfra.config.data import failed"
pass "titaninfra.config.data"

$PYTHON -c "from titaninfra.data.schema import SFTSample, StreamStats" \
    || fail "titaninfra.data.schema import failed"
pass "titaninfra.data.schema"

$PYTHON -c "from titaninfra.data.templates import get_template, list_templates" \
    || fail "titaninfra.data.templates import failed"
pass "titaninfra.data.templates"

$PYTHON -c "from titaninfra.tools import compare_losses, diagnose_hf_checkpoint" \
    || fail "titaninfra.tools import failed"
pass "titaninfra.tools"

$PYTHON -c "from titaninfra.data.loss import resolve_weight" \
    || fail "titaninfra.data.loss import failed"
pass "titaninfra.data.loss"

$PYTHON -c "from titaninfra.models import _SUPPORTED_FAMILIES" \
    || fail "titaninfra.models import failed"
pass "titaninfra.models"

# Eager-chain purity: `import titaninfra` must NOT transitively pull in any optional
# heavy dependency. Production installs with `pip install --no-deps`, so a stray
# top-level `import transformers` in an eager module silently breaks `import titaninfra`
# in the field. This guard fails the build the moment that regression lands.
$PYTHON -c "
import sys
HEAVY = {'transformers', 'datasets', 'pandas', 'pyarrow', 'vllm', 'sglang',
         'omegaconf', 'hydra', 'boto3', 'streaming', 'wandb'}
before = set(sys.modules)
import titaninfra  # noqa: F401
leaked = sorted(m for m in (set(sys.modules) - before)
                if m.split('.')[0] in HEAVY)
if leaked:
    print('  eager import leaked optional deps: ' + ', '.join(leaked), file=sys.stderr)
    sys.exit(1)
print('  eager chain pulls no optional heavy deps')
" || fail "eager-chain purity check failed (an eager module top-level-imports an optional dep)"
pass "Eager-chain purity"

$PYTHON -c "
from importlib.metadata import version
v = version('titaninfra')
assert v, 'version is empty'
print(f'  version={v}')
" || fail "version metadata missing"
pass "Version metadata"

echo ""
echo -e "${GREEN}All checks passed.${NC}"
