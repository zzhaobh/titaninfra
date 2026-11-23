"""Test-only legacy adapters preserved for the Jinja oracle path.

PR-C (2026-05-28) deleted production gemma4 / future-family tool-use adapters
from ``titaninfra/data/sft/tool_use_adapters/`` once the native encoder became the
source of truth. The Jinja path is retained ONLY as an oracle for parity tests.

Tool-use fixtures (e.g. gemma4 06/08) still require the family-specific boundary
logic to render via Jinja — without it, ``tokenize_with_hf_template`` falls back
to ``DefaultAdapter`` and fails fail-loud on continue-same-turn assistants. To
keep these fixtures usable as oracles, the adapters live here and are registered
into the production ``_REGISTRY`` only for the duration of oracle tests (see
the ``gemma4_oracle_adapter_registered`` fixture in ``tests/oracle/conftest.py``).

The production runtime never imports this package.
"""
