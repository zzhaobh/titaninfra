"""Renderers oracle baseline — test-only, independent of the Jinja oracle.

This package validates ``tokenize_sample`` (titaninfra's native SFT encoder
dispatch) against the `renderers <https://pypi.org/project/renderers/>`_
PyPI package as a **second, independent** reference implementation, parallel
to (and never replacing) ``tests/oracle/_jinja_oracle/``. ``renderers`` ships
hand-coded Python renderers that mirror each model family's chat template —
no Jinja involved on either the production or the oracle side for families
covered here.

Scope (2026-11, prototype round): **qwen3 only**. See
``docs/renderers_oracle_feasibility_report.md`` for the full investigation —
which of the existing ``test_qwen3_parity.py`` scenarios ``renderers``
reproduces exactly, which show a *known, documented* divergence (not a bug
in either system), and which ``renderers`` cannot express at all.

Nothing under ``titaninfra/`` imports from here — this lives entirely under
``tests/`` and never runs on any production hot path.
"""
