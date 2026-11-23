"""
Jinja oracle baseline — test-only encoder-drift detector.

This package is the reference renderer used by ``tests/oracle/test_*_parity.py``
to detect drift in titaninfra's single-path SFT encoders. It re-renders each
fixture sample through the legacy HF Jinja path (``apply_chat_template``) and
the parity tests assert that ``dispatch.tokenize_sample`` produces identical
output.

Nothing under ``titaninfra/`` imports from here — it lives under ``tests/`` and
is off the runtime hot path entirely. The companion
``expected_deviations.yaml`` registry catalogs known acceptable divergences.
"""
