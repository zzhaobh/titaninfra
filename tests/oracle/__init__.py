"""Oracle harness tests — fixture-driven parity between native and jinja paths.

M1.1 ships:

* ``_identity_encoder.IdentityEncoder`` — test-only encoder that
  round-trips ``NormalizedSample`` through ``tokenize_with_hf_template``,
  used to wire up the harness before real native encoders land.
* Fixture loader in ``conftest.py``.
* First Qwen3 single-turn plain-text fixture under
  ``tests/fixtures/sft/qwen3/``.

Oracle code (``compare_encoded``, ``_jinja_baseline``) lives in
``tests.oracle._jinja_oracle`` — a test-only module-private package.
"""
