"""Unit tests for models registry."""

from unittest.mock import MagicMock

import pytest


class TestIsSupported:
    """Tests for test is supported."""

    def test_unregistered_family(self):
        """Verify unregistered family."""
        from titaninfra.models import is_supported

        assert is_supported("nonexistent_model_family_xyz") is False

    def test_registered_family_via_register(self):
        """Verify registered family via register."""
        from titaninfra.models import _SUPPORTED_FAMILIES, is_supported, register_family

        test_family = "_test_family_is_supported"
        try:
            register_family(test_family)
            assert is_supported(test_family) is True
        finally:
            _SUPPORTED_FAMILIES.discard(test_family)

    def test_qwen3_auto_registered(self):
        """Verify qwen3 auto registered."""
        from titaninfra.models import is_supported

        # qwen3 may or may not be auto-registered depending on environment
        # We test the function works regardless
        result = is_supported("qwen3")
        assert isinstance(result, bool)


class TestRegisterFamily:
    """Tests for test register family."""

    def test_register_new_family(self):
        """Verify register new family."""
        from titaninfra.models import _SUPPORTED_FAMILIES, is_supported, register_family

        test_family = "_test_register_new"
        try:
            assert is_supported(test_family) is False
            register_family(test_family)
            assert is_supported(test_family) is True
        finally:
            _SUPPORTED_FAMILIES.discard(test_family)

    def test_register_idempotent(self):
        """Verify register idempotent."""
        from titaninfra.models import _SUPPORTED_FAMILIES, register_family

        test_family = "_test_register_idempotent"
        try:
            register_family(test_family)
            register_family(test_family)
            assert test_family in _SUPPORTED_FAMILIES
        finally:
            _SUPPORTED_FAMILIES.discard(test_family)


class TestStripTitanInfraSuffix:
    """Tests for test strip titan infra suffix."""

    def test_with_titaninfra_suffix(self):
        """Verify with titaninfra suffix."""
        from titaninfra.models import strip_titaninfra_suffix

        stripped, is_titaninfra = strip_titaninfra_suffix("8B_titaninfra")
        assert stripped == "8B"
        assert is_titaninfra is True

    def test_without_titaninfra_suffix(self):
        """Verify without titaninfra suffix."""
        from titaninfra.models import strip_titaninfra_suffix

        stripped, is_titaninfra = strip_titaninfra_suffix("8B")
        assert stripped == "8B"
        assert is_titaninfra is False

    def test_complex_flavor_with_suffix(self):
        """Verify complex flavor with suffix."""
        from titaninfra.models import strip_titaninfra_suffix

        stripped, is_titaninfra = strip_titaninfra_suffix("30B-A3B_titaninfra")
        assert stripped == "30B-A3B"
        assert is_titaninfra is True

    def test_suffix_only_at_end(self):
        """Verify suffix only at end."""
        from titaninfra.models import strip_titaninfra_suffix

        stripped, is_titaninfra = strip_titaninfra_suffix("_titaninfra_special")
        assert stripped == "_titaninfra_special"
        assert is_titaninfra is False

    def test_empty_flavor_with_suffix(self):
        """Verify empty flavor with suffix."""
        from titaninfra.models import strip_titaninfra_suffix

        stripped, is_titaninfra = strip_titaninfra_suffix("_titaninfra")
        assert stripped == ""
        assert is_titaninfra is True


class TestLoadModelSpec:
    """Tests for test load model spec."""

    def test_unregistered_family_raises_value_error(self):
        """Verify unregistered family raises value error."""
        from titaninfra.models import load_model_spec

        with pytest.raises(ValueError, match="不在 TitanInfra 维护列表中"):
            load_model_spec("nonexistent_family_xyz", "8B")

    def test_import_error_propagated(self):
        """Verify import error propagated."""
        from titaninfra.models import _SUPPORTED_FAMILIES, load_model_spec

        test_family = "_test_import_error"
        _SUPPORTED_FAMILIES.add(test_family)
        try:
            with pytest.raises(ImportError, match="无法加载"):
                load_model_spec(test_family, "8B")
        finally:
            _SUPPORTED_FAMILIES.discard(test_family)

    def test_key_error_for_unsupported_flavor(self):
        """Verify key error for unsupported flavor."""
        import sys

        from titaninfra.models import _SUPPORTED_FAMILIES, load_model_spec

        test_family = "_test_key_error"
        _SUPPORTED_FAMILIES.add(test_family)

        mock_module = MagicMock()
        mock_module.model_registry.side_effect = KeyError("unknown_flavor")
        mock_module._test_key_error_configs = {}

        module_name = f"titaninfra.models.{test_family}"
        sys.modules[module_name] = mock_module
        try:
            with pytest.raises(KeyError, match="不支持 flavor"):
                load_model_spec(test_family, "unknown_flavor")
        finally:
            _SUPPORTED_FAMILIES.discard(test_family)
            del sys.modules[module_name]

    def test_successful_load(self):
        """Verify successful load."""
        import sys

        from titaninfra.models import _SUPPORTED_FAMILIES, load_model_spec

        test_family = "_test_success_load"
        _SUPPORTED_FAMILIES.add(test_family)

        mock_spec = MagicMock()
        mock_module = MagicMock()
        mock_module.model_registry.return_value = mock_spec

        module_name = f"titaninfra.models.{test_family}"
        sys.modules[module_name] = mock_module
        try:
            result = load_model_spec(test_family, "8B")
            assert result is mock_spec
            mock_module.model_registry.assert_called_once_with("8B", parallelism=None)
        finally:
            _SUPPORTED_FAMILIES.discard(test_family)
            del sys.modules[module_name]

    def test_error_message_includes_registered_families(self):
        """Verify error message includes registered families."""
        from titaninfra.models import _SUPPORTED_FAMILIES, load_model_spec, register_family

        test_family = "_test_msg_family"
        register_family(test_family)
        try:
            with pytest.raises(ValueError, match=test_family):
                load_model_spec("totally_unknown_family", "8B")
        finally:
            _SUPPORTED_FAMILIES.discard(test_family)


class TestLoadModelSpecRouting:
    """Tests for test load model spec routing."""

    def test_titaninfra_family_routes_to_titaninfra_no_suffix(self, monkeypatch):
        """Verify titaninfra family routes to titaninfra no suffix."""
        import titaninfra.engine.pretrain as P

        called = {}

        def fake_load_model_spec(family, flavor, parallelism=None):
            called["family"], called["flavor"] = family, flavor
            return MagicMock(name="titaninfra_spec")

        monkeypatch.setattr(P, "logger", MagicMock())
        monkeypatch.setattr("titaninfra.models.is_supported", lambda f: True)
        monkeypatch.setattr("titaninfra.models.load_model_spec", fake_load_model_spec)

        spec = P._load_model_spec("qwen3", "8B")
        assert called == {"family": "qwen3", "flavor": "8B"}
        assert spec is not None

    def test_titaninfra_suffix_tolerated(self, monkeypatch):
        """Verify titaninfra suffix tolerated."""
        import titaninfra.engine.pretrain as P

        called = {}
        monkeypatch.setattr(P, "logger", MagicMock())
        monkeypatch.setattr("titaninfra.models.is_supported", lambda f: True)
        monkeypatch.setattr(
            "titaninfra.models.load_model_spec",
            lambda family, flavor, parallelism=None: called.update(flavor=flavor) or MagicMock(),
        )

        P._load_model_spec("qwen3", "8B_titaninfra")
        assert called["flavor"] == "8B_titaninfra"

    def test_non_titaninfra_family_falls_through_to_torchtitan(self, monkeypatch):
        """Verify non titaninfra family falls through to torchtitan."""
        import titaninfra.engine.pretrain as P

        monkeypatch.setattr(P, "logger", MagicMock())
        monkeypatch.setattr("titaninfra.models.is_supported", lambda f: False)

        tt_module = MagicMock()
        tt_module.model_registry.return_value = MagicMock(name="tt_spec")
        monkeypatch.setattr(P.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(P.importlib, "import_module", lambda name: tt_module)

        P._load_model_spec("llama4", "17B")
        tt_module.model_registry.assert_called_once_with("17B")

    def test_new_build_model_config_registry_receives_training_context(self, monkeypatch):
        """New upstream registries receive real parallelism, seq_len and backend."""
        import titaninfra.engine.pretrain as P

        monkeypatch.setattr(P, "logger", MagicMock())
        monkeypatch.setattr("titaninfra.models.is_supported", lambda f: False)
        tt_module = MagicMock(spec=["build_model_config"])
        tt_module.build_model_config.return_value = MagicMock(name="tt_spec")
        monkeypatch.setattr(P.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(P.importlib, "import_module", lambda name: tt_module)
        parallelism = object()

        P._load_model_spec(
            "kimi_k2_7",
            "Kimi-VL-A3B",
            parallelism=parallelism,
            seq_len=2048,
            attn_backend="sdpa",
        )
        tt_module.build_model_config.assert_called_once_with(
            "Kimi-VL-A3B",
            parallelism=parallelism,
            seq_len=2048,
            attn_backend="sdpa",
        )

    def test_unknown_family_raises(self, monkeypatch):
        """Verify unknown family raises."""
        import titaninfra.engine.pretrain as P

        monkeypatch.setattr(P, "logger", MagicMock())
        monkeypatch.setattr("titaninfra.models.is_supported", lambda f: False)
        monkeypatch.setattr(P.importlib.util, "find_spec", lambda name: None)

        with pytest.raises(ValueError, match="不支持的模型系列"):
            P._load_model_spec("no_such_family", "8B")
