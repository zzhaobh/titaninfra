from types import SimpleNamespace

from titaninfra.engine import _pretrain_helpers


def _should_disable():
    fn = getattr(_pretrain_helpers, "_should_disable_accelerator_graphs", None)
    assert fn is not None
    return fn


def test_multimodal_models_disable_accelerator_graphs():
    should_disable = _should_disable()
    model_spec = SimpleNamespace(multimodal_encoder_fqns=("vision_encoder",))
    assert should_disable(model_spec, 1) is True


def test_multimodal_owner_class_disables_accelerator_graphs():
    should_disable = _should_disable()
    model_spec = SimpleNamespace(
        _owner=SimpleNamespace(multimodal_encoder_fqns=("vision_encoder",))
    )
    assert should_disable(model_spec, 1) is True


def test_text_only_models_only_disable_accelerator_graphs_for_ep():
    should_disable = _should_disable()
    model_spec = SimpleNamespace(multimodal_encoder_fqns=())
    assert should_disable(model_spec, 1) is False
    assert should_disable(model_spec, 2) is True
