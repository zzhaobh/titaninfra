"""Unit tests for mm collate."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from titaninfra.data.multimodal.mm_collate import MMCollator  # noqa: E402

_VOCAB = {
    "<|image_pad|>": 248056,
    "<|video_pad|>": 248057,
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|endoftext|>": 151643,
}


class _MockTok:
    def get_vocab(self):
        return dict(_VOCAB)


def _sample(input_ids, *, pixel_values=None, grid_thw=None, positions=None):
    """Sample."""
    out = {"input": torch.LongTensor(input_ids)}
    if positions is not None:
        out["positions"] = torch.LongTensor(positions)
    if pixel_values is not None:
        out["pixel_values"] = pixel_values
        out["grid_thw"] = grid_thw
    labels = torch.LongTensor(input_ids)
    return out, labels


def _make_collator(batch, **kwargs):
    """Make collator."""
    total = sum(len(b[0]["input"]) for b in batch)
    kwargs.setdefault("num_tokens_per_microbatch", total)
    kwargs.setdefault("max_context_length", max(len(b[0]["input"]) for b in batch))
    kwargs.setdefault("pad_id", 151643)
    return MMCollator(_MockTok(), **kwargs)


def test_pure_text_batch_cats_to_flat_microbatch():
    """Verify pure text batch cats to flat microbatch."""
    batch = [_sample([1, 2, 3]), _sample([4, 5, 6])]
    collate = _make_collator(batch)
    result = collate(batch)
    assert result.input.shape == (6,)
    assert result.input.tolist() == [1, 2, 3, 4, 5, 6]
    assert "pixel_values" not in result.model_kwargs
    assert "mrope_positions" not in result.model_kwargs
    assert result.model_kwargs["special_tokens"]["image_id"] == 248056
    assert result.labels.shape == (6,)
    assert result.labels.tolist() == [1, 2, 3, 4, 5, 6]
    assert result.loss_token_counts.item() == 6
    assert not result.padding_mask.any()


def test_mm_batch_pads_tail_when_fetcher_short():
    """Verify mm batch pads tail when fetcher short."""
    batch = [_sample([1, 2, 3])]
    collate = _make_collator(batch, num_tokens_per_microbatch=9, max_context_length=3)
    result = collate(batch)
    assert result.input.shape == (9,)
    assert result.input[:3].tolist() == [1, 2, 3]
    assert torch.all(result.input[3:] == 151643)  # pad_id
    assert result.padding_mask[:3].tolist() == [False, False, False]
    assert result.padding_mask[3:].all()
    assert result.loss_token_counts.item() == 3


def test_mm_batch_cats_images_flat_and_injects_special_tokens():
    """Verify mm batch cats images flat and injects special tokens."""
    ids1 = [248053, 248056, 248056, 248056, 248056, 248054, 7] + [151643] * 5
    ids2 = [248053] + [248056] * 8 + [248054, 9, 151643]
    s1 = _sample(ids1, pixel_values=torch.ones(4, 8), grid_thw=[(1, 2, 2)])
    s2 = _sample(ids2, pixel_values=torch.ones(8, 8) * 3, grid_thw=[(1, 2, 4)])
    batch = [s1, s2]
    collate = _make_collator(batch)
    result = collate(batch)
    assert result.input.shape == (24,)
    pv = result.model_kwargs["pixel_values"]
    assert pv.shape == (12, 8)
    assert torch.all(pv[:4] == 1.0)
    assert torch.all(pv[4:] == 3.0)
    gthw = result.model_kwargs["grid_thw"]
    assert gthw.shape == (2, 3)
    assert gthw.tolist() == [[1, 2, 2], [1, 2, 4]]
    expected_total = sum(t * h * w for t, h, w in gthw.tolist())
    assert pv.shape[0] == expected_total
    assert result.model_kwargs["special_tokens"]["image_id"] == 248056
    assert result.model_kwargs["special_tokens"]["video_id"] == 248057


def test_mm_collate_slices_multi_image_sample_back_per_image():
    """Verify mm collate slices multi image sample back per image."""
    pv = torch.cat([torch.ones(2, 8), torch.ones(4, 8) * 5], dim=0)
    s = _sample([248053, 248056, 248056, 248054, 248053] + [248056] * 4 + [248054],
                pixel_values=pv, grid_thw=[(1, 1, 2), (1, 2, 2)])
    batch = [s]
    collate = _make_collator(batch)
    result = collate(batch)
    out_pv = result.model_kwargs["pixel_values"]
    assert out_pv.shape == (6, 8)
    assert torch.all(out_pv[:2] == 1.0)
    assert torch.all(out_pv[2:6] == 5.0)
    assert result.model_kwargs["grid_thw"].tolist() == [[1, 1, 2], [1, 2, 2]]


@pytest.mark.xdist_group("torchtitan_import")
def test_mrope_positions_built_when_torchtitan_available():
    """Verify mrope positions built when torchtitan available."""
    pytest.importorskip("torchtitan.components.data.multimodal.mm_collator")
    s = _sample([248053, 248056, 248056, 248056, 248056, 248054, 7],
                pixel_values=torch.ones(4, 8), grid_thw=[(1, 2, 2)],
                positions=[0, 1, 2, 3, 4, 5, 6])
    batch = [s]
    collate = _make_collator(batch)
    result = collate(batch)
    assert "mrope_positions" in result.model_kwargs
    assert result.model_kwargs["mrope_positions"].shape == (7, 3)


@pytest.mark.xdist_group("torchtitan_import")
def test_mrope_positions_cat_across_samples_not_stacked():
    """Verify mrope positions cat across samples not stacked."""
    pytest.importorskip("torchtitan.components.data.multimodal.mm_collator")
    ids1 = [248053, 248056, 248056, 248056, 248056, 248054, 7]
    ids2 = [248053, 248056, 248056, 248054, 8, 9, 151643]
    s1 = _sample(ids1, pixel_values=torch.ones(4, 8), grid_thw=[(1, 2, 2)],
                 positions=[0, 1, 2, 3, 4, 5, 6])
    s2 = _sample(ids2, pixel_values=torch.ones(2, 8) * 2, grid_thw=[(1, 1, 2)],
                 positions=[0, 1, 2, 3, 4, 5, 6])
    batch = [s1, s2]
    collate = _make_collator(batch)
    result = collate(batch)
    mrope = result.model_kwargs["mrope_positions"]
    assert mrope.shape == (14, 3)
    assert result.model_kwargs["grid_thw"].tolist() == [[1, 2, 2], [1, 1, 2]]


def test_mm_batch_patch_count_matches_grid_thw_sum_multi_sample():
    """Verify mm batch patch count matches grid thw sum multi sample."""
    pv1 = torch.cat([torch.ones(2, 8), torch.ones(4, 8) * 2], dim=0)
    ids1 = (
        [248053, 248056, 248056, 248054]
        + [248053] + [248056] * 4 + [248054]
        + [7]
    )
    s1 = _sample(ids1, pixel_values=pv1, grid_thw=[(1, 1, 2), (1, 1, 4)])

    pv2 = torch.ones(4, 8) * 3
    ids2 = [248053] + [248056] * 4 + [248054, 9]
    s2 = _sample(ids2, pixel_values=pv2, grid_thw=[(1, 2, 2)])

    batch = [s1, s2]
    collate = _make_collator(batch)
    result = collate(batch)
    pv_out = result.model_kwargs["pixel_values"]
    gthw_out = result.model_kwargs["grid_thw"]
    expected_total = sum(t * h * w for t, h, w in gthw_out.tolist())
    assert pv_out.shape[0] == expected_total == 10  # 2+4+4
    assert gthw_out.tolist() == [[1, 1, 2], [1, 1, 4], [1, 2, 2]]
