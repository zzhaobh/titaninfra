# Multimodal test fixture — datatrove-schema LanceDB

Small, self-contained LanceDB that mirrors the **source format the multimodal
team writes** (datatrove: one row per document). Feeds development of the
TitanInfra multimodal converter (module 1, `docs/LanceDB_multimod_streaming.md §八`).

## Build / rebuild

```bash
python tests/fixtures/multimodal/build_fixture.py                       # defaults: 15 caption + 8 interleaved
python tests/fixtures/multimodal/build_fixture.py --n-caption 20 --n-interleaved 10
```

Real images stream from the flickr30k parquet (first N rows only, not a full
download). Synthetic images reuse `tests/fixtures/images/*`. If the network is
unavailable the builder degrades to synthetic-only and still emits a valid
fixture.

## Schema (6 columns, one row = one document)

| column | type | content |
|--------|------|---------|
| `id` | str | document id (`caption_*` / `interleaved_*` / `synthetic_*` / `dirty_*`) |
| `text` | str | raw text, no image placeholders |
| `metadata` | str (JSON) | holds `mml_text` (with `[IMG_i]` markers) + provenance |
| `media` | list<struct> | per-image `media_bytes` (binary) / `position_in_text` (int32) / `url` (str) |
| `ctype` | str | `caption` \| `interleaved` \| `text` |
| `image_sizes` | list<struct> | per-image `width` / `height` (int32), aligned with `media` |

**Interleaving contract**: `mml_text` carries `[IMG_i]` markers; `media[i]`
supplies the bytes and `position_in_text` (char offset of `[IMG_i]` in
`mml_text`). Marker count == `len(media)` == `len(image_sizes)`.

## Composition (~32 rows)

| group | count | purpose |
|-------|-------|---------|
| `caption_*` | ~15 | single real image + one caption — loss-mask boundary (image weight=0, caption weight=1) |
| `interleaved_*` | ~8 | two real images woven into one `mml_text` — `[IMG_i]` / `position_in_text` restoration |
| `synthetic_*` | 5 | committed synthetic images, extreme/non-32-multiple sizes — smart_resize / est_tokens edge |
| `interleaved_syn` | 1 | synthetic interleaved — guarantees interleaved coverage offline |
| `dirty_*` | 3 | **invariant #1 negatives** — converter fail-loud gate must filter/backfill |

The 3 dirty rows: `dirty_empty_bytes` (empty `media_bytes`),
`dirty_zero_size` (undecodable bytes, `image_sizes={0,0}`),
`dirty_url_only` (URL present, bytes never fetched).

## What it validates

- **Interleaving restoration**: `mml_text` + `[IMG_i]` + `position_in_text` → TitanInfra blocks.
- **est_tokens correctness**: `image_sizes` drives offline `N = f(w, h, ViT)`; 21/23 sizes are non-32-multiples, exercising `smart_resize` rounding.
- **Image-bytes inlining**: real jpeg bytes decode; recorded `image_sizes` match decoded dims (converter can trust the column).
- **Invariant #1 gate**: the 3 dirty rows are all detectable via empty-bytes or `{0,0}` size.

## Notes

- Requires `pylance` (in `requirements-dev.txt` — the team stores source MM data in Lance).
- `datatrove_sample.lance/` is generated locally by `build_fixture.py` and is not tracked. Pytest bootstrap builds a small synthetic-only fixture with `build(n_caption=0, n_interleaved=0)` when Lance is installed, so roundtrip tests do not require network access.
