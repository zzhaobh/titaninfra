"""Unit tests for build fixture."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FIXTURE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = FIXTURE_DIR.parent / "images"
OUTPUT = FIXTURE_DIR / "datatrove_sample.lance"

FLICKR_PARQUET = (
    "https://huggingface.co/api/datasets/nlphuji/flickr30k/parquet/TEST/test/0.parquet"
)

# ── datatrove 6-column schema (see module docstring) ──────────────────────────
MEDIA_STRUCT = pa.struct(
    [
        ("media_bytes", pa.binary()),
        ("position_in_text", pa.int32()),
        ("url", pa.string()),
    ]
)
SIZE_STRUCT = pa.struct([("width", pa.int32()), ("height", pa.int32())])
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("metadata", pa.string()),
        ("media", pa.list_(MEDIA_STRUCT)),
        ("ctype", pa.string()),
        ("image_sizes", pa.list_(SIZE_STRUCT)),
    ]
)


def _img_size(raw: bytes) -> tuple[int, int]:
    """Decode just enough to read (width, height). Returns (0, 0) on failure."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def _load_synthetic() -> list[tuple[str, bytes, int, int]]:
    """Reuse committed synthetic images for extreme-size coverage."""
    out: list[tuple[str, bytes, int, int]] = []
    for p in sorted(IMAGES_DIR.glob("*.[jp][pn]g")):
        raw = p.read_bytes()
        w, h = _img_size(raw)
        out.append((p.name, raw, w, h))
    return out


def _stream_flickr(n: int) -> list[tuple[str, bytes, list[str]]]:
    """Read the first `n` rows of the flickr30k parquet (streamed, not full DL).

    Returns list of (filename, jpeg_bytes, captions). Empty list if unreachable.
    Zero rows is an explicit offline path and never creates a network client.
    """
    if n <= 0:
        return []
    try:
        import fsspec
    except ImportError:
        print("  fsspec unavailable — skipping real images", file=sys.stderr)
        return []
    try:
        fs = fsspec.filesystem("http")
        pf = pq.ParquetFile(fs.open(FLICKR_PARQUET))
        rows: list[tuple[str, bytes, list[str]]] = []
        for batch in pf.iter_batches(batch_size=min(n, 64)):
            for row in batch.to_pylist():
                rows.append(
                    (row["filename"], row["image"]["bytes"], list(row["caption"]))
                )
                if len(rows) >= n:
                    return rows
        return rows
    except Exception as e:  # network / parquet errors: degrade to synthetic-only
        print(f"  flickr30k unreachable ({e!r}) — skipping real images", file=sys.stderr)
        return []


def _caption_row(rid: str, fname: str, raw: bytes, w: int, h: int, cap: str) -> dict:
    """Single-image caption doc: mml_text = image placeholder then caption."""
    mml = f"[IMG_0] {cap}"
    return {
        "id": rid,
        "text": cap,
        "metadata": json.dumps({"mml_text": mml, "source_file": fname}),
        "media": [{"media_bytes": raw, "position_in_text": 0, "url": ""}],
        "ctype": "caption",
        "image_sizes": [{"width": w, "height": h}],
    }


def _interleaved_row(rid: str, imgs: list[tuple[bytes, int, int]], caps: list[str]) -> dict:
    """Multi-image interleaved doc: text woven with [IMG_i] markers."""
    parts: list[str] = []
    media: list[dict] = []
    sizes: list[dict] = []
    for i, (raw, w, h) in enumerate(imgs):
        prefix = caps[i] if i < len(caps) else "See the image below."
        marker = f"[IMG_{i}]"
        # position_in_text = char offset of the marker in the accumulated mml
        running = " ".join(parts + [prefix]) + " "
        pos = len(running)
        parts.append(prefix)
        parts.append(marker)
        media.append({"media_bytes": raw, "position_in_text": pos, "url": ""})
        sizes.append({"width": w, "height": h})
    mml = " ".join(parts)
    text = " ".join(caps[: len(imgs)])
    return {
        "id": rid,
        "text": text,
        "metadata": json.dumps({"mml_text": mml}),
        "media": media,
        "ctype": "interleaved",
        "image_sizes": sizes,
    }


def _dirty_rows() -> list[dict]:
    """Invariant #1 negatives — converter fail-loud gate must catch these."""
    return [
        {  # empty media_bytes
            "id": "dirty_empty_bytes",
            "text": "caption with a broken image",
            "metadata": json.dumps({"mml_text": "[IMG_0] broken", "defect": "empty_bytes"}),
            "media": [{"media_bytes": b"", "position_in_text": 0, "url": ""}],
            "ctype": "caption",
            "image_sizes": [{"width": 640, "height": 480}],
        },
        {  # decode failed → size sentinel {0,0}
            "id": "dirty_zero_size",
            "text": "caption with undecodable image",
            "metadata": json.dumps({"mml_text": "[IMG_0] undecodable", "defect": "zero_size"}),
            "media": [{"media_bytes": b"\xff\xd8not_a_real_jpeg", "position_in_text": 0, "url": ""}],
            "ctype": "caption",
            "image_sizes": [{"width": 0, "height": 0}],
        },
        {  # URL only, never downloaded
            "id": "dirty_url_only",
            "text": "caption where bytes were never fetched",
            "metadata": json.dumps({"mml_text": "[IMG_0] url only", "defect": "url_only"}),
            "media": [
                {"media_bytes": b"", "position_in_text": 0, "url": "http://example.com/missing.jpg"}
            ],
            "ctype": "caption",
            "image_sizes": [{"width": 0, "height": 0}],
        },
    ]


def build(n_caption: int, n_interleaved: int) -> None:
    import lance

    synthetic = _load_synthetic()
    print(f"synthetic images: {len(synthetic)} ({[s[0] for s in synthetic]})")

    n_real = n_caption + n_interleaved * 2
    real = _stream_flickr(n_real) if n_real > 0 else []
    print(f"flickr30k rows fetched: {len(real)}")

    rows: list[dict] = []

    # ── single-image caption (real images) ──
    for i, (fname, raw, caps) in enumerate(real[:n_caption]):
        w, h = _img_size(raw)
        rows.append(_caption_row(f"caption_{i:03d}", fname, raw, w, h, caps[0]))

    # ── multi-image interleaved (pair up remaining real images) ──
    pool = real[n_caption : n_caption + n_interleaved * 2]
    for i in range(0, len(pool) - 1, 2):
        (_, r0, c0), (_, r1, c1) = pool[i], pool[i + 1]
        imgs = [(r0, *_img_size(r0)), (r1, *_img_size(r1))]
        rows.append(_interleaved_row(f"interleaved_{i // 2:03d}", imgs, [c0[0], c1[0]]))

    # ── synthetic extreme sizes (always present, no network needed) ──
    for i, (fname, raw, w, h) in enumerate(synthetic):
        cap = f"A synthetic {w}x{h} test image ({fname})."
        rows.append(_caption_row(f"synthetic_{i:03d}", fname, raw, w, h, cap))

    # one synthetic interleaved (guarantees interleaved coverage offline)
    if len(synthetic) >= 2:
        imgs = [
            (synthetic[0][1], synthetic[0][2], synthetic[0][3]),
            (synthetic[1][1], synthetic[1][2], synthetic[1][3]),
        ]
        rows.append(
            _interleaved_row(
                "interleaved_syn", imgs, ["First a small patch.", "Then a gradient."]
            )
        )

    # ── dirty rows (invariant #1) ──
    rows.extend(_dirty_rows())

    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    if OUTPUT.exists():
        import shutil

        shutil.rmtree(OUTPUT)
    lance.write_dataset(table, str(OUTPUT))

    ds = lance.dataset(str(OUTPUT))
    by_type: dict[str, int] = {}
    for r in ds.to_table(columns=["ctype"]).to_pylist():
        by_type[r["ctype"]] = by_type.get(r["ctype"], 0) + 1
    print(f"\nwrote {ds.count_rows()} rows to {OUTPUT}")
    print(f"  by ctype: {by_type}")
    print("  dirty rows (invariant #1): 3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-caption", type=int, default=15, help="single-image caption rows")
    ap.add_argument("--n-interleaved", type=int, default=8, help="multi-image interleaved rows")
    args = ap.parse_args()
    build(args.n_caption, args.n_interleaved)


if __name__ == "__main__":
    main()
