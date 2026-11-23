"""Runtime sequence packing and document-boundary handling.

PackedStream supports hard truncation, sentence boundaries, full samples, and
no-pack operation. positions resets to zero at document starts and -1 denotes
padding; these positions drive FlexAttention document masks."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple


class MMDoc(NamedTuple):
    """Mmdoc configuration and behavior."""

    tokens: list[tuple[int, int]]
    images: tuple[Any, list[tuple[int, int, int]]]


class PackMode(Enum):
    """Pack mode configuration and behavior."""
    NONE = "none"
    SIMPLE = "simple"
    SFT = "sft"
    PRETRAIN = "pretrain"


@dataclass
class PackedSample:
    """Packed sample configuration and behavior."""
    input_ids: list[int]
    labels: list[int]
    positions: list[int] | None = None
    loss_weights: list[float] | None = None
    pixel_values: Any | None = None
    grid_thw: list[tuple[int, int, int]] | None = None

    def to_dict(self) -> dict:
        """To dict."""
        result = {
            "input_ids": self.input_ids,
            "labels": self.labels,
        }
        if self.positions is not None:
            result["positions"] = self.positions
        if self.pixel_values is not None:
            result["pixel_values"] = self.pixel_values
        if self.grid_thw is not None:
            result["grid_thw"] = self.grid_thw
        return result


class SamplePacker:
    """Sample packer configuration and behavior."""

    def __init__(self, seq_len: int, mode: PackMode = PackMode.NONE):
        """Init."""
        self.seq_len = seq_len
        self.mode = mode

    def pack(self, samples: list[dict]) -> list[dict]:
        """Pack."""
        if self.mode == PackMode.NONE:
            return samples
        elif self.mode == PackMode.SIMPLE:
            return self._pack_simple(samples)
        elif self.mode == PackMode.SFT:
            return self._pack_sft(samples)
        else:
            raise ValueError(f"未知的 Packing 模式: {self.mode}")

    def _pack_simple(self, samples: list[dict]) -> list[dict]:
        """Pack simple."""
        if not samples:
            return []

        packed_samples = []
        current_input_ids: list[int] = []
        current_labels: list[int] = []
        current_positions: list[int] = []

        for sample in samples:
            sample_input_ids = sample["input_ids"]
            sample_labels = sample["labels"]
            sample_len = len(sample_input_ids)

            sample_positions = list(range(sample_len))

            if not current_input_ids:
                if sample_len > self.seq_len:
                    packed_samples.append({
                        "input_ids": sample_input_ids[:self.seq_len],
                        "labels": sample_labels[:self.seq_len],
                        "positions": list(range(self.seq_len)),
                    })
                else:
                    current_input_ids = sample_input_ids.copy()
                    current_labels = sample_labels.copy()
                    current_positions = sample_positions.copy()
                continue

            if len(current_input_ids) + sample_len <= self.seq_len:
                current_input_ids.extend(sample_input_ids)
                current_labels.extend(sample_labels)
                current_positions.extend(sample_positions)
            else:
                packed_samples.append({
                    "input_ids": current_input_ids,
                    "labels": current_labels,
                    "positions": current_positions,
                })

                if sample_len > self.seq_len:
                    packed_samples.append({
                        "input_ids": sample_input_ids[:self.seq_len],
                        "labels": sample_labels[:self.seq_len],
                        "positions": list(range(self.seq_len)),
                    })
                    current_input_ids = []
                    current_labels = []
                    current_positions = []
                else:
                    current_input_ids = sample_input_ids.copy()
                    current_labels = sample_labels.copy()
                    current_positions = sample_positions.copy()

        if current_input_ids:
            packed_samples.append({
                "input_ids": current_input_ids,
                "labels": current_labels,
                "positions": current_positions,
            })

        return packed_samples

    def _pack_sft(self, samples: list[dict]) -> list[dict]:
        """Pack sft."""
        if not samples:
            return []

        packed_samples = []
        current_input_ids: list[int] = []
        current_labels: list[int] = []
        current_positions: list[int] = []

        for sample in samples:
            sample_input_ids = sample["input_ids"]
            sample_labels = sample["labels"]
            sample_len = len(sample_input_ids)

            sample_positions = list(range(sample_len))

            if sample_len > self.seq_len:
                if current_input_ids:
                    packed_samples.append({
                        "input_ids": current_input_ids,
                        "labels": current_labels,
                        "positions": current_positions,
                    })
                    current_input_ids = []
                    current_labels = []
                    current_positions = []

                for i in range(0, sample_len, self.seq_len):
                    chunk_end = min(i + self.seq_len, sample_len)
                    chunk_input_ids = sample_input_ids[i:chunk_end]
                    chunk_labels = sample_labels[i:chunk_end]
                    chunk_positions = list(range(chunk_end - i))

                    packed_samples.append({
                        "input_ids": chunk_input_ids,
                        "labels": chunk_labels,
                        "positions": chunk_positions,
                    })

            elif not current_input_ids:
                current_input_ids = sample_input_ids.copy()
                current_labels = sample_labels.copy()
                current_positions = sample_positions.copy()

            elif len(current_input_ids) + sample_len <= self.seq_len:
                current_input_ids.extend(sample_input_ids)
                current_labels.extend(sample_labels)
                current_positions.extend(sample_positions)

            else:
                packed_samples.append({
                    "input_ids": current_input_ids,
                    "labels": current_labels,
                    "positions": current_positions,
                })

                current_input_ids = sample_input_ids.copy()
                current_labels = sample_labels.copy()
                current_positions = sample_positions.copy()

        if current_input_ids:
            packed_samples.append({
                "input_ids": current_input_ids,
                "labels": current_labels,
                "positions": current_positions,
            })

        return packed_samples


class StreamingPacker:
    """Streaming packer configuration and behavior."""

    def __init__(
        self,
        seq_len: int,
        mode: PackMode = PackMode.SIMPLE,
        sentence_end_tokens: list[int] | None = None,
    ):
        """Init."""
        self.seq_len = seq_len
        self.mode = mode

        self.sentence_end_tokens = sentence_end_tokens or []

        self._buffer: list[tuple[int, int]] = []
        self._weight_buffer: list[float] | None = None
        self._position_buffer: list[int] = []
        self._doc_boundaries: list[int] = []
        self._image_buffer: list[tuple[int, Any, list[tuple[int, int, int]]]] = []
        self._total_tokens_seen: int = 0
        self._no_pack_ready: list[PackedSample] = []

    def reset(self) -> None:
        """Reset."""
        self._buffer = []
        self._weight_buffer = None
        self._position_buffer = []
        self._doc_boundaries = []
        self._image_buffer = []
        self._no_pack_ready = []

    @staticmethod
    def _renumber_positions(positions: list[int]) -> list[int]:
        """Renumber positions so each document fragment restarts from 0.

        After a buffer cut, a document's tail may have positions like
        [2049, 2050, ..., 3696, 0, 1, ..., 400].  The first fragment
        would exceed the RoPE cache (sized to seq_len).  This detects
        document boundaries (position decreases) and renumbers each
        fragment starting from 0.
        """
        if not positions:
            return positions
        result = [0] * len(positions)
        counter = 0
        for i, pos in enumerate(positions):
            if i > 0 and pos <= positions[i - 1]:
                counter = 0
            result[i] = counter
            counter += 1
        return result

    def _pop_images_for_chunk(
        self, end: int
    ) -> tuple[Any, list[tuple[int, int, int]] | None]:
        """Pop images for chunk."""
        if not self._image_buffer:
            return None, None

        taken: list[tuple[Any, list[tuple[int, int, int]]]] = []
        remaining: list[tuple[int, Any, list[tuple[int, int, int]]]] = []
        for offset, pv, gthw in self._image_buffer:
            if offset <= end:
                taken.append((pv, gthw))
            else:
                remaining.append((offset - end, pv, gthw))
        self._image_buffer = remaining

        if not taken:
            return None, None

        import torch

        pixel_values = torch.cat([pv for pv, _ in taken], dim=0)
        grid_thw: list[tuple[int, int, int]] = []
        for _, gthw in taken:
            grid_thw.extend(gthw)
        return pixel_values, grid_thw

    def add_document(
        self,
        tokens: list[tuple[int, int]] | list[tuple[int, int, float]],
        images: "tuple[Any, list[tuple[int, int, int]]] | None" = None,
    ) -> None:
        """Add document."""
        if not tokens:
            return

        has_weights = len(tokens[0]) == 3
        if has_weights:
            weights = [t[2] for t in tokens]
            tokens_2 = [(t[0], t[1]) for t in tokens]
            if self._weight_buffer is None:
                self._weight_buffer = [
                    0.0 if lbl == -100 else 1.0 for _, lbl in self._buffer
                ]
            self._weight_buffer.extend(weights)
            tokens = tokens_2
        elif self._weight_buffer is not None:
            self._weight_buffer.extend(
                0.0 if lbl == -100 else 1.0 for _, lbl in tokens
            )

        if self.mode == PackMode.NONE:
            self._add_document_no_pack(tokens)
        elif self.mode == PackMode.SFT:
            self._add_document_sft(tokens)
        else:
            self._buffer.extend(tokens)
            positions = list(range(len(tokens)))
            self._position_buffer.extend(positions)

        if images is not None:
            pixel_values, grid_thw = images
            self._image_buffer.append((len(self._buffer), pixel_values, grid_thw))

    def _add_document_no_pack(self, tokens: list[tuple[int, int]]) -> None:
        """Add document no pack."""
        self._total_tokens_seen += len(tokens)
        truncated = tokens[: self.seq_len + 1]
        input_ids = [t for t, _ in truncated]
        labels = [lbl for _, lbl in truncated]
        positions = list(range(len(truncated)))
        weights = None
        if self._weight_buffer is not None:
            weights = self._weight_buffer[-len(truncated):]
            self._weight_buffer = self._weight_buffer[:-len(tokens)]
        self._no_pack_ready.append(PackedSample(
            input_ids=input_ids, labels=labels, positions=positions,
            loss_weights=weights,
        ))

    def _add_document_sft(self, tokens: list[tuple[int, int]]) -> None:
        """Add document sft."""
        self._buffer.extend(tokens)
        positions = list(range(len(tokens)))
        self._position_buffer.extend(positions)

        self._doc_boundaries.append(len(self._buffer))

    def flush_ready(self) -> Iterator[PackedSample]:
        """Flush ready."""
        if self.mode == PackMode.NONE:
            yield from self._no_pack_ready
            self._no_pack_ready.clear()
        elif self.mode == PackMode.PRETRAIN:
            yield from self._flush_pretrain()
        elif self.mode == PackMode.SFT:
            yield from self._flush_sft()
        else:
            yield from self._flush_simple()

    def flush_all(self) -> Iterator[PackedSample]:
        """Flush all."""
        if self.mode == PackMode.SFT:
            target = self.seq_len + 1
            while self._doc_boundaries:
                yield from self._emit_one_sft_chunk(target)
            return

        yield from self.flush_ready()
        if len(self._buffer) >= 2:
            chunk = self._buffer
            positions_chunk = self._renumber_positions(self._position_buffer)
            weights_chunk = self._weight_buffer if self._weight_buffer is not None else None
            self._buffer = []
            self._position_buffer = []
            self._weight_buffer = None

            input_ids = [t for t, _ in chunk]
            labels = [lbl for _, lbl in chunk]
            self._total_tokens_seen += len(chunk) - 1

            yield PackedSample(
                input_ids=input_ids,
                labels=labels,
                positions=positions_chunk,
                loss_weights=weights_chunk,
            )

    def _flush_simple(self) -> Iterator[PackedSample]:
        """Flush simple."""
        max_buffer_len = self.seq_len + 1

        while len(self._buffer) >= max_buffer_len:
            chunk = self._buffer[:max_buffer_len]
            positions_chunk = self._position_buffer[:max_buffer_len]
            weights_chunk = self._weight_buffer[:max_buffer_len] if self._weight_buffer is not None else None

            self._buffer = self._buffer[max_buffer_len:]
            self._position_buffer = self._renumber_positions(
                self._position_buffer[max_buffer_len:]
            )
            if self._weight_buffer is not None:
                self._weight_buffer = self._weight_buffer[max_buffer_len:]

            input_ids = [t for t, _ in chunk]
            labels = [lbl for _, lbl in chunk]

            self._total_tokens_seen += len(chunk) - 1

            yield PackedSample(
                input_ids=input_ids,
                labels=labels,
                positions=positions_chunk,
                loss_weights=weights_chunk,
            )

    def _flush_sft(self) -> Iterator[PackedSample]:
        """Flush sft."""
        target = self.seq_len + 1
        while len(self._buffer) > target and self._doc_boundaries:
            chunks_before = len(self._doc_boundaries)
            yield from self._emit_one_sft_chunk(target)
            if len(self._doc_boundaries) >= chunks_before:
                break

    def _emit_one_sft_chunk(self, target: int) -> Iterator[PackedSample]:
        """Emit one sft chunk."""
        fit_count = 0
        for boundary in self._doc_boundaries:
            if boundary <= target:
                fit_count += 1
            else:
                break

        if fit_count == 0:
            first_end = self._doc_boundaries[0]
            self._buffer = self._buffer[first_end:]
            self._position_buffer = self._position_buffer[first_end:]
            if self._weight_buffer is not None:
                self._weight_buffer = self._weight_buffer[first_end:]
            self._doc_boundaries = [b - first_end for b in self._doc_boundaries[1:]]
            self._pop_images_for_chunk(first_end)
            return

        end = self._doc_boundaries[fit_count - 1]

        if end < 2:
            self._buffer = self._buffer[end:]
            self._position_buffer = self._position_buffer[end:]
            if self._weight_buffer is not None:
                self._weight_buffer = self._weight_buffer[end:]
            self._doc_boundaries = [b - end for b in self._doc_boundaries[fit_count:]]
            self._pop_images_for_chunk(end)
            return

        chunk = self._buffer[:end]
        positions_chunk = self._position_buffer[:end]
        weights_chunk = self._weight_buffer[:end] if self._weight_buffer is not None else None

        self._buffer = self._buffer[end:]
        self._position_buffer = self._position_buffer[end:]
        if self._weight_buffer is not None:
            self._weight_buffer = self._weight_buffer[end:]
        self._doc_boundaries = [b - end for b in self._doc_boundaries[fit_count:]]

        pixel_values, grid_thw = self._pop_images_for_chunk(end)

        self._total_tokens_seen += len(chunk) - 1

        yield PackedSample(
            input_ids=[t for t, _ in chunk],
            labels=[lbl for _, lbl in chunk],
            positions=positions_chunk,
            loss_weights=weights_chunk,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
        )

    def _flush_pretrain(self) -> Iterator[PackedSample]:
        """Flush pretrain."""
        max_buffer_len = self.seq_len + 1

        while len(self._buffer) >= max_buffer_len:
            if self.sentence_end_tokens:
                split_pos = self._find_sentence_boundary(max_buffer_len)
            else:
                split_pos = max_buffer_len

            chunk = self._buffer[:split_pos]
            positions_chunk = self._position_buffer[:split_pos]
            weights_chunk = self._weight_buffer[:split_pos] if self._weight_buffer is not None else None

            self._buffer = self._buffer[split_pos:]
            self._position_buffer = self._renumber_positions(
                self._position_buffer[split_pos:]
            )
            if self._weight_buffer is not None:
                self._weight_buffer = self._weight_buffer[split_pos:]

            input_ids = [t for t, _ in chunk]
            labels = [lbl for _, lbl in chunk]

            self._total_tokens_seen += len(chunk) - 1

            yield PackedSample(
                input_ids=input_ids,
                labels=labels,
                positions=positions_chunk,
                loss_weights=weights_chunk,
            )

    def _find_document_boundary(self, max_pos: int) -> int:
        """Find document boundary."""
        for i in range(1, min(max_pos, len(self._position_buffer))):
            if self._position_buffer[i] == 0:
                return i

        return max_pos

    def _find_sentence_boundary(self, max_pos: int) -> int:
        """Find sentence boundary."""
        if not self.sentence_end_tokens:
            return max_pos

        for i in range(min(max_pos, len(self._buffer)) - 1, max(0, max_pos - 100), -1):
            token_id = self._buffer[i][0]
            if token_id in self.sentence_end_tokens:
                return i + 1

        return max_pos

    def get_total_tokens_seen(self) -> int:
        """Return the requested value."""
        return self._total_tokens_seen


def strategy_to_pack_mode(strategy: str) -> PackMode:
    """Strategy to pack mode."""
    mapping = {
        "no_pack": PackMode.NONE,
        "full_sample": PackMode.SFT,
        "sentence_boundary": PackMode.PRETRAIN,
        "hard_truncate": PackMode.SIMPLE,
    }
    if strategy not in mapping:
        raise ValueError(
            f"未知的 pack_strategy: '{strategy}'，"
            f"可选值: {list(mapping.keys())}"
        )
    return mapping[strategy]
