"""Unit tests for sft mask position alignment."""

from titaninfra.data.packer import PackMode, StreamingPacker
from titaninfra.data.sft.types import IGNORE_INDEX


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def make_sft_doc(prompt_len: int, resp_len: int, resp_weight: float = 1.0,
                 base: int = 100):
    """Make sft doc."""
    doc = []
    tid = base
    for _ in range(prompt_len):
        doc.append((tid, IGNORE_INDEX, 0.0))
        tid += 1
    for _ in range(resp_len):
        doc.append((tid, tid, resp_weight))
        tid += 1
    return doc


def causal_shift(sample):
    """Causal shift."""
    input_ids = sample.input_ids[:-1]
    labels = sample.labels[1:]
    positions = sample.positions[:-1] if sample.positions is not None else None
    weights = sample.loss_weights[1:] if sample.loss_weights is not None else None
    return input_ids, labels, positions, weights


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class TestBufferLengthAlignment:
    """Tests for test buffer length alignment."""

    def _assert_sample_aligned(self, s):
        n = len(s.input_ids)
        assert len(s.labels) == n, "labels 长度与 input_ids 不等 → mask 漂移"
        assert s.positions is not None and len(s.positions) == n, \
            "positions 长度与 input_ids 不等 → position 漂移"
        assert s.loss_weights is not None and len(s.loss_weights) == n, \
            "loss_weights 长度与 input_ids 不等 → 权重漂移"

    def test_single_doc(self):
        packer = StreamingPacker(seq_len=32, mode=PackMode.SFT)
        packer.add_document(make_sft_doc(3, 4))
        samples = list(packer.flush_all())
        assert len(samples) == 1
        self._assert_sample_aligned(samples[0])

    def test_multi_doc_binpack(self):
        packer = StreamingPacker(seq_len=32, mode=PackMode.SFT)
        packer.add_document(make_sft_doc(2, 3, base=100))
        packer.add_document(make_sft_doc(4, 2, resp_weight=0.5, base=200))
        packer.add_document(make_sft_doc(1, 5, base=300))
        samples = list(packer.flush_all())
        assert samples
        for s in samples:
            self._assert_sample_aligned(s)

    def test_cross_batch_cut(self):
        """Verify cross batch cut."""
        packer = StreamingPacker(seq_len=8, mode=PackMode.SFT)
        for b in (100, 200, 300):
            packer.add_document(make_sft_doc(2, 4, base=b))
        samples = list(packer.flush_all())
        assert len(samples) >= 2, "应跨多个 batch 输出"
        for s in samples:
            self._assert_sample_aligned(s)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class TestMaskFollowsToken:
    """Tests for test mask follows token."""

    def test_response_weighted_prompt_masked(self):
        packer = StreamingPacker(seq_len=64, mode=PackMode.SFT)
        packer.add_document(make_sft_doc(3, 4, resp_weight=1.0, base=100))
        packer.add_document(make_sft_doc(5, 2, resp_weight=0.3, base=200))
        samples = list(packer.flush_all())

        seen_resp = 0
        for s in samples:
            for tid, lbl, w in zip(s.input_ids, s.labels, s.loss_weights, strict=True):
                if lbl == IGNORE_INDEX:
                    assert w == 0.0, f"token {tid}: 被 mask 但 weight={w}（应 0）"
                else:
                    assert lbl == tid, f"token {tid}: label={lbl} 漂移（应==tid）"
                    assert w > 0.0, f"token {tid}: response 但 weight={w}（应>0）"
                    seen_resp += 1
        assert seen_resp == 4 + 2, "response token 数不符 → 有 token 归属漂移"

    def test_thinking_scale_lands_on_right_tokens(self):
        """Verify thinking scale lands on right tokens."""
        packer = StreamingPacker(seq_len=64, mode=PackMode.SFT)
        packer.add_document(make_sft_doc(2, 3, resp_weight=1.0, base=100))   # resp: 102,103,104
        packer.add_document(make_sft_doc(2, 3, resp_weight=0.25, base=200))  # resp: 202,203,204
        samples = list(packer.flush_all())

        weight_by_tid = {}
        for s in samples:
            for tid, lbl, w in zip(s.input_ids, s.labels, s.loss_weights, strict=True):
                if lbl != IGNORE_INDEX:
                    weight_by_tid[tid] = w
        for tid in (102, 103, 104):
            assert weight_by_tid[tid] == 1.0, f"token {tid} 权重漂移: {weight_by_tid.get(tid)}"
        for tid in (202, 203, 204):
            assert weight_by_tid[tid] == 0.25, f"token {tid} 权重漂移: {weight_by_tid.get(tid)}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class TestPositionResetAtDocStart:
    """Tests for test position reset at doc start."""

    def test_positions_reset_only_at_doc_boundary(self):
        packer = StreamingPacker(seq_len=64, mode=PackMode.SFT)
        packer.add_document(make_sft_doc(2, 2, base=100))  # 4 token
        packer.add_document(make_sft_doc(1, 3, base=200))  # 4 token
        packer.add_document(make_sft_doc(3, 1, base=300))  # 4 token
        samples = list(packer.flush_all())

        for s in samples:
            expected = 0
            for i, p in enumerate(s.positions):
                if p == 0:
                    expected = 0
                assert p == expected, (
                    f"index {i}: position={p} 期望 {expected} "
                    f"→ per-doc reset 漂移（positions={s.positions}）"
                )
                expected += 1

    def test_doc_start_token_has_position_zero(self):
        """Verify doc start token has position zero."""
        packer = StreamingPacker(seq_len=64, mode=PackMode.SFT)
        packer.add_document(make_sft_doc(2, 2, base=100))
        packer.add_document(make_sft_doc(2, 2, base=200))
        packer.add_document(make_sft_doc(2, 2, base=300))
        samples = list(packer.flush_all())
        assert len(samples) == 1, "三条短文档应打进同一 sample"
        zero_idx = [i for i, p in enumerate(samples[0].positions) if p == 0]
        assert zero_idx == [0, 4, 8], f"文档边界漂移: positions==0 在 {zero_idx}，应 [0,4,8]"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class TestAlignmentSurvivesCausalShift:
    """Tests for test alignment survives causal shift."""

    def test_shift_preserves_lengths_and_ignore_zero(self):
        packer = StreamingPacker(seq_len=64, mode=PackMode.SFT)
        packer.add_document(make_sft_doc(3, 4, resp_weight=1.0, base=100))
        packer.add_document(make_sft_doc(2, 3, resp_weight=0.5, base=200))
        samples = list(packer.flush_all())

        for s in samples:
            inp, lbl, pos, w = causal_shift(s)
            n = len(inp)
            assert len(lbl) == n and len(pos) == n and len(w) == n, \
                "causal shift 后四者长度不等 → 漂移"
            for i in range(n):
                if lbl[i] == IGNORE_INDEX:
                    assert w[i] == 0.0, f"shift 后 index {i}: label=IGNORE 但 weight={w[i]}"
                else:
                    assert w[i] > 0.0, f"shift 后 index {i}: label!=IGNORE 但 weight={w[i]}"

    def test_shift_target_weight_matches_source_token(self):
        """Verify shift target weight matches source token."""
        packer = StreamingPacker(seq_len=64, mode=PackMode.SFT)
        packer.add_document(make_sft_doc(2, 3, resp_weight=0.7, base=100))
        samples = list(packer.flush_all())
        assert len(samples) == 1
        s = samples[0]
        orig_w = list(s.loss_weights)
        inp, lbl, pos, w = causal_shift(s)
        for i in range(len(w)):
            assert w[i] == orig_w[i + 1], (
                f"index {i}: shift 后 weight={w[i]} 应 == 原 weight[{i+1}]={orig_w[i+1]} "
                "→ loss_weights 用错切片（漂移）"
            )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_two_tuple_docs_positions_still_aligned():
    """Verify two tuple docs positions still aligned."""
    packer = StreamingPacker(seq_len=64, mode=PackMode.SFT)
    packer.add_document([(1, 1), (2, 2), (3, 3)])
    packer.add_document([(4, 4), (5, 5)])
    samples = list(packer.flush_all())
    for s in samples:
        n = len(s.input_ids)
        assert len(s.labels) == n
        assert len(s.positions) == n
        expected = 0
        for p in s.positions:
            if p == 0:
                expected = 0
            assert p == expected
            expected += 1
