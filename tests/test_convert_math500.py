"""Unit tests for convert math500."""

import json
import os

import pytest

from scripts.eval.convert_math500 import raw_to_hf_record, raw_to_titaninfra_record

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH500_RAW = os.path.join(REPO_ROOT, "tests/fixtures/datasets/math500.jsonl")

SAMPLE_RAW = {"problem": "What is 2+2?", "solution": "The answer is $\\boxed{4}$.", "answer": "4"}


class TestHfRecord:
    """Tests for test hf record."""

    def test_uses_content_not_blocks(self):
        record = raw_to_hf_record(SAMPLE_RAW)
        for msg in record["messages"]:
            assert "content" in msg
            assert "blocks" not in msg

    def test_user_content_is_problem(self):
        record = raw_to_hf_record(SAMPLE_RAW)
        assert record["messages"][0]["content"] == SAMPLE_RAW["problem"]

    def test_assistant_content_is_solution(self):
        record = raw_to_hf_record(SAMPLE_RAW)
        assert record["messages"][1]["content"] == SAMPLE_RAW["solution"]

    def test_no_extra_keys(self):
        raw = {**SAMPLE_RAW, "subject": "Algebra", "level": 3}
        record = raw_to_hf_record(raw)
        assert set(record.keys()) == {"messages"}


class TestTitanInfraRecord:
    """Tests for test titan infra record."""

    def test_uses_blocks_not_content(self):
        record = raw_to_titaninfra_record(SAMPLE_RAW)
        for msg in record["messages"]:
            assert "blocks" in msg
            assert "content" not in msg

    def test_blocks_are_text_type(self):
        record = raw_to_titaninfra_record(SAMPLE_RAW)
        for msg in record["messages"]:
            assert len(msg["blocks"]) == 1
            assert msg["blocks"][0]["type"] == "text"

    def test_user_text_is_problem(self):
        record = raw_to_titaninfra_record(SAMPLE_RAW)
        assert record["messages"][0]["blocks"][0]["text"] == SAMPLE_RAW["problem"]

    def test_assistant_text_is_solution(self):
        record = raw_to_titaninfra_record(SAMPLE_RAW)
        assert record["messages"][1]["blocks"][0]["text"] == SAMPLE_RAW["solution"]

    def test_no_thinking_blocks(self):
        record = raw_to_titaninfra_record(SAMPLE_RAW)
        for msg in record["messages"]:
            for block in msg["blocks"]:
                assert block["type"] != "thinking"

    def test_no_extra_keys(self):
        raw = {**SAMPLE_RAW, "subject": "Algebra", "level": 3}
        record = raw_to_titaninfra_record(raw)
        assert set(record.keys()) == {"messages"}


class TestFormatAlignment:
    """Tests for test format alignment."""

    def test_same_text_content(self):
        hf = raw_to_hf_record(SAMPLE_RAW)
        jd = raw_to_titaninfra_record(SAMPLE_RAW)

        assert hf["messages"][0]["content"] == jd["messages"][0]["blocks"][0]["text"]
        assert hf["messages"][1]["content"] == jd["messages"][1]["blocks"][0]["text"]

    def test_boxed_preserved_in_both(self):
        raw = {"problem": "P", "solution": "Therefore $\\boxed{\\frac{1}{2}}$.", "answer": "A"}
        hf = raw_to_hf_record(raw)
        jd = raw_to_titaninfra_record(raw)

        assert "\\boxed{" in hf["messages"][1]["content"]
        assert "\\boxed{" in jd["messages"][1]["blocks"][0]["text"]


class TestRealData:
    """Tests for test real data."""

    def test_roundtrip(self):
        if not os.path.exists(MATH500_RAW):
            pytest.skip("math500.jsonl not found")

        with open(MATH500_RAW) as f:
            lines = f.readlines()

        assert len(lines) == 500

        for line in lines[:10]:
            raw = json.loads(line)
            hf = raw_to_hf_record(raw)
            jd = raw_to_titaninfra_record(raw)

            assert hf["messages"][0]["content"] == raw["problem"]
            assert hf["messages"][1]["content"] == raw["solution"]
            assert jd["messages"][0]["blocks"][0]["text"] == raw["problem"]
            assert jd["messages"][1]["blocks"][0]["text"] == raw["solution"]
