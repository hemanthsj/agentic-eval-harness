"""Tests for the deterministic mock message generator."""

from eval_harness.generator import MessageGenerator
from eval_harness.llm import MockProvider
from eval_harness.models import CHANNELS


def test_generate_returns_exactly_n_with_unique_ids():
    gen = MessageGenerator(MockProvider())
    messages = gen.generate(7)

    assert len(messages) == 7
    ids = [m.id for m in messages]
    assert len(set(ids)) == 7  # unique ids
    for m in messages:
        assert m.channel in CHANNELS
        assert m.body  # non-empty body


def test_generate_is_deterministic():
    gen = MessageGenerator(MockProvider())
    a = gen.generate(5, seed=1)
    b = gen.generate(5, seed=1)
    assert [m.to_dict() for m in a] == [m.to_dict() for m in b]


def test_generate_zero():
    gen = MessageGenerator(MockProvider())
    assert gen.generate(0) == []
