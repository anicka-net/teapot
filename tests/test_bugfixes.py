"""Regression tests for correctness bugs found in the 2026-07-03 audit.

Covers the pure-logic fixes (no torch/network needed). The eval-aggregate,
orchestrator-gate, and compose-license fixes are verified separately (they live
inside network/subprocess/fixture-heavy functions); see the audit notes.
"""
from teapot.eval import strip_thinking
from teapot.hardware import estimate_model_memory


# --- strip_thinking: unclosed reasoning traces must not leak to the scorer ---
def test_strip_closed_inner_prefix():
    assert strip_thinking("<|inner_prefix|>secret<|inner_suffix|>answer") == "answer"


def test_strip_closed_think():
    assert strip_thinking("<think>reasoning</think>final answer") == "final answer"


def test_strip_unclosed_inner_prefix_leaks_nothing():
    # truncated generation: opener with no suffix. The whole trace used to leak.
    out = strip_thinking("<|inner_prefix|>how to do the bad thing, step 1")
    assert "bad thing" not in out
    assert out == ""


def test_strip_unclosed_think_leaks_nothing():
    out = strip_thinking("visible. <think>hidden reasoning cut off")
    assert "hidden" not in out
    assert out == "visible."


def test_strip_preserves_content_after_closed_block():
    assert strip_thinking("<think>x</think>The answer is 42.") == "The answer is 42."


def test_strip_noop_without_tags():
    assert strip_thinking("just a normal reply") == "just a normal reply"


# --- estimate_model_memory: "3b" must not match "13b" (word-boundary fix) ---
def test_13b_is_not_sized_as_3b():
    # the bug: substring "3b" in "13b" resolved 13B to the 3B tier (4GB)
    assert estimate_model_memory("Llama-13B", "qlora") == 14
    assert estimate_model_memory("Llama-13B", "qlora") != 4


def test_boundary_sizes_resolve_correctly():
    assert estimate_model_memory("apertus-8b", "qlora") == 8
    assert estimate_model_memory("model-3b", "qlora") == 4
    assert estimate_model_memory("Qwen-14B", "qlora") == 14
    assert estimate_model_memory("apertus-70b", "qlora") == 40


def test_bigger_model_never_estimates_smaller():
    assert (estimate_model_memory("Llama-13B", "qlora")
            > estimate_model_memory("Llama-3B", "qlora"))
