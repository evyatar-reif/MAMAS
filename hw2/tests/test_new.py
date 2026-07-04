"""
New tests added by Evyatar Reif
"""

from __future__ import annotations

from src.lab_utils import evaluate_predictor


def _evaluate_trace(predictor_module, traces_dir, trace_name, cfg):
    """Run one .trace file through the predictor and return the summary."""
    trace_file = str(traces_dir / trace_name)
    return evaluate_predictor(cfg, trace_file, predictor_module)


def test_nested_loop_accuracy(predictor_module, traces_dir):
    """A local-local predictor should do better than chance on the nested-loop trace."""
    cfg = predictor_module.PredictorConfig(
        history_size=4,
        history_table_size=16,
        history_mode="local",
        table_mode="local",
        share=False,
    )

    results = _evaluate_trace(predictor_module, traces_dir, "nested_loop.trace", cfg)
    print(f"Nested loop accuracy: {results['accuracy']}")

    assert results["accuracy"] > 0.20, "Predictor failed to learn the nested loop pattern."


def test_always_taken_trace_accuracy(predictor_module, traces_dir):
    """A very simple taken-heavy trace should achieve high accuracy quickly."""
    cfg = predictor_module.PredictorConfig(
        history_size=4,
        history_table_size=16,
        history_mode="global",
        table_mode="global",
        share=False,
    )

    results = _evaluate_trace(predictor_module, traces_dir, "always_taken.trace", cfg)
    print(f"Always-taken accuracy: {results['accuracy']}")

    assert results["accuracy"] >= 0.70, "The predictor should quickly learn an always-taken branch."


def test_alternating_trace_accuracy(predictor_module, traces_dir):
    """A simple alternating pattern should still be learned well with history."""
    cfg = predictor_module.PredictorConfig(
        history_size=2,
        history_table_size=16,
        history_mode="global",
        table_mode="global",
        share=False,
    )

    results = _evaluate_trace(predictor_module, traces_dir, "alternating.trace", cfg)
    print(f"Alternating accuracy: {results['accuracy']}")

    assert results["accuracy"] >= 0.85, "The predictor should learn the alternating pattern well."


def test_loop_pattern_trace_accuracy(predictor_module, traces_dir):
    """A repeated loop pattern should be predictable with enough history."""
    cfg = predictor_module.PredictorConfig(
        history_size=4,
        history_table_size=16,
        history_mode="global",
        table_mode="global",
        share=False,
    )

    results = _evaluate_trace(predictor_module, traces_dir, "loop_pattern.trace", cfg)
    print(f"Loop pattern accuracy: {results['accuracy']}")

    assert results["accuracy"] >= 0.80, "The predictor should learn the repeating loop pattern."


def test_private_patterns_favor_local_history(predictor_module, traces_dir):
    """Interleaved private patterns should be easier for a local predictor to track."""
    global_cfg = predictor_module.PredictorConfig(
        history_size=4,
        history_table_size=16,
        history_mode="global",
        table_mode="global",
        share=False,
    )
    local_cfg = predictor_module.PredictorConfig(
        history_size=4,
        history_table_size=16,
        history_mode="local",
        table_mode="local",
        share=False,
    )

    global_results = _evaluate_trace(predictor_module, traces_dir, "private_patterns.trace", global_cfg)
    local_results = _evaluate_trace(predictor_module, traces_dir, "private_patterns.trace", local_cfg)
    print(
        "Private patterns accuracy: global="
        f"{global_results['accuracy']}, local={local_results['accuracy']}"
    )

    assert local_results["accuracy"] >= 0.85, "The local predictor should do well on private branch patterns."
    assert local_results["accuracy"] > global_results["accuracy"], (
        "The local-history predictor should beat a shared global-history predictor on this trace."
    )