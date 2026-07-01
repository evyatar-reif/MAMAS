"""Tests for the global-history / global-table predictor configuration."""

from __future__ import annotations

import pytest


def _make_global_predictor(predictor_module, history_size=4):
    cfg = predictor_module.PredictorConfig(
        history_size=history_size,
        history_table_size=1,
        history_mode="global",
        table_mode="global",
        share=False,
    )
    return predictor_module.BranchPredictor(cfg)


def test_update_history_basic(predictor_module):
    uh = predictor_module.update_history
    assert uh(0b0000, True, 4) == 0b0001
    assert uh(0b0001, True, 4) == 0b0011
    assert uh(0b1111, True, 4) == 0b1111  # masked to 4 bits
    assert uh(0b1111, False, 4) == 0b1110
    assert uh(0b1010, False, 4) == 0b0100


def test_global_predicts_always_taken(predictor_module, traces_dir):
    """After a short warm-up the predictor should reach perfect accuracy."""
    from src.lab_utils import load_trace, run_trace
    trace = load_trace(str(traces_dir / "always_taken.trace"))
    bp = _make_global_predictor(predictor_module)
    res = run_trace(bp, trace)
    # Warm-up requires one miss per distinct GHR state encountered before the
    # PHT can predict T. With history_size=4 the predictor visits at most 5
    # distinct GHRs before stabilising on GHR=0xF.
    assert res["mispredictions"] <= 5
    # After the warm-up phase every prediction must be Taken.
    assert all(res["predicted"][-10:])


def test_global_learns_loop_pattern(predictor_module, traces_dir):
    """With history_size=4 the global predictor must learn the T T T T N loop."""
    from src.lab_utils import load_trace, run_trace
    cfg = predictor_module.PredictorConfig(
        history_size=4,
        history_table_size=1,
        history_mode="global",
        table_mode="global",
        share=False,
    )
    bp = predictor_module.BranchPredictor(cfg)
    trace = load_trace(str(traces_dir / "loop_pattern.trace"))
    res = run_trace(bp, trace)
    # After the first iteration or two, every prediction must be correct.
    tail = res["correct"][-15:]
    assert all(tail), f"tail had mispredictions: {tail}"


def test_predict_does_not_modify_state(predictor_module):
    """``predict`` must be a pure observer; only ``update`` mutates state."""
    bp = _make_global_predictor(predictor_module)
    bp.predict_and_update(0x100, True)  # warm up one step
    snapshot_ghr = bp.ghr
    snapshot_pht = list(bp.global_pht)
    for _ in range(5):
        bp.predict(0x100)
    assert bp.ghr == snapshot_ghr
    assert list(bp.global_pht) == snapshot_pht


def test_predict_before_update_order(predictor_module):
    """The prediction returned by predict_and_update is made *before* the update.

    Concretely: feed one T, then check that the second prediction uses the
    counter state from after a single increment (WNT->WT), not after two.
    """
    bp = _make_global_predictor(predictor_module, history_size=1)
    # GHR starts at 0, PHT[0]=WNT(1).
    p1 = bp.predict_and_update(0x100, True)
    assert p1 is False  # initial counter is WNT -> predicts N
    # After one update: PHT[0]=WT(2). GHR=1 -> PHT[1]=WNT(1) is what next reads.
    p2 = bp.predict_and_update(0x100, True)
    assert p2 is False  # PHT[1] is still WNT -> N (predict before update)


def test_storage_bits_global(predictor_module):
    bp = _make_global_predictor(predictor_module, history_size=4)
    # history_size + 2 * 2^history_size = 4 + 32 = 36.
    assert bp.storage_bits() == 36
