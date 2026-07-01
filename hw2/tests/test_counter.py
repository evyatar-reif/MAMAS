"""Tests for the 2-bit saturating counter (counter_predict + counter_update)."""

from __future__ import annotations

import pytest


def test_counter_predict_truth_table(predictor_module):
    pp = predictor_module.counter_predict
    assert pp(0) is False   # SNT
    assert pp(1) is False   # WNT
    assert pp(2) is True    # WT
    assert pp(3) is True    # ST


def test_counter_update_saturates_low(predictor_module):
    cu = predictor_module.counter_update
    # Repeated N from SNT must stay at SNT.
    c = 0
    for _ in range(5):
        c = cu(c, False)
    assert c == 0


def test_counter_update_saturates_high(predictor_module):
    cu = predictor_module.counter_update
    # Repeated T from ST must stay at ST.
    c = 3
    for _ in range(5):
        c = cu(c, True)
    assert c == 3


def test_counter_update_one_step(predictor_module):
    cu = predictor_module.counter_update
    # Single-step transitions cover every starting state.
    assert cu(0, True) == 1
    assert cu(1, True) == 2
    assert cu(2, True) == 3
    assert cu(3, True) == 3
    assert cu(0, False) == 0
    assert cu(1, False) == 0
    assert cu(2, False) == 1
    assert cu(3, False) == 2


def test_counter_update_round_trip(predictor_module):
    """Take a counter on a full T/N round trip and verify state at each step."""
    cu = predictor_module.counter_update
    pp = predictor_module.counter_predict

    # Start at WNT, see four T's then four N's.
    c = 1
    trace = [True] * 4 + [False] * 4
    expected_after = [2, 3, 3, 3, 2, 1, 0, 0]
    for step, taken in enumerate(trace):
        c = cu(c, taken)
        assert c == expected_after[step], (step, c, expected_after[step])

    # And the predictions follow naturally:
    c = 1
    expected_preds = []
    for taken in trace:
        expected_preds.append(pp(c))
        c = cu(c, taken)
    assert expected_preds == [False, True, True, True, True, True, False, False]
