"""Tests for gshare/lshare indexing and aliasing behavior."""

from __future__ import annotations


def test_get_pht_index_no_share(predictor_module):
    gi = predictor_module.get_pht_index
    # share=False: result == history (masked to history_size bits).
    assert gi(0x123, 0b0101, 4, False) == 0b0101
    assert gi(0xABCD, 0b1111, 4, False) == 0b1111
    assert gi(0x000, 0b10101, 4, False) == 0b0101  # masked


def test_get_pht_index_with_share(predictor_module):
    gi = predictor_module.get_pht_index
    # share=True: result == history XOR (address & mask).
    assert gi(0b0011, 0b0101, 4, True) == (0b0101 ^ 0b0011)
    assert gi(0xFF, 0b0000, 4, True) == 0b1111
    # Higher-order address bits must be ignored:
    assert gi(0xAB13, 0b0000, 4, True) == 0b0011


def test_gshare_identical_on_aliasing_bad(predictor_module, traces_dir):
    """On aliasing_bad the addresses (0x100, 0x200) share their low 4 bits,
    so gshare's XOR is a no-op and it must produce *exactly* the same
    predictions as plain global-global."""
    from src.lab_utils import evaluate_predictor
    cfg_plain = predictor_module.PredictorConfig(
        history_size=4, history_table_size=16,
        history_mode="global", table_mode="global", share=False)
    cfg_gshare = predictor_module.PredictorConfig(
        history_size=4, history_table_size=16,
        history_mode="global", table_mode="global", share=True)
    bad = str(traces_dir / "aliasing_bad.trace")
    r_plain = evaluate_predictor(cfg_plain, bad, predictor_module)
    r_gshare = evaluate_predictor(cfg_gshare, bad, predictor_module)
    assert r_gshare["predicted"] == r_plain["predicted"]
    assert r_gshare["mispredictions"] == r_plain["mispredictions"]


def test_gshare_helps_aliasing_good(predictor_module, traces_dir):
    """On aliasing_good the low address bits differ, so gshare must clearly
    beat plain global indexing by separating the two branches in the PHT."""
    from src.lab_utils import evaluate_predictor
    cfg_plain = predictor_module.PredictorConfig(
        history_size=4, history_table_size=16,
        history_mode="global", table_mode="global", share=False)
    cfg_gshare = predictor_module.PredictorConfig(
        history_size=4, history_table_size=16,
        history_mode="global", table_mode="global", share=True)
    good = str(traces_dir / "aliasing_good.trace")
    r_plain = evaluate_predictor(cfg_plain, good, predictor_module)
    r_gshare = evaluate_predictor(cfg_gshare, good, predictor_module)
    assert r_gshare["accuracy"] > r_plain["accuracy"]


def test_gshare_helps_mixed_workload(predictor_module, traces_dir):
    """On the mixed workload, gshare should help over plain global indexing."""
    from src.lab_utils import evaluate_predictor
    cfg_plain = predictor_module.PredictorConfig(
        history_size=4, history_table_size=16,
        history_mode="global", table_mode="global", share=False)
    cfg_gshare = predictor_module.PredictorConfig(
        history_size=4, history_table_size=16,
        history_mode="global", table_mode="global", share=True)
    mixed = str(traces_dir / "mixed_workload.trace")
    r_plain = evaluate_predictor(cfg_plain, mixed, predictor_module)
    r_gshare = evaluate_predictor(cfg_gshare, mixed, predictor_module)
    assert r_gshare["accuracy"] > r_plain["accuracy"]
