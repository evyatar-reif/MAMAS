"""Tests for the local-history predictor and tag-mismatch behavior."""

from __future__ import annotations


def _make_local_global(predictor_module, history_size=4, hts=16):
    cfg = predictor_module.PredictorConfig(
        history_size=history_size,
        history_table_size=hts,
        history_mode="local",
        table_mode="global",
        share=False,
    )
    return predictor_module.BranchPredictor(cfg)


def _make_local_local(predictor_module, history_size=2, hts=16):
    cfg = predictor_module.PredictorConfig(
        history_size=history_size,
        history_table_size=hts,
        history_mode="local",
        table_mode="local",
        share=False,
    )
    return predictor_module.BranchPredictor(cfg)


def test_local_index_and_tag(predictor_module):
    gi = predictor_module.get_local_index
    gt = predictor_module.get_local_tag
    # history_table_size = 16 -> index = low 4 bits, tag = address >> 4.
    assert gi(0x10A, 16) == 0xA
    assert gt(0x10A, 16) == 0x10
    assert gi(0xABC, 16) == 0xC
    assert gt(0xABC, 16) == 0xAB
    # Power-of-two sanity:
    assert gi(0xFF, 8) == 0x7
    assert gt(0xFF, 8) == 0x1F


def test_local_learns_alternating_per_branch(predictor_module, traces_dir):
    """The local predictor must learn an alternating pattern after warm-up."""
    from src.lab_utils import load_trace, run_trace
    trace = load_trace(str(traces_dir / "alternating.trace"))
    bp = _make_local_global(predictor_module)
    res = run_trace(bp, trace)
    # Tail must be fully correct.
    assert all(res["correct"][-10:])


def test_local_beats_global_on_private_patterns(predictor_module, traces_dir):
    """Two interleaved branches with private periodic patterns: per-branch
    history must clearly outperform a shared global history."""
    from src.lab_utils import evaluate_predictor
    cfg_local = predictor_module.PredictorConfig(
        history_size=4, history_table_size=16,
        history_mode="local", table_mode="global", share=False)
    cfg_global = predictor_module.PredictorConfig(
        history_size=4, history_table_size=16,
        history_mode="global", table_mode="global", share=False)
    trace = str(traces_dir / "private_patterns.trace")
    r_local = evaluate_predictor(cfg_local, trace, predictor_module)
    r_global = evaluate_predictor(cfg_global, trace, predictor_module)
    assert r_local["accuracy"] > r_global["accuracy"] + 0.10
    # After warm-up the local predictor must track both private patterns.
    assert all(r_local["correct"][-20:])


def test_tag_mismatch_predicts_not_taken(predictor_module):
    """Two addresses that share the index but differ in the tag must reset.

    history_table_size = 4 -> low 2 bits index. 0x100 and 0x140 both have low
    2 bits == 0, but their tags differ (0x40 vs 0x50).
    """
    bp = _make_local_global(predictor_module, history_size=2, hts=4)

    # Train 0x100 to "always Taken" so its counter is biased toward T.
    for _ in range(8):
        bp.predict_and_update(0x100, True)
    # The predictor should now strongly believe 0x100 is Taken.
    assert bp.predict(0x100) is True

    # Now a new branch at 0x140 lands on the same local entry. The first
    # prediction for 0x140 must default to Not Taken because of the tag miss.
    first_pred = bp.predict(0x140)
    assert first_pred is False


def test_tag_mismatch_resets_entry_on_update(predictor_module):
    """After a tag miss the entry must be reset to a clean state."""
    bp = _make_local_local(predictor_module, history_size=2, hts=4)

    # Bias 0x100 hard towards Taken, training a couple of PHT entries.
    for _ in range(10):
        bp.predict_and_update(0x100, True)

    # Now hit the same direct-mapped entry with a different tag (0x140) and
    # observe that the entry is reset cleanly (BHR=0, fresh PHT).
    pred = bp.predict_and_update(0x140, False)  # tag miss -> defaults to N
    assert pred is False

    idx = predictor_module.get_local_index(0x140, 4)
    entry = bp.local_table[idx]
    expected_tag = predictor_module.get_local_tag(0x140, 4)
    assert entry.tag == expected_tag
    # After one N update, BHR should hold a single 0 bit -> still 0.
    assert entry.bhr == 0
    # In local-local mode the entry has its own PHT. The one counter at the
    # post-reset BHR=0 (pre-update) should have been decremented from WNT(1)
    # by one step (to SNT=0).
    assert entry.pht[0] == 0


def test_local_local_isolation_between_entries(predictor_module):
    """Two different local entries must not pollute each other."""
    bp = _make_local_local(predictor_module, history_size=2, hts=16)
    # Train 0x100 strongly Taken, 0x108 strongly Not Taken.
    for _ in range(8):
        bp.predict_and_update(0x100, True)
        bp.predict_and_update(0x108, False)

    # After warm-up each branch should predict its own bias.
    assert bp.predict(0x100) is True
    assert bp.predict(0x108) is False
