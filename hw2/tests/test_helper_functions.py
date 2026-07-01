"""
Tests for the helper functions
"""

from __future__ import annotations
import pytest

def test_counter_predict(predictor_module):
    """Test 2-bit counter prediction logic (0/1=False, 2/3=True)."""
    pp = predictor_module.counter_predict
    assert pp(0) is False   # Strongly Not Taken
    assert pp(1) is False   # Weakly Not Taken
    assert pp(2) is True    # Weakly Taken
    assert pp(3) is True    # Strongly Taken

def test_counter_update(predictor_module):
    """Test saturating logic of the 2-bit counter."""
    cu = predictor_module.counter_update
    # Clamping at boundaries
    assert cu(0, False) == 0
    assert cu(3, True) == 3
    # Normal transitions
    assert cu(1, True) == 2
    assert cu(2, True) == 3
    assert cu(2, False) == 1
    assert cu(1, False) == 0

def test_update_history(predictor_module):
    """Test history register shifting and masking."""
    uh = predictor_module.update_history
    assert uh(0b0000, True, 4) == 0b0001
    assert uh(0b0101, True, 4) == 0b1011
    assert uh(0b1111, False, 4) == 0b1110
    assert uh(0b1111, True, 3) == 0b111 # Masks correctly to 3 bits

def test_local_index_and_tag(predictor_module):
    """Test address splitting for local history tables."""
    gi = predictor_module.get_local_index
    gt = predictor_module.get_local_tag
    # Table size 16 = 4 bits for index
    assert gi(0x10A, 16) == 0xA
    assert gt(0x10A, 16) == 0x10
    # Table size 8 = 3 bits for index
    assert gi(0b101011, 8) == 0b011
    assert gt(0b101011, 8) == 0b101

def test_get_pht_index(predictor_module):
    """Test PHT indexing with and without gshare/lshare XORing."""
    gi = predictor_module.get_pht_index
    # share=False -> just masked history
    assert gi(0x123, 0b1101, 4, False) == 0b1101
    # share=True -> history XOR (address & mask)
    assert gi(0b0011, 0b0101, 4, True) == (0b0101 ^ 0b0011)
    assert gi(0xFF, 0b0000, 4, True) == 0b1111

def test_storage_bits(predictor_module):
    """Test hardware storage cost calculation."""
    # Global/Global
    cfg_g = predictor_module.PredictorConfig(
        history_size=4, history_table_size=1, 
        history_mode="global", table_mode="global", share=False
    )
    bp_g = predictor_module.BranchPredictor(cfg_g)
    assert bp_g.storage_bits() == 4 + (16 * 2) # 4-bit GHR + 16 2-bit counters = 36
    
    # Local/Local (4 entries, 2-bit history, 32-bit address)
    cfg_ll = predictor_module.PredictorConfig(
        history_size=2, history_table_size=4, address_bits=32,
        history_mode="local", table_mode="local", share=False
    )
    bp_ll = predictor_module.BranchPredictor(cfg_ll)
    # per entry: 1(valid) + 30(tag) + 2(bhr) + 4*2(pht) = 41. Total = 4 * 41 = 164
    assert bp_ll.storage_bits() == 164
